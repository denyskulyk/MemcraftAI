"""`python -m memcraft.worker`: the worker with the real pipeline.

It runs the idea lane (Gemma on the CPU) and the render lane (the GPU) with the app's own
ComfyUI on 127.0.0.1:8199, started through deploy/comfyui.ps1 when it does not answer, and
the app's own CPU llama-server for Gemma on 127.0.0.1:8098, started when a step needs it and
stopped after --gemma-idle-minutes. RAM and VRAM are logged around every step, and before a
GPU step with less than --min-free-ram-gb free RAM, Gemma and ComfyUI's models are unloaded
first (worker/memory.py). A pause frees ComfyUI's models and stops Gemma. The
watchdog restarts only these two. At start the ComfyUI install is checked against
deploy/pins.json; a mismatch is logged, not refused.

`--dry` runs the old dry steps, which only sleep; they would mark real jobs done, so the dry run
refuses any database whose name does not end in _test.

A file named by --stop-file (default data/worker.stop) asks the worker to stop after its current
steps; deploy/worker.ps1 stop uses it.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
from pathlib import Path

from sqlalchemy import Engine, create_engine, text

from memcraft.config import get_settings
from memcraft.models import JobLane
from memcraft.storage import FileStore
from memcraft.worker.loop import Worker, WorkerConfig
from memcraft.worker.schedule import Schedule, default_tasks
from memcraft.worker.steps import dry_registry

PROJECT = Path(__file__).resolve().parents[3]
ANSWER_LIMIT = 330.0
log = logging.getLogger("memcraft.worker")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m memcraft.worker")
    parser.add_argument("--database-url", help="default: MEMCRAFT_DATABASE_URL")
    parser.add_argument("--worker-id")
    parser.add_argument("--dry", action="store_true", help="dry steps that only sleep (*_test)")
    parser.add_argument("--step-seconds", type=float, default=0.5, help="how long a dry step takes")
    parser.add_argument("--once", action="store_true", help="work until the queues are empty")
    parser.add_argument("--no-schedule", action="store_true", help="skip expiry and purges")
    parser.add_argument("--no-pins-check", action="store_true", help="skip the pins check")
    parser.add_argument("--gemma-idle-minutes", type=float, default=5.0)
    parser.add_argument(
        "--gemma",
        choices=("swap", "own"),
        default="swap",
        help="swap: the memcraft-gemma model of llama-swap; own: a llama-server on 8098",
    )
    parser.add_argument(
        "--min-free-ram-gb",
        type=float,
        default=6.0,
        help="before a GPU step, unload Gemma and ComfyUI's models below this much free RAM "
        "(settings.memory_min_free_gb overrides it; 0 turns it off)",
    )
    parser.add_argument("--memory-wait-seconds", type=float, default=180.0)
    parser.add_argument("--stop-file", type=Path, default=PROJECT / "data" / "worker.stop")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(threadName)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # every health probe otherwise

    settings = get_settings()
    engine = create_engine(args.database_url or settings.database_url, pool_pre_ping=True)
    if args.dry:
        return _dry(engine, args, settings)
    return _real(engine, args, settings)


def _database_name(engine: Engine) -> str:
    with engine.connect() as connection:
        return connection.scalar(text("SELECT current_database()")) or ""


def _dry(engine: Engine, args: argparse.Namespace, settings: object) -> int:
    name = _database_name(engine)
    if not name.endswith("_test"):
        print(f"refusing to run dry steps on {name!r}: not a *_test database", file=sys.stderr)
        return 2
    store = None
    shared = settings.shared_dir  # type: ignore[attr-defined]
    if shared.is_dir():
        store = FileStore(shared, settings.private_dir)  # type: ignore[attr-defined]
    schedule = None if args.no_schedule else Schedule(engine, default_tasks(store))
    worker = Worker(
        engine,
        dry_registry(seconds=args.step_seconds),
        store=store,
        worker_id=args.worker_id,
        unload=lambda lane: logging.info("unload %s models (dry)", lane.value),
        schedule=schedule,
        config=WorkerConfig(retry_delay=1.0),
    )
    return _serve(worker, args, schedule, cleanup=lambda: None)


def _real(engine: Engine, args: argparse.Namespace, settings: object) -> int:
    from memcraft.pipeline import build_registry
    from memcraft.pipeline.services import (
        ComfyServer,
        GemmaServer,
        SwapGemma,
        memory_watch,
        stop_all,
    )
    from memcraft.pipeline.steps import Deps
    from memcraft.worker.watchdog import Watchdog, comfy_service, llama_server_service

    shared = Path(settings.shared_dir)  # type: ignore[attr-defined]
    private = settings.private_dir  # type: ignore[attr-defined]
    if private is None or not shared.is_dir() or not Path(private).is_dir():
        print("the worker needs MEMCRAFT_SHARED_DIR and MEMCRAFT_PRIVATE_DIR", file=sys.stderr)
        return 2
    store = FileStore(shared, Path(private))
    log.info("database %s", _database_name(engine))
    comfy = ComfyServer()
    gemma: GemmaServer | SwapGemma = (
        SwapGemma() if args.gemma == "swap" else GemmaServer(idle_minutes=args.gemma_idle_minutes)
    )
    deps = Deps(
        store=store,
        comfy=comfy,
        gemma=gemma,
        work_dir=PROJECT / "data" / "work",
        workflows_dir=PROJECT / "workflows",
    )

    def unload(lane: JobLane) -> None:
        # Each lane unloads only its own model: the render lane pausing must not pull Gemma
        # away from an idea the idea lane is still asking about.
        if lane is JobLane.RENDER:
            comfy.free()
        else:
            gemma.stop()

    # A server that a step starts (ComfyUI up to 5 min, Gemma about 20 s from the RAM cache,
    # longer from disk) does not answer meanwhile, so the answer limit covers a start.
    watchdog = Watchdog(
        [comfy_service(comfy.restart, comfy.base_url)]
        # llama-swap looks after its own servers; the app restarts only its own llama-server.
        + (
            [] if isinstance(gemma, SwapGemma) else [llama_server_service(gemma.restart, gemma.url)]
        ),
        answer_limit=ANSWER_LIMIT,
    )
    schedule = None if args.no_schedule else Schedule(engine, default_tasks(store))
    worker = Worker(
        engine,
        build_registry(deps),
        store=store,
        worker_id=args.worker_id,
        unload=unload,
        watchdog=watchdog,
        schedule=schedule,
        memory=memory_watch(
            comfy,
            gemma,
            min_free_gb=args.min_free_ram_gb,
            wait_seconds=args.memory_wait_seconds,
        ),
    )
    if not args.no_pins_check:
        threading.Thread(target=_check_pins, name="pins", daemon=True).start()
    idle = threading.Thread(
        target=gemma.run_idle_stopper, args=(worker.stopping,), name="gemma-idle", daemon=True
    )
    idle.start()

    def cleanup() -> None:
        stop_all(comfy.free, gemma.stop)

    return _serve(worker, args, schedule, cleanup)


def _check_pins() -> None:
    from memcraft.pins import check_installation

    try:
        problems = check_installation()
    except Exception:
        log.exception("the pins check failed")
        return
    for line in problems:
        log.warning("pins: %s", line)
    log.info("pins: %s", "all match" if not problems else f"{len(problems)} mismatch(es)")


def _serve(worker: Worker, args: argparse.Namespace, schedule: Schedule | None, cleanup) -> int:
    stop_file: Path = args.stop_file
    stop_file.unlink(missing_ok=True)
    if args.once:
        try:
            if schedule is not None:
                schedule.run_pending()
            print(f"worked on {worker.drain()} claims")
        finally:
            cleanup()
        return 0

    done = threading.Event()

    def on_signal(_number: int, _frame: object) -> None:
        if worker.stopping.is_set():
            raise KeyboardInterrupt  # a second Ctrl+C does not wait for the step
        logging.info("stopping after the current steps; Ctrl+C again to quit now")
        worker.stopping.set()
        done.set()

    signal.signal(signal.SIGINT, on_signal)
    worker.start()
    logging.info("worker %s running", worker.worker_id)
    try:
        while not done.wait(1.0):
            if stop_file.exists():
                logging.info("stop file found: stopping after the current steps")
                stop_file.unlink(missing_ok=True)
                worker.stopping.set()
                break
        worker.stop()
    finally:
        cleanup()
    logging.info("worker %s stopped", worker.worker_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
