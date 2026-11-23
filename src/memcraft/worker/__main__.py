"""`python -m memcraft.worker`: run the worker with the dry step handlers.

Until the pipeline registers real handlers every step only sleeps and reports progress, which
would mark real jobs done. So the dry run refuses any database whose name does not end in _test.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading

from sqlalchemy import create_engine, text

from memcraft.config import get_settings
from memcraft.storage import FileStore
from memcraft.worker.loop import Worker, WorkerConfig
from memcraft.worker.schedule import Schedule, default_tasks
from memcraft.worker.steps import dry_registry


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m memcraft.worker")
    parser.add_argument("--database-url", help="default: MEMCRAFT_DATABASE_URL")
    parser.add_argument("--worker-id")
    parser.add_argument("--step-seconds", type=float, default=0.5, help="how long a dry step takes")
    parser.add_argument("--once", action="store_true", help="work until the queues are empty")
    parser.add_argument("--no-schedule", action="store_true", help="skip expiry and purges")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(threadName)s %(message)s")

    settings = get_settings()
    engine = create_engine(args.database_url or settings.database_url, pool_pre_ping=True)
    with engine.connect() as connection:
        name = connection.scalar(text("SELECT current_database()")) or ""
    if not name.endswith("_test"):
        print(f"refusing to run dry steps on {name!r}: not a *_test database", file=sys.stderr)
        return 2

    store = None
    if settings.shared_dir.is_dir():
        store = FileStore(settings.shared_dir, settings.private_dir)
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
    if args.once:
        if schedule is not None:
            schedule.run_pending()
        print(f"worked on {worker.drain()} claims")
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
    while not done.wait(1.0):
        pass
    worker.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
