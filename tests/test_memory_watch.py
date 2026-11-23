import threading

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from memcraft.models import (
    Job,
    JobLane,
    JobLength,
    JobQuality,
    JobStatus,
    JobStep,
    Setting,
    StepName,
    StepStatus,
    User,
)
from memcraft.worker.loop import Worker, WorkerConfig
from memcraft.worker.memory import GB, SETTING_MIN_FREE, MemoryProbe, MemoryWatch, describe
from memcraft.worker.steps import DRY_GPU_MODELS, DryHandler, DryPlanner, StepRegistry


class FakeMachine:
    """Free RAM that goes up when Gemma or ComfyUI's models are unloaded."""

    def __init__(self, free_gb=4.0, total_gb=64.0, gemma_gb=16.0, comfy_gb=3.0):
        self.free_gb, self.total_gb = free_gb, total_gb
        self.gemma_gb, self.comfy_gb = gemma_gb, comfy_gb
        self.calls = []
        self.pids = {"comfyui": [200], "gemma": [300]}
        self.alive = {100: (1 * GB, GB // 2), 200: (4 * GB, 12 * GB), 300: (16 * GB, 2 * GB)}
        self.scans = 0

    def ram(self):
        return int(self.free_gb * GB), int(self.total_gb * GB)

    def process(self, pid):
        return self.alive.get(pid)

    def finder(self):
        self.scans += 1
        return {name: list(pids) for name, pids in self.pids.items()}

    def gpu(self):
        return {"used_mb": 9000, "total_mb": 12282, "free_mb": 3282}

    def stop_gemma(self):
        self.calls.append("gemma")
        self.free_gb += self.gemma_gb
        self.alive.pop(300, None)

    def free_comfy(self):
        self.calls.append("comfy")
        self.free_gb += self.comfy_gb

    def probe(self, clock=None):
        return MemoryProbe(
            ram=self.ram,
            process=self.process,
            finder=self.finder,
            gpu=self.gpu,
            own_pid=100,
            clock=clock or (lambda: 0.0),
        )


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class TickingStop(threading.Event):
    """A stop event whose wait() moves the fake clock instead of sleeping."""

    def __init__(self, clock):
        super().__init__()
        self.clock = clock

    def wait(self, timeout=None):
        self.clock.now += timeout or 0
        return self.is_set()


def watch_for(machine, clock=None, **kw):
    clock = clock or FakeClock()
    return MemoryWatch(
        probe=machine.probe(clock),
        free_gemma=machine.stop_gemma,
        free_comfy=machine.free_comfy,
        clock=clock,
        poll=5.0,
        **kw,
    )


def test_a_snapshot_has_ram_processes_and_vram():
    machine = FakeMachine(free_gb=20.0)
    snap = machine.probe().snapshot()
    assert snap["ram_free_gb"] == 20.0
    assert snap["ram_total_gb"] == 64.0
    assert snap["processes"]["worker"] == {"working_set_mb": 1024, "private_mb": 512}
    assert snap["processes"]["comfyui"]["working_set_mb"] == 4096
    assert snap["processes"]["gemma"]["working_set_mb"] == 16 * 1024
    assert snap["vram"]["used_mb"] == 9000
    line = describe(snap)
    assert "RAM free 20.0/64 GB" in line and "gemma 16.0 GB" in line and "VRAM 8.8/12 GB" in line


def test_a_snapshot_never_raises():
    def broken(*_args):
        raise OSError("no")

    probe = MemoryProbe(ram=broken, process=broken, finder=broken, gpu=broken, own_pid=1)
    snap = probe.snapshot()
    assert snap["processes"] == {}
    assert "ram_free_gb" not in snap and "vram" not in snap
    assert describe(snap) == "no memory readings"


def test_processes_are_looked_up_again_only_when_needed():
    machine = FakeMachine(free_gb=20.0)
    clock = FakeClock()
    probe = machine.probe(clock)
    probe.snapshot()
    probe.snapshot()
    assert machine.scans == 1  # all found and alive: no new scan
    machine.alive.pop(300)  # Gemma unloaded: its pid is gone
    machine.pids["gemma"] = []
    assert "gemma" not in probe.snapshot()["processes"]
    assert machine.scans == 2
    probe.snapshot()
    assert machine.scans == 2  # missing, but scanned less than a minute ago
    clock.now = 61
    machine.pids["gemma"] = [301]
    machine.alive[301] = (15 * GB, GB)
    assert probe.snapshot()["processes"]["gemma"]["working_set_mb"] == 15 * 1024
    assert machine.scans == 3


def test_enough_ram_does_nothing():
    machine = FakeMachine(free_gb=10.0)
    record = watch_for(machine).before_step(needs_gpu=True, service="comfyui")
    assert "made_room" not in record
    assert record["start"]["ram_free_gb"] == 10.0
    assert machine.calls == []


def test_a_cpu_step_never_unloads():
    machine = FakeMachine(free_gb=1.0)
    record = watch_for(machine).before_step(needs_gpu=False, service="gemma")
    assert "made_room" not in record and machine.calls == []


def test_tight_ram_unloads_gemma_then_comfy_before_a_gpu_step():
    machine = FakeMachine(free_gb=4.0)
    watch = watch_for(machine)
    record = watch.before_step(needs_gpu=True, service="comfyui")
    watch.after_step(record)
    made = record["made_room"]
    assert machine.calls == ["gemma", "comfy"]
    assert made["actions"] == ["unloaded Gemma", "freed ComfyUI models"]
    assert made["free_gb_before"] == 4.0 and made["free_gb_after"] == 23.0
    assert made["enough"] is True and made["waited_seconds"] == 0
    assert record["start"]["ram_free_gb"] == 23.0
    assert "gemma" not in record["start"]["processes"]
    assert record["end"]["ram_free_gb"] == 23.0


def test_a_gemma_step_keeps_gemma():
    machine = FakeMachine(free_gb=4.0)
    made = watch_for(machine).before_step(needs_gpu=True, service="gemma")["made_room"]
    assert machine.calls == ["comfy"]
    assert made["actions"] == ["freed ComfyUI models"]
    assert made["enough"] is True  # 7 GB


def test_the_limit_comes_from_the_call():
    machine = FakeMachine(free_gb=7.0)
    watch = watch_for(machine)
    assert "made_room" not in watch.before_step(needs_gpu=True)
    assert "made_room" in watch.before_step(needs_gpu=True, min_free_gb=8.0)
    machine.calls.clear()
    assert "made_room" not in watch.before_step(needs_gpu=True, min_free_gb=0)


def test_ram_that_comes_back_slowly_is_waited_for():
    machine = FakeMachine(free_gb=2.0, gemma_gb=0.0, comfy_gb=0.0)
    clock = FakeClock()
    stop = TickingStop(clock)
    reads = iter([2.0, 2.0, 3.0, 5.0, 6.5])
    machine.ram = lambda: (int(next(reads, 6.5) * GB), 64 * GB)
    watch = watch_for(machine, clock)
    made = watch.make_room(6.0, stop=stop)
    assert made["enough"] is True
    assert made["free_gb_after"] == 6.5
    assert made["waited_seconds"] == 15.0


def test_ram_that_stays_low_runs_the_step_anyway_after_the_wait():
    machine = FakeMachine(free_gb=1.0, gemma_gb=0.5, comfy_gb=0.5)
    clock = FakeClock()
    watch = watch_for(machine, clock, wait_seconds=60)
    record = watch.before_step(needs_gpu=True, stop=TickingStop(clock))
    made = record["made_room"]
    assert made["enough"] is False
    assert made["free_gb_after"] == 2.0
    assert made["waited_seconds"] == 60.0
    assert "start" in record


def test_a_stop_ends_the_wait():
    machine = FakeMachine(free_gb=1.0, gemma_gb=0, comfy_gb=0)
    clock = FakeClock()
    stop = TickingStop(clock)
    stop.set()
    made = watch_for(machine, clock, wait_seconds=600).make_room(6.0, stop=stop)
    assert made["waited_seconds"] == 5.0 and made["enough"] is False


def test_a_failing_unload_is_recorded_and_the_rest_still_runs():
    machine = FakeMachine(free_gb=4.0)

    def broken():
        raise RuntimeError("llama-swap is down")

    watch = watch_for(machine)
    watch.free_gemma = broken
    made = watch.make_room(6.0)
    assert made["actions"] == ["unloaded Gemma failed: llama-swap is down", "freed ComfyUI models"]
    assert machine.calls == ["comfy"]


def test_unknown_free_ram_does_nothing():
    machine = FakeMachine()
    watch = watch_for(machine)
    watch.probe.ram = lambda: None
    assert watch.make_room(6.0) is None
    assert machine.calls == []


def test_the_services_factory_wires_gemma_and_comfy():
    from memcraft.pipeline.services import memory_watch

    class Thing:
        def __init__(self):
            self.called = 0

        def stop(self):
            self.called += 1

        def free(self):
            self.called += 1

    comfy, gemma = Thing(), Thing()
    watch = memory_watch(comfy, gemma, min_free_gb=5.0, wait_seconds=30)
    assert watch.min_free_gb == 5.0 and watch.wait_seconds == 30
    watch.free_gemma()
    watch.free_comfy()
    assert gemma.called == comfy.called == 1


# The worker writes the record to job_steps.output (Postgres).


@pytest.mark.parametrize(
    "engine", [pytest.param("postgres", marks=pytest.mark.postgres)], indirect=True
)
def test_the_worker_keeps_memory_in_every_step_output(engine):
    machine = FakeMachine(free_gb=4.0)
    handlers = {
        StepName.IDEA: DryHandler(service="gemma", seconds=0),
        StepName.SCENARIO: DryHandler(service="gemma", seconds=0),
        **{
            name: DryHandler(needs_gpu=model is not None, model=model, seconds=0)
            for name, model in DRY_GPU_MODELS.items()
        },
    }
    registry = StepRegistry(DryPlanner(clips=1))
    for name, handler in handlers.items():
        registry.register(name, handler)
    with Session(engine) as session:
        user = User(tailscale_login="memory-watch")
        session.add(user)
        session.add(Setting(key=SETTING_MIN_FREE, value=5))
        session.flush()
        job = Job(
            owner_id=user.id,
            idea_text="a cat",
            photo_count=1,
            length=JobLength.SHORT,
            quality=JobQuality.HD720,
            lane=JobLane.RENDER,
            status=JobStatus.RENDER_QUEUED,
        )
        session.add(job)
        session.commit()
        job_id = job.id
    worker = Worker(
        engine,
        registry,
        config=WorkerConfig(poll=0.02, heartbeat=0.1, retry_delay=0, progress_interval=0),
        memory=watch_for(machine),
    )
    assert worker.drain() == 1
    with Session(engine) as session:
        rows = session.scalars(
            select(JobStep).where(JobStep.job_id == job_id).order_by(JobStep.created_at)
        ).all()
        assert session.get(Job, job_id).status is JobStatus.PREVIEW_READY
    assert rows and all(row.status is StepStatus.DONE for row in rows)
    for row in rows:
        memory = row.output["memory"]
        assert {"start", "end"} <= set(memory)
        assert memory["start"]["processes"]["worker"]["working_set_mb"] == 1024
        assert row.output["dry"] is True
    # The first GPU step made room (4 GB < 5 GB from settings); later ones had enough.
    made = [row.output["memory"].get("made_room") for row in rows]
    first_gpu = next(i for i, row in enumerate(rows) if row.step is StepName.CLIP)
    assert all(m is None for m in made[:first_gpu])
    assert made[first_gpu]["limit_gb"] == 5.0
    assert made[first_gpu]["actions"] == ["unloaded Gemma", "freed ComfyUI models"]
    assert all(m is None for m in made[first_gpu + 1 :])
    assert machine.calls == ["gemma", "comfy"]


@pytest.mark.parametrize(
    "engine", [pytest.param("postgres", marks=pytest.mark.postgres)], indirect=True
)
def test_a_failed_step_keeps_its_memory_record(engine):
    machine = FakeMachine(free_gb=20.0)

    class Boom:
        needs_gpu, model, service = True, "dry-video", None

        def run(self, ctx):
            raise RuntimeError("out of memory")

    registry = StepRegistry(DryPlanner(clips=1))
    for name, model in DRY_GPU_MODELS.items():
        handler = Boom() if name is StepName.CLIP else DryHandler(needs_gpu=model is not None)
        registry.register(name, handler)
    for name in (StepName.IDEA, StepName.SCENARIO):
        registry.register(name, DryHandler(seconds=0))
    with Session(engine) as session:
        user = User(tailscale_login="memory-watch-fail")
        session.add(user)
        session.flush()
        job = Job(
            owner_id=user.id,
            idea_text="a cat",
            photo_count=1,
            length=JobLength.SHORT,
            quality=JobQuality.HD720,
            lane=JobLane.RENDER,
            status=JobStatus.RENDER_QUEUED,
        )
        session.add(job)
        session.commit()
        job_id = job.id
    worker = Worker(
        engine,
        registry,
        config=WorkerConfig(poll=0.02, heartbeat=0.1, retry_delay=0, max_attempts=1),
        memory=watch_for(machine),
    )
    worker.drain()
    with Session(engine) as session:
        failed = session.scalars(
            select(JobStep).where(JobStep.job_id == job_id, JobStep.step == StepName.CLIP)
        ).one()
    assert failed.status is StepStatus.FAILED
    assert "out of memory" in failed.error
    assert {"start", "end"} <= set(failed.output["memory"])
