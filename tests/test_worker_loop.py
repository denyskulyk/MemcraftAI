import itertools
import threading
import time
import uuid
from datetime import timedelta

import pytest
from sqlalchemy import event, func, select, update
from sqlalchemy.orm import Session

from memcraft.models import (
    Job,
    JobLane,
    JobLength,
    JobQuality,
    JobRegime,
    JobStatus,
    JobStep,
    StepName,
    StepStatus,
    User,
)
from memcraft.queue import requeue_stale, transition
from memcraft.worker.loop import INTERRUPTED, Worker, WorkerConfig
from memcraft.worker.pause import is_paused, set_paused
from memcraft.worker.steps import (
    DRY_GPU_MODELS,
    DryHandler,
    DryPlanner,
    StepRegistry,
    StepStopped,
)
from memcraft.worker.watchdog import HUNG, Service, Watchdog

S = JobStatus
WAIT = 30
FAST = WorkerConfig(poll=0.02, heartbeat=0.1, retry_delay=0, progress_interval=0)
RENDER_STEPS = [
    (StepName.SCENARIO, 0),
    (StepName.CLIP, 0),
    (StepName.CLIP, 1),
    (StepName.CHECK, 0),
    (StepName.JOIN, 0),
]

pytestmark = pytest.mark.parametrize(
    "engine", [pytest.param("postgres", marks=pytest.mark.postgres)], indirect=True
)


def limit_lock_waits(_dialect, _record, _args, params):
    params["options"] = "-c lock_timeout=15s"


@pytest.fixture
def session(engine):
    engine.dispose()
    event.listen(engine, "do_connect", limit_lock_waits)
    with Session(engine) as session:
        yield session


class Crash(BaseException):
    """Stands for the process dying: nothing in the worker catches it."""


class Scripted:
    """A handler that records its calls and runs `script(ctx)` between two progress reports."""

    def __init__(self, script=None, *, needs_gpu=False, model=None, service=None):
        self.needs_gpu, self.model, self.service = needs_gpu, model, service
        self.script = script or (lambda ctx: None)
        self.calls = []
        self.lock = threading.Lock()

    def run(self, ctx):
        with self.lock:
            self.calls.append((ctx.job_id, ctx.ordinal, ctx.attempt))
        ctx.progress(0.5)
        output = self.script(ctx)
        ctx.progress(1.0)
        return output or {"ok": True}


def registry(**handlers):
    """The dry pipeline with some steps replaced, by step name."""
    defaults = {
        StepName.IDEA: DryHandler(service="gemma", seconds=0),
        StepName.SCENARIO: DryHandler(service="gemma", seconds=0),
        **{
            name: DryHandler(needs_gpu=model is not None, model=model, seconds=0)
            for name, model in DRY_GPU_MODELS.items()
        },
    }
    result = StepRegistry(DryPlanner(clips=2))
    for name, handler in defaults.items():
        result.register(name, handlers.get(name.value, handler))
    return result


def worker(engine, reg=None, **kw):
    return Worker(engine, reg or registry(), config=kw.pop("config", FAST), **kw)


def new_job(session, status=S.RENDER_QUEUED, idea_text="a cat", regime=None):
    owner_id = session.scalar(select(User.id))
    if owner_id is None:
        user = User(tailscale_login=uuid.uuid4().hex)
        session.add(user)
        session.flush()
        owner_id = user.id
    lane = JobLane.IDEA if status is S.IDEA_QUEUED else JobLane.RENDER
    job = Job(
        owner_id=owner_id,
        idea_text=idea_text,
        photo_count=1,
        length=JobLength.SHORT,
        quality=JobQuality.HD720,
        lane=lane,
        status=status,
        regime=regime,
    )
    session.add(job)
    session.commit()
    return job


def steps_of(session, job_id):
    session.expire_all()
    return [
        (row.step, row.ordinal, row.attempt, row.status)
        for row in session.scalars(
            select(JobStep)
            .where(JobStep.job_id == job_id)
            .order_by(JobStep.created_at, JobStep.step, JobStep.ordinal, JobStep.attempt)
        )
    ]


def status_of(session, job_id):
    session.expire_all()
    return session.get(Job, job_id).status


def wait_for(condition, timeout=WAIT):
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError("timed out")
        time.sleep(0.02)


def done_once(steps):
    return [(step, ordinal, 1, StepStatus.DONE) for step, ordinal in steps]


def test_a_job_goes_from_idea_to_preview(engine, session):
    job = new_job(session, S.IDEA_QUEUED)
    assert worker(engine).drain() == 2  # once in each lane
    session.expire_all()
    assert job.status is S.PREVIEW_READY
    assert job.progress == 1
    assert (job.claimed_by, job.claimed_at) == (None, None)
    assert steps_of(session, job.id) == done_once([(StepName.IDEA, 0), *RENDER_STEPS])


def test_a_proposed_idea_waits_for_approval(engine, session):
    job = new_job(session, S.IDEA_QUEUED, idea_text=None)
    assert worker(engine).drain() == 1
    assert status_of(session, job.id) is S.AWAITING_APPROVAL


def test_keyframes_regime_and_release(engine, session):
    job = new_job(session, regime=JobRegime.KEYFRAMES)
    worker(engine).drain()
    steps = steps_of(session, job.id)
    assert [step for step, *_ in steps].count(StepName.KEYFRAME) == 2
    assert status_of(session, job.id) is S.PREVIEW_READY
    transition(session, job, S.RELEASE_QUEUED, expected=S.PREVIEW_READY)
    session.commit()
    worker(engine).drain()
    assert status_of(session, job.id) is S.RELEASED


def test_ideas_answer_while_a_render_runs(engine, session):
    rendering, go_on = threading.Event(), threading.Event()

    def slow(ctx):
        rendering.set()
        assert go_on.wait(WAIT)

    render = new_job(session)
    me = worker(engine, registry(clip=Scripted(slow, needs_gpu=True, model="v")))
    me.start()
    try:
        assert rendering.wait(WAIT)
        idea = new_job(session, S.IDEA_QUEUED, idea_text=None)
        wait_for(lambda: status_of(session, idea.id) is S.AWAITING_APPROVAL)
        assert status_of(session, render.id) is S.RENDERING
        go_on.set()
        wait_for(lambda: status_of(session, render.id) is S.PREVIEW_READY)
    finally:
        go_on.set()
        assert me.stop(WAIT)


def test_two_workers_share_the_queue_and_never_the_gpu(engine, session):
    jobs = [new_job(session) for _ in range(4)] + [
        new_job(session, S.IDEA_QUEUED) for _ in range(2)
    ]
    active, most, lock = [0], [0], threading.Lock()

    def on_gpu(ctx):
        with lock:
            active[0] += 1
            most[0] = max(most[0], active[0])
        time.sleep(0.02)
        with lock:
            active[0] -= 1

    clip = Scripted(on_gpu, needs_gpu=True, model="v")
    check = Scripted(on_gpu, needs_gpu=True, model="c")
    workers = [worker(engine, registry(clip=clip, check=check)) for _ in range(2)]
    for each in workers:
        each.start()
    try:
        wait_for(lambda: all(status_of(session, j.id) is S.PREVIEW_READY for j in jobs))
    finally:
        assert all(each.stop(WAIT) for each in workers)
    assert most[0] == 1
    for n, job in enumerate(jobs):
        idea = [(StepName.IDEA, 0)] if n >= 4 else []
        assert steps_of(session, job.id) == done_once(idea + RENDER_STEPS)
    # Every job was worked on by one worker per claim, and each GPU step ran once.
    assert len(clip.calls) == 2 * len(jobs)
    assert len(check.calls) == len(jobs)


def test_pause_finishes_the_step_unloads_and_resumes(engine, session):
    inside, go_on = threading.Event(), threading.Event()

    def scenario(ctx):
        inside.set()
        assert go_on.wait(WAIT)

    clip = Scripted(needs_gpu=True, model="v")
    unloaded = []
    job = new_job(session)
    me = worker(engine, registry(scenario=Scripted(scenario, service="gemma"), clip=clip))
    me.unload = unloaded.append
    me.start()
    try:
        assert inside.wait(WAIT)
        set_paused(session, True)
        session.commit()
        go_on.set()
        wait_for(lambda: unloaded)
        assert unloaded[0] is JobLane.RENDER
        time.sleep(0.3)  # several heartbeats and pause polls
        assert steps_of(session, job.id) == done_once([(StepName.SCENARIO, 0)])
        assert clip.calls == []
        assert status_of(session, job.id) is S.RENDERING  # the claim stays alive
        assert not me.lease.held
        set_paused(session, False)
        session.commit()
        wait_for(lambda: status_of(session, job.id) is S.PREVIEW_READY)
    finally:
        go_on.set()
        assert me.stop(WAIT)
    assert steps_of(session, job.id) == done_once(RENDER_STEPS)


def test_paused_worker_claims_nothing(engine, session):
    set_paused(session, True)
    session.commit()
    assert is_paused(session)
    job = new_job(session, S.IDEA_QUEUED)
    assert worker(engine).drain() == 0
    assert status_of(session, job.id) is S.IDEA_QUEUED


def test_a_crash_between_steps_redoes_only_the_unfinished_step(engine, session):
    crashed = []

    def second_clip(ctx):
        if ctx.ordinal == 1 and not crashed:
            crashed.append(ctx.attempt)
            raise Crash

    scenario = Scripted(service="gemma")
    clip = Scripted(second_clip, needs_gpu=True, model="v")
    reg = registry(scenario=scenario, clip=clip)
    job = new_job(session)
    first = worker(engine, reg)
    with pytest.raises(Crash):
        first.run_one(JobLane.RENDER)
    first.lease.release()  # the dead process's connection closes

    assert status_of(session, job.id) is S.RENDERING
    assert steps_of(session, job.id)[-1] == (StepName.CLIP, 1, 1, StepStatus.RUNNING)
    # Its heartbeat died with it: the claim gets old and goes back in line.
    session.execute(
        update(Job).where(Job.id == job.id).values(claimed_at=func.now() - timedelta(minutes=5))
    )
    session.commit()
    assert requeue_stale(session, JobLane.RENDER, timedelta(seconds=90)) == 1

    assert worker(engine, reg).drain() == 1
    assert status_of(session, job.id) is S.PREVIEW_READY
    assert len(scenario.calls) == 1
    assert [(ordinal, attempt) for _, ordinal, attempt in clip.calls] == [(0, 1), (1, 1), (1, 2)]
    rows = session.scalars(
        select(JobStep).where(JobStep.job_id == job.id, JobStep.step == StepName.CLIP)
    ).all()
    interrupted = [row for row in rows if row.status is StepStatus.FAILED]
    assert [(row.ordinal, row.attempt, row.error) for row in interrupted] == [(1, 1, INTERRUPTED)]
    assert all(row.finished_at is not None for row in rows)


def test_a_live_claim_is_not_requeued(engine, session):
    inside, go_on = threading.Event(), threading.Event()

    def slow(ctx):
        inside.set()
        assert go_on.wait(WAIT)

    job = new_job(session)
    me = worker(engine, registry(scenario=Scripted(slow)))
    me.start()
    try:
        assert inside.wait(WAIT)
        session.execute(
            update(Job).where(Job.id == job.id).values(claimed_at=func.now() - timedelta(hours=1))
        )
        session.commit()
        time.sleep(0.4)  # the heartbeat renews it
        assert requeue_stale(session, JobLane.RENDER, timedelta(seconds=2)) == 0
        go_on.set()
        wait_for(lambda: status_of(session, job.id) is S.PREVIEW_READY)
    finally:
        go_on.set()
        assert me.stop(WAIT)


def test_a_failed_step_is_retried(engine, session):
    def flaky(ctx):
        if ctx.attempt == 1:
            raise ValueError("once")

    job = new_job(session)
    worker(engine, registry(join=Scripted(flaky))).drain()
    assert status_of(session, job.id) is S.PREVIEW_READY
    assert steps_of(session, job.id)[-2:] == [
        (StepName.JOIN, 0, 1, StepStatus.FAILED),
        (StepName.JOIN, 0, 2, StepStatus.DONE),
    ]


def test_after_the_retry_limit_the_job_fails(engine, session):
    def broken(ctx):
        raise ValueError("the model said no")

    clip = Scripted(broken, needs_gpu=True, model="v")
    job = new_job(session)
    worker(engine, registry(clip=clip)).drain()
    session.expire_all()
    assert job.status is S.FAILED
    assert job.error == "ValueError: the model said no"
    assert (job.claimed_by, job.claimed_at) == (None, None)
    assert [attempt for *_, attempt in clip.calls] == [1, 2, 3]
    assert steps_of(session, job.id)[1:] == [
        (StepName.CLIP, 0, n, StepStatus.FAILED) for n in (1, 2, 3)
    ]
    rows = session.scalars(select(JobStep).where(JobStep.status == StepStatus.FAILED)).all()
    assert {row.error for row in rows} == {"ValueError: the model said no"}


def test_a_failed_step_rolls_back_its_files(engine, session):
    """What the handler wrote to ctx.session is not committed when the step fails."""

    def half(ctx):
        ctx.session.add(User(tailscale_login="half"))
        ctx.session.flush()
        raise ValueError("after the write")

    new_job(session)
    worker(engine, registry(join=Scripted(half))).drain()
    assert session.scalar(select(func.count()).where(User.tailscale_login == "half")) == 0


def test_a_cancelled_job_stops_its_step(engine, session):
    inside = threading.Event()
    stopped = []

    def long(ctx):
        inside.set()
        try:
            ctx.wait(WAIT)
        except StepStopped:
            stopped.append(True)
            raise

    clip = Scripted(long, needs_gpu=True, model="v")
    job = new_job(session)
    me = worker(engine, registry(clip=clip))
    runner = threading.Thread(target=me.run_one, args=(JobLane.RENDER,))
    runner.start()
    assert inside.wait(WAIT)
    session.expire_all()
    transition(session, session.get(Job, job.id), S.CANCELLED, expected=S.RENDERING)
    session.commit()
    runner.join(WAIT)
    assert not runner.is_alive()
    assert stopped == [True]
    assert status_of(session, job.id) is S.CANCELLED
    assert steps_of(session, job.id)[-1] == (StepName.CLIP, 0, 1, StepStatus.FAILED)
    assert len(clip.calls) == 1
    me.lease.release()


def test_a_hung_step_is_retried_once(engine, session):
    restarts = []
    ticks = itertools.count()
    dog = Watchdog(
        # It answers again once restarted.
        [
            Service(
                "comfyui", 8199, probe=lambda: bool(restarts), restart=lambda: restarts.append(1)
            )
        ],
        gpu=lambda: None,
        answer_limit=0.5,
        stall_limit=1e9,
        clock=lambda: next(ticks),
    )

    def hangs_first(ctx):
        if (ctx.ordinal, ctx.attempt) == (0, 1):
            ctx.wait(WAIT)

    job = new_job(session)
    clip = Scripted(hangs_first, needs_gpu=True, model="v", service="comfyui")
    me = worker(engine, registry(clip=clip), watchdog=dog)
    runner = threading.Thread(target=me.drain)
    runner.start()
    while runner.is_alive():
        dog.check()
        time.sleep(0.02)
    assert status_of(session, job.id) is S.PREVIEW_READY
    assert restarts == [1]
    rows = session.scalars(
        select(JobStep)
        .where(JobStep.job_id == job.id, JobStep.step == StepName.CLIP, JobStep.ordinal == 0)
        .order_by(JobStep.attempt)
    ).all()
    assert [row.status for row in rows] == [StepStatus.FAILED, StepStatus.DONE]
    assert rows[0].error.startswith(HUNG)


def test_a_step_hung_twice_fails_the_job(engine, session):
    restarts = []
    ticks = itertools.count()
    dog = Watchdog(
        [Service("comfyui", 8199, probe=lambda: False, restart=lambda: restarts.append(1))],
        gpu=lambda: None,
        answer_limit=0.5,
        stall_limit=1e9,
        clock=lambda: next(ticks),
    )
    job = new_job(session)
    clip = Scripted(lambda ctx: ctx.wait(WAIT), needs_gpu=True, model="v", service="comfyui")
    me = worker(engine, registry(clip=clip), watchdog=dog)
    runner = threading.Thread(target=me.drain)
    runner.start()
    while runner.is_alive():
        dog.check()
        time.sleep(0.02)
    session.expire_all()
    assert job.status is S.FAILED
    assert job.error.startswith(HUNG)
    assert len(restarts) == len(clip.calls) == 2


def test_a_done_step_with_other_inputs_runs_again(engine, session):
    job = new_job(session)
    worker(engine).drain()
    session.execute(
        update(JobStep)
        .where(JobStep.job_id == job.id, JobStep.step == StepName.JOIN)
        .values(inputs_hash="changed")
    )
    session.execute(
        update(Job)
        .where(Job.id == job.id)
        .values(status=S.RENDER_QUEUED, claimed_by=None, claimed_at=None)
    )
    session.commit()
    join = Scripted()
    worker(engine, registry(join=join)).drain()
    assert len(join.calls) == 1
    assert status_of(session, job.id) is S.PREVIEW_READY


def test_the_dry_command(engine, session, capsys):
    from memcraft.worker.__main__ import main

    job = new_job(session, S.IDEA_QUEUED)
    url = engine.url.render_as_string(hide_password=False)
    assert (
        main(["--dry", "--database-url", url, "--once", "--no-schedule", "--step-seconds", "0"])
        == 0
    )
    assert "worked on 2 claims" in capsys.readouterr().out
    assert status_of(session, job.id) is S.PREVIEW_READY


def test_step_times_are_real(engine, session):
    def slow(ctx):
        ctx.session.connection()  # the step's transaction begins here
        time.sleep(0.3)

    job = new_job(session)
    worker(engine, registry(join=Scripted(slow))).drain()
    row = session.scalars(
        select(JobStep).where(JobStep.job_id == job.id, JobStep.step == StepName.JOIN)
    ).one()
    assert row.finished_at - row.started_at >= timedelta(seconds=0.3)
