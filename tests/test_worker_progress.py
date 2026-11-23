import asyncio
import os
import uuid

import psycopg
import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from memcraft.models import (
    Job,
    JobLane,
    JobLength,
    JobQuality,
    JobRegime,
    JobStatus,
    StepName,
    User,
)
from memcraft.worker.lease import LocalLease
from memcraft.worker.loop import Worker, WorkerConfig
from memcraft.worker.progress import (
    CHANNEL,
    IDEA_WEIGHTS,
    RENDER_WEIGHTS,
    ProgressEvent,
    job_events,
    job_fraction,
    psycopg_conninfo,
    sse,
    weights_for,
)
from memcraft.worker.steps import dry_registry

S = JobStatus
on_postgres = pytest.mark.parametrize(
    "engine", [pytest.param("postgres", marks=pytest.mark.postgres)], indirect=True
)
FAST = WorkerConfig(poll=0.02, heartbeat=0.1, retry_delay=0, progress_interval=0)
MULTI = RENDER_WEIGHTS[JobRegime.MULTISHOT]


def test_weights_add_up():
    for weights in [*RENDER_WEIGHTS.values(), IDEA_WEIGHTS]:
        assert sum(weights.values()) == pytest.approx(1)


def test_weights_for_each_phase():
    assert weights_for(S.IDEA_RUNNING, None) is IDEA_WEIGHTS
    assert weights_for(S.RENDERING, None) is MULTI
    assert weights_for(S.RENDERING, JobRegime.KEYFRAMES) is RENDER_WEIGHTS[JobRegime.KEYFRAMES]
    assert weights_for(S.RELEASING, JobRegime.KEYFRAMES) == {StepName.RELEASE: 1.0}


def test_job_fraction():
    assert job_fraction(MULTI, StepName.SCENARIO, 0, 1, 0) == 0
    assert job_fraction(MULTI, StepName.SCENARIO, 0, 1, 1) == pytest.approx(0.05)
    # Two clips share 0.80.
    assert job_fraction(MULTI, StepName.CLIP, 0, 2, 0.5) == pytest.approx(0.05 + 0.2)
    assert job_fraction(MULTI, StepName.CLIP, 1, 2, 1) == pytest.approx(0.85)
    assert job_fraction(MULTI, StepName.JOIN, 0, 1, 1) == pytest.approx(1)
    assert job_fraction(MULTI, StepName.JOIN, 0, 1, 7) == pytest.approx(1)
    assert job_fraction(MULTI, StepName.KEYFRAME, 0, 1, 0.5) == -1  # not in this regime


def test_events_round_trip():
    event = ProgressEvent(uuid.uuid4(), S.RENDERING, 0.123456, StepName.CLIP)
    back = ProgressEvent.parse(event.payload())
    assert (back.job_id, back.status, back.step) == (event.job_id, S.RENDERING, StepName.CLIP)
    assert back.progress == pytest.approx(0.1235)
    assert sse(event).startswith("event: progress\ndata: {")
    assert sse(event).endswith("\n\n")
    assert sse(None) == ": keepalive\n\n"


def test_conninfo():
    info = psycopg_conninfo("postgresql+psycopg://u:p%40ss@127.0.0.1:5432/db_test")
    assert info == "postgresql://u:p%40ss@127.0.0.1:5432/db_test"


def new_job(session, status=S.RENDER_QUEUED):
    user = User(tailscale_login=uuid.uuid4().hex)
    session.add(user)
    session.flush()
    job = Job(
        owner_id=user.id,
        idea_text="a cat",
        photo_count=1,
        length=JobLength.SHORT,
        quality=JobQuality.HD720,
        lane=JobLane.RENDER,
        status=status,
    )
    session.add(job)
    session.commit()
    return job


def conninfo():
    return psycopg_conninfo(os.environ["MEMCRAFT_TEST_DATABASE_URL"])


@on_postgres
def test_progress_is_written_and_notified(engine):
    with Session(engine) as session:
        job = new_job(session)
        other = new_job(session)
        job_id, other_id = job.id, other.id
    with psycopg.connect(conninfo(), autocommit=True) as listener:
        listener.execute(f"LISTEN {CHANNEL}")
        worker = Worker(engine, dry_registry(seconds=0.01), lease=LocalLease(), config=FAST)
        assert worker.drain() == 2
        events = [ProgressEvent.parse(note.payload) for note in listener.notifies(timeout=1.0)]
    mine = [event for event in events if event.job_id == job_id]
    assert {event.job_id for event in events} == {job_id, other_id}
    assert mine[0].status is S.RENDERING
    assert mine[0].progress == 0
    assert mine[-1].status is S.PREVIEW_READY
    assert mine[-1].progress == 1
    values = [event.progress for event in mine]
    assert values == sorted(values)
    assert len(set(values)) > 5
    assert {event.step for event in mine} >= {StepName.SCENARIO, StepName.CLIP, StepName.JOIN}
    with Session(engine) as session:
        assert session.scalar(select(Job.progress).where(Job.id == job_id)) == 1


@on_postgres
def test_job_events_streams_until_the_job_stops_moving(engine):
    with Session(engine) as session:
        job_id = new_job(session).id
    worker = Worker(engine, dry_registry(seconds=0.01), lease=LocalLease(), config=FAST)
    url = os.environ["MEMCRAFT_TEST_DATABASE_URL"]

    async def stream():
        seen = []
        async for event in job_events(url, job_id, keepalive=0.2):
            seen.append(event)
            if len(seen) == 1:
                asyncio.get_running_loop().run_in_executor(None, worker.drain)
        return seen

    seen = asyncio.run(asyncio.wait_for(stream(), 60), loop_factory=asyncio.SelectorEventLoop)
    events = [event for event in seen if event is not None]
    assert events[0].status is S.RENDER_QUEUED
    assert events[-1].status is S.PREVIEW_READY
    assert events[-1].progress == 1
    assert all(event.job_id == job_id for event in events)


@on_postgres
def test_job_events_of_a_finished_or_unknown_job(engine):
    with Session(engine) as session:
        job_id = new_job(session, S.PREVIEW_READY).id
    url = os.environ["MEMCRAFT_TEST_DATABASE_URL"]

    async def collect(wanted):
        return [event async for event in job_events(url, wanted)]

    def run(wanted):
        return asyncio.run(collect(wanted), loop_factory=asyncio.SelectorEventLoop)

    (only,) = run(job_id)
    assert only.status is S.PREVIEW_READY
    assert run(uuid.uuid4()) == []
