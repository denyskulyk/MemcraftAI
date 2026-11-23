import threading
import uuid
from datetime import timedelta

import pytest
from sqlalchemy import func, select, text, update
from sqlalchemy.orm import Session

from memcraft.models import Job, JobLane, JobLength, JobQuality, JobRegime, JobStatus, User
from memcraft.worker.lease import GpuLease, LocalLease, acquire, claim_render
from memcraft.worker.loop import Worker
from memcraft.worker.steps import dry_registry

S = JobStatus
on_postgres = pytest.mark.parametrize(
    "engine", [pytest.param("postgres", marks=pytest.mark.postgres)], indirect=True
)


def new_job(session, regime=None, waited=timedelta(0)):
    owner_id = session.scalar(select(User.id))
    if owner_id is None:
        user = User(tailscale_login=uuid.uuid4().hex)
        session.add(user)
        session.flush()
        owner_id = user.id
    job = Job(
        owner_id=owner_id,
        photo_count=1,
        length=JobLength.SHORT,
        quality=JobQuality.HD720,
        lane=JobLane.RENDER,
        status=S.RENDER_QUEUED,
        regime=regime,
    )
    session.add(job)
    session.flush()
    session.execute(update(Job).where(Job.id == job.id).values(queued_at=func.now() - waited))
    session.commit()
    return job


@on_postgres
def test_one_holder_across_connections(engine):
    first, second = GpuLease(engine), GpuLease(engine)
    assert first.try_acquire()
    assert first.try_acquire()  # again: still one hold
    assert first.check()
    assert not second.try_acquire()
    first.release()
    assert not first.held
    assert second.try_acquire()
    assert not first.try_acquire()
    second.release()
    second.release()  # twice is fine


@on_postgres
def test_a_dead_holder_frees_the_lease(engine):
    first, second = GpuLease(engine), GpuLease(engine)
    assert first.try_acquire()
    pid = first._connection.scalar(text("SELECT pg_backend_pid()"))
    with engine.connect() as admin:
        admin.execute(text("SELECT pg_terminate_backend(:pid)"), {"pid": pid})
        admin.commit()
    assert second.try_acquire()
    assert not first.check()  # the holder learns that it lost the lease
    assert not first.held
    second.release()


@on_postgres
def test_acquire_waits_until_released(engine):
    first, second = GpuLease(engine), GpuLease(engine)
    assert first.try_acquire()
    timer = threading.Timer(0.2, first.release)
    timer.start()
    assert acquire(second, lambda: True, poll=0.02)
    second.release()
    timer.join()


def test_acquire_gives_up_when_told():
    lock = threading.Lock()
    holder, other = LocalLease(lock), LocalLease(lock)
    assert holder.try_acquire()
    assert not acquire(other, lambda: False, poll=0.01)
    holder.release()
    assert acquire(other, lambda: False, poll=0.01)


def test_gpu_lease_needs_postgres():
    from sqlalchemy import create_engine

    with pytest.raises(ValueError, match="Postgres"):
        GpuLease(create_engine("sqlite://"))


# Grouping by model: the dry pipeline's multishot jobs first need "dry-video", keyframes jobs
# "dry-image".


@pytest.fixture
def session(engine):
    with Session(engine) as session:
        yield session


def model_of(engine):
    return Worker(engine, dry_registry(), lease=LocalLease()).next_model


@on_postgres
def test_the_next_model(engine, session):
    multishot, keyframes = new_job(session), new_job(session, JobRegime.KEYFRAMES)
    assert model_of(engine)(session, multishot) == "dry-video"
    assert model_of(engine)(session, keyframes) == "dry-image"


@on_postgres
@pytest.mark.parametrize(
    ("loaded", "waited", "first"),
    [
        (None, timedelta(0), 0),  # nothing loaded: the oldest
        ("dry-image", timedelta(0), 1),  # the job that needs the loaded model
        ("dry-image", timedelta(minutes=20), 0),  # but not past the longest wait
        ("dry-video", timedelta(0), 0),  # the oldest already needs it
        ("other", timedelta(0), 0),  # nobody needs it
    ],
)
def test_claim_prefers_the_loaded_model(engine, session, loaded, waited, first):
    jobs = [
        new_job(session, waited=waited + timedelta(seconds=2)),
        new_job(session, JobRegime.KEYFRAMES, waited=waited + timedelta(seconds=1)),
        new_job(session),
    ]
    job = claim_render(
        session,
        "w",
        loaded=loaded,
        model_of=model_of(engine),
        max_wait=timedelta(minutes=15),
    )
    assert not session.in_transaction()  # committed
    assert job.id == jobs[first].id
    assert job.status is S.RENDERING
    assert job.claimed_by == "w"


@on_postgres
def test_claim_with_an_empty_queue(engine, session):
    assert claim_render(session, "w", loaded="x", model_of=model_of(engine)) is None


@on_postgres
def test_a_broken_model_lookup_falls_back_to_the_oldest(engine, session):
    jobs = [new_job(session, waited=timedelta(seconds=2)), new_job(session)]

    def broken(_session, _job):
        raise RuntimeError("no")

    assert claim_render(session, "w", loaded="x", model_of=broken).id == jobs[0].id


@on_postgres
def test_a_preferred_job_taken_meanwhile(engine, session, monkeypatch):
    """If another worker takes the chosen job first, the claim falls back to the queue."""
    oldest = new_job(session, waited=timedelta(seconds=2))
    wanted = new_job(session, JobRegime.KEYFRAMES)

    def taken(lookup_session, job):
        model = model_of(engine)(lookup_session, job)
        if job.id == wanted.id:
            with Session(engine) as other:
                other.execute(update(Job).where(Job.id == wanted.id).values(status=S.RENDERING))
                other.commit()
        return model

    job = claim_render(session, "w", loaded="dry-image", model_of=taken)
    assert job.id == oldest.id
