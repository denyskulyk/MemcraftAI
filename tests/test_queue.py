import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import event, select, update
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from memcraft.models import Job, JobLane, JobLength, JobQuality, JobStatus, User
from memcraft.queue import (
    FINAL,
    QUEUED,
    RUNNING,
    STARTS,
    TRANSITIONS,
    StatusConflictError,
    TransitionError,
    claim,
    queue_position,
    requeue_stale,
    transition,
)

HOUR = timedelta(hours=1)
WAIT = 30
S = JobStatus
LANES = {
    **{status: JobLane.IDEA for status in (S.IDEA_QUEUED, S.IDEA_RUNNING, S.AWAITING_APPROVAL)},
    S.CLOSED: JobLane.IDEA,
}
on_postgres = pytest.mark.parametrize(
    "engine", [pytest.param("postgres", marks=pytest.mark.postgres)], indirect=True
)


def limit_lock_waits(_dialect, _record, _args, params):
    params["options"] = "-c lock_timeout=15s"


@pytest.fixture
def session(engine):
    if engine.dialect.name == "postgresql":  # a broken lock must fail, not hang
        engine.dispose()
        event.listen(engine, "do_connect", limit_lock_waits)
    with Session(engine) as session:
        yield session


def ago(**kw):
    return datetime.now(UTC) - timedelta(**kw)


def new_job(session, status=S.IDEA_QUEUED, **kw):
    owner_id = session.scalar(select(User.id))
    if owner_id is None:
        user = User(tailscale_login=uuid.uuid4().hex)
        session.add(user)
        session.flush()
        owner_id = user.id
    values = {"lane": LANES.get(status, JobLane.RENDER)}
    if status in RUNNING:
        values.update(claimed_by="w", claimed_at=ago(minutes=5))
    job = Job(
        owner_id=owner_id,
        photo_count=1,
        length=JobLength.SHORT,
        quality=JobQuality.HD720,
        status=status,
        **(values | kw),
    )
    session.add(job)
    session.commit()
    return job


# The state machine.


def test_the_table():
    assert set(TRANSITIONS) == set(JobStatus)
    for status, following in TRANSITIONS.items():
        assert status not in following
        assert (S.CANCELLED in following) == (status not in FINAL)
    assert {s for s, following in TRANSITIONS.items() if not following} == FINAL
    for lane in JobLane:
        for queued, running in STARTS[lane].items():
            assert running in TRANSITIONS[queued]


@pytest.mark.parametrize("old", JobStatus)
def test_every_pair(session, old):
    allowed = {
        S.IDEA_QUEUED: {S.CANCELLED},
        S.IDEA_RUNNING: {S.AWAITING_APPROVAL, S.RENDER_QUEUED, S.FAILED, S.CANCELLED},
        S.AWAITING_APPROVAL: {S.RENDER_QUEUED, S.CLOSED, S.CANCELLED},
        S.RENDER_QUEUED: {S.CANCELLED},
        S.RENDERING: {S.PREVIEW_READY, S.FAILED, S.CANCELLED},
        S.PREVIEW_READY: {S.RELEASE_QUEUED, S.EXPIRED, S.CANCELLED},
        S.RELEASE_QUEUED: {S.CANCELLED},
        S.RELEASING: {S.RELEASED, S.FAILED, S.CANCELLED},
    }.get(old, set())
    for new in JobStatus:
        job = new_job(session, old)
        if new in allowed:
            assert transition(session, job, new, expected=old) is job
            session.commit()
            assert job.status is new
        else:
            with pytest.raises(TransitionError):
                transition(session, job, new, expected=old)
            session.rollback()
            assert job.status is old


def test_transition_leaves_the_commit_to_the_caller(session):
    job = new_job(session)
    transition(session, job, S.CANCELLED, expected=S.IDEA_QUEUED)
    session.rollback()
    assert job.status is S.IDEA_QUEUED


def test_transition_refuses_a_stale_status(session):
    job = new_job(session, S.PREVIEW_READY)
    # Written behind the session's back, so the object still says preview_ready.
    session.execute(
        update(Job).values(status=S.EXPIRED).execution_options(synchronize_session=False)
    )
    with pytest.raises(StatusConflictError):
        transition(session, job, S.RELEASE_QUEUED, expected=S.PREVIEW_READY)
    assert job.status is S.EXPIRED


def test_transition_of_a_deleted_job(session):
    job = new_job(session)
    session.connection().execute(Job.__table__.delete())
    session.expire(job)
    with pytest.raises(StatusConflictError):
        transition(session, job, S.CANCELLED, expected=S.IDEA_QUEUED)


def test_queueing_moves_to_the_render_lane_and_the_back_of_the_line(session):
    waiting = new_job(session, S.RENDER_QUEUED, queued_at=ago(hours=1))
    for old, lane, new in [
        (S.IDEA_RUNNING, JobLane.IDEA, S.RENDER_QUEUED),
        (S.AWAITING_APPROVAL, JobLane.IDEA, S.RENDER_QUEUED),
        (S.PREVIEW_READY, JobLane.RENDER, S.RELEASE_QUEUED),
    ]:
        job = new_job(session, old, lane=lane, queued_at=ago(hours=9))
        transition(session, job, new, expected=old)
        session.commit()
        assert job.lane is JobLane.RENDER
        assert job.queued_at > waiting.queued_at
        assert queue_position(session, job.id) > 0


def test_queued_at_stays_when_not_queueing(session):
    job = new_job(session, S.RENDERING, queued_at=ago(hours=9))
    before = job.queued_at
    transition(session, job, S.PREVIEW_READY, expected=S.RENDERING)
    session.commit()
    assert job.queued_at == before


@pytest.mark.parametrize("new", [S.PREVIEW_READY, S.FAILED, S.CANCELLED])
def test_leaving_a_running_status_clears_the_claim(session, new):
    job = new_job(session, S.RENDERING)
    transition(session, job, new, expected=S.RENDERING)
    session.commit()
    assert job.claimed_by is None
    assert job.claimed_at is None


def test_a_worker_that_lost_its_claim_cannot_finish(session):
    job = new_job(session, S.RENDERING, claimed_by="slow", claimed_at=ago(hours=2))
    assert requeue_stale(session, JobLane.RENDER, HOUR) == 1
    assert claim(session, JobLane.RENDER, "new").id == job.id
    with pytest.raises(StatusConflictError):
        transition(session, job, S.PREVIEW_READY, expected=S.RENDERING, worker_id="slow")
    transition(session, job, S.PREVIEW_READY, expected=S.RENDERING, worker_id="new")
    session.commit()
    assert job.status is S.PREVIEW_READY


# Claiming.


def test_claim(session):
    job = new_job(session)
    claimed = claim(session, JobLane.IDEA, "w1")
    assert claimed is job
    assert not session.in_transaction()
    assert claimed.status is S.IDEA_RUNNING
    assert claimed.claimed_by == "w1"
    assert claimed.claimed_at is not None
    assert claim(session, JobLane.IDEA, "w2") is None
    assert not session.in_transaction()


@pytest.mark.parametrize("status", JobStatus)
def test_lanes(session, status):
    job = new_job(session, status)
    takes = {
        S.IDEA_QUEUED: (JobLane.IDEA, S.IDEA_RUNNING),
        S.RENDER_QUEUED: (JobLane.RENDER, S.RENDERING),
        S.RELEASE_QUEUED: (JobLane.RENDER, S.RELEASING),
    }
    lane, running = takes.get(status, (None, None))
    for other in JobLane:
        if other is not lane:
            assert claim(session, other, "w") is None
    if lane is not None:
        assert claim(session, lane, "w").id == job.id
        assert job.status is running


def test_claim_order(session):
    stamp = ago(hours=2)
    ids = [uuid.UUID(int=n) for n in (3, 1, 2)]
    for each in ids:
        new_job(session, S.RENDER_QUEUED, id=each, queued_at=stamp)
    first = new_job(session, S.RELEASE_QUEUED, queued_at=ago(hours=3))
    last = new_job(session, S.RENDER_QUEUED, queued_at=ago(hours=1))
    expected = [first.id, *sorted(ids), last.id]
    order = []
    while (job := claim(session, JobLane.RENDER, "w")) is not None:
        order.append(job.id)
    assert order == expected


# Waiting.


def test_queue_position(session):
    stamp = ago(hours=2)
    low, high = uuid.UUID(int=1), uuid.UUID(int=2)
    new_job(session, S.RENDER_QUEUED, id=high, queued_at=stamp)
    new_job(session, S.RENDER_QUEUED, id=low, queued_at=stamp)
    release = new_job(session, S.RELEASE_QUEUED, queued_at=ago(hours=1))
    new_job(session, S.RENDERING, queued_at=ago(hours=3))
    new_job(session, S.PREVIEW_READY, queued_at=ago(hours=3))
    idea = new_job(session, S.IDEA_QUEUED, queued_at=ago(minutes=1))
    assert queue_position(session, low) == 0
    assert queue_position(session, high) == 1
    assert queue_position(session, release.id) == 2
    assert queue_position(session, idea.id) == 0


def test_queue_position_of_jobs_queued_by_the_database_clock(session):
    jobs = [new_job(session) for _ in range(3)]
    assert sorted(queue_position(session, job.id) for job in jobs) == [0, 1, 2]


@pytest.mark.parametrize("status", sorted(set(JobStatus) - QUEUED))
def test_no_position_when_not_waiting(session, status):
    assert queue_position(session, new_job(session, status).id) is None


def test_no_position_for_an_unknown_job(session):
    assert queue_position(session, uuid.uuid4()) is None


# A dead worker.


@pytest.mark.parametrize(
    ("running", "queued"),
    [
        (S.IDEA_RUNNING, S.IDEA_QUEUED),
        (S.RENDERING, S.RENDER_QUEUED),
        (S.RELEASING, S.RELEASE_QUEUED),
    ],
)
def test_requeue_stale(session, running, queued):
    lane = LANES.get(running, JobLane.RENDER)
    old = new_job(session, running, queued_at=ago(hours=9), claimed_at=ago(hours=2))
    young = new_job(session, running, claimed_at=ago(minutes=59))
    before = old.queued_at
    other = next(each for each in JobLane if each is not lane)
    assert requeue_stale(session, other, HOUR) == 0
    assert requeue_stale(session, lane, HOUR) == 1
    assert (old.status, old.queued_at, old.claimed_by, old.claimed_at) == (
        queued,
        before,
        None,
        None,
    )
    assert young.status is running
    assert young.claimed_by == "w"
    assert young.claimed_at is not None


@pytest.mark.parametrize("status", sorted(set(JobStatus) - RUNNING))
def test_requeue_touches_only_running_jobs(session, status):
    job = new_job(session, status, claimed_by="w", claimed_at=ago(hours=2))
    for lane in JobLane:
        assert requeue_stale(session, lane, HOUR) == 0
    assert job.status is status


def test_requeue_needs_a_positive_age(session):
    with pytest.raises(ValueError, match="positive"):
        requeue_stale(session, JobLane.IDEA, timedelta(0))


# Threads, each with its own session and connection.


def run_all(engine, count, work):
    """Start `work(session, n)` in all threads at once; their results in order."""
    engine.dispose()
    barrier = threading.Barrier(count)

    def run(n):
        with Session(engine) as session:
            session.connection()
            session.rollback()
            barrier.wait(WAIT)
            return work(session, n)

    with ThreadPoolExecutor(count) as pool:
        return [future.result(WAIT) for future in [pool.submit(run, n) for n in range(count)]]


@on_postgres
@pytest.mark.parametrize("workers", [2, 4, 8])
def test_workers_never_take_the_same_job(engine, session, workers):
    for n in range(100):
        status = S.RELEASE_QUEUED if n % 3 == 0 else S.RENDER_QUEUED
        new_job(session, status, queued_at=ago(minutes=n % 7))
    jobs = set(session.scalars(select(Job.id)))
    session.commit()

    def work(session, n):
        taken = []
        while (job := claim(session, JobLane.RENDER, f"w{n}")) is not None:
            assert job.claimed_by == f"w{n}"
            taken.append(job.id)
        return taken

    results = run_all(engine, workers, work)
    taken = [job_id for result in results for job_id in result]
    assert len(taken) == 100
    assert set(taken) == jobs
    for n, result in enumerate(results):
        mine = select(Job.id).where(Job.claimed_by == f"w{n}", Job.status.in_(RUNNING))
        assert set(session.scalars(mine)) == set(result)


@on_postgres
def test_claim_skips_a_locked_job_without_waiting(engine, session):
    first = new_job(session, queued_at=ago(hours=2))
    second = new_job(session, queued_at=ago(hours=1))
    first_id, second_id = first.id, second.id
    session.commit()
    with Session(engine) as holder:
        holder.execute(select(Job.id).where(Job.id == first_id).with_for_update()).all()
        session.connection().exec_driver_sql("SET lock_timeout = '2s'")
        assert claim(session, JobLane.IDEA, "w").id == second_id
        assert claim(session, JobLane.IDEA, "w") is None
    assert claim(session, JobLane.IDEA, "w").id == first_id


@on_postgres
def test_a_lock_that_only_blocks_deletes_does_not_hide_a_job(engine, session):
    job_id = new_job(session).id
    session.commit()
    with Session(engine) as saver:
        saver.execute(
            select(Job.id).where(Job.id == job_id).with_for_update(read=True, key_share=True)
        ).all()
        assert claim(session, JobLane.IDEA, "w").id == job_id


@on_postgres
def test_only_one_of_many_transitions_wins(engine, session):
    for _ in range(10):
        job_id = new_job(session, S.PREVIEW_READY).id
        session.commit()
        targets = [S.RELEASE_QUEUED, S.EXPIRED, S.CANCELLED, S.RELEASE_QUEUED]

        def work(session, n, job_id=job_id, targets=targets):
            job = session.get(Job, job_id)
            try:
                transition(session, job, targets[n], expected=S.PREVIEW_READY)
            except StatusConflictError:
                return None
            session.commit()
            return targets[n]

        winners = [result for result in run_all(engine, 4, work) if result is not None]
        assert len(winners) == 1
        session.expire_all()
        assert session.get(Job, job_id).status is winners[0]
        session.commit()


@on_postgres
def test_a_second_transition_waits_for_the_first(engine, session):
    """Compare-and-set holds against an open transaction, not only a committed one."""
    job_id = new_job(session, S.PREVIEW_READY).id
    session.commit()
    job = session.get(Job, job_id)
    transition(session, job, S.RELEASE_QUEUED, expected=S.PREVIEW_READY)
    with Session(engine) as other:
        theirs = other.get(Job, job_id)
        other.connection().exec_driver_sql("SET LOCAL lock_timeout = '200ms'")
        with pytest.raises(OperationalError):
            transition(other, theirs, S.EXPIRED, expected=S.PREVIEW_READY)
        other.rollback()
        session.commit()
        with pytest.raises(StatusConflictError):
            transition(other, theirs, S.EXPIRED, expected=S.PREVIEW_READY)


@on_postgres
def test_requeue_racing_with_the_worker(engine, session):
    """Either the worker finishes the job or it goes back in line, never both."""
    for _ in range(10):
        job_id = new_job(session, S.RENDERING, claimed_at=ago(hours=2)).id
        session.commit()

        def work(session, n, job_id=job_id):
            if n == 0:
                return requeue_stale(session, JobLane.RENDER, HOUR) == 1
            job = session.get(Job, job_id)
            try:
                transition(session, job, S.PREVIEW_READY, expected=S.RENDERING, worker_id="w")
            except StatusConflictError:
                return False
            session.commit()
            return True

        requeued, finished = run_all(engine, 2, work)
        assert requeued != finished
        session.expire_all()
        job = session.get(Job, job_id)
        assert job.status is (S.RENDER_QUEUED if requeued else S.PREVIEW_READY)
        assert job.claimed_by is None
        session.commit()
