"""The jobs table as a queue, and the only code that may change a job's status.

Every change is compare-and-set: UPDATE ... WHERE status = the status the caller expects, so of
two callers exactly one wins. claim() and requeue_stale() COMMIT the session, including work the
caller left open in it. transition() leaves the commit to the caller, so that a status and the
files that go with it land together.

Times come from the database clock, because the web container's clock may differ.
"""

from __future__ import annotations

import uuid
from datetime import timedelta
from typing import Any

from sqlalchemy import CursorResult, func, inspect, select, tuple_, update
from sqlalchemy.orm import Session, aliased

from memcraft.models import Job, JobLane, JobStatus

# What each lane takes, and the status the job gets while a worker has it.
STARTS: dict[JobLane, dict[JobStatus, JobStatus]] = {
    JobLane.IDEA: {JobStatus.IDEA_QUEUED: JobStatus.IDEA_RUNNING},
    JobLane.RENDER: {
        JobStatus.RENDER_QUEUED: JobStatus.RENDERING,
        JobStatus.RELEASE_QUEUED: JobStatus.RELEASING,
    },
}
_LANE_OF_QUEUED = {queued: lane for lane, starts in STARTS.items() for queued in starts}
QUEUED = frozenset(_LANE_OF_QUEUED)
RUNNING = frozenset(running for starts in STARTS.values() for running in starts.values())

_NEXT: dict[JobStatus, set[JobStatus]] = {
    JobStatus.IDEA_QUEUED: {JobStatus.IDEA_RUNNING},
    JobStatus.IDEA_RUNNING: {
        JobStatus.AWAITING_APPROVAL,  # the app proposed the idea
        JobStatus.RENDER_QUEUED,  # the friend gave one
        JobStatus.FAILED,
    },
    JobStatus.AWAITING_APPROVAL: {JobStatus.RENDER_QUEUED, JobStatus.CLOSED},
    JobStatus.RENDER_QUEUED: {JobStatus.RENDERING},
    JobStatus.RENDERING: {JobStatus.PREVIEW_READY, JobStatus.FAILED},
    JobStatus.PREVIEW_READY: {JobStatus.RELEASE_QUEUED, JobStatus.EXPIRED},
    JobStatus.RELEASE_QUEUED: {JobStatus.RELEASING},
    JobStatus.RELEASING: {JobStatus.RELEASED, JobStatus.FAILED},
    JobStatus.CLOSED: set(),
    JobStatus.EXPIRED: set(),
    JobStatus.RELEASED: set(),
    JobStatus.FAILED: set(),
    JobStatus.CANCELLED: set(),
}
FINAL = frozenset(status for status, following in _NEXT.items() if not following)

# The whole state machine: anything not here is refused. The one move outside it is
# requeue_stale(), which undoes a claim.
TRANSITIONS: dict[JobStatus, frozenset[JobStatus]] = {
    status: frozenset(following | {JobStatus.CANCELLED} if following else ())
    for status, following in _NEXT.items()
}


class QueueError(Exception):
    pass


class TransitionError(QueueError):
    """The state machine does not allow this change."""


class StatusConflictError(QueueError):
    """The job is no longer as the caller expects: somebody else changed it first."""


def claim(session: Session, lane: JobLane, worker_id: str) -> Job | None:
    """Take the oldest waiting job of the lane, or None. COMMITS the session."""
    # E5.4: while settings.paused is set, return None here.
    starts = STARTS[lane]
    row = session.execute(
        select(Job.id, Job.status)
        .where(Job.lane == lane, Job.status.in_(starts))
        .order_by(Job.queued_at, Job.id)
        .limit(1)
        # FOR NO KEY UPDATE: a file being saved for the job does not hide it from the queue.
        .with_for_update(skip_locked=True, key_share=True)
    ).first()
    job = None
    if row is not None:
        job = _set(
            session,
            row.id,
            row.status,
            status=starts[row.status],
            claimed_by=worker_id,
            claimed_at=func.now(),
        )
    session.commit()
    return job


def transition(
    session: Session,
    job: Job,
    new_status: JobStatus,
    *,
    expected: JobStatus,
    worker_id: str | None = None,
) -> Job:
    """Move the job from `expected` to `new_status`. The caller commits.

    A worker passes its `worker_id` when it leaves a running status. Without it a worker that
    was slow, not dead, could finish a job that requeue_stale() has given to another one.

    Save files before this call, not after: save() locks the user and then the job, the order
    in which FileStore.delete_user() locks them.
    """
    if new_status not in TRANSITIONS[expected] or new_status in RUNNING:
        raise TransitionError(
            f"a job cannot go from {expected.value} to {new_status.value}"
            + (" without claim()" if new_status in RUNNING else "")
        )
    values: dict[str, Any] = {"status": new_status}
    if expected in RUNNING:
        values.update(claimed_by=None, claimed_at=None)
    lane = _LANE_OF_QUEUED.get(new_status)
    if lane is not None:
        values.update(lane=lane, queued_at=func.now())
    session.flush()
    (job_id,) = inspect(job).identity  # job.id would fail on a row that has gone
    moved = _set(session, job_id, expected, worker_id, **values)
    if moved is None:
        if job in session:
            session.expire(job)
        raise StatusConflictError(f"job {job_id} is not {expected.value} any more")
    return moved


def queue_position(session: Session, job_id: uuid.UUID) -> int | None:
    """How many jobs of the same lane are ahead, or None if the job is not waiting."""
    me, other = aliased(Job), aliased(Job)
    ahead = (
        select(func.count())
        .select_from(other)
        .where(
            other.lane == me.lane,
            other.status.in_(QUEUED),
            tuple_(other.queued_at, other.id) < tuple_(me.queued_at, me.id),
        )
        .scalar_subquery()
    )
    return session.scalar(select(ahead).where(me.id == job_id, me.status.in_(QUEUED)))


def requeue_stale(session: Session, lane: JobLane, older_than: timedelta) -> int:
    """Put jobs whose worker died back in line, at their old place. COMMITS the session.

    A claim is not renewed while a job runs, so `older_than` must be longer than any job takes.
    """
    if older_than <= timedelta(0):
        raise ValueError("older_than must be positive")
    cutoff = session.scalar(select(func.now())) - older_than
    count = 0
    for queued, running in STARTS[lane].items():
        result: CursorResult[Any] = session.execute(
            update(Job)
            .where(Job.lane == lane, Job.status == running, Job.claimed_at < cutoff)
            .values(status=queued, claimed_by=None, claimed_at=None)
            .execution_options(synchronize_session=False)
        )
        count += result.rowcount
    session.commit()
    return count


def _set(
    session: Session,
    job_id: uuid.UUID,
    expected: JobStatus,
    worker_id: str | None = None,
    **values: Any,
) -> Job | None:
    statement = update(Job).where(Job.id == job_id, Job.status == expected)
    if worker_id is not None:
        statement = statement.where(Job.claimed_by == worker_id)
    return session.scalars(
        statement.values(**values).returning(Job),
        execution_options={"populate_existing": True, "synchronize_session": False},
    ).one_or_none()
