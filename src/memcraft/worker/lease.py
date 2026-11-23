"""The GPU lease, and claiming render jobs grouped by the model they need.

The lease is a Postgres session-level advisory lock held on a connection of its own, so at most
one process on any connection holds the GPU, and the lock goes away with a process that dies.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from datetime import timedelta
from typing import Protocol

from sqlalchemy import Connection, Engine, func, select, text, update
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from memcraft.models import Job, JobLane
from memcraft.queue import STARTS, claim

log = logging.getLogger(__name__)

# "memcraft" in ASCII, as the advisory lock key of the GPU.
GPU_LOCK_KEY = 0x6D656D6372616674
# A job that has waited this long is taken next, whatever model is loaded.
MAX_WAIT = timedelta(minutes=15)
# How many waiting jobs are looked at for one with the loaded model.
LOOK_AHEAD = 20


class Lease(Protocol):
    @property
    def held(self) -> bool: ...

    def try_acquire(self) -> bool: ...

    def release(self) -> None: ...

    def check(self) -> bool:
        """True if the lease is still ours. A lost connection loses the lease."""
        ...


class GpuLease:
    """One holder of the GPU across all processes. Not for use by two threads at once."""

    def __init__(self, engine: Engine, key: int = GPU_LOCK_KEY) -> None:
        if engine.dialect.name != "postgresql":
            raise ValueError("the GPU lease needs Postgres; use LocalLease elsewhere")
        self._engine = engine
        self._key = key
        self._connection: Connection | None = None

    @property
    def held(self) -> bool:
        return self._connection is not None

    def try_acquire(self) -> bool:
        if self._connection is not None:
            return True
        connection = self._engine.connect().execution_options(isolation_level="AUTOCOMMIT")
        try:
            got = connection.scalar(text("SELECT pg_try_advisory_lock(:key)"), {"key": self._key})
        except BaseException:
            connection.close()
            raise
        if not got:
            connection.close()
            return False
        self._connection = connection
        return True

    def release(self) -> None:
        connection, self._connection = self._connection, None
        if connection is None:
            return
        try:
            connection.execute(text("SELECT pg_advisory_unlock(:key)"), {"key": self._key})
        except SQLAlchemyError:
            log.warning("GPU lease unlock failed; closing its connection frees it")
            connection.invalidate()
        finally:
            connection.close()

    def check(self) -> bool:
        if self._connection is None:
            return False
        try:
            mine = self._connection.scalar(
                text(
                    "SELECT count(*) FROM pg_locks WHERE locktype = 'advisory' AND granted"
                    " AND pid = pg_backend_pid()"
                    " AND ((classid::bigint << 32) | objid::bigint) = :key"
                ),
                {"key": self._key},
            )
        except SQLAlchemyError:
            mine = 0
        if not mine:
            log.warning("GPU lease was lost")
            connection, self._connection = self._connection, None
            connection.invalidate()
            connection.close()
            return False
        return True


class LocalLease:
    """A lease within one process, for SQLite and for tests."""

    def __init__(self, lock: threading.Lock | None = None) -> None:
        self._lock = lock or threading.Lock()
        self._held = False

    @property
    def held(self) -> bool:
        return self._held

    def try_acquire(self) -> bool:
        if not self._held:
            self._held = self._lock.acquire(blocking=False)
        return self._held

    def release(self) -> None:
        if self._held:
            self._held = False
            self._lock.release()

    def check(self) -> bool:
        return self._held


def acquire(
    lease: Lease, keep_waiting: Callable[[], bool], poll: float = 1.0, log_after: float = 60.0
) -> bool:
    """Wait for the lease while keep_waiting() is true. True once it is ours."""
    started = time.monotonic()
    warned = False
    while not lease.try_acquire():
        if not keep_waiting():
            return False
        if not warned and time.monotonic() - started > log_after:
            log.info("waiting for the GPU lease held by another process")
            warned = True
        time.sleep(poll)
    return True


def claim_render(
    session: Session,
    worker_id: str,
    *,
    loaded: str | None,
    model_of: Callable[[Session, Job], str | None],
    max_wait: timedelta = MAX_WAIT,
) -> Job | None:
    """Claim a render-lane job, preferring one whose next GPU step uses the loaded model.

    The oldest job goes first when nothing is loaded, when it has waited `max_wait`, or when no
    waiting job needs the loaded model. COMMITS the session, as queue.claim() does.
    """
    starts = STARTS[JobLane.RENDER]
    if loaded is None:
        return claim(session, JobLane.RENDER, worker_id)
    now = session.scalar(select(func.now()))
    waiting = session.scalars(
        select(Job)
        .where(Job.lane == JobLane.RENDER, Job.status.in_(starts))
        .order_by(Job.queued_at, Job.id)
        .limit(LOOK_AHEAD)
    ).all()
    chosen = None
    if waiting and now - waiting[0].queued_at < max_wait:
        for job in waiting:
            try:
                model = model_of(session, job)
            except Exception:
                log.exception("cannot tell the model of job %s", job.id)
                continue
            if model == loaded:
                chosen = job
                break
    if chosen is None or chosen is waiting[0]:
        return claim(session, JobLane.RENDER, worker_id)
    job = _claim_job(session, chosen, worker_id)
    if job is None:  # somebody else was faster
        return claim(session, JobLane.RENDER, worker_id)
    session.commit()
    return job


def _claim_job(session: Session, job: Job, worker_id: str) -> Job | None:
    """The same compare-and-set as queue.claim(), for one chosen job."""
    status = job.status
    running = STARTS[JobLane.RENDER].get(status)
    if running is None:
        return None
    locked = session.scalar(
        select(Job.id)
        .where(Job.id == job.id, Job.status == status, Job.lane == JobLane.RENDER)
        .with_for_update(skip_locked=True, key_share=True)
    )
    if locked is None:
        return None
    return session.scalars(
        update(Job)
        .where(Job.id == job.id, Job.status == status)
        .values(status=running, claimed_by=worker_id, claimed_at=func.now())
        .returning(Job),
        execution_options={"populate_existing": True, "synchronize_session": False},
    ).one_or_none()
