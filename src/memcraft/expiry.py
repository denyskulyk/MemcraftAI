"""Previews nobody released: everything the job made is hard deleted after 24 hours.

The job row stays, as expired, without any text or file. Jobs that ended closed, cancelled or
failed lose their photos, files and text the same way 24 hours after their last change, and
keep their status.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

from sqlalchemy import ColumnElement, and_, delete, exists, func, null, or_, select, update
from sqlalchemy.orm import Session

from memcraft.models import Asset, AssetKind, Idea, Job, JobStatus, JobStep
from memcraft.queue import transition
from memcraft.storage import FileStore

log = logging.getLogger(__name__)

PREVIEW_LIFETIME = timedelta(hours=24)
FINISHED_LIFETIME = timedelta(hours=24)
FINISHED = (JobStatus.CLOSED, JobStatus.CANCELLED, JobStatus.FAILED)


def expire_previews(session: Session, store: FileStore, now: datetime | None = None) -> int:
    """Expire every preview_ready job whose preview is too old. COMMITS the session.

    Returns how many jobs expired. A job that fails or is locked is left for the next run.
    """
    return _sweep(session, store, now, _due, JobStatus.EXPIRED)


def expire_finished(session: Session, store: FileStore, now: datetime | None = None) -> int:
    """Strip closed, cancelled and failed jobs untouched for a day. COMMITS the session.

    Their files, ideas, text and step data go; the row keeps its status. Returns how many jobs
    were stripped. A job that fails or is locked is left for the next run.
    """
    return _sweep(session, store, now, _finished_due, None)


def _sweep(
    session: Session,
    store: FileStore,
    now: datetime | None,
    due_at: Callable[[datetime], ColumnElement[bool]],
    new_status: JobStatus | None,
) -> int:
    session.commit()  # a failed job rolls back, which must not cost the caller's work
    if now is None:
        now = session.scalar(select(func.now()))  # the clock created_at comes from
    elif now.tzinfo is None:
        raise ValueError("now needs a time zone")
    else:
        now = now.astimezone(UTC)
    due = due_at(now)
    job_ids = session.scalars(select(Job.id).where(due).order_by(Job.id)).all()
    session.commit()
    expired = 0
    for job_id in job_ids:
        try:
            expired += _expire(session, store, job_id, due, new_status)
        except Exception:
            session.rollback()
            log.exception("job %s could not be expired", job_id)
    return expired


def _finished_due(now: datetime) -> ColumnElement[bool]:
    step_data = or_(
        JobStep.inputs_hash.is_not(None),
        JobStep.inputs.is_not(None),
        JobStep.output.is_not(None),
        JobStep.error.is_not(None),
    )
    left = or_(
        Job.idea_text.is_not(None),
        Job.error.is_not(None),
        exists().where(Asset.job_id == Job.id),
        exists().where(Idea.job_id == Job.id),
        exists().where(JobStep.job_id == Job.id, step_data),
    )
    return and_(Job.status.in_(FINISHED), Job.updated_at <= now - FINISHED_LIFETIME, left)


def _due(now: datetime) -> ColumnElement[bool]:
    preview = and_(Asset.job_id == Job.id, Asset.kind == AssetKind.PREVIEW)
    live = or_(
        Asset.expires_at > now,
        and_(Asset.expires_at.is_(None), Asset.created_at > now - PREVIEW_LIFETIME),
    )
    return and_(
        Job.status == JobStatus.PREVIEW_READY,
        exists().where(preview),
        ~exists().where(preview, live),
    )


def _expire(
    session: Session,
    store: FileStore,
    job_id: uuid.UUID,
    due: ColumnElement[bool],
    new_status: JobStatus | None = JobStatus.EXPIRED,
) -> int:
    # A release or a save in progress holds a lock on the row: it wins, and this run skips it.
    # Once the lock is ours, a release waits and then finds the job expired.
    job = session.scalars(
        select(Job)
        .where(Job.id == job_id, due)
        .with_for_update(skip_locked=True)
        .execution_options(populate_existing=True)
    ).one_or_none()
    if job is None:
        session.rollback()
        return 0
    if new_status is not None:
        transition(session, job, new_status, expected=JobStatus.PREVIEW_READY)
    job.idea_text = None
    job.error = None
    session.execute(delete(Idea).where(Idea.job_id == job_id))
    session.execute(
        update(JobStep)
        .where(JobStep.job_id == job_id)
        .values(inputs_hash=None, inputs=null(), output=null(), error=None)
    )
    # Commits everything above together with the asset rows and the list of files to remove,
    # and only then removes files. A crash before that commit changes nothing; a crash after
    # it leaves the files in file_deletions for the worker.
    store.delete_assets(session, where=Asset.job_id == job_id)
    return 1
