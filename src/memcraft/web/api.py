"""The friend's own jobs: list, one job, its idea, progress, release, files and delete.

Every status change goes through memcraft.queue as compare-and-set: of the friend, the
worker and the expiry run, exactly one wins, and the others get 409.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import AsyncIterator
from datetime import datetime
from typing import Annotated, Any

from fastapi import APIRouter, Body, HTTPException, Query, Response
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict
from sqlalchemy import Engine, and_, func, or_, select, update
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

from memcraft import queue
from memcraft.expiry import PREVIEW_LIFETIME
from memcraft.models import (
    Asset,
    AssetKind,
    Idea,
    IdeaSource,
    Job,
    JobLength,
    JobQuality,
    JobRegime,
    JobStatus,
    User,
    UserRole,
    UserStatus,
)
from memcraft.queue import StatusConflictError, TransitionError
from memcraft.storage import StorageError
from memcraft.web.auth import ApprovedUser, CurrentUser, SessionDep, StoreDep
from memcraft.web.owned import assets_of, get_asset, get_job, jobs_of

router = APIRouter(prefix="/api")

# Progress stream. Polling is the whole mechanism for now; a LISTEN/NOTIFY wake-up can
# replace the sleep in _events() without changing what the browser sees.
POLL_SECONDS = 1.0
HEARTBEAT_SECONDS = 15.0
# After this the server ends the stream without "end", and EventSource reconnects by itself,
# so no connection lives for hours.
STREAM_SECONDS = 600.0
RETRY_MS = 3000
# The stream ends here too: nothing moves until the friend acts, and the page then opens a
# new stream.
WAITING_FOR_USER = frozenset({JobStatus.AWAITING_APPROVAL, JobStatus.PREVIEW_READY})

NOT_WAITING = "This idea is not waiting for an answer any more. Please reload."
BUSY = "This video is being worked on right now. Please wait until it finishes, then delete it."


class _Out(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class Me(_Out):
    tailscale_login: str
    display_name: str | None
    role: UserRole
    status: UserStatus


class JobOut(_Out):
    id: uuid.UUID
    idea_text: str | None
    length: JobLength
    quality: JobQuality
    regime: JobRegime | None
    status: JobStatus
    progress: float
    error: str | None
    created_at: datetime


class JobListItem(JobOut):
    # The released mp4, for the download link in the history.
    video_id: uuid.UUID | None = None


class AssetOut(_Out):
    id: uuid.UUID
    kind: AssetKind
    mime: str
    size_bytes: int
    expires_at: datetime | None


class IdeaOut(_Out):
    id: uuid.UUID
    text: str
    # How the app understood the friend's own idea.
    understood: str | None = None
    score: int
    # The app's understanding of the idea, or why it suggests this one.
    reason: str
    proposed_by: IdeaSource
    created_at: datetime
    approved_at: datetime | None


class JobDetail(JobOut):
    assets: list[AssetOut]
    idea: IdeaOut | None
    # How many jobs of the same lane are ahead: 0 means it is next. None when not waiting.
    queue_position: int | None


class ReleaseIn(BaseModel):
    # Empty keeps the quality chosen at upload.
    quality: JobQuality | None = None


@router.get("/me", response_model=Me)
def me(user: CurrentUser) -> object:
    """Works for pending and blocked people too, so the page can tell them."""
    return user


@router.get("/jobs")
def list_jobs(user: ApprovedUser, session: SessionDep) -> list[JobListItem]:
    jobs = session.scalars(jobs_of(user).order_by(Job.created_at.desc(), Job.id).limit(200)).all()
    videos: dict[uuid.UUID, uuid.UUID] = {}
    if jobs:
        rows = session.execute(
            assets_of(user)
            .with_only_columns(Asset.job_id, Asset.id)
            .where(Asset.kind == AssetKind.RELEASE, Asset.job_id.in_([job.id for job in jobs]))
        )
        videos = {row.job_id: row.id for row in rows}
    return [
        JobListItem.model_validate(job).model_copy(update={"video_id": videos.get(job.id)})
        for job in jobs
    ]


@router.get("/jobs/{job_id}")
def show_job(job_id: uuid.UUID, user: ApprovedUser, session: SessionDep) -> JobDetail:
    job = _job(session, user, job_id)
    assets = session.scalars(
        assets_of(user).where(Asset.job_id == job.id).order_by(Asset.kind, Asset.ordinal, Asset.id)
    )
    idea = _latest_idea(session, job.id)
    return JobDetail(
        **JobOut.model_validate(job).model_dump(),
        assets=[AssetOut.model_validate(asset) for asset in assets],
        idea=IdeaOut.model_validate(idea) if idea is not None else None,
        queue_position=queue.queue_position(session, job.id),
    )


@router.post("/jobs/{job_id}/approve")
def approve_idea(job_id: uuid.UUID, user: ApprovedUser, session: SessionDep) -> JobDetail:
    """The app's idea is fine: the video goes into the render queue."""
    job = _job(session, user, job_id)
    idea = _latest_idea(session, job.id)
    if idea is None:
        raise HTTPException(409, NOT_WAITING)
    session.execute(
        update(Idea)
        .where(Idea.id == idea.id)
        .values(approved_at=func.now())
        .execution_options(synchronize_session=False)
    )
    _move(session, job, JobStatus.RENDER_QUEUED, JobStatus.AWAITING_APPROVAL, NOT_WAITING)
    return show_job(job_id, user, session)


@router.post("/jobs/{job_id}/reject")
def reject_idea(job_id: uuid.UUID, user: ApprovedUser, session: SessionDep) -> JobDetail:
    """The app's idea is not wanted: the job closes. The photos go with a delete."""
    job = _job(session, user, job_id)
    _move(session, job, JobStatus.CLOSED, JobStatus.AWAITING_APPROVAL, NOT_WAITING)
    return show_job(job_id, user, session)


@router.post("/jobs/{job_id}/release")
def release(
    job_id: uuid.UUID,
    user: ApprovedUser,
    session: SessionDep,
    body: Annotated[ReleaseIn | None, Body()] = None,
) -> JobDetail:
    """The preview is good: make the final video, in the quality chosen now or at upload."""
    # The row lock makes a release and the expiry run take turns (see memcraft.expiry).
    # FOR NO KEY UPDATE, so that the worker can still save files for the job meanwhile.
    job = session.scalar(
        jobs_of(user)
        .where(Job.id == job_id)
        .with_for_update(key_share=True)
        .execution_options(populate_existing=True)
    )
    if job is None:
        raise HTTPException(404, "not found")
    if job.status != JobStatus.PREVIEW_READY:
        session.rollback()
        raise HTTPException(409, "This video has no preview waiting for release. Please reload.")
    if not _preview_is_live(session, job.id):
        session.rollback()
        raise HTTPException(409, "The preview has expired. Please make the video again.")
    if body is not None and body.quality is not None:
        job.quality = body.quality
    _move(session, job, JobStatus.RELEASE_QUEUED, JobStatus.PREVIEW_READY, NOT_WAITING)
    return show_job(job_id, user, session)


@router.delete("/jobs/{job_id}", status_code=204)
def delete_job(job_id: uuid.UUID, user: ApprovedUser, session: SessionDep, store: StoreDep) -> None:
    """Hard delete: the job, its idea, steps, photos and every file the worker made.

    A job a worker is running right now is refused with 409. Deleting under the worker's
    feet would leave it writing files for a row that is gone; once the step ends the job is
    in a state that can go. The row lock keeps a worker from claiming the job between the
    check and the delete. Files the web container cannot see wait in file_deletions for
    the worker.
    """
    job = session.scalar(jobs_of(user).where(Job.id == job_id).with_for_update())
    if job is None:
        raise HTTPException(404, "not found")
    if job.status in queue.RUNNING:
        session.rollback()
        raise HTTPException(409, BUSY)
    try:
        store.delete_job(session, job_id)
    except StorageError:
        raise HTTPException(503, "Files cannot be deleted right now. Please try later.") from None


@router.get("/jobs/{job_id}/events")
async def job_events(job_id: uuid.UUID, user: ApprovedUser, session: SessionDep) -> Response:
    """Server-sent events: the job's status, progress and queue place whenever they change.

    Each poll uses its own short transaction, so no connection stays checked out while the
    stream is open. Comments keep the line alive. The stream sends "end" once nothing will
    move without the friend, and "gone" when the job was deleted.
    """
    first, bind = await run_in_threadpool(_open_stream, session, user, job_id)
    if first is None:
        raise HTTPException(404, "not found")
    return StreamingResponse(
        _events(bind, first["owner_id"], job_id, first["state"]),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )


@router.get("/files/{asset_id}")
def get_file(
    asset_id: uuid.UUID,
    user: ApprovedUser,
    session: SessionDep,
    store: StoreDep,
    download: Annotated[bool, Query()] = False,
) -> FileResponse:
    asset = get_asset(session, user, asset_id)
    if asset is None:
        raise HTTPException(404, "not found")
    try:
        path = store.path_for(asset)
    except StorageError:
        raise HTTPException(503, "files are not available right now") from None
    if not path.is_file():
        raise HTTPException(404, "not found")
    return FileResponse(
        path,
        media_type=asset.mime,
        filename=f"memcraft-{asset.kind.value}{path.suffix}",
        content_disposition_type="attachment" if download else "inline",
        headers={"Cache-Control": "private, no-store"},
    )


def _job(session: Session, user: User, job_id: uuid.UUID) -> Job:
    job = get_job(session, user, job_id)
    if job is None:
        raise HTTPException(404, "not found")
    return job


def _latest_idea(session: Session, job_id: uuid.UUID) -> Idea | None:
    return session.scalar(
        select(Idea).where(Idea.job_id == job_id).order_by(Idea.created_at.desc(), Idea.id).limit(1)
    )


def _move(
    session: Session, job: Job, new_status: JobStatus, expected: JobStatus, message: str
) -> None:
    """One compare-and-set transition, committed; 409 when somebody else was first."""
    try:
        queue.transition(session, job, new_status, expected=expected)
        session.commit()
    except (StatusConflictError, TransitionError):
        session.rollback()
        raise HTTPException(409, message) from None


def _preview_is_live(session: Session, job_id: uuid.UUID) -> bool:
    """The same rule as memcraft.expiry, on the database clock."""
    now = session.scalar(select(func.now()))
    live = or_(
        Asset.expires_at > now,
        and_(Asset.expires_at.is_(None), Asset.created_at > now - PREVIEW_LIFETIME),
    )
    found = session.scalar(
        select(Asset.id)
        .where(Asset.job_id == job_id, Asset.kind == AssetKind.PREVIEW, live)
        .limit(1)
    )
    return found is not None


def _state(session: Session, owner_id: uuid.UUID, job_id: uuid.UUID) -> dict[str, Any] | None:
    # The owner filter of memcraft.web.owned, on columns only.
    row = session.execute(
        select(Job.status, Job.progress, Job.error).where(
            Job.id == job_id, Job.owner_id == owner_id
        )
    ).first()
    if row is None:
        return None
    return {
        "status": row.status.value,
        "progress": round(float(row.progress), 3),
        "queue_position": queue.queue_position(session, job_id)
        if row.status in queue.QUEUED
        else None,
        "error": row.error,
    }


def _open_stream(
    session: Session, user: User, job_id: uuid.UUID
) -> tuple[dict[str, Any] | None, Engine]:
    owner_id = user.id
    bind = session.get_bind()
    try:
        state = _state(session, owner_id, job_id)
    finally:
        session.close()  # hands the connection back before the stream starts
    if state is None:
        return None, bind
    return {"owner_id": owner_id, "state": state}, bind


def _poll(bind: Engine, owner_id: uuid.UUID, job_id: uuid.UUID) -> dict[str, Any] | None:
    with Session(bind) as session:
        return _state(session, owner_id, job_id)


def _message(data: dict[str, Any], event: str | None = None) -> str:
    head = f"event: {event}\n" if event else ""
    return f"{head}data: {json.dumps(data, separators=(',', ':'))}\n\n"


def _done(state: dict[str, Any]) -> bool:
    status = JobStatus(state["status"])
    return status in queue.FINAL or status in WAITING_FOR_USER


async def _events(
    bind: Engine, owner_id: uuid.UUID, job_id: uuid.UUID, last: dict[str, Any]
) -> AsyncIterator[str]:
    loop = asyncio.get_running_loop()
    started = quiet_since = loop.time()
    yield f"retry: {RETRY_MS}\n\n"
    yield _message(last)
    while not _done(last):
        if loop.time() - started >= STREAM_SECONDS:
            return  # without "end": the browser reconnects
        await asyncio.sleep(POLL_SECONDS)
        state = await run_in_threadpool(_poll, bind, owner_id, job_id)
        if state is None:
            yield _message({}, "gone")
            return
        if state != last:
            last, quiet_since = state, loop.time()
            yield _message(state)
        elif loop.time() - quiet_since >= HEARTBEAT_SECONDS:
            quiet_since = loop.time()
            yield ": keep-alive\n\n"
    yield _message(last, "end")
