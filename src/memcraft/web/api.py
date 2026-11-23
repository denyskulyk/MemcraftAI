from __future__ import annotations

import uuid
from datetime import datetime

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict

from memcraft.models import (
    Asset,
    AssetKind,
    Job,
    JobLength,
    JobQuality,
    JobStatus,
    UserRole,
    UserStatus,
)
from memcraft.storage import StorageError
from memcraft.web.auth import ApprovedUser, CurrentUser, SessionDep, StoreDep
from memcraft.web.owned import assets_of, get_asset, get_job, jobs_of

router = APIRouter(prefix="/api")


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
    status: JobStatus
    progress: float
    error: str | None
    created_at: datetime


class AssetOut(_Out):
    id: uuid.UUID
    kind: AssetKind
    mime: str
    size_bytes: int
    expires_at: datetime | None


class JobDetail(JobOut):
    assets: list[AssetOut]


@router.get("/me", response_model=Me)
def me(user: CurrentUser) -> object:
    """Works for pending and blocked people too, so the page can tell them."""
    return user


@router.get("/jobs", response_model=list[JobOut])
def list_jobs(user: ApprovedUser, session: SessionDep) -> object:
    return session.scalars(jobs_of(user).order_by(Job.created_at.desc(), Job.id).limit(200)).all()


@router.get("/jobs/{job_id}")
def show_job(job_id: uuid.UUID, user: ApprovedUser, session: SessionDep) -> JobDetail:
    job = get_job(session, user, job_id)
    if job is None:
        raise HTTPException(404, "not found")
    assets = session.scalars(
        assets_of(user).where(Asset.job_id == job.id).order_by(Asset.kind, Asset.ordinal, Asset.id)
    )
    return JobDetail(
        **JobOut.model_validate(job).model_dump(),
        assets=[AssetOut.model_validate(asset) for asset in assets],
    )


@router.get("/files/{asset_id}")
def get_file(
    asset_id: uuid.UUID, user: ApprovedUser, session: SessionDep, store: StoreDep
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
        content_disposition_type="inline",
        headers={"Cache-Control": "private, no-store"},
    )
