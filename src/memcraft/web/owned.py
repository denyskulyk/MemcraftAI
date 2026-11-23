"""The only place web code looks up jobs and files: always filtered by the owner.

Somebody else's id and an unknown id give the same answer, so ids cannot be probed.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import Select, or_, select
from sqlalchemy.orm import Session

from memcraft.models import Asset, AssetKind, AssetStore, Job, User

# Keyframes, latents and clips are the worker's business.
VISIBLE_KINDS = (AssetKind.PHOTO, AssetKind.PREVIEW, AssetKind.RELEASE)


def jobs_of(user: User) -> Select[tuple[Job]]:
    return select(Job).where(Job.owner_id == user.id)


def assets_of(user: User) -> Select[tuple[Asset]]:
    """The caller's files, as long as the job they belong to is the caller's too."""
    return (
        select(Asset)
        .join(Job, Job.id == Asset.job_id)
        .where(
            Asset.owner_id == user.id,
            Job.owner_id == user.id,
            Asset.kind.in_(VISIBLE_KINDS),
            Asset.store == AssetStore.SHARED,
            or_(Asset.expires_at.is_(None), Asset.expires_at > datetime.now(UTC)),
        )
    )


def get_job(session: Session, user: User, job_id: uuid.UUID) -> Job | None:
    return session.scalar(jobs_of(user).where(Job.id == job_id))


def get_asset(session: Session, user: User, asset_id: uuid.UUID) -> Asset | None:
    return session.scalar(assets_of(user).where(Asset.id == asset_id))
