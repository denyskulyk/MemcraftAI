"""Schema v1. Every status is text with a CHECK, so a later migration can add a value."""

from __future__ import annotations

import enum
import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    LargeBinary,
    MetaData,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class UserRole(enum.StrEnum):
    USER = "user"
    ADMIN = "admin"


class UserStatus(enum.StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    BLOCKED = "blocked"


class JobLength(enum.StrEnum):
    SHORT = "short"
    LONG = "long"


class JobQuality(enum.StrEnum):
    HD720 = "hd720"
    FHD1080 = "fhd1080"


class JobLane(enum.StrEnum):
    IDEA = "idea"
    RENDER = "render"


class JobRegime(enum.StrEnum):
    MULTISHOT = "multishot"
    KEYFRAMES = "keyframes"
    # Admin only (2026-11-26): 1 to 3 photos and a typed idea; Qwen3.8 Flash-Next directs with
    # thinking on, no inventory, no SAM, people cut out 1:1, no critic. Uses keyframes recipes.
    REGIME3 = "regime3"

    @property
    def recipe_regime(self) -> JobRegime:
        """The regime of the recipes this regime renders with."""
        return JobRegime.KEYFRAMES if self is JobRegime.REGIME3 else self


class JobStatus(enum.StrEnum):
    IDEA_QUEUED = "idea_queued"
    IDEA_RUNNING = "idea_running"
    AWAITING_APPROVAL = "awaiting_approval"
    CLOSED = "closed"
    RENDER_QUEUED = "render_queued"
    RENDERING = "rendering"
    PREVIEW_READY = "preview_ready"
    EXPIRED = "expired"
    RELEASE_QUEUED = "release_queued"
    RELEASING = "releasing"
    RELEASED = "released"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StepName(enum.StrEnum):
    IDEA = "idea"
    SCENARIO = "scenario"
    KEYFRAME = "keyframe"
    CLIP = "clip"
    CHECK = "check"
    JOIN = "join"
    RELEASE = "release"


class StepStatus(enum.StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class IdeaSource(enum.StrEnum):
    USER = "user"
    GEMMA = "gemma"


class AssetKind(enum.StrEnum):
    PHOTO = "photo"
    GEMMA_COPY = "gemma_copy"
    KEYFRAME = "keyframe"
    LATENT = "latent"
    CLIP = "clip"
    PREVIEW = "preview"
    RELEASE = "release"


# Keyframe step ordinals from here on are a scenario's asset pictures: background, people and
# props, drawn before the keyframes (E16).
ASSET_ORDINAL = 200


class AssetStore(enum.StrEnum):
    SHARED = "shared"
    PRIVATE = "private"


class VersionStatus(enum.StrEnum):
    DRAFT = "draft"
    CANDIDATE = "candidate"
    ACTIVE = "active"
    RETIRED = "retired"


class PromptRole(enum.StrEnum):
    CHAT = "chat"
    DIRECTOR_MULTISHOT = "director_multishot"
    DIRECTOR_KEYFRAMES = "director_keyframes"
    CRITIC = "critic"


class ChallengePurpose(enum.StrEnum):
    REGISTER = "register"
    SIGN_IN = "sign_in"


class EngineSlot(enum.StrEnum):
    KEYFRAME = "keyframe"
    CLIP = "clip"
    CRITIC = "critic"


def _text_enum(kind: type[enum.StrEnum], name: str) -> Enum:
    return Enum(
        kind,
        name=name,
        native_enum=False,
        create_constraint=True,
        length=32,
        values_callable=lambda members: [member.value for member in members],
    )


def _one_active(table: str, *columns: str) -> Index:
    active = text("status = 'active'")
    return Index(
        f"uq_{table}_active", *columns, unique=True, postgresql_where=active, sqlite_where=active
    )


class Base(DeclarativeBase):
    metadata = MetaData(
        naming_convention={
            "ix": "ix_%(table_name)s_%(column_0_N_name)s",
            "uq": "uq_%(table_name)s_%(column_0_N_name)s",
            "ck": "ck_%(table_name)s_%(constraint_name)s",
            "fk": "fk_%(table_name)s_%(column_0_name)s",
            "pk": "pk_%(table_name)s",
        }
    )
    type_annotation_map = {  # noqa: RUF012
        datetime: DateTime(timezone=True),
        dict[str, Any]: JSONB,
    }


def _id() -> Mapped[uuid.UUID]:
    return mapped_column(primary_key=True, default=uuid.uuid4)


def _now() -> Mapped[datetime]:
    return mapped_column(server_default=func.now())


class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = _id()
    tailscale_login: Mapped[str] = mapped_column(unique=True)
    display_name: Mapped[str | None]
    role: Mapped[UserRole] = mapped_column(
        _text_enum(UserRole, "role"), server_default=UserRole.USER.value
    )
    status: Mapped[UserStatus] = mapped_column(
        _text_enum(UserStatus, "status"), server_default=UserStatus.PENDING.value
    )
    created_at: Mapped[datetime] = _now()
    approved_at: Mapped[datetime | None]


class WorkflowVersion(Base):
    __tablename__ = "workflow_versions"
    __table_args__ = (UniqueConstraint("name", "version"), _one_active("workflow_versions", "name"))

    id: Mapped[uuid.UUID] = _id()
    name: Mapped[str]
    version: Mapped[int]
    api_json: Mapped[dict[str, Any]]
    manifest: Mapped[dict[str, Any]]
    sha256: Mapped[str]
    status: Mapped[VersionStatus] = mapped_column(
        _text_enum(VersionStatus, "status"), server_default=VersionStatus.DRAFT.value
    )
    created_at: Mapped[datetime] = _now()


class PromptVersion(Base):
    __tablename__ = "prompt_versions"
    __table_args__ = (
        UniqueConstraint("role", "model", "version"),
        _one_active("prompt_versions", "role", "model"),
    )

    id: Mapped[uuid.UUID] = _id()
    role: Mapped[PromptRole] = mapped_column(_text_enum(PromptRole, "role"))
    # The critic has one prompt per model, so the model is part of the key.
    model: Mapped[str]
    version: Mapped[int]
    text: Mapped[str]
    status: Mapped[VersionStatus] = mapped_column(
        _text_enum(VersionStatus, "status"), server_default=VersionStatus.DRAFT.value
    )
    created_at: Mapped[datetime] = _now()


class EngineVersion(Base):
    """One model in a swappable slot. "Off" for the critic is simply no engine on the job."""

    __tablename__ = "engines"
    __table_args__ = (
        UniqueConstraint("slot", "name", "version"),
        _one_active("engines", "slot", "name"),
    )

    id: Mapped[uuid.UUID] = _id()
    slot: Mapped[EngineSlot] = mapped_column(_text_enum(EngineSlot, "slot"))
    name: Mapped[str]
    version: Mapped[int]
    # Empty for Gemma, which runs through llama.cpp and not a ComfyUI workflow.
    workflow_version_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("workflow_versions.id")
    )
    prompt_version_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("prompt_versions.id"))
    # Frame rule, longest clip, side divisor, cuts, sound, number of references.
    limits: Mapped[dict[str, Any]] = mapped_column(server_default=text("'{}'::jsonb"))
    status: Mapped[VersionStatus] = mapped_column(
        _text_enum(VersionStatus, "status"), server_default=VersionStatus.DRAFT.value
    )
    created_at: Mapped[datetime] = _now()


class Recipe(Base):
    __tablename__ = "recipes"
    __table_args__ = (UniqueConstraint("name", "version"), _one_active("recipes", "name"))

    id: Mapped[uuid.UUID] = _id()
    name: Mapped[str]
    version: Mapped[int]
    regime: Mapped[JobRegime] = mapped_column(_text_enum(JobRegime, "regime"))
    keyframe_engine_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("engines.id"))
    clip_engine_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("engines.id"))
    status: Mapped[VersionStatus] = mapped_column(
        _text_enum(VersionStatus, "status"), server_default=VersionStatus.DRAFT.value
    )
    created_at: Mapped[datetime] = _now()


class Job(Base):
    """This table is also the queue: workers claim rows by lane and status."""

    __tablename__ = "jobs"
    __table_args__ = (
        CheckConstraint("photo_count BETWEEN 1 AND 3", name="photo_count"),
        CheckConstraint("progress BETWEEN 0 AND 1", name="progress"),
        Index("ix_jobs_queue", "lane", "status", "queued_at"),
        Index("ix_jobs_owner_id_created_at", "owner_id", "created_at"),
    )

    id: Mapped[uuid.UUID] = _id()
    owner_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    # What the friend typed; empty means the app proposes an idea.
    idea_text: Mapped[str | None]
    photo_count: Mapped[int]
    length: Mapped[JobLength] = mapped_column(_text_enum(JobLength, "length"))
    quality: Mapped[JobQuality] = mapped_column(_text_enum(JobQuality, "quality"))
    lane: Mapped[JobLane] = mapped_column(
        _text_enum(JobLane, "lane"), server_default=JobLane.IDEA.value
    )
    status: Mapped[JobStatus] = mapped_column(
        _text_enum(JobStatus, "status"), server_default=JobStatus.IDEA_QUEUED.value
    )
    regime: Mapped[JobRegime | None] = mapped_column(_text_enum(JobRegime, "regime"))
    # The people the idea names, as memcraft.cast stores them: a list of
    # {"tag", "photo", "position"}. Null when the friend named no one.
    cast: Mapped[list[dict[str, Any]] | None] = mapped_column(JSONB)
    # Pinned when the job starts. A null critic means the critic was off.
    recipe_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("recipes.id"))
    critic_engine_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("engines.id"))
    progress: Mapped[float] = mapped_column(server_default=text("0"))
    error: Mapped[str | None]
    queued_at: Mapped[datetime] = _now()
    claimed_by: Mapped[str | None]
    claimed_at: Mapped[datetime | None]
    created_at: Mapped[datetime] = _now()
    updated_at: Mapped[datetime] = mapped_column(server_default=func.now(), onupdate=func.now())


class JobStep(Base):
    __tablename__ = "job_steps"
    __table_args__ = (
        UniqueConstraint("job_id", "step", "ordinal", "attempt"),
        CheckConstraint("progress BETWEEN 0 AND 1", name="progress"),
        CheckConstraint("attempt >= 1", name="attempt"),
    )

    id: Mapped[uuid.UUID] = _id()
    job_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("jobs.id", ondelete="CASCADE"))
    step: Mapped[StepName] = mapped_column(_text_enum(StepName, "step"))
    # Which keyframe or clip of the job this is, starting at 0.
    ordinal: Mapped[int] = mapped_column(server_default=text("0"))
    attempt: Mapped[int] = mapped_column(server_default=text("1"))
    status: Mapped[StepStatus] = mapped_column(
        _text_enum(StepStatus, "status"), server_default=StepStatus.PENDING.value
    )
    inputs_hash: Mapped[str | None]
    inputs: Mapped[dict[str, Any] | None]
    output: Mapped[dict[str, Any] | None]
    workflow_version_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("workflow_versions.id")
    )
    prompt_version_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("prompt_versions.id"))
    progress: Mapped[float] = mapped_column(server_default=text("0"))
    error: Mapped[str | None]
    created_at: Mapped[datetime] = _now()
    started_at: Mapped[datetime | None]
    finished_at: Mapped[datetime | None]


class Idea(Base):
    __tablename__ = "ideas"
    __table_args__ = (CheckConstraint("score BETWEEN 1 AND 10", name="score"),)

    id: Mapped[uuid.UUID] = _id()
    job_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("jobs.id", ondelete="CASCADE"), index=True)
    text: Mapped[str]
    # How the chat agent understood the friend's own idea; empty for its own ideas.
    understood: Mapped[str | None]
    # Null since 2026-11-24: ideas are no longer scored. Older rows keep their 1-10.
    score: Mapped[int | None]
    reason: Mapped[str]
    proposed_by: Mapped[IdeaSource] = mapped_column(_text_enum(IdeaSource, "proposed_by"))
    prompt_version_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("prompt_versions.id"))
    created_at: Mapped[datetime] = _now()
    # Empty while the app's own idea waits for the friend.
    approved_at: Mapped[datetime | None]


class Asset(Base):
    """One file. Rows are removed only through memcraft.storage, which removes the file too."""

    __tablename__ = "assets"
    __table_args__ = (
        UniqueConstraint("store", "path"),
        Index(
            "ix_assets_expires_at", "expires_at", postgresql_where=text("expires_at IS NOT NULL")
        ),
    )

    id: Mapped[uuid.UUID] = _id()
    owner_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="RESTRICT"), index=True
    )
    job_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("jobs.id", ondelete="RESTRICT"), index=True
    )
    step_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("job_steps.id", ondelete="SET NULL")
    )
    kind: Mapped[AssetKind] = mapped_column(_text_enum(AssetKind, "kind"))
    # Which photo, keyframe or clip of the job this is, starting at 0.
    ordinal: Mapped[int] = mapped_column(server_default=text("0"))
    store: Mapped[AssetStore] = mapped_column(_text_enum(AssetStore, "store"))
    # Relative to the store's folder, with forward slashes: the web container and Windows
    # see that folder at different places.
    path: Mapped[str]
    mime: Mapped[str]
    size_bytes: Mapped[int] = mapped_column(BigInteger)
    sha256: Mapped[str]
    created_at: Mapped[datetime] = _now()
    expires_at: Mapped[datetime | None]


class FileDeletion(Base):
    """Files whose row is gone but which still have to leave the disk.

    The web container cannot see private files, and Windows refuses to delete an open file.
    The worker empties this table. A path is a random id, so it says nothing about a person.
    """

    __tablename__ = "file_deletions"
    __table_args__ = (UniqueConstraint("store", "path"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    store: Mapped[AssetStore] = mapped_column(_text_enum(AssetStore, "store"))
    path: Mapped[str]
    created_at: Mapped[datetime] = _now()


class Setting(Base):
    __tablename__ = "settings"

    key: Mapped[str] = mapped_column(primary_key=True)
    value: Mapped[Any] = mapped_column(JSONB)
    updated_at: Mapped[datetime] = mapped_column(server_default=func.now(), onupdate=func.now())


class Passkey(Base):
    """An admin's passkey. Only the public key is kept."""

    __tablename__ = "passkeys"

    id: Mapped[uuid.UUID] = _id()
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    credential_id: Mapped[bytes] = mapped_column(LargeBinary, unique=True)
    public_key: Mapped[bytes] = mapped_column(LargeBinary)
    sign_count: Mapped[int] = mapped_column(BigInteger, server_default=text("0"))
    name: Mapped[str]
    created_at: Mapped[datetime] = _now()
    last_used_at: Mapped[datetime | None]


class AuthChallenge(Base):
    """A WebAuthn challenge, used once and only before it expires."""

    __tablename__ = "auth_challenges"

    id: Mapped[uuid.UUID] = _id()
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    purpose: Mapped[ChallengePurpose] = mapped_column(_text_enum(ChallengePurpose, "purpose"))
    challenge: Mapped[bytes] = mapped_column(LargeBinary)
    expires_at: Mapped[datetime]


class AdminSession(Base):
    """A signed-in admin. The cookie holds the token; only its sha256 is stored."""

    __tablename__ = "admin_sessions"

    token_sha256: Mapped[str] = mapped_column(primary_key=True)
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    passkey_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("passkeys.id", ondelete="CASCADE"))
    created_at: Mapped[datetime] = _now()
    expires_at: Mapped[datetime]
