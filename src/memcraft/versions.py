"""Versions of workflows, prompts, engines and recipes, and which of them a job uses.

A version is a draft until it is published as a candidate; from then on only its status moves:
draft -> candidate -> active -> retired, or candidate -> retired. A change is a new version. In
Postgres the triggers of migration 0002 enforce the same for every other writer.

Nothing here commits: every function flushes and leaves the commit to the caller.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from memcraft.models import (
    EngineSlot,
    EngineVersion,
    Idea,
    Job,
    JobRegime,
    JobStep,
    PromptRole,
    PromptVersion,
    Recipe,
    VersionStatus,
    WorkflowVersion,
)
from memcraft.workflows import (
    InvalidWorkflowError,
    Manifest,
    bind,
    check_workflow,
    workflow_sha256,
)

type Version = WorkflowVersion | PromptVersion | EngineVersion | Recipe

TRANSITIONS: dict[VersionStatus, frozenset[VersionStatus]] = {
    VersionStatus.DRAFT: frozenset({VersionStatus.CANDIDATE}),
    VersionStatus.CANDIDATE: frozenset({VersionStatus.ACTIVE, VersionStatus.RETIRED}),
    VersionStatus.ACTIVE: frozenset({VersionStatus.RETIRED}),
    VersionStatus.RETIRED: frozenset(),
}

# Rows with the same key are versions of one thing; at most one of them is active.
KEYS: dict[type[Version], tuple[str, ...]] = {
    WorkflowVersion: ("name",),
    PromptVersion: ("role", "model"),
    EngineVersion: ("slot", "name"),
    Recipe: ("name",),
}

# What a draft may still change. The key and the number stay.
_EDITABLE: dict[type[Version], frozenset[str]] = {
    WorkflowVersion: frozenset({"api_json", "manifest"}),
    PromptVersion: frozenset({"text"}),
    EngineVersion: frozenset({"workflow_version_id", "prompt_version_id", "limits"}),
    Recipe: frozenset({"regime", "keyframe_engine_id", "clip_engine_id"}),
}

_OPEN = frozenset({VersionStatus.CANDIDATE, VersionStatus.ACTIVE})


class VersionError(Exception):
    pass


class NotFoundError(VersionError):
    pass


class LifecycleError(VersionError):
    """This status does not allow it."""


class PublishedError(LifecycleError):
    """A published version cannot be edited; make a new version."""


class InvalidVersionError(VersionError):
    """The version does not fit the versions it is made of."""


class TamperedError(VersionError):
    """The stored content is not what was published."""


class AlreadyPinnedError(VersionError):
    """The row already records another version."""


def create_workflow_version(
    session: Session, api_json: dict[str, Any], manifest: dict[str, Any]
) -> WorkflowVersion:
    """A draft under the name and version of its manifest.

    Storing the same content again returns the row that is there; other content needs a new
    version number.
    """
    parsed = check_workflow(api_json, manifest)
    sha256 = workflow_sha256(api_json, manifest)
    existing = session.scalar(
        select(WorkflowVersion).where(
            WorkflowVersion.name == parsed.name, WorkflowVersion.version == parsed.version
        )
    )
    if existing is not None:
        verify_workflow(existing)
        if existing.sha256 != sha256:
            raise LifecycleError(
                f"{parsed.name} version {parsed.version} already exists with other content"
            )
        return existing
    return _add(
        session,
        WorkflowVersion(
            name=parsed.name,
            version=parsed.version,
            api_json=api_json,
            manifest=manifest,
            sha256=sha256,
        ),
    )


def create_prompt_version(
    session: Session, *, role: PromptRole, model: str, text: str
) -> PromptVersion:
    if not model or not text.strip():
        raise InvalidVersionError("a prompt needs a model and a text")
    row = PromptVersion(role=role, model=model, text=text)
    return _add_next(session, row)


def create_engine_version(
    session: Session,
    *,
    slot: EngineSlot,
    name: str,
    workflow_version_id: uuid.UUID | None = None,
    prompt_version_id: uuid.UUID | None = None,
    limits: dict[str, Any] | None = None,
) -> EngineVersion:
    row = EngineVersion(
        slot=slot,
        name=name,
        workflow_version_id=workflow_version_id,
        prompt_version_id=prompt_version_id,
        limits=limits or {},
    )
    _check_parts(session, row)
    return _add_next(session, row)


def create_recipe_version(
    session: Session,
    *,
    name: str,
    regime: JobRegime,
    clip_engine_id: uuid.UUID,
    keyframe_engine_id: uuid.UUID | None = None,
) -> Recipe:
    row = Recipe(
        name=name,
        regime=regime,
        clip_engine_id=clip_engine_id,
        keyframe_engine_id=keyframe_engine_id,
    )
    _check_parts(session, row)
    return _add_next(session, row)


def edit_draft(session: Session, version: Version, **changes: Any) -> None:
    """Change the content of a draft. A workflow gets its new sha256 at the same time."""
    unknown = sorted(set(changes) - _EDITABLE[type(version)])
    if unknown:
        raise InvalidVersionError(f"cannot change {', '.join(unknown)}")
    _lock(session, version)
    if version.status is not VersionStatus.DRAFT:
        raise PublishedError(f"{_label(version)} is {version.status.value}; make a new version")
    if isinstance(version, WorkflowVersion):
        api_json = changes.get("api_json", version.api_json)
        manifest = changes.get("manifest", version.manifest)
        _check_number(version, check_workflow(api_json, manifest))
        changes["sha256"] = workflow_sha256(api_json, manifest)
    elif isinstance(version, PromptVersion) and not changes.get("text", "x").strip():
        raise InvalidVersionError("a prompt needs a text")
    before = {name: getattr(version, name) for name in changes}
    for name, value in changes.items():
        setattr(version, name, value)
    try:
        with session.no_autoflush:
            _check_parts(session, version)
    except VersionError:
        for name, value in before.items():
            setattr(version, name, value)
        raise
    session.flush()


def promote(session: Session, version: Version, to: VersionStatus) -> None:
    """One step forward, to candidate or active.

    Publishing freezes the content. Activating retires the version that was active until now,
    in the same transaction; jobs that pinned it keep it.
    """
    if to not in _OPEN:
        raise LifecycleError("promote() goes to candidate or active; use retire() to retire")
    # The whole family, so that of two activations at once the second sees the first.
    family = _lock(session, version)
    _check_transition(version, to)
    if isinstance(version, WorkflowVersion):
        verify_workflow(version)
    _check_parts(session, version, published=to)
    if to is VersionStatus.ACTIVE:
        for other in family:
            if other.status is VersionStatus.ACTIVE:
                other.status = VersionStatus.RETIRED
        session.flush()  # before the new one, or the unique index would see two
    version.status = to
    session.flush()


def retire(session: Session, version: Version) -> None:
    """Final. New jobs cannot use it; jobs that pinned it keep it."""
    _lock(session, version)
    _check_transition(version, VersionStatus.RETIRED)
    version.status = VersionStatus.RETIRED
    session.flush()


def get_active[V: Version](session: Session, kind: type[V], **key: Any) -> V | None:
    """The active version, such as get_active(session, PromptVersion, role=..., model=...)."""
    if set(key) != set(KEYS[kind]):
        raise TypeError(f"{kind.__name__} is found by {', '.join(KEYS[kind])}")
    row = session.scalar(select(kind).filter_by(status=VersionStatus.ACTIVE, **key))
    if isinstance(row, WorkflowVersion):
        verify_workflow(row)
    return row


def verify_workflow(version: WorkflowVersion) -> Manifest:
    """The manifest, once the stored content is proved to be what the sha256 was taken of."""
    if workflow_sha256(version.api_json, version.manifest) != version.sha256:
        raise TamperedError(f"{_label(version)} does not match its sha256")
    parsed = check_workflow(version.api_json, version.manifest)
    _check_number(version, parsed)
    return parsed


def bind_workflow(version: WorkflowVersion, values: Mapping[str, Any]) -> dict[str, Any]:
    """A filled graph ready for ComfyUI. Refuses content that fails verify_workflow()."""
    return bind(version.api_json, verify_workflow(version), values)


def pin_job(
    session: Session, job: Job, *, recipe: Recipe, critic: EngineVersion | None = None
) -> None:
    """Record once which recipe and critic the job uses. No critic means it was off."""
    if _is_pinned(session, job, "recipe_id", recipe_id=recipe.id, critic_engine_id=_id(critic)):
        return
    for version in (recipe, critic):
        if version is not None:
            _refresh(session, version)
            if version.status not in _OPEN:
                raise LifecycleError(f"{_label(version)} is {version.status.value}")
    if critic is not None and critic.slot is not EngineSlot.CRITIC:
        raise InvalidVersionError(f"{_label(critic)} is not a critic")
    if job.regime is not None and job.regime.recipe_regime is not recipe.regime:
        raise InvalidVersionError(f"the job is {job.regime.value}, the recipe is not")
    job.recipe_id = recipe.id
    job.critic_engine_id = _id(critic)
    session.flush()


def pin_step(
    session: Session,
    step: JobStep,
    *,
    workflow: WorkflowVersion | None = None,
    prompt: PromptVersion | None = None,
) -> None:
    """Record once which workflow and prompt the step uses.

    A retired version is fine, because the job may have pinned it earlier; a draft never is.
    """
    pins = {}
    if workflow is not None:
        pins["workflow_version_id"] = workflow
    if prompt is not None:
        pins["prompt_version_id"] = prompt
    _pin_parts(session, step, pins)


def pin_idea(session: Session, idea: Idea, prompt: PromptVersion) -> None:
    """Record once which prompt the idea came from."""
    _pin_parts(session, idea, {"prompt_version_id": prompt})


def pinned_workflow(session: Session, job: Job, slot: EngineSlot) -> WorkflowVersion:
    """The workflow the job has pinned for a slot, verified, whatever is active now."""
    engine_id = None
    if slot is EngineSlot.CRITIC:
        engine_id = job.critic_engine_id
    elif job.recipe_id is not None:
        recipe = session.get_one(Recipe, job.recipe_id)
        engine_id = recipe.clip_engine_id if slot is EngineSlot.CLIP else recipe.keyframe_engine_id
    engine = session.get(EngineVersion, engine_id) if engine_id is not None else None
    if engine is None or engine.workflow_version_id is None:
        raise NotFoundError(f"the job has no workflow for {slot.value}")
    workflow = session.get_one(WorkflowVersion, engine.workflow_version_id)
    verify_workflow(workflow)
    return workflow


def _add[V: Version](session: Session, row: V) -> V:
    row.status = VersionStatus.DRAFT
    session.add(row)
    session.flush()
    return row


def _add_next[V: Version](session: Session, row: V) -> V:
    # Two at once get the same number, and the unique constraint refuses the second.
    kind = type(row)
    last = session.scalar(select(func.max(kind.version)).filter_by(**_key(row)))
    row.version = (last or 0) + 1
    return _add(session, row)


def _key(row: Version) -> dict[str, Any]:
    return {column: getattr(row, column) for column in KEYS[type(row)]}


def _id(row: Version | None) -> uuid.UUID | None:
    return None if row is None else row.id


def _label(row: Version) -> str:
    parts = [str(part) for part in _key(row).values()]
    return f"{type(row).__tablename__} {' '.join(parts)} version {row.version}"


def _lock[V: Version](session: Session, version: V) -> list[V]:
    """Lock and reload every version of the same thing, in one order for every caller."""
    session.flush()
    kind = type(version)
    family = session.scalars(
        select(kind)
        .filter_by(**_key(version))
        .order_by(kind.version)
        .with_for_update()
        .execution_options(populate_existing=True)
    ).all()
    if version not in family:
        raise NotFoundError(f"{_label(version)} is not in the database")
    return list(family)


def _refresh(session: Session, version: Version) -> None:
    session.flush()
    # FOR SHARE, so that a version cannot retire between this check and the caller's commit.
    session.refresh(version, with_for_update={"read": True})


def _check_transition(version: Version, to: VersionStatus) -> None:
    if to not in TRANSITIONS[version.status]:
        raise LifecycleError(
            f"{_label(version)} cannot go from {version.status.value} to {to.value}"
        )


def _check_number(version: WorkflowVersion, manifest: Manifest) -> None:
    if (manifest.name, manifest.version) != (version.name, version.version):
        raise InvalidWorkflowError(
            f"{_label(version)} has the manifest of {manifest.name} version {manifest.version}"
        )


def _check_parts(
    session: Session, version: Version, published: VersionStatus | None = None
) -> None:
    """The versions this one is made of fit, and are as far as it is about to be."""
    parts: list[Version | None] = []
    if isinstance(version, EngineVersion):
        workflow = _part(session, WorkflowVersion, version.workflow_version_id)
        prompt = _part(session, PromptVersion, version.prompt_version_id)
        if version.slot is not EngineSlot.CRITIC and workflow is None:
            raise InvalidVersionError(f"a {version.slot.value} engine needs a workflow")
        if prompt is not None and (
            (version.slot is EngineSlot.CRITIC) != (prompt.role is PromptRole.CRITIC)
        ):
            raise InvalidVersionError("a critic prompt belongs to a critic, and only there")
        parts = [workflow, prompt]
    elif isinstance(version, Recipe):
        clip = _part(session, EngineVersion, version.clip_engine_id)
        keyframe = _part(session, EngineVersion, version.keyframe_engine_id)
        if clip is None or clip.slot is not EngineSlot.CLIP:
            raise InvalidVersionError("a recipe needs a clip engine")
        if keyframe is not None and keyframe.slot is not EngineSlot.KEYFRAME:
            raise InvalidVersionError("the keyframe engine is for another slot")
        if version.regime is JobRegime.KEYFRAMES and keyframe is None:
            raise InvalidVersionError("the keyframes regime needs a keyframe engine")
        parts = [clip, keyframe]
    if published is None:
        return
    enough = {VersionStatus.ACTIVE} if published is VersionStatus.ACTIVE else _OPEN
    for part in parts:
        if part is not None:
            _refresh(session, part)
            if part.status not in enough:
                raise LifecycleError(
                    f"{_label(version)} cannot be {published.value}: "
                    f"{_label(part)} is {part.status.value}"
                )


def _part[V: Version](session: Session, kind: type[V], part_id: uuid.UUID | None) -> V | None:
    if part_id is None:
        return None
    part = session.get(kind, part_id)
    if part is None:
        raise NotFoundError(f"no row {part_id} in {kind.__tablename__}")
    return part


def _pin_parts(
    session: Session, row: JobStep | Idea, pins: dict[str, WorkflowVersion | PromptVersion]
) -> None:
    for column, version in pins.items():
        if _is_pinned(session, row, column, **{column: version.id}):
            continue
        _refresh(session, version)
        if version.status is VersionStatus.DRAFT:
            raise LifecycleError(f"{_label(version)} is a draft")
        if isinstance(version, WorkflowVersion):
            verify_workflow(version)
        setattr(row, column, version.id)
    session.flush()


def _is_pinned(session: Session, row: Job | JobStep | Idea, marker: str, **wanted: Any) -> bool:
    """False if the row is free, True if it has exactly these pins. Locks and reloads the row."""
    session.flush()
    session.refresh(row, with_for_update=True)
    if getattr(row, marker) is None:
        return False
    if any(getattr(row, column) != value for column, value in wanted.items()):
        raise AlreadyPinnedError(f"{row.__tablename__} {row.id} already has another version")
    return True
