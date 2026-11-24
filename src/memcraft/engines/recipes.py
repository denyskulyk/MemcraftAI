"""Recipes as data: how a job of each regime and length is cut into keyframes and runs.

A recipe row names a clip engine and, for keyframes, a keyframe engine; the job pins it with
versions.pin_job. The rules below hold what E0 settled. The release reuses the clip engine the
job pinned (versions.pinned_workflow), never the one active now.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from memcraft import versions
from memcraft.engines.base import Adapter, EngineError
from memcraft.engines.clips import LAST, LtxClips
from memcraft.engines.keyframes import QwenImageKeyframes
from memcraft.models import (
    EngineSlot,
    EngineVersion,
    Job,
    JobLength,
    JobRegime,
    Recipe,
    VersionStatus,
    WorkflowVersion,
)
from memcraft.workflows import load_workflow

ADAPTERS: dict[str, type[Adapter]] = {
    QwenImageKeyframes.workflow_name: QwenImageKeyframes,
    LtxClips.workflow_name: LtxClips,
}
WORKFLOW_FOLDERS = {
    QwenImageKeyframes.workflow_name: "keyframe-qwen-image-2.1-v4",
    LtxClips.workflow_name: "clip-ltx-2.5-v2",
}


class First(StrEnum):
    PHOTO = "photo"
    KEYFRAME = "keyframe"
    # The real last frame of the run before (E0.5: continue, do not meet at a shared keyframe).
    PREVIOUS_LAST_FRAME = "previous_last_frame"


@dataclass(frozen=True)
class RunPlan:
    index: int
    frames: int
    first: First
    # Index into Plan.keyframes when first is KEYFRAME.
    first_keyframe: int | None
    # (index into Plan.keyframes, frame of the clip; -1 is the last frame).
    guides: tuple[tuple[int, int], ...]
    first_strength: float
    # Shots the director writes into this run by default, and at most; 1 is a single take.
    shots: int
    max_shots: int
    # How hard the middle and end keyframes pull. 1.0 makes LTX jump to them and hold still;
    # 0.5 leaves room to draw the motion, with no face loss in Denys's review (E15).
    guide_strength: float = 1.0


@dataclass(frozen=True)
class Plan:
    regime: JobRegime
    # Names of the keyframes to draw, in order; the first one is drawn before the others and
    # is an extra reference of each later one when the scene has 1 or 2 people (the director
    # decides, memcraft.roles.director.CHAINED_PEOPLE).
    keyframes: tuple[str, ...]
    runs: tuple[RunPlan, ...]

    @property
    def frames(self) -> int:
        """Frames of the joined video: each later run drops its first frame."""
        return sum(run.frames for run in self.runs) - (len(self.runs) - 1)


@dataclass(frozen=True)
class RecipeRules:
    regime: JobRegime
    frames_per_run: int = 241
    runs_short: int = 1
    runs_long: int = 2
    # Multi-shot: one photo makes a single take only. With 2 shots the faces held in 2 of 4
    # videos, in a single take in 4 of 5 (E0 verdicts).
    default_shots: int = 1
    max_shots: int = 1
    # The first run of a multi-shot job starts from the photo at the strength of LTX v8.
    photo_first_strength: float = 0.7
    guide_strength: float = 0.5


RULES: dict[str, RecipeRules] = {
    "multishot": RecipeRules(JobRegime.MULTISHOT),
    "keyframes": RecipeRules(JobRegime.KEYFRAMES, default_shots=1, max_shots=1),
}


def plan(
    rules: RecipeRules,
    length: JobLength,
    frames: int | Sequence[int] | None = None,
    keyframes: int | None = None,
) -> Plan:
    """Keyframes to draw and runs to render.

    ``frames`` overrides the default length of every run, or gives one length per run.
    ``keyframes`` is how many keyframes the scenario holds: by default 2 (start, end) for one
    run and 3 (start, middle, end) for two; 3 for one run is how scenarios from before E16 were
    planned.
    """
    count = rules.runs_long if length is JobLength.LONG else rules.runs_short
    if frames is None or isinstance(frames, int):
        lengths = [frames or rules.frames_per_run] * count
    else:
        lengths = list(frames)
        if len(lengths) != count:
            raise EngineError(f"a {length.value} job has {count} run(s), not {len(lengths)}")
    runs = []
    if rules.regime is JobRegime.MULTISHOT:
        for i in range(count):
            runs.append(
                RunPlan(
                    index=i,
                    frames=lengths[i],
                    first=First.PHOTO if i == 0 else First.PREVIOUS_LAST_FRAME,
                    first_keyframe=None,
                    guides=(),
                    first_strength=rules.photo_first_strength if i == 0 else 1.0,
                    shots=rules.default_shots,
                    max_shots=rules.max_shots,
                )
            )
        return Plan(rules.regime, (), tuple(runs))
    middle = 8 * round(lengths[0] / 16)
    g = rules.guide_strength
    if count == 1:
        if (keyframes or 2) == 2:
            # E16: a middle keyframe is one more picture to drift (light, duplicates) and held
            # the action (E15), so one run goes from the start keyframe to the end keyframe.
            runs.append(RunPlan(0, lengths[0], First.KEYFRAME, 0, ((1, LAST),), 1.0, 1, 1, g))
            return Plan(rules.regime, ("start", "end"), tuple(runs))
        # E0.4: start, middle and end in one run move properly when the keyframes match.
        runs.append(
            RunPlan(0, lengths[0], First.KEYFRAME, 0, ((1, middle), (2, LAST)), 1.0, 1, 1, g)
        )
    else:
        # E16: run 1 ends on the middle keyframe (guide 0.5). Its free end (E0.5) let LTX
        # drift into a close-up, and run 2 had to zoom back out to the end keyframe.
        runs.append(RunPlan(0, lengths[0], First.KEYFRAME, 0, ((1, LAST),), 1.0, 1, 1, g))
        for i in range(1, count):
            guides = ((2, LAST),) if i == count - 1 else ()
            runs.append(
                RunPlan(i, lengths[i], First.PREVIOUS_LAST_FRAME, None, guides, 1.0, 1, 1, g)
            )
    return Plan(rules.regime, ("start", "middle", "end"), tuple(runs))


def rules_for(recipe: Recipe) -> RecipeRules:
    try:
        rules = RULES[recipe.name]
    except KeyError:
        raise EngineError(f"no rules for recipe {recipe.name}") from None
    if rules.regime is not recipe.regime:
        raise EngineError(f"recipe {recipe.name} is {recipe.regime.value}, its rules are not")
    return rules


def adapter_for(workflow: WorkflowVersion) -> Adapter:
    """The adapter of a stored workflow version, after its content is verified."""
    manifest = versions.verify_workflow(workflow)
    try:
        kind = ADAPTERS[manifest.name]
    except KeyError:
        raise EngineError(f"no adapter runs {manifest.name}") from None
    return kind(workflow.api_json, manifest)


def pinned_adapter(session: Session, job: Job, slot: EngineSlot) -> Adapter:
    """The adapter of the workflow the job pinned for a slot. Release uses this too."""
    return adapter_for(versions.pinned_workflow(session, job, slot))


@dataclass(frozen=True)
class Registered:
    keyframe_engine: EngineVersion
    clip_engine: EngineVersion
    recipes: dict[str, Recipe]


def register_defaults(
    session: Session, workflows_dir: Path, *, activate: bool = False
) -> Registered:
    """Store the two workflows, their engines and both recipes as drafts, or activate them.

    Running it again returns the rows already there when their content is the same.
    """
    stored = {}
    for name, folder in WORKFLOW_FOLDERS.items():
        package = load_workflow(Path(workflows_dir) / folder)
        stored[name] = versions.create_workflow_version(
            session, package.api_json, package.manifest_json
        )
    keyframe = _engine(
        session,
        EngineSlot.KEYFRAME,
        QwenImageKeyframes.name,
        stored[QwenImageKeyframes.workflow_name],
        QwenImageKeyframes.limits.as_json(),
    )
    clip = _engine(
        session,
        EngineSlot.CLIP,
        LtxClips.name,
        stored[LtxClips.workflow_name],
        LtxClips.limits.as_json(),
    )
    recipes = {
        "multishot": _recipe(session, "multishot", JobRegime.MULTISHOT, clip, None),
        "keyframes": _recipe(session, "keyframes", JobRegime.KEYFRAMES, clip, keyframe),
    }
    if activate:
        for row in (*stored.values(), keyframe, clip, *recipes.values()):
            _activate(session, row)
    return Registered(keyframe, clip, recipes)


def _engine(
    session: Session,
    slot: EngineSlot,
    name: str,
    workflow: WorkflowVersion,
    limits: dict[str, Any],
) -> EngineVersion:
    existing = session.scalars(
        select(EngineVersion)
        .filter_by(slot=slot, name=name, workflow_version_id=workflow.id)
        .order_by(EngineVersion.version.desc())
    ).all()
    for row in existing:
        if row.limits == limits and row.status is not VersionStatus.RETIRED:
            return row
    return versions.create_engine_version(
        session, slot=slot, name=name, workflow_version_id=workflow.id, limits=limits
    )


def _recipe(
    session: Session,
    name: str,
    regime: JobRegime,
    clip: EngineVersion,
    keyframe: EngineVersion | None,
) -> Recipe:
    existing = session.scalars(
        select(Recipe)
        .filter_by(
            name=name,
            regime=regime,
            clip_engine_id=clip.id,
            keyframe_engine_id=None if keyframe is None else keyframe.id,
        )
        .order_by(Recipe.version.desc())
    ).first()
    if existing is not None and existing.status is not VersionStatus.RETIRED:
        return existing
    return versions.create_recipe_version(
        session,
        name=name,
        regime=regime,
        clip_engine_id=clip.id,
        keyframe_engine_id=None if keyframe is None else keyframe.id,
    )


def _activate(session: Session, row: versions.Version) -> None:
    if row.status is VersionStatus.DRAFT:
        versions.promote(session, row, VersionStatus.CANDIDATE)
    if row.status is VersionStatus.CANDIDATE:
        versions.promote(session, row, VersionStatus.ACTIVE)
