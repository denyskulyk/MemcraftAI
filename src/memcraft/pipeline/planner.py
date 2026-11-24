"""The pipeline's planner: the steps of a job by its phase, and the status it ends in.

- idea lane (idea_queued, idea_running): IDEA. With the friend's idea the job goes on to
  render_queued, without one it waits in awaiting_approval.
- render lane, render phase (render_queued, rendering): SCENARIO, then for keyframes
  KEYFRAME 0..2 and CHECK of the keyframes (before LTX loads), CLIP per run, for multi-shot a
  CHECK of the last frame, and JOIN when there are two runs or more. A CHECK judges faces only
  and only warns: it never redraws or holds the job. The job
  becomes preview_ready, and the PREVIEW asset is saved in the same transaction (outcome()).
- render lane, release phase (release_queued, releasing): RELEASE -> released.

The plan reads only, because the worker also plans queued jobs to group them by model. Until
the scenario has run (and pinned the recipe and the critic) the counts come from the recipe
and critic a render would get now.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from memcraft.expiry import PREVIEW_LIFETIME
from memcraft.models import (
    Asset,
    AssetKind,
    Job,
    JobLength,
    JobRegime,
    JobStatus,
    JobStep,
    StepName,
    StepStatus,
)
from memcraft.pipeline import choices
from memcraft.pipeline.steps import MODEL_CLIPS, MODEL_KEYFRAMES, MP4, PipelineError, asset_path
from memcraft.pipeline.steps import job_regime as _job_regime
from memcraft.storage import FileStore
from memcraft.worker.steps import PlannedStep

IDEA_PHASE = (JobStatus.IDEA_QUEUED, JobStatus.IDEA_RUNNING)
RELEASE_PHASE = (JobStatus.RELEASE_QUEUED, JobStatus.RELEASING)


class PipelinePlanner:
    def __init__(self, store: FileStore) -> None:
        self.store = store

    def plan(self, session: Session, job: Job, done: Sequence[JobStep]) -> Sequence[PlannedStep]:
        if job.status in IDEA_PHASE:
            return [PlannedStep(StepName.IDEA, inputs={"idea": (job.idea_text or "").strip()})]
        if job.status in RELEASE_PHASE:
            return [PlannedStep(StepName.RELEASE, inputs={"quality": job.quality.value})]
        return render_plan(session, job, done)

    def outcome(self, session: Session, job: Job) -> JobStatus:
        if job.status is JobStatus.IDEA_RUNNING:
            return (
                JobStatus.RENDER_QUEUED
                if (job.idea_text or "").strip()
                else (JobStatus.AWAITING_APPROVAL)
            )
        if job.status is JobStatus.RELEASING:
            return JobStatus.RELEASED
        if job.status is JobStatus.RENDERING:
            save_preview(session, self.store, job)
            return JobStatus.PREVIEW_READY
        raise PipelineError(f"a job in {job.status.value} has no outcome")


def render_plan(session: Session, job: Job, done: Sequence[JobStep]) -> list[PlannedStep]:
    regime = _job_regime(job)
    outputs = {(row.step, row.ordinal): row.output for row in done}
    scenario = outputs.get((StepName.SCENARIO, 0))
    if scenario is not None:
        runs = len(scenario["runs"])
        keyframes = len(scenario.get("keyframes") or [])
    else:
        rules = choices.rules_for_plan(session, job, regime)
        runs = rules.runs_long if job.length is JobLength.LONG else rules.runs_short
        keyframes = (3 if runs > 1 else 2) if regime is JobRegime.KEYFRAMES else 0
    critic = choices.critic_planned(session, job)
    steps = [PlannedStep(StepName.SCENARIO, inputs={"regime": regime.value})]
    steps += [PlannedStep(StepName.KEYFRAME, n, keyframes, {"n": n}) for n in range(keyframes)]
    if critic and keyframes:
        steps.append(
            PlannedStep(StepName.CHECK, inputs={"kind": "keyframes"}, model=MODEL_KEYFRAMES)
        )
    steps += [PlannedStep(StepName.CLIP, n, runs, {"n": n}) for n in range(runs)]
    if critic and not keyframes:
        steps.append(PlannedStep(StepName.CHECK, inputs={"kind": "last_frame"}, model=MODEL_CLIPS))
    if runs > 1:
        steps.append(PlannedStep(StepName.JOIN, inputs={"runs": runs}))
    return steps


def final_video(session: Session, job_id: uuid.UUID) -> str:
    """The asset id of the finished render: the joined video, or the only run."""
    outputs: dict[tuple[StepName, int], Any] = {}
    for row in session.scalars(
        select(JobStep)
        .where(
            JobStep.job_id == job_id,
            JobStep.status == StepStatus.DONE,
            JobStep.step.in_((StepName.JOIN, StepName.CLIP)),
        )
        .order_by(JobStep.created_at)
    ):
        outputs[(row.step, row.ordinal)] = row.output
    joined = outputs.get((StepName.JOIN, 0))
    if joined:
        return joined["video"]
    clip = outputs.get((StepName.CLIP, 0))
    if not clip:
        raise PipelineError("the render has no video")
    return clip["video"]


def save_preview(session: Session, store: FileStore, job: Job) -> Asset:
    """The PREVIEW asset in the shared store, alive for 24 hours. The caller commits."""
    existing = session.scalar(
        select(Asset).where(Asset.job_id == job.id, Asset.kind == AssetKind.PREVIEW).limit(1)
    )
    if existing is not None:
        return existing
    source = asset_path(session, store, final_video(session, job.id))
    now = session.scalar(select(func.now()))
    with source.open("rb") as handle:
        return store.save(
            session,
            owner_id=job.owner_id,
            job_id=job.id,
            kind=AssetKind.PREVIEW,
            mime=MP4,
            source=handle,
            expires_at=now + PREVIEW_LIFETIME,
        )
