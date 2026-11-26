"""What the pipeline gives the worker: step handlers and a plan per job.

The pipeline registers one handler per StepName and one planner. The worker asks the planner
for the job's steps after every step, runs the first one that is not done yet, and owns
everything around it: claims, job_steps rows, retries, pause, the GPU lease and heartbeats.

A handler runs outside any open transaction. It may use ctx.session to save files with
FileStore.save() and to pin versions on ctx.step_id; the worker commits that session together
with the step's done mark, and rolls it back, files included, if the step fails. Touch it only
at the end of the step, because sweep_orphans() needs save transactions to be short.
"""

from __future__ import annotations

import hashlib
import json
import threading
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from sqlalchemy.orm import Session

from memcraft.models import Job, JobRegime, JobStatus, JobStep, StepName
from memcraft.storage import FileStore


class StepStopped(Exception):
    """Raised by ctx.check_stop(): the step must end now (claim lost, hung, shutting down)."""


@dataclass(frozen=True)
class PlannedStep:
    """One step the job needs. `count` is how many ordinals of this step the job has."""

    step: StepName
    ordinal: int = 0
    count: int = 1
    inputs: Mapping[str, Any] = field(default_factory=dict)
    # Overrides the handler's model, for a step whose model depends on the job's recipe.
    model: str | None = None

    @property
    def inputs_hash(self) -> str:
        return inputs_hash(self.inputs)


def inputs_hash(inputs: Mapping[str, Any]) -> str:
    canonical = json.dumps(inputs, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


@dataclass
class StepContext:
    job_id: uuid.UUID
    owner_id: uuid.UUID
    regime: JobRegime | None
    step: StepName
    ordinal: int
    count: int
    attempt: int
    step_id: uuid.UUID
    inputs: Mapping[str, Any]
    session: Session
    store: FileStore | None
    # Outputs of the job's done steps, by (step, ordinal).
    outputs: Mapping[tuple[StepName, int], Mapping[str, Any] | None]
    _report: Callable[[float], None]
    _stop: threading.Event

    def progress(self, fraction: float) -> None:
        """Report how far this step is, from 0 to 1. Cheap: the worker throttles writes."""
        self._report(min(1.0, max(0.0, float(fraction))))

    @property
    def stopping(self) -> bool:
        return self._stop.is_set()

    def check_stop(self) -> None:
        if self._stop.is_set():
            raise StepStopped("the step was stopped")

    def wait(self, seconds: float) -> None:
        """Sleep, but wake and raise StepStopped as soon as the step is stopped."""
        if self._stop.wait(seconds):
            raise StepStopped("the step was stopped")


@runtime_checkable
class StepHandler(Protocol):
    needs_gpu: bool
    # Which model the step loads, for grouping render jobs; None for no model.
    model: str | None
    # Which watched service the step talks to ("comfyui", "gemma"), or None.
    service: str | None

    def run(self, ctx: StepContext) -> Mapping[str, Any] | None:
        """Do the step and return its output, which goes to job_steps.output."""
        ...


class Planner(Protocol):
    def plan(self, session: Session, job: Job, done: Sequence[JobStep]) -> Sequence[PlannedStep]:
        """Every step the job needs in its current status, in order, as far as known now.

        `done` holds the job's done step rows. The list may grow as steps finish, for example
        once the scenario says how many clips there are. To group render jobs by model the
        worker also asks about jobs that are still queued, so plan by the phase, not only by
        the running status.
        """
        ...

    def outcome(self, session: Session, job: Job) -> JobStatus:
        """The status the job takes once every planned step is done."""
        ...


class StepRegistry:
    def __init__(self, planner: Planner) -> None:
        self.planner = planner
        self._handlers: dict[StepName, StepHandler] = {}

    def register(self, name: StepName, handler: StepHandler) -> None:
        if name in self._handlers:
            raise ValueError(f"step {name.value} already has a handler")
        self._handlers[name] = handler

    def handler(self, name: StepName) -> StepHandler:
        try:
            return self._handlers[name]
        except KeyError:
            raise LookupError(f"no handler for step {name.value}") from None

    def model_of(self, planned: PlannedStep) -> str | None:
        return planned.model or self.handler(planned.step).model


# A dry run: every step sleeps a little and reports progress. For `python -m memcraft.worker`
# before the pipeline exists, and for tests.

DRY_CLIPS = 2


@dataclass
class DryHandler:
    needs_gpu: bool = False
    model: str | None = None
    service: str | None = None
    seconds: float = 0.05
    ticks: int = 4

    def run(self, ctx: StepContext) -> Mapping[str, Any] | None:
        for tick in range(1, self.ticks + 1):
            ctx.wait(self.seconds / self.ticks)
            ctx.progress(tick / self.ticks)
        return {"dry": True, "step": ctx.step.value, "ordinal": ctx.ordinal}


class DryPlanner:
    """The order of the architecture: idea, then scenario, keyframes, clips, check and join."""

    def __init__(self, clips: int = DRY_CLIPS) -> None:
        self.clips = clips

    def plan(self, session: Session, job: Job, done: Sequence[JobStep]) -> Sequence[PlannedStep]:
        if job.status in (JobStatus.IDEA_QUEUED, JobStatus.IDEA_RUNNING):
            return [PlannedStep(StepName.IDEA, inputs={"idea": job.idea_text})]
        if job.status in (JobStatus.RELEASE_QUEUED, JobStatus.RELEASING):
            return [PlannedStep(StepName.RELEASE)]
        steps = [PlannedStep(StepName.SCENARIO, inputs={"regime": _regime(job)})]
        if job.regime in (JobRegime.KEYFRAMES, JobRegime.REGIME3):
            steps += [
                PlannedStep(StepName.KEYFRAME, n, self.clips, {"n": n}) for n in range(self.clips)
            ]
        steps += [PlannedStep(StepName.CLIP, n, self.clips, {"n": n}) for n in range(self.clips)]
        steps += [PlannedStep(StepName.CHECK), PlannedStep(StepName.JOIN)]
        return steps

    def outcome(self, session: Session, job: Job) -> JobStatus:
        if job.status is JobStatus.IDEA_RUNNING:
            # The friend gave an idea: go on to render. Otherwise the app proposes one.
            return JobStatus.RENDER_QUEUED if job.idea_text else JobStatus.AWAITING_APPROVAL
        if job.status is JobStatus.RELEASING:
            return JobStatus.RELEASED
        return JobStatus.PREVIEW_READY


def _regime(job: Job) -> str:
    return (job.regime or JobRegime.MULTISHOT).value


DRY_GPU_MODELS = {
    StepName.KEYFRAME: "dry-image",
    StepName.CLIP: "dry-video",
    StepName.CHECK: "dry-critic",
    StepName.JOIN: None,
    StepName.RELEASE: None,
}


def dry_registry(seconds: float = 0.05, clips: int = DRY_CLIPS) -> StepRegistry:
    registry = StepRegistry(DryPlanner(clips))
    registry.register(StepName.IDEA, DryHandler(service="gemma", seconds=seconds))
    registry.register(StepName.SCENARIO, DryHandler(service="gemma", seconds=seconds))
    for name, model in DRY_GPU_MODELS.items():
        registry.register(
            name,
            DryHandler(
                needs_gpu=model is not None,
                model=model,
                service="comfyui" if model else None,
                seconds=seconds,
            ),
        )
    return registry
