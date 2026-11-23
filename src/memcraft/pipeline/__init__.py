"""The pipeline: what turns a job into a video, as step handlers and a planner for the worker.

    from memcraft.pipeline import build_registry
    from memcraft.pipeline.steps import Deps
    registry = build_registry(Deps(store, comfy, gemma, work_dir, workflows_dir))
    Worker(engine, registry, store=store, ...)

- regime.regime_for(photo_count, length): multi-shot or keyframes (the upload uses it too).
- planner.PipelinePlanner: the steps of a job by its phase, and its outcome.
- steps: IdeaStep, ScenarioStep, KeyframeStep, CheckStep, ClipStep, JoinStep, ReleaseStep.
- choices: which recipe and critic a render pins (settings active_recipes, active_critic_engine).
- services: the app's ComfyUI (8199) and its own CPU llama-server for Gemma (8098).
- media: the ffmpeg join, frame counts, contact sheets.

This file imports nothing heavy, so that the web app can use regime_for alone.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from memcraft.pipeline.steps import Deps
    from memcraft.worker.steps import StepRegistry

__all__ = ["build_registry"]


def build_registry(deps: Deps) -> StepRegistry:
    from memcraft.models import StepName
    from memcraft.pipeline.planner import PipelinePlanner
    from memcraft.pipeline.steps import (
        CheckStep,
        ClipStep,
        IdeaStep,
        JoinStep,
        KeyframeStep,
        ReleaseStep,
        ScenarioStep,
    )
    from memcraft.worker.steps import StepRegistry

    registry = StepRegistry(PipelinePlanner(deps.store))
    registry.register(StepName.IDEA, IdeaStep(deps))
    registry.register(StepName.SCENARIO, ScenarioStep(deps))
    registry.register(StepName.KEYFRAME, KeyframeStep(deps))
    registry.register(StepName.CHECK, CheckStep(deps))
    registry.register(StepName.CLIP, ClipStep(deps))
    registry.register(StepName.JOIN, JoinStep(deps))
    registry.register(StepName.RELEASE, ReleaseStep(deps))
    return registry
