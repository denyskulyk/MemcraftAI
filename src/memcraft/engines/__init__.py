"""Engines in swappable slots, each an adapter over one pinned ComfyUI workflow.

- keyframe slot: QwenImageKeyframes.render(client, KeyframeRequest, dest) -> KeyframeResult
- clip slot: LtxClips.render(client, ClipRequest, dest_dir) -> ClipResult (preview, last frame,
  stage-1 latents) and LtxClips.release(client, request, latents..., quality, dest)
- recipes: plan(rules_for(recipe), length) -> Plan of keyframes and runs;
  pinned_adapter(session, job, slot) builds the adapter of the job's pinned workflow;
  register_defaults(session, workflows_dir) stores workflows, engines and recipes.

Adapters check the engine's limits (8n+1 frames, at most 297, sides divisible by 32, 30 fps,
guides, references) and raise LimitError before anything reaches ComfyUI.
"""

from memcraft.engines.base import (
    EngineError,
    LimitError,
    OutputMissingError,
    RunOptions,
    frames_for,
    is_8n_plus_1,
)
from memcraft.engines.clips import (
    LAST,
    ClipLimits,
    ClipRequest,
    ClipResult,
    Guide,
    LtxClips,
    ReleaseResult,
)
from memcraft.engines.keyframes import (
    KeyframeLimits,
    KeyframeRequest,
    KeyframeResult,
    QwenImageKeyframes,
    match_colours,
    matching_prompt,
)
from memcraft.engines.recipes import (
    ADAPTERS,
    RULES,
    First,
    Plan,
    RecipeRules,
    RunPlan,
    adapter_for,
    pinned_adapter,
    plan,
    register_defaults,
    rules_for,
)

__all__ = [
    "ADAPTERS",
    "LAST",
    "RULES",
    "ClipLimits",
    "ClipRequest",
    "ClipResult",
    "EngineError",
    "First",
    "Guide",
    "KeyframeLimits",
    "KeyframeRequest",
    "KeyframeResult",
    "LimitError",
    "LtxClips",
    "OutputMissingError",
    "Plan",
    "QwenImageKeyframes",
    "RecipeRules",
    "ReleaseResult",
    "RunOptions",
    "RunPlan",
    "adapter_for",
    "frames_for",
    "is_8n_plus_1",
    "match_colours",
    "matching_prompt",
    "pinned_adapter",
    "plan",
    "register_defaults",
    "rules_for",
]
