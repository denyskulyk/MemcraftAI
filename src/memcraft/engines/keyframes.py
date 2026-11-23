"""The keyframe slot: a still picture drawn from the photos, and its Qwen-Image-2.1 adapter.

Keyframes of one clip must match, or LTX dissolves between them instead of moving (E0.3/E0.4).
So the start keyframe is drawn first, and middle and end keyframes take it as an extra reference
with the words of SAME_SCENE. matching_prompt() writes those prefixes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from memcraft.comfy import ComfyClient
from memcraft.engines.base import Adapter, LimitError, RunOptions, upload_input

NEW_PLACE = (
    "New place, replace every photo background completely: {place} {light} "
    "Take only the people (face, hair, clothing) from {photos}. "
)
SAME_SCENE = (
    "Same scene as {start}: same place, light, camera position and framing; only the action "
    "and expressions change. Faces come from {photos}. "
)
DEFAULT_NEGATIVE = (
    "blurry, low quality, deformed face, extra fingers, different person, text, watermark, frame"
)


@dataclass(frozen=True)
class KeyframeLimits:
    references: int = 4
    width: int = 768
    height: int = 1344
    side_divisor: int = 32
    sound: bool = False

    def as_json(self) -> dict[str, Any]:
        return {
            "references": self.references,
            "size": [self.width, self.height],
            "side_divisor": self.side_divisor,
            "sound": self.sound,
        }


@dataclass(frozen=True)
class KeyframeRequest:
    """One keyframe.

    ``photos`` are the friend's photos, ``extra_references`` pictures drawn before (the start
    keyframe). In the prompt they are <image1>, <image2>, ... in that order, photos first.
    """

    prompt: str
    photos: tuple[Path, ...]
    seed: int
    extra_references: tuple[Path, ...] = ()
    negative_prompt: str = DEFAULT_NEGATIVE
    width: int = 768
    height: int = 1344
    steps: int = 16
    cfg: float = 3.0
    # A subfolder of ComfyUI's input and output folders, such as the job id in hex.
    folder: str = "memcraft"

    @property
    def references(self) -> tuple[Path, ...]:
        return (*self.photos, *self.extra_references)


@dataclass(frozen=True)
class KeyframeResult:
    image: Path
    seconds: float
    prompt_id: str
    width: int
    height: int


def matching_prompt(
    action: str, *, place: str, light: str, photos: int, start_reference: bool
) -> str:
    """The prompt of a keyframe: the new place first, or the start keyframe as the scene.

    With ``start_reference`` the start keyframe is the reference after the photos.
    """
    if not 1 <= photos <= 3:
        raise LimitError("a keyframe takes 1 to 3 photos")
    tags = ", ".join(f"<image{i}>" for i in range(1, photos + 1))
    if start_reference:
        prefix = SAME_SCENE.format(start=f"<image{photos + 1}>", photos=tags)
    else:
        prefix = NEW_PLACE.format(place=place.strip(), light=light.strip(), photos=tags)
    return prefix + action.strip()


class QwenImageKeyframes(Adapter):
    slot = "keyframe"
    name = "qwen-image-2.1"
    workflow_name = "keyframe-qwen-image-2.1"
    limits = KeyframeLimits()
    default_timeout = 600.0

    def check(self, request: KeyframeRequest) -> None:
        refs = request.references
        if not request.photos:
            raise LimitError("a keyframe needs at least one photo")
        if len(refs) > self.limits.references:
            raise LimitError(f"at most {self.limits.references} references, not {len(refs)}")
        for side in (request.width, request.height):
            if side % self.limits.side_divisor:
                raise LimitError(f"sides must divide by {self.limits.side_divisor}, not {side}")
        if not request.prompt.strip():
            raise LimitError("a keyframe needs a prompt")

    def values(self, request: KeyframeRequest, names: list[str]) -> dict[str, Any]:
        """The manifest values for uploaded reference names, in the request's order."""
        values: dict[str, Any] = {
            "prompt": request.prompt,
            "negative_prompt": request.negative_prompt,
            "seed": request.seed,
            "steps": request.steps,
            "cfg": float(request.cfg),
            "width": request.width,
            "height": request.height,
            "prefix": f"{request.folder}/keyframe",
        }
        for i, name in enumerate(names, start=1):
            values[f"image_{i}"] = name
        return values

    def render(
        self,
        client: ComfyClient,
        request: KeyframeRequest,
        dest: Path,
        options: RunOptions | None = None,
    ) -> KeyframeResult:
        """Draw the keyframe and download it to ``dest`` (a .png path)."""
        options = options or RunOptions()
        self.check(request)
        names = [upload_input(client, path, request.folder) for path in request.references]
        result = self.run(
            client, self.values(request, names), ["image"], options, self.default_timeout
        )
        image = self.fetch(client, result, "image", Path(dest))
        return KeyframeResult(
            image, result.seconds, result.prompt_id, request.width, request.height
        )
