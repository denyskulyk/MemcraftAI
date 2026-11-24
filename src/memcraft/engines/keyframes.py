"""The keyframe slot: a still picture drawn from the photos, and its Qwen-Image-2.1 adapter.

Keyframes of one clip must match, or LTX dissolves between them instead of moving (E0.3/E0.4).
So the start keyframe is drawn first, and middle and end keyframes take it as an extra reference
with the words of SAME_SCENE. matching_prompt() writes those prefixes.

Every reference adds its pixels to the sequence Qwen-Image samples, so image_1 (the first photo)
goes in at the canvas size, which sets the size of the picture, and the others at about half a
megapixel (reference_size()): with two photos and the start keyframe at full size a keyframe took
173 s instead of about 60.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
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
    "Same scene as {start}: same place, light, camera position, shot size and framing, and the "
    "same objects in the same colours; only the action and expressions change. "
    "The place: {place} The light: {light} "
    "Faces come from {photos}. "
)
# E16 asset-first: a person whose photo holds a prop (the troll's log, wide in both hands) is drawn
# alone and empty-handed first, and a prop the story uses is drawn alone; the keyframes take
# those pictures instead, so the photo's grip and pose do not come back in every keyframe.
PROP_TAKE = "Take {words} from {image}."
ONCE = "Each person appears exactly once. "
# Middle and end keyframes drawn as an edit of the start keyframe (when there are asset pictures).
EDIT_SCENE = (
    "Edit <image1>: keep the place, the light, the colours, the camera position, the shot size, "
    "the framing and every object exactly as in <image1>. " + ONCE + "Change only the poses and "
    "expressions as follows. The place: {place} The light: {light} Faces come from {photos}. "
)
# Not only people: a photo may show a cat (E16: "the same people" turned a cat into two men).
CLEAN_PERSON = (
    "Edit <image1>: exactly the same {looks}, with the same face, hair or fur, body and "
    "clothing as in <image1>, and nobody else. Show them alone in a relaxed neutral pose, "
    "the whole body from head to feet, hands or paws empty. Remove every object they hold "
    "completely. Plain light grey studio background, even soft light."
)
PROP_FROM_PHOTO = (
    "Show only the object from <image1>: {words}. Alone and whole at its full length and size, "
    "with the same look and colours as in <image1>, on a plain light grey studio background, "
    "even soft light. Leave out every person and creature, no hands, no body parts."
)
PROP_NEW = (
    "On the plain light grey <image1>, draw only one object: {words}. Alone and whole, in the "
    "middle, photorealistic, even soft light. No people, no creatures, no hands, no body parts."
)
# Pixels of image_2 to image_4; image_1 is drawn at the canvas size.
REFERENCE_PIXELS = 512 * 896
# The empty place of the background-first layout: the director's camera sentence names the
# people ("capturing both men"), and Qwen drew them (E16).
BACKGROUND_NEGATIVE = (
    "people, person, man, woman, child, crowd, figure, creature, animal, face, body, "
    "blurry, low quality, text, watermark, frame"
)
DEFAULT_NEGATIVE = (
    "blurry, low quality, deformed face, extra fingers, different person, text, watermark, frame"
)


@dataclass(frozen=True)
class KeyframeLimits:
    # Workflow v3 (keyframe-qwen-image-2.1-v2) has 6 picture slots; v2 had 4.
    references: int = 6
    width: int = 768
    height: int = 1344
    side_divisor: int = 32
    sound: bool = False
    reference_pixels: int = REFERENCE_PIXELS

    def as_json(self) -> dict[str, Any]:
        return {
            "references": self.references,
            "size": [self.width, self.height],
            "side_divisor": self.side_divisor,
            "sound": self.sound,
            "reference_pixels": self.reference_pixels,
        }


def reference_size(width: int, height: int, pixels: int, divisor: int = 32) -> tuple[int, int]:
    """The size of a later reference: the canvas's aspect at about ``pixels``, sides that
    divide by ``divisor``, never larger than the canvas."""
    scale = min(1.0, math.sqrt(pixels / (width * height)))
    return (
        max(divisor, round(width * scale / divisor) * divisor),
        max(divisor, round(height * scale / divisor) * divisor),
    )


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
    # About how many pixels image_2 to image_4 have; None draws them at the canvas size.
    reference_pixels: int | None = REFERENCE_PIXELS
    # A subfolder of ComfyUI's input and output folders, such as the job id in hex.
    folder: str = "memcraft"
    # Cut the drawn picture out of its background (BiRefNet) and save it as RGBA (E16).
    cutout: bool = False

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
    action: str,
    *,
    place: str,
    light: str,
    photos: int,
    start_reference: bool,
    props: Sequence[str] = (),
) -> str:
    """The prompt of a keyframe: the new place first, or the start keyframe as the scene.

    With ``start_reference`` the start keyframe is the reference after the photos. ``props``
    are the words of prop pictures that follow the photos (the start keyframe only).
    """
    if not 1 <= photos <= 3:
        raise LimitError("a keyframe takes 1 to 3 photos")
    if props and start_reference:
        raise LimitError("prop pictures go into the start keyframe only")
    tags = ", ".join(f"<image{i}>" for i in range(1, photos + 1))
    if props:
        takes = " ".join(
            PROP_TAKE.format(words=words.strip(), image=f"<image{photos + i}>")
            for i, words in enumerate(props, start=1)
        )
        prefix = NEW_PLACE.format(place=place.strip(), light=light.strip(), photos=tags)
        return f"{prefix}{takes} {ONCE}{action.strip()}"
    if start_reference:
        # The place and light sentences too, word for word as in the start keyframe: the start
        # keyframe alone goes in at reference size and the light drifted (E16).
        prefix = SAME_SCENE.format(
            start=f"<image{photos + 1}>", photos=tags, place=place.strip(), light=light.strip()
        )
    else:
        prefix = NEW_PLACE.format(place=place.strip(), light=light.strip(), photos=tags)
    return prefix + action.strip()


COMPOSE = (
    "Put the people and objects into the empty place of <image1>: keep its place, light and "
    "colours{keep}. The people come from {people}, each shown alone there: give them the poses "
    "below. "
)


def compose_prompt(
    action: str, *, people: int, props: Sequence[str] = (), camera_moves: bool = False
) -> str:
    """A keyframe put together from asset pictures (E16, Denys's layout): <image1> is the empty
    place, then one cut-out picture per photo, then the prop pictures. The action names the
    photos as <imageN>; they move up by one. No other keyframe is a reference."""
    if not 1 <= people <= 3:
        raise LimitError("a keyframe takes 1 to 3 photos")
    tags = ", ".join(f"<image{i}>" for i in range(2, people + 2))
    keep = "" if camera_moves else ", the camera position and the framing"
    takes = "".join(
        PROP_TAKE.format(words=words.strip(), image=f"<image{people + 1 + i}>") + " "
        for i, words in enumerate(props, start=1)
    )
    moved = re.sub(r"<image([1-9])>", lambda m: f"<image{int(m.group(1)) + 1}>", action.strip())
    return COMPOSE.format(keep=keep, people=tags) + takes + ONCE + moved


def edit_prompt(action: str, *, place: str, light: str, photos: int) -> str:
    """A middle or end keyframe as an edit of the start keyframe, which is <image1>; the photos
    (or their clean pictures) follow, so the action's <imageN> move up by one (E16)."""
    if not 1 <= photos <= 3:
        raise LimitError("a keyframe takes 1 to 3 photos")
    tags = ", ".join(f"<image{i}>" for i in range(2, photos + 2))
    moved = re.sub(r"<image([1-9])>", lambda m: f"<image{int(m.group(1)) + 1}>", action.strip())
    return EDIT_SCENE.format(place=place.strip(), light=light.strip(), photos=tags) + moved


def match_colours(image: Path, reference: Path, strength: float = 1.0) -> None:
    """Shift ``image`` (in place) to the brightness and tint of ``reference``: Reinhard colour
    transfer, the mean and spread of each Lab channel. Middle and end keyframes drifted up to
    50 of 255 in brightness from the start keyframe; this took it out in 0.1 s (E16)."""
    import cv2
    import numpy as np

    source = cv2.imread(str(image), cv2.IMREAD_COLOR)
    target = cv2.imread(str(reference), cv2.IMREAD_COLOR)
    if source is None or target is None:
        raise LimitError(f"cannot read {image if source is None else reference}")
    a = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(np.float32)
    b = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float32)
    mean_a, spread_a = a.reshape(-1, 3).mean(0), a.reshape(-1, 3).std(0) + 1e-6
    mean_b, spread_b = b.reshape(-1, 3).mean(0), b.reshape(-1, 3).std(0)
    moved = (a - mean_a) / spread_a * spread_b + mean_b
    mixed = a + (moved - a) * strength
    out = cv2.cvtColor(np.clip(mixed, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    if not cv2.imwrite(str(image), out):
        raise LimitError(f"cannot write {image}")


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
        slots = sum(1 for name in self.manifest.inputs if name.startswith("image_"))
        if len(refs) > min(self.limits.references, slots):
            raise LimitError(
                f"at most {min(self.limits.references, slots)} references, not {len(refs)}"
            )
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
        # A workflow version from before reference_width draws every reference at the canvas.
        if request.cutout:
            if "cutout" not in self.manifest.inputs:
                raise LimitError("this keyframe workflow cannot cut pictures out")
            values["cutout"] = True
        if request.reference_pixels and "reference_width" in self.manifest.inputs:
            values["reference_width"], values["reference_height"] = reference_size(
                request.width, request.height, request.reference_pixels
            )
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
