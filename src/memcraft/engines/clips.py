"""The clip slot: a short video with sound from a first frame, and its LTX-2.5 adapter.

render() is stage 1: the preview at 544x960, the real last frame (where a next run starts) and
the stage-1 latents, which are the resume file of the release. release() turns those latents
into HD (RTX upscale to 720x1280) or Full HD (x2 refine to 1080x1920). The release must use the
same engine version and the same request as the preview.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from memcraft.comfy import ComfyClient
from memcraft.engines.base import Adapter, LimitError, RunOptions, is_8n_plus_1, upload_input
from memcraft.models import JobQuality

LAST = -1


@dataclass(frozen=True)
class ClipLimits:
    frame_rule: str = "8n+1"
    max_frames: int = 297
    fps: int = 30
    side_divisor: int = 32
    width: int = 544
    height: int = 960
    # Shots one run may hold: E0.2 kept faces with 1 and 2, lost them with 4.
    cuts: int = 2
    # Keyframes after the first frame that one run can be steered to.
    guides: int = 2
    sound: bool = True
    resume: str = "latents"
    hd: tuple[int, int] = (720, 1280)
    fhd: tuple[int, int] = (1080, 1920)

    def as_json(self) -> dict[str, Any]:
        return {
            "frame_rule": self.frame_rule,
            "max_frames": self.max_frames,
            "fps": self.fps,
            "side_divisor": self.side_divisor,
            "size": [self.width, self.height],
            "cuts": self.cuts,
            "guides": self.guides,
            "sound": self.sound,
            "resume": self.resume,
            "hd": list(self.hd),
            "fhd": list(self.fhd),
        }


@dataclass(frozen=True)
class Guide:
    """A keyframe the clip must pass through; ``frame`` -1 is the last frame."""

    image: Path
    frame: int = LAST


@dataclass(frozen=True)
class ClipRequest:
    prompt: str
    first_frame: Path
    frames: int
    seed: int
    guides: tuple[Guide, ...] = ()
    fps: int = 30
    width: int = 544
    height: int = 960
    # 1.0 in keyframe runs and continued runs: the cut between runs jumps less (E0.5).
    first_strength: float = 1.0
    guide_strength: float = 1.0
    negative_prompt: str = "watermark, text"
    folder: str = "memcraft"


@dataclass(frozen=True)
class ClipResult:
    video: Path
    last_frame: Path
    latent_video: Path
    latent_audio: Path
    seconds: float
    prompt_id: str
    frames: int
    fps: int
    has_sound: bool = True


@dataclass(frozen=True)
class ReleaseResult:
    video: Path
    seconds: float
    prompt_id: str
    quality: JobQuality
    width: int
    height: int


class LtxClips(Adapter):
    slot = "clip"
    name = "ltx-2.5"
    workflow_name = "clip-ltx-2.5"
    limits = ClipLimits()
    timeouts: ClassVar[dict[str, float]] = {
        "clip": 900.0,
        JobQuality.HD720: 600.0,
        JobQuality.FHD1080: 2400.0,
    }

    def check(self, request: ClipRequest) -> None:
        limits = self.limits
        if not is_8n_plus_1(request.frames):
            raise LimitError(f"frames must be 8n+1, not {request.frames}")
        if request.frames > limits.max_frames:
            raise LimitError(f"at most {limits.max_frames} frames, not {request.frames}")
        if request.fps != limits.fps:
            raise LimitError(f"the engine runs at {limits.fps} fps, not {request.fps}")
        for side in (request.width, request.height):
            if side % limits.side_divisor:
                raise LimitError(f"sides must divide by {limits.side_divisor}, not {side}")
        if len(request.guides) > limits.guides:
            raise LimitError(f"at most {limits.guides} guides, not {len(request.guides)}")
        for guide in request.guides:
            if guide.frame != LAST and not 0 < guide.frame < request.frames:
                raise LimitError(f"guide frame {guide.frame} is outside the clip")
        if not request.prompt.strip():
            raise LimitError("a clip needs a prompt")

    def values(self, request: ClipRequest, first: str, guides: list[str]) -> dict[str, Any]:
        values: dict[str, Any] = {
            "first_frame": first,
            "prompt": request.prompt,
            "negative_prompt": request.negative_prompt,
            "seed": request.seed,
            "frames": request.frames,
            "fps": float(request.fps),
            "width": request.width,
            "height": request.height,
            # References at the refine size, so the Full HD pass sees them sharp.
            "reference_width": request.width * 2,
            "reference_height": request.height * 2,
            "first_strength": float(request.first_strength),
            "guide_strength": float(request.guide_strength),
        }
        for i, (name, guide) in enumerate(zip(guides, request.guides, strict=True), start=1):
            values[f"guide_{i}"] = name
            values[f"guide_{i}_frame"] = guide.frame
        return values

    def _upload(self, client: ComfyClient, request: ClipRequest) -> tuple[str, list[str]]:
        first = upload_input(client, request.first_frame, request.folder)
        guides = [upload_input(client, guide.image, request.folder) for guide in request.guides]
        return first, guides

    def render(
        self,
        client: ComfyClient,
        request: ClipRequest,
        dest: Path,
        options: RunOptions | None = None,
    ) -> ClipResult:
        """Stage 1 into the folder ``dest``: preview.mp4, last.png, latent_v/a.latent."""
        options = options or RunOptions()
        self.check(request)
        first, guides = self._upload(client, request)
        values = self.values(request, first, guides)
        folder = request.folder
        values |= {
            "preview_prefix": f"{folder}/preview",
            "latent_video_prefix": f"{folder}/latent_v",
            "latent_audio_prefix": f"{folder}/latent_a",
            "last_frame_prefix": f"{folder}/last",
        }
        outputs = ["preview", "last_frame", "latent_video", "latent_audio"]
        result = self.run(client, values, outputs, options, self.timeouts["clip"])
        dest = Path(dest)
        return ClipResult(
            video=self.fetch(client, result, "preview", dest / "preview.mp4"),
            last_frame=self.fetch(client, result, "last_frame", dest / "last.png"),
            latent_video=self.fetch(client, result, "latent_video", dest / "latent_v.latent"),
            latent_audio=self.fetch(client, result, "latent_audio", dest / "latent_a.latent"),
            seconds=result.seconds,
            prompt_id=result.prompt_id,
            frames=request.frames,
            fps=request.fps,
        )

    def release(
        self,
        client: ComfyClient,
        request: ClipRequest,
        latent_video: Path,
        latent_audio: Path,
        quality: JobQuality,
        dest: Path,
        *,
        refine_seed: int = 42,
        options: RunOptions | None = None,
    ) -> ReleaseResult:
        """HD or Full HD from the stage-1 latents of ``request``, to the file ``dest``."""
        options = options or RunOptions()
        self.check(request)
        folder = request.folder
        latents = {
            "latent_video": upload_input(client, latent_video, folder),
            "latent_audio": upload_input(client, latent_audio, folder),
        }
        if quality is JobQuality.HD720:
            width, height = self.limits.hd
            # This branch reads no image; the required first frame only has to name a file.
            values = {
                "first_frame": latents["latent_video"],
                "prompt": request.prompt,
                "seed": request.seed,
                "fps": float(request.fps),
                **latents,
                "hd_scale": height / request.height,
                "hd_width": width,
                "hd_height": height,
                "hd_prefix": f"{folder}/hd",
            }
            output = "hd"
        else:
            width, height = self.limits.fhd
            first, guides = self._upload(client, request)
            values = self.values(request, first, guides) | latents
            values |= {
                "refine_seed": refine_seed,
                "fhd_width": width,
                "fhd_height": height,
                "fhd_prefix": f"{folder}/fhd",
            }
            output = "fhd"
        result = self.run(client, values, [output], options, self.timeouts[quality])
        video = self.fetch(client, result, output, Path(dest))
        return ReleaseResult(video, result.seconds, result.prompt_id, quality, width, height)
