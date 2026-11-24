"""The step handlers of the pipeline, one per StepName, and what they share.

Every handler reads what it needs, does its slow work, and only then writes: it saves its
files with FileStore.save() into ctx.session and returns its output, which the worker commits
together with the step's done mark. A worker that dies in a step leaves nothing of it, so a
restart repeats only that step. Files go to the private store (Gemma copies, keyframes, last
frames, latents, clips) or the shared one (preview, release), as memcraft.storage decides by
kind.

Pins that lock rows (versions.pin_job on the job, versions.pin_step on the step row) come last:
the worker's progress reports update those rows from this thread and would wait on the lock.

Assets a step makes, by kind and ordinal:
- gemma_copy n: the small JPEG of photo n (photos.make_gemma_copies)
- keyframe 0..2: start, middle, end (jobs from before the E0 verdicts may hold a redraw too)
- keyframe 100 + r: the real last frame of run r, where run r + 1 starts
- keyframe 200 + k: asset picture k of the scenario (E16: a photo's people alone and
  empty-handed, or a prop alone), drawn before the keyframes and never shown to friends
- latent 2r, 2r + 1: the stage-1 video and audio latents of run r (the release resumes there)
- clip r: the 544x960 preview of run r; clip 100: the joined runs of a long video
- preview 0 (shared): the video the friend sees, saved by the planner's outcome
- release 0 (shared): the final video
"""

from __future__ import annotations

import contextlib
import io
import logging
import math
import shutil
import time
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from PIL import Image
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from memcraft import versions
from memcraft.comfy import Progress
from memcraft.engines import (
    BACKGROUND_NEGATIVE,
    LAST,
    ClipRequest,
    First,
    Guide,
    KeyframeRequest,
    RunOptions,
    adapter_for,
    match_colours,
    pinned_adapter,
    plan,
    rules_for,
)
from memcraft.models import (
    ASSET_ORDINAL,
    Asset,
    AssetKind,
    EngineSlot,
    EngineVersion,
    Idea,
    IdeaSource,
    Job,
    JobQuality,
    JobRegime,
    JobStep,
    PromptRole,
    Recipe,
    StepName,
    StepStatus,
    WorkflowVersion,
)
from memcraft.photos import list_photos, make_gemma_copies
from memcraft.pipeline import choices, media
from memcraft.pipeline.regime import regime_for
from memcraft.roles.chat import record_idea, run_chat, user_idea
from memcraft.roles.critic import Critic, pinned_critic
from memcraft.roles.director import (
    direct_keyframes,
    direct_multishot,
    no_music,
    take_inventory,
)
from memcraft.roles.qwen import ComfyTextRunner
from memcraft.roles.seed import GEMMA, QWEN, active_prompt, parts
from memcraft.storage import FileStore
from memcraft.worker.steps import StepContext, StepStopped

log = logging.getLogger(__name__)

MODEL_DIRECTOR = "qwen3.8"
MODEL_KEYFRAMES = "qwen-image"
MODEL_CLIPS = "ltx"
COMFYUI = "comfyui"
GEMMA_SERVICE = "gemma"

LAST_FRAME_ORDINAL = 100
JOINED_ORDINAL = 100
KEYFRAME_NAMES = ("start", "middle", "end")
# Keyframe ordinals from ASSET_ORDINAL (memcraft.models) on are the scenario's asset pictures.
# The critic sees every picture at the size E0.14 tested.
CRITIC_SIZE = (540, 960)
# Prop and people pictures that are cut out of their grey background: 8 Qwen-Image steps
# looked the same as 16 and take 20 s instead of 38-42 s (2026-11-24 test).
CUTOUT_STEPS = 8
FPS = 30

PNG = "image/png"
MP4 = "video/mp4"
LATENT = "application/x-safetensors"


class PipelineError(Exception):
    pass


# What the handlers need from outside, so tests can put fakes in.


class ComfyLike(Protocol):
    def ensure_up(self) -> None: ...

    def client(self) -> Any: ...

    def clean(self, folder: str) -> None: ...


class GemmaLike(Protocol):
    def client(self) -> contextlib.AbstractContextManager[Any]: ...


type AdapterFor = Callable[[Session, Job, EngineSlot], Any]
type RecipeAdapterFor = Callable[[Session, Recipe, EngineSlot], Any]


def recipe_adapter(session: Session, recipe: Recipe, slot: EngineSlot) -> Any:
    """The adapter of a recipe's engine, for a step that runs before the job pins it."""
    engine_id = recipe.keyframe_engine_id if slot is EngineSlot.KEYFRAME else recipe.clip_engine_id
    engine = session.get_one(EngineVersion, engine_id)
    return adapter_for(session.get_one(WorkflowVersion, engine.workflow_version_id))


type TextRunnerFor = Callable[[Any, str, Callable[[], bool]], Any]
type Joiner = Callable[[Sequence[Path], Path], media.VideoInfo]


def _text_runner(client: Any, folder: str, should_cancel: Callable[[], bool]) -> Any:
    return ComfyTextRunner(client, subfolder=folder, should_cancel=should_cancel)


def _join(clips: Sequence[Path], dest: Path) -> media.VideoInfo:
    return media.join(clips, dest, fps=FPS)


@dataclass
class Deps:
    store: FileStore
    comfy: ComfyLike
    gemma: GemmaLike
    work_dir: Path
    workflows_dir: Path
    adapter: AdapterFor = pinned_adapter
    recipe_adapter: RecipeAdapterFor = recipe_adapter
    text_runner: TextRunnerFor = _text_runner
    join: Joiner = _join
    # The director on its own llama-server (memcraft.pipeline.services.LlamaDirector), or None
    # for Qwen 3.8 inside ComfyUI. The critic keeps text_runner.
    director: Any = None
    # Seconds a ComfyUI run is expected to take, for the progress it reports while it waits.
    expected: dict[str, float] = field(
        default_factory=lambda: {"director": 150.0, "keyframe": 60.0, "clip": 120.0}
    )


# Helpers.


def folder_of(job_id: uuid.UUID) -> str:
    """The job's subfolder in ComfyUI's input and output folders: its id, nothing personal."""
    return job_id.hex


@contextlib.contextmanager
def work_folder(deps: Deps, ctx: StepContext) -> Iterator[Path]:
    folder = deps.work_dir / folder_of(ctx.job_id) / f"{ctx.step.value}-{ctx.ordinal}-{ctx.attempt}"
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    try:
        yield folder
    finally:
        shutil.rmtree(folder, ignore_errors=True)
        with contextlib.suppress(OSError):
            folder.parent.rmdir()  # only when empty


def save_file(
    deps: Deps,
    ctx: StepContext,
    kind: AssetKind,
    path: Path,
    mime: str,
    ordinal: int,
    *,
    step_id: uuid.UUID | None = None,
) -> Asset:
    with Path(path).open("rb") as source:
        return deps.store.save(
            ctx.session,
            owner_id=ctx.owner_id,
            job_id=ctx.job_id,
            kind=kind,
            mime=mime,
            source=source,
            ordinal=ordinal,
            step_id=step_id or ctx.step_id,
        )


def asset_path(session: Session, store: FileStore, asset_id: str | uuid.UUID) -> Path:
    asset = session.get(Asset, uuid.UUID(str(asset_id)))
    if asset is None:
        raise PipelineError(f"asset {asset_id} is gone")
    path = store.path_for(asset)
    if not path.is_file():
        raise PipelineError(f"the file of asset {asset_id} is missing")
    return path


def gemma_images(session: Session, store: FileStore, job_id: uuid.UUID) -> list[bytes]:
    """The small JPEG copies of the job's photos, made now if missing."""
    copies = make_gemma_copies(session, store, job_id)
    return [store.path_for(copy).read_bytes() for copy in copies]


def critic_jpeg(path: Path, size: tuple[int, int] = CRITIC_SIZE) -> bytes:
    with Image.open(path) as image:
        small = image.convert("RGB").resize(size, Image.Resampling.LANCZOS)
    out = io.BytesIO()
    small.save(out, "JPEG", quality=90)
    return out.getvalue()


def approved_idea(session: Session, job: Job) -> str:
    """The idea the video is made of: the latest idea row, else what the friend typed."""
    idea = session.scalar(
        select(Idea).where(Idea.job_id == job.id).order_by(Idea.created_at.desc()).limit(1)
    )
    text = (idea.text if idea is not None else None) or job.idea_text or ""
    if not text.strip():
        raise PipelineError("the job has no idea to render")
    return text.strip()


def job_regime(job: Job) -> JobRegime:
    return job.regime or regime_for(job.photo_count, job.length)


def job_seed(job_id: uuid.UUID, attempt: int = 1) -> int:
    """A seed that stays the same for a job, and changes with a new attempt."""
    return (job_id.int % 1_000_000_007) + 101 * (attempt - 1)


def comfy_progress(
    ctx: StepContext, expected: float, start: float = 0.05, end: float = 0.95
) -> Callable[[Progress], None]:
    """Progress from ComfyUI's polls: rises with the time spent, never reaching `end`.

    Only a poll that ComfyUI answered calls it, so a hung ComfyUI reports nothing and the
    watchdog sees the stall.
    """

    def report(progress: Progress) -> None:
        share = 1 - math.exp(-progress.elapsed / max(expected, 1.0))
        ctx.progress(start + (end - start) * share)

    return report


class _Reporting:
    """A ComfyClient whose run() reports progress, for callers that cannot pass on_progress."""

    def __init__(self, client: Any, on_progress: Callable[[Progress], None]) -> None:
        self._client = client
        self._on_progress = on_progress

    def run(self, graph: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("on_progress", self._on_progress)
        return self._client.run(graph, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


def _progressing(run_text: Any, ctx: StepContext) -> Any:
    """A TextRunner that moves the step's progress after each answer, so the watchdog sees the
    llama-server director working."""
    calls = [0]

    def run(system: str, user: str, images: Sequence[bytes], seed: int) -> str:
        ctx.check_stop()
        answer = run_text(system, user, images, seed)
        calls[0] += 1
        ctx.progress(min(0.9, 0.05 + 0.3 * calls[0]))
        return answer

    return run


def run_options(ctx: StepContext, expected: float, start: float, end: float) -> RunOptions:
    return RunOptions(
        on_progress=comfy_progress(ctx, expected, start, end),
        should_cancel=lambda: ctx.stopping,
    )


def output_of(ctx: StepContext, step: StepName, ordinal: int = 0) -> Mapping[str, Any]:
    output = ctx.outputs.get((step, ordinal))
    if output is None:
        raise PipelineError(f"step {step.value} {ordinal} has not run")
    return output


def chosen_keyframes(outputs: Mapping[tuple[StepName, int], Any]) -> dict[int, str]:
    """Asset ids of the keyframes the clips use: the critic's choice where an older check chose."""
    chosen = {
        ordinal: output["asset_id"]
        for (step, ordinal), output in outputs.items()
        if step is StepName.KEYFRAME and output
    }
    check = outputs.get((StepName.CHECK, 0)) or {}
    if check.get("kind") == "keyframes":
        for entry in check.get("keyframes", []):
            chosen[entry["ordinal"]] = entry["asset_id"]
    return chosen


def _row(ctx: StepContext) -> JobStep:
    return ctx.session.get_one(JobStep, ctx.step_id)


# Handlers.


@dataclass
class IdeaStep:
    """The friend's own idea is kept as typed, without Gemma; with no idea, Gemma reads the
    photos and proposes one."""

    deps: Deps
    needs_gpu: bool = False
    model: str | None = None
    service: str | None = GEMMA_SERVICE

    def run(self, ctx: StepContext) -> Mapping[str, Any]:
        session = ctx.session
        job = session.get_one(Job, ctx.job_id)
        idea, length = job.idea_text, job.length
        if idea and idea.strip():
            row = record_idea(session, ctx.job_id, user_idea(idea), None)
            row.approved_at = func.now()  # the friend's own idea needs no approval
            session.flush()
            return {"idea_id": str(row.id), "proposed_by": IdeaSource.USER.value, "skipped": True}
        ctx.progress(0.05)
        images = gemma_images(session, self.deps.store, ctx.job_id)
        prompt = active_prompt(session, PromptRole.CHAT, GEMMA)
        started = time.monotonic()
        with self.deps.gemma.client() as client:
            result = run_chat(
                client, prompt, images, idea, length=length, seed=ctx.attempt, cast=job.cast
            )
        ctx.progress(0.95)
        row = record_idea(session, ctx.job_id, result, prompt)
        if result.proposed_by is IdeaSource.USER:
            row.approved_at = func.now()  # the friend's own idea needs no approval
        session.flush()
        output = result.as_output()
        output.update(idea_id=str(row.id), wall_seconds=round(time.monotonic() - started, 1))
        return output


@dataclass
class ScenarioStep:
    """Qwen 3.8 directs: LTX prompts per run, and for keyframes the bible and three pictures.

    Pins the recipe and the critic, the first thing of the render that stays with the job.
    """

    deps: Deps
    needs_gpu: bool = True
    model: str | None = MODEL_DIRECTOR
    service: str | None = COMFYUI

    def run(self, ctx: StepContext) -> Mapping[str, Any]:
        session, deps = ctx.session, self.deps
        job = session.get_one(Job, ctx.job_id)
        regime = job_regime(job)
        idea = approved_idea(session, job)
        images = gemma_images(session, deps.store, ctx.job_id)
        recipe = choices.render_recipe(session, job, regime, deps.workflows_dir)
        critic = (
            session.get(EngineVersion, job.critic_engine_id)
            if job.recipe_id is not None
            else choices.render_critic(session)
        )
        rules = rules_for(recipe)
        runs = rules.runs_long if job.length.value == "long" else rules.runs_short
        frames = rules.frames_per_run * runs - (runs - 1)
        role = (
            PromptRole.DIRECTOR_MULTISHOT
            if regime is JobRegime.MULTISHOT
            else PromptRole.DIRECTOR_KEYFRAMES
        )
        prompt = active_prompt(session, role, QWEN)
        seed = job_seed(ctx.job_id, ctx.attempt)
        folder = folder_of(ctx.job_id)
        ctx.progress(0.02)
        deps.comfy.ensure_up()
        started = time.monotonic()
        session_of = deps.director.session if deps.director is not None else contextlib.nullcontext
        with session_of() as llama, deps.comfy.client() as client:
            if llama is not None:
                runner = _progressing(llama, ctx)
            else:
                reporting = _Reporting(client, comfy_progress(ctx, deps.expected["director"]))
                runner = deps.text_runner(reporting, folder, lambda: ctx.stopping)
            if regime is JobRegime.MULTISHOT:
                scenario: Any = direct_multishot(
                    runner,
                    prompt,
                    images[0],
                    idea,
                    length=job.length,
                    shots=rules.default_shots,
                    frames=frames,
                    seed=seed,
                    cast=job.cast,
                )
            else:
                inventory, confirm = None, None
                found = parts(prompt.text)
                if "inventory" in found and "background" in found:
                    inventory = self._inventory(ctx, client, runner, recipe, prompt, images, seed)
                    confirm = self._confirm(ctx, client, recipe, inventory)
                scenario = direct_keyframes(
                    runner,
                    prompt,
                    images,
                    idea,
                    length=job.length,
                    frames=frames,
                    seed=seed,
                    cast=job.cast,
                    inventory=inventory,
                    confirm=confirm,
                    faces=face_boxes(session, deps.store, ctx.job_id) if confirm else None,
                )
        deps.comfy.clean(folder)
        output = scenario.as_output()
        if regime is not JobRegime.MULTISHOT and inventory is not None:
            output["inventory"] = inventory
        output.update(
            regime=regime.value,
            idea=idea,
            recipe={"id": str(recipe.id), "name": recipe.name, "version": recipe.version},
            critic_engine_id=None if critic is None else str(critic.id),
            wall_seconds=round(time.monotonic() - started, 1),
        )
        # Last: these lock the job and the step row until the worker commits.
        if job.regime is None:
            job.regime = regime
        versions.pin_job(session, job, recipe=recipe, critic=critic)
        versions.pin_step(session, _row(ctx), prompt=prompt)
        return output

    def _inventory(
        self,
        ctx: StepContext,
        client: Any,
        runner: Any,
        recipe: Recipe,
        prompt: Any,
        images: list[bytes],
        seed: int,
    ) -> list[dict[str, Any]]:
        """E16: Qwen 3.8 lists what each photo's people hold, and SAM checks every object on the
        photo itself; only what SAM finds counts. A failed check counts as not found."""
        entries, _raw = take_inventory(runner, prompt, images, seed)
        for entry in entries:
            entry["checked"] = {
                item: self._find(ctx, client, recipe, entry["photo"], item)
                for item in entry["holds"]
            }
        return entries

    def _confirm(
        self, ctx: StepContext, client: Any, recipe: Recipe, inventory: list[dict[str, Any]]
    ) -> Any:
        """A check for the director's "from <imageN>" props: SAM looks for the prop's own words
        on that photo. Every answer is kept in the photo's inventory entry."""
        by_photo = {entry["photo"]: entry for entry in inventory}

        def confirm(photo: int, words: str) -> bool:
            entry = by_photo.setdefault(photo, {"photo": photo, "who": "", "holds": []})
            claims = entry.setdefault("claims", {})
            if words not in claims:
                claims[words] = self._find(ctx, client, recipe, photo, words)
            return bool(claims[words])

        return confirm

    def _find(self, ctx: StepContext, client: Any, recipe: Recipe, photo: int, what: str) -> bool:
        """Whether SAM finds ``what`` on photo ``photo``; a check that fails counts as no."""
        session, deps = ctx.session, self.deps
        if deps.director is not None:
            # SAM runs in ComfyUI and needs the GPU memory the director's llama-server holds;
            # the next director call starts the server again.
            deps.director.release()
        try:
            path = deps.store.path_for(list_photos(session, ctx.job_id)[photo - 1])
            adapter = deps.recipe_adapter(session, recipe, EngineSlot.KEYFRAME)
            request = KeyframeRequest(
                prompt="find", photos=(path,), seed=1, detect=what, folder=folder_of(ctx.job_id)
            )
            with work_folder(deps, ctx) as work:
                found = adapter.render(client, request, work / "found.png")
                return mask_share(found.image) >= HELD_SHARE
        except Exception as error:  # a check that fails never fails the job
            log.warning("job %s: SAM check of %r failed: %s", ctx.job_id, what, error)
            return False


def face_boxes(session: Session, store: FileStore, job_id: uuid.UUID) -> dict[int, list[float]]:
    """The largest face of each photo, as fractions [x, y, w, h], for the face crops of the
    background-first keyframes (E16). A photo without a face found has none."""
    from memcraft.web.faces import faces_in

    boxes: dict[int, list[float]] = {}
    for number, photo in enumerate(list_photos(session, job_id), start=1):
        try:
            found = faces_in(store.path_for(photo).read_bytes())
        except Exception as error:  # no face crop is no reason to fail the scenario
            log.warning("job %s: no face search in photo %d: %s", job_id, number, error)
            continue
        if found:
            box = max(found, key=lambda b: b.w * b.h)
            boxes[number] = [box.x, box.y, box.w, box.h]
    return boxes


def crop_face(photo: Path, box: Sequence[float], dest: Path) -> Path:
    """The face with hair and chin, square, from the upright photo, at least 512 px."""
    with Image.open(photo) as opened:
        from PIL import ImageOps

        image = ImageOps.exif_transpose(opened).convert("RGB")
    x, y, w, h = (v * s for v, s in zip(box, (*image.size, *image.size), strict=True))
    side = max(w, h) * 2.0
    cx, cy = x + w / 2, y + h / 2 - h * 0.1
    left, top = max(0, cx - side / 2), max(0, cy - side / 2)
    right, bottom = min(image.width, left + side), min(image.height, top + side)
    face = image.crop((round(left), round(top), round(right), round(bottom)))
    if face.width < 512:
        face = face.resize((512, round(512 * face.height / face.width)), Image.Resampling.LANCZOS)
    face.save(dest)
    return dest


def mask_share(path: Path) -> float:
    """How much of a white-on-black mask picture is white."""
    with Image.open(path) as image:
        gray = image.convert("L")
        hist = gray.histogram()
    return sum(hist[128:]) / max(1, sum(hist))


# The share of a photo SAM must mark for an object to count as held (a knife is about 0.5 %).
HELD_SHARE = 0.001


@dataclass
class KeyframeStep:
    """One keyframe with Qwen-Image; middle and end take the start keyframe as a reference
    when the director chose so (1 or 2 people)."""

    deps: Deps
    needs_gpu: bool = True
    model: str | None = MODEL_KEYFRAMES
    service: str | None = COMFYUI

    def run(self, ctx: StepContext) -> Mapping[str, Any]:
        session, deps = ctx.session, self.deps
        job = session.get_one(Job, ctx.job_id)
        scenario = output_of(ctx, StepName.SCENARIO)
        is_asset = ctx.ordinal >= ASSET_ORDINAL
        if is_asset and scenario["assets"][ctx.ordinal - ASSET_ORDINAL]["kind"] == "face":
            return self._face(ctx, scenario["assets"][ctx.ordinal - ASSET_ORDINAL])
        if is_asset:
            keyframe = scenario["assets"][ctx.ordinal - ASSET_ORDINAL]
            references: list[int | str] = [keyframe["photo"] or "canvas"]
        else:
            keyframe = scenario["keyframes"][ctx.ordinal]
            references = list(keyframe["references"])
        photos = [deps.store.path_for(photo) for photo in list_photos(session, ctx.job_id)]
        adapter = deps.adapter(session, job, EngineSlot.KEYFRAME)
        workflow = versions.pinned_workflow(session, job, EngineSlot.KEYFRAME)
        deps.comfy.ensure_up()
        with work_folder(deps, ctx) as work:
            paths = [self._reference(ctx, ref, photos, work) for ref in references]
            # The start keyframe last is the older layout: photos, then the scene to match.
            extra = tuple(paths[-1:]) if references[-1] == "start" else ()
            request = KeyframeRequest(
                prompt=keyframe["prompt"],
                photos=tuple(paths[: len(paths) - len(extra)]),
                seed=keyframe["seed"],
                extra_references=extra,
                folder=folder_of(ctx.job_id),
                cutout=bool(is_asset and keyframe.get("cutout")),
                cut_only=is_asset and keyframe.get("kind") == "cut",
                **(
                    {"steps": CUTOUT_STEPS}
                    if is_asset and keyframe.get("cutout") and keyframe.get("kind") != "cut"
                    else {}
                ),
                **(
                    {"negative_prompt": BACKGROUND_NEGATIVE}
                    if is_asset and keyframe.get("kind") == "background"
                    else {}
                ),
            )
            with deps.comfy.client() as client:
                result = adapter.render(
                    client,
                    request,
                    work / "keyframe.png",
                    run_options(ctx, deps.expected["keyframe"], 0.02, 0.95),
                )
            matched = False
            if 0 < ctx.ordinal < ASSET_ORDINAL:
                # Middle and end keyframes take the start keyframe's brightness and tint (E16).
                start = output_of(ctx, StepName.KEYFRAME, 0)
                match_colours(result.image, asset_path(session, deps.store, start["asset_id"]))
                matched = True
            asset = save_file(deps, ctx, AssetKind.KEYFRAME, result.image, PNG, ctx.ordinal)
        deps.comfy.clean(request.folder)
        versions.pin_step(session, _row(ctx), workflow=workflow)
        return {
            "asset_id": str(asset.id),
            "at": "asset" if is_asset else keyframe["at"],
            "seed": request.seed,
            "seconds": round(result.seconds, 1),
            "prompt_id": result.prompt_id,
            "colour_matched": matched,
        }

    def _face(self, ctx: StepContext, asset: Mapping[str, Any]) -> Mapping[str, Any]:
        """A face crop, cut from the photo here; nothing runs on the GPU for it."""
        session, deps = ctx.session, self.deps
        photo = deps.store.path_for(list_photos(session, ctx.job_id)[asset["photo"] - 1])
        with work_folder(deps, ctx) as work:
            path = crop_face(photo, asset["box"], work / "face.png")
            saved = save_file(deps, ctx, AssetKind.KEYFRAME, path, PNG, ctx.ordinal)
        return {"asset_id": str(saved.id), "at": "asset", "seed": 0, "seconds": 0.0}

    def _reference(self, ctx: StepContext, ref: int | str, photos: list[Path], work: Path) -> Path:
        """The file of one reference: a photo number, "start", "asset:K" or "canvas"."""
        session, store = ctx.session, self.deps.store
        if isinstance(ref, int):
            return photos[ref - 1]
        if ref == "start":
            return asset_path(session, store, output_of(ctx, StepName.KEYFRAME, 0)["asset_id"])
        if ref.startswith("asset:"):
            drawn = output_of(ctx, StepName.KEYFRAME, ASSET_ORDINAL + int(ref.split(":")[1]))
            return asset_path(session, store, drawn["asset_id"])
        if ref == "canvas":
            canvas = work / "canvas.png"
            Image.new("RGB", (768, 1344), (200, 200, 200)).save(canvas)
            return canvas
        raise PipelineError(f"unknown keyframe reference {ref!r}")


@dataclass
class CheckStep:
    """The critic looks at faces only: every keyframe (before LTX loads), or the last frame.

    Its verdict is a warning that is recorded and logged: nothing is redrawn, and the step
    never fails the job, not even when the critic cannot run (E0: faces were the only check
    worth keeping).
    """

    deps: Deps
    needs_gpu: bool = True
    model: str | None = MODEL_KEYFRAMES
    service: str | None = GEMMA_SERVICE

    def run(self, ctx: StepContext) -> Mapping[str, Any]:
        session, deps = ctx.session, self.deps
        kind = ctx.inputs.get("kind")
        job = session.get_one(Job, ctx.job_id)
        engine = session.get(EngineVersion, job.critic_engine_id) if job.critic_engine_id else None
        if engine is None:
            return {"kind": kind, "skipped": "the critic is off"}
        started = time.monotonic()
        critic: Critic | None = None
        try:
            photos = gemma_images(session, deps.store, ctx.job_id)
            with contextlib.ExitStack() as stack:
                gemma = run_text = None
                if engine.name == QWEN:
                    deps.comfy.ensure_up()
                    client = stack.enter_context(deps.comfy.client())
                    run_text = deps.text_runner(client, folder_of(ctx.job_id), lambda: ctx.stopping)
                else:
                    gemma = stack.enter_context(deps.gemma.client())
                critic = pinned_critic(session, job, gemma=gemma, run_text=run_text)
                if critic is None:
                    return {"kind": kind, "skipped": "the critic is off"}
                if kind == "keyframes":
                    output = self._keyframes(ctx, critic, photos)
                else:
                    output = self._last_frame(ctx, critic, photos)
        except StepStopped:
            raise
        except Exception as error:  # a warning step never fails the job
            log.warning("job %s: the face check could not run: %s", ctx.job_id, error)
            output = {"kind": kind, "faces_ok": None, "error": str(error)[:300]}
        output["wall_seconds"] = round(time.monotonic() - started, 1)
        if critic is not None:
            versions.pin_step(session, _row(ctx), prompt=critic.prompt)
        return output

    def _verdict(
        self, ctx: StepContext, critic: Critic, photos: list[bytes], path: Path, label: str
    ) -> dict[str, Any]:
        """The face verdict of one picture, logged; a critic that fails is a warning too."""
        try:
            verdict = critic.check(photos, critic_jpeg(path), "", seed=1)
        except StepStopped:
            raise
        except Exception as error:  # a warning step never fails the job
            log.warning("job %s: no face verdict for %s: %s", ctx.job_id, label, error)
            return {"face": None, "face_why": "", "error": str(error)[:300]}
        if verdict.face:
            log.info("job %s: faces of %s pass: %s", ctx.job_id, label, verdict.face_why)
        else:
            log.warning("job %s: faces of %s fail: %s", ctx.job_id, label, verdict.face_why)
        return {
            "face": verdict.face,
            "face_why": verdict.face_why,
            "model": verdict.model,
            "seconds": verdict.seconds,
            "attempts": verdict.attempts,
        }

    def _keyframes(self, ctx: StepContext, critic: Critic, photos: list[bytes]) -> dict[str, Any]:
        session, deps = ctx.session, self.deps
        scenario = output_of(ctx, StepName.SCENARIO)
        drawn = chosen_keyframes(ctx.outputs)
        count = len(scenario["keyframes"])
        entries: list[dict[str, Any]] = []
        for index, keyframe in enumerate(scenario["keyframes"]):
            ctx.check_stop()
            path = asset_path(session, deps.store, drawn[index])
            entry: dict[str, Any] = {
                "ordinal": index,
                "at": keyframe["at"],
                "asset_id": drawn[index],
            }
            entry |= self._verdict(ctx, critic, photos, path, f"keyframe {keyframe['at']}")
            entries.append(entry)
            ctx.progress((index + 1) / count)
        return {"kind": "keyframes", "keyframes": entries, "faces_ok": _faces_ok(entries)}

    def _last_frame(self, ctx: StepContext, critic: Critic, photos: list[bytes]) -> dict[str, Any]:
        session, deps = ctx.session, self.deps
        scenario = output_of(ctx, StepName.SCENARIO)
        last_run = len(scenario["runs"]) - 1
        clip = output_of(ctx, StepName.CLIP, last_run)
        path = asset_path(session, deps.store, clip["last_frame"])
        entry = self._verdict(ctx, critic, photos, path, f"the last frame of run {last_run}")
        ctx.progress(1.0)
        return {"kind": "last_frame", "run": last_run, **entry, "faces_ok": _faces_ok([entry])}


def _faces_ok(entries: Sequence[Mapping[str, Any]]) -> bool | None:
    """False when any face failed, None when no verdict came back, else True."""
    faces = [entry.get("face") for entry in entries]
    if any(face is False for face in faces):
        return False
    return None if any(face is None for face in faces) else True


@dataclass
class ClipStep:
    """One LTX-2.5 run: the 544x960 preview with sound, its real last frame and its latents."""

    deps: Deps
    needs_gpu: bool = True
    model: str | None = MODEL_CLIPS
    service: str | None = COMFYUI

    def run(self, ctx: StepContext) -> Mapping[str, Any]:
        session, deps = ctx.session, self.deps
        job = session.get_one(Job, ctx.job_id)
        request, described = clip_request(session, deps.store, job, ctx.outputs, ctx.ordinal)
        adapter = deps.adapter(session, job, EngineSlot.CLIP)
        workflow = versions.pinned_workflow(session, job, EngineSlot.CLIP)
        deps.comfy.ensure_up()
        run = ctx.ordinal
        with work_folder(deps, ctx) as work:
            with deps.comfy.client() as client:
                result = adapter.render(
                    client,
                    request,
                    work,
                    run_options(ctx, deps.expected["clip"], 0.02, 0.95),
                )
            video = save_file(deps, ctx, AssetKind.CLIP, result.video, MP4, run)
            last = save_file(
                deps, ctx, AssetKind.KEYFRAME, result.last_frame, PNG, LAST_FRAME_ORDINAL + run
            )
            latent_v = save_file(deps, ctx, AssetKind.LATENT, result.latent_video, LATENT, 2 * run)
            latent_a = save_file(
                deps, ctx, AssetKind.LATENT, result.latent_audio, LATENT, 2 * run + 1
            )
        deps.comfy.clean(request.folder)
        versions.pin_step(session, _row(ctx), workflow=workflow)
        return {
            "video": str(video.id),
            "last_frame": str(last.id),
            "latent_video": str(latent_v.id),
            "latent_audio": str(latent_a.id),
            "frames": result.frames,
            "fps": result.fps,
            "seconds": round(result.seconds, 1),
            "prompt_id": result.prompt_id,
            "request": described,
        }


def clip_request(
    session: Session,
    store: FileStore,
    job: Job,
    outputs: Mapping[tuple[StepName, int], Any],
    run: int,
) -> tuple[ClipRequest, dict[str, Any]]:
    """The request of run `run`, and the same as JSON with asset ids, for the release."""
    scenario = outputs.get((StepName.SCENARIO, 0))
    if scenario is None:
        raise PipelineError("the scenario has not run")
    director_run = scenario["runs"][run]
    if job.recipe_id is None:
        raise PipelineError("the job has no pinned recipe")
    recipe = session.get_one(Recipe, job.recipe_id)
    frames = [r["frames"] for r in scenario["runs"]]
    count = len(scenario.get("keyframes") or []) or None
    run_plan = plan(rules_for(recipe), job.length, frames, count).runs[run]
    keyframes = chosen_keyframes(outputs)
    if run_plan.first is First.PHOTO:
        photo = list_photos(session, job.id)[0]
        first_id = str(photo.id)
    elif run_plan.first is First.KEYFRAME:
        first_id = keyframes[run_plan.first_keyframe or 0]
    else:
        before = outputs.get((StepName.CLIP, run - 1))
        if before is None:
            raise PipelineError(f"run {run} needs the last frame of run {run - 1}")
        first_id = before["last_frame"]
    guides = [[keyframes[index], frame] for index, frame in run_plan.guides]
    described = {
        # Never music, also for a scenario written before the director removed it.
        "prompt": no_music(director_run["prompt"]),
        "seed": director_run["seed"],
        "frames": director_run["frames"],
        "fps": FPS,
        "first_strength": run_plan.first_strength,
        "guide_strength": run_plan.guide_strength,
        "first_frame": first_id,
        "guides": guides,
    }
    return request_from(session, store, job.id, described), described


def request_from(
    session: Session, store: FileStore, job_id: uuid.UUID, described: Mapping[str, Any]
) -> ClipRequest:
    return ClipRequest(
        prompt=described["prompt"],
        first_frame=asset_path(session, store, described["first_frame"]),
        frames=described["frames"],
        seed=described["seed"],
        guides=tuple(
            Guide(asset_path(session, store, asset_id), frame if frame >= 0 else LAST)
            for asset_id, frame in described["guides"]
        ),
        fps=described.get("fps", FPS),
        first_strength=described["first_strength"],
        # Scenarios planned before E15 have no guide strength and keep the old 1.0.
        guide_strength=described.get("guide_strength", 1.0),
        folder=folder_of(job_id),
    )


@dataclass
class JoinStep:
    """A long video's runs as one: run 2 drops its first frame, one encode, sound crossfades."""

    deps: Deps
    needs_gpu: bool = False
    model: str | None = None
    service: str | None = None

    def run(self, ctx: StepContext) -> Mapping[str, Any]:
        session, deps = ctx.session, self.deps
        runs = int(ctx.inputs["runs"])
        clips = [output_of(ctx, StepName.CLIP, run) for run in range(runs)]
        paths = [asset_path(session, deps.store, clip["video"]) for clip in clips]
        started = time.monotonic()
        with work_folder(deps, ctx) as work:
            info = deps.join(paths, work / "joined.mp4")
            ctx.progress(0.9)
            asset = save_file(deps, ctx, AssetKind.CLIP, work / "joined.mp4", MP4, JOINED_ORDINAL)
        return {
            "video": str(asset.id),
            "frames": info.frames,
            "run_frames": [clip["frames"] for clip in clips],
            "seconds": round(info.seconds, 2),
            "has_audio": info.has_audio,
            "wall_seconds": round(time.monotonic() - started, 1),
        }


@dataclass
class ReleaseStep:
    """HD (RTX upscale) or Full HD (x2 refine) from the saved latents, with the clip engine
    version the preview used; a long video is joined again the same way."""

    deps: Deps
    needs_gpu: bool = True
    model: str | None = MODEL_CLIPS
    service: str | None = COMFYUI

    def run(self, ctx: StepContext) -> Mapping[str, Any]:
        session, deps = ctx.session, self.deps
        job = session.get_one(Job, ctx.job_id)
        quality = JobQuality(ctx.inputs["quality"])
        clips = clip_outputs(session, ctx.job_id)
        if not clips:
            raise PipelineError("the job has no clips to release")
        adapter = deps.adapter(session, job, EngineSlot.CLIP)
        jobs = [
            (
                request_from(session, deps.store, ctx.job_id, clip["request"]),
                asset_path(session, deps.store, clip["latent_video"]),
                asset_path(session, deps.store, clip["latent_audio"]),
            )
            for clip in clips
        ]
        deps.comfy.ensure_up()
        started = time.monotonic()
        with work_folder(deps, ctx) as work:
            videos, seconds, size = [], [], (0, 0)
            with deps.comfy.client() as client:
                for run, (request, latent_v, latent_a) in enumerate(jobs):
                    ctx.check_stop()
                    share = 0.9 / len(jobs)
                    expected = 120.0 if quality is JobQuality.HD720 else 420.0
                    result = adapter.release(
                        client,
                        request,
                        latent_v,
                        latent_a,
                        quality,
                        work / f"release-{run}.mp4",
                        options=run_options(ctx, expected, run * share, (run + 1) * share),
                    )
                    videos.append(result.video)
                    seconds.append(round(result.seconds, 1))
                    size = (result.width, result.height)
            final = work / "release.mp4"
            info = deps.join(videos, final)
            ctx.progress(0.95)
            asset = save_file(deps, ctx, AssetKind.RELEASE, final, MP4, 0)
        deps.comfy.clean(folder_of(ctx.job_id))
        return {
            "asset_id": str(asset.id),
            "quality": quality.value,
            "width": size[0],
            "height": size[1],
            "frames": info.frames,
            "run_seconds": seconds,
            "wall_seconds": round(time.monotonic() - started, 1),
        }


def clip_outputs(session: Session, job_id: uuid.UUID) -> list[Mapping[str, Any]]:
    """The done CLIP outputs of the job, by run."""
    rows = session.scalars(
        select(JobStep)
        .where(
            JobStep.job_id == job_id,
            JobStep.step == StepName.CLIP,
            JobStep.status == StepStatus.DONE,
        )
        .order_by(JobStep.ordinal, JobStep.attempt.desc())
    ).all()
    by_run: dict[int, Mapping[str, Any]] = {}
    for row in rows:
        if row.ordinal not in by_run and row.output:
            by_run[row.ordinal] = row.output
    return [by_run[run] for run in sorted(by_run)]
