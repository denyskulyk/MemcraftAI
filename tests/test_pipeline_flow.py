"""The pipeline end to end with fake engines and fake roles, on Postgres.

Gemma answers through GemmaClient with a fake transport, Qwen through a fake text runner, and
the keyframe and clip adapters write small files (the clips are real mp4s made by ffmpeg, so the
join is real). Everything else is the real thing: the worker, the queue, the file store, the
versions and the pins.
"""

import contextlib
import io
import json
import threading
import uuid
from pathlib import Path

import pytest
from PIL import Image
from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from memcraft.engines import ClipResult, KeyframeResult, ReleaseResult, register_defaults
from memcraft.models import (
    Asset,
    AssetKind,
    AssetStore,
    Idea,
    Job,
    JobLane,
    JobLength,
    JobQuality,
    JobRegime,
    JobStatus,
    JobStep,
    PromptRole,
    Setting,
    StepName,
    StepStatus,
    User,
)
from memcraft.photos import CropBox, intake
from memcraft.pipeline import build_registry, media
from memcraft.pipeline.planner import PipelinePlanner
from memcraft.pipeline.steps import Deps, folder_of
from memcraft.queue import transition
from memcraft.roles import seed as seed_module
from memcraft.roles.gemma import GemmaClient
from memcraft.roles.seed import seed_prompts
from memcraft.storage import FileStore
from memcraft.worker.loop import Worker, WorkerConfig

S = JobStatus
ROOT = Path(__file__).resolve().parents[1]
FAST = WorkerConfig(poll=0.02, heartbeat=0.2, retry_delay=0, progress_interval=0)

try:
    FFMPEG = media.find_ffmpeg()
    media.find_tool("ffprobe")
except media.MediaError:
    FFMPEG = None

pytestmark = [
    pytest.mark.parametrize(
        "engine", [pytest.param("postgres", marks=pytest.mark.postgres)], indirect=True
    ),
    pytest.mark.skipif(FFMPEG is None, reason="ffmpeg is not installed"),
]


# Fakes.


class FakeComfy:
    def __init__(self):
        self.cleaned = []
        self.started = 0

    def ensure_up(self):
        self.started += 1

    def client(self):
        return contextlib.nullcontext(object())

    def clean(self, folder):
        self.cleaned.append(folder)


class FakeGemma:
    """Gemma behind the real GemmaClient: the transport answers by the schema it is sent."""

    def __init__(self):
        self.calls = []
        self.verdicts = []  # queued critic answers; default: everything passes
        self.lock = threading.Lock()

    @contextlib.contextmanager
    def client(self):
        yield GemmaClient("http://127.0.0.1:8098", post=self.post)

    def post(self, url, body, timeout):
        schema = body["response_format"]["json_schema"]["schema"]["properties"]
        with self.lock:
            if "face" in schema:
                self.calls.append("critic")
                answer = self.verdicts.pop(0) if self.verdicts else passing()
            else:
                self.calls.append("chat")
                key = "understood" if "understood" in schema else "idea"
                answer = {key: "a cat steals the show", "weakness": "seen before", "score": 6}
        return {"choices": [{"message": {"content": json.dumps(answer)}, "finish_reason": "stop"}]}


def passing():
    return verdict(True)


def verdict(face, why="ok"):
    return {"face_why": why, "face": "pass" if face else "fail"}


SECTIONS = """TITLE: The test
PEOPLE: <image1>: the man, red jacket | <image2>: the woman, glasses | <image3>: old man, cap
PLACE: a sunny park
LIGHT: soft morning light.
CAMERA: medium shot, eye level.
KEYFRAME START: the man from image1 stands by a bench.
KEYFRAME MIDDLE: the man from image1 opens an umbrella.
KEYFRAME END: the man from image1 floats above the bench.
RUN 1: A man in a red jacket opens an umbrella and rises slowly. Single take, no cuts.
RUN 2: He floats higher and waves. Single take, no cuts.
"""


class FakeDirector:
    def __init__(self):
        self.calls = 0
        self.extra = ""  # sections put before KEYFRAME START, like PROPS, CLEAN, PROP PICTURES

    def __call__(self, client, folder, should_cancel):
        def run(system, user, images, seed):
            self.calls += 1
            # The bible lists one person per photo.
            head, people, tail = SECTIONS.partition(SECTIONS.splitlines()[1])
            listed = " | ".join(people.removeprefix("PEOPLE: ").split(" | ")[: len(images)])
            tail = tail.replace("KEYFRAME START:", f"{self.extra}KEYFRAME START:")
            return f"{head}PEOPLE: {listed}{tail}"

        return run


class FakeAdapters:
    def __init__(self, tmp_path):
        self.keyframes = []
        self.clips = []
        self.releases = []
        self.fail_clip = None  # an exception class to raise once in render()
        self.lock = threading.Lock()

    def __call__(self, session, job, slot):
        return self

    def render(self, client, request, dest, options=None):
        if hasattr(request, "extra_references"):
            return self._keyframe(request, dest)
        return self._clip(request, dest)

    def _keyframe(self, request, dest):
        with self.lock:
            self.keyframes.append(request)
        Image.new("RGB", (96, 168), (len(self.keyframes) * 20 % 255, 80, 120)).save(dest)
        return KeyframeResult(Path(dest), 1.5, uuid.uuid4().hex, 768, 1344)

    def _clip(self, request, dest):
        with self.lock:
            self.clips.append(request)
        if self.fail_clip is not None:
            error, self.fail_clip = self.fail_clip, None
            raise error("the clip died")
        dest = Path(dest)
        video = make_clip(dest / "preview.mp4", request.frames)
        Image.new("RGB", (64, 112), "blue").save(dest / "last.png")
        (dest / "latent_v.latent").write_bytes(b"latent video " + request.prompt.encode())
        (dest / "latent_a.latent").write_bytes(b"latent audio")
        return ClipResult(
            video,
            dest / "last.png",
            dest / "latent_v.latent",
            dest / "latent_a.latent",
            2.0,
            uuid.uuid4().hex,
            request.frames,
            request.fps,
        )

    def release(self, client, request, latent_video, latent_audio, quality, dest, **kw):
        with self.lock:
            self.releases.append((request, latent_video.read_bytes(), quality))
        make_clip(Path(dest), request.frames, size=(72, 128))
        return ReleaseResult(Path(dest), 3.0, uuid.uuid4().hex, quality, 720, 1280)


def make_clip(path, frames, size=(64, 112)):
    import subprocess

    seconds = frames / 30
    subprocess.run(  # noqa: S603 - ffmpeg with arguments made here
        [
            FFMPEG,
            "-hide_banner",
            "-nostdin",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"testsrc=s={size[0]}x{size[1]}:r=30:d={seconds}",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=300:sample_rate=48000:d={seconds}",
            "-frames:v",
            str(frames),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(path),
        ],
        check=True,
        capture_output=True,
    )
    return Path(path)


class Crash(BaseException):
    """Stands for the process dying: nothing in the worker catches it."""


# Fixtures.


@pytest.fixture
def session(engine, request, monkeypatch):
    if request.node.get_closest_marker("compose") is None:
        # The keyframe director as before E16: its prompt without the background part.
        shipped = seed_module.read_prompt

        def older(role, model):
            text = shipped(role, model)
            if role is PromptRole.DIRECTOR_KEYFRAMES:
                text = text.split("=== background ===")[0]
            return text

        monkeypatch.setattr(seed_module, "read_prompt", older)
    with Session(engine) as session:
        register_defaults(session, ROOT / "workflows", activate=True)
        seed_prompts(session)
        session.commit()
        yield session


@pytest.fixture
def store(tmp_path):
    shared, private = tmp_path / "shared", tmp_path / "private"
    shared.mkdir()
    private.mkdir()
    return FileStore(shared, private)


@pytest.fixture
def fakes(tmp_path, store):
    comfy, gemma, director, adapters = (
        FakeComfy(),
        FakeGemma(),
        FakeDirector(),
        FakeAdapters(tmp_path),
    )
    deps = Deps(
        store=store,
        comfy=comfy,
        gemma=gemma,
        work_dir=tmp_path / "work",
        workflows_dir=ROOT / "workflows",
        adapter=adapters,
        text_runner=director,
    )
    return type(
        "Fakes",
        (),
        {"deps": deps, "comfy": comfy, "gemma": gemma, "director": director, "adapters": adapters},
    )


def worker(engine, fakes, store):
    return Worker(engine, build_registry(fakes.deps), store=store, config=FAST)


def photo_bytes(color):
    target = io.BytesIO()
    Image.new("RGB", (360, 640), color).save(target, "PNG")
    return target.getvalue()


def new_job(
    session, store, photos=1, length=JobLength.SHORT, idea="a cat", quality=JobQuality.HD720
):
    user = User(tailscale_login=uuid.uuid4().hex)
    session.add(user)
    session.flush()
    job = Job(
        owner_id=user.id,
        idea_text=idea,
        photo_count=photos,
        length=length,
        quality=quality,
        lane=JobLane.IDEA,
        status=S.IDEA_QUEUED,
    )
    session.add(job)
    session.flush()
    colors = ["red", "green", "blue"]
    uploads = [(io.BytesIO(photo_bytes(colors[i])), CropBox(0, 0, 1, 1)) for i in range(photos)]
    intake(session, store, owner_id=user.id, job_id=job.id, uploads=uploads)
    session.commit()
    return job


def status_of(session, job):
    session.expire_all()
    return session.get(Job, job.id).status


def steps_of(session, job):
    session.expire_all()
    return [
        (row.step, row.ordinal)
        for row in session.scalars(
            select(JobStep)
            .where(JobStep.job_id == job.id, JobStep.status == StepStatus.DONE)
            .order_by(JobStep.created_at)
        )
    ]


def output(session, job, step, ordinal=0):
    session.expire_all()
    return session.scalar(
        select(JobStep.output).where(
            JobStep.job_id == job.id,
            JobStep.step == step,
            JobStep.ordinal == ordinal,
            JobStep.status == StepStatus.DONE,
        )
    )


def assets(session, job, kind):
    session.expire_all()
    return session.scalars(
        select(Asset).where(Asset.job_id == job.id, Asset.kind == kind).order_by(Asset.ordinal)
    ).all()


def approve(session, job):
    session.execute(update(Idea).where(Idea.job_id == job.id).values(approved_at=func.now()))
    transition(session, session.get(Job, job.id), S.RENDER_QUEUED, expected=S.AWAITING_APPROVAL)
    session.commit()


def release(session, job, quality=None):
    row = session.get(Job, job.id)
    if quality is not None:
        row.quality = quality
    transition(session, row, S.RELEASE_QUEUED, expected=S.PREVIEW_READY)
    session.commit()


def plan_names(session, job, done=()):
    session.expire_all()
    job = session.get(Job, job.id)
    return [
        (p.step, p.ordinal, dict(p.inputs).get("kind"))
        for p in PipelinePlanner(None).plan(session, job, list(done))
    ]


def force_status(session, job, status, regime=None):
    lane = JobLane.IDEA if status in (S.IDEA_QUEUED, S.IDEA_RUNNING) else JobLane.RENDER
    values = {"status": status, "lane": lane}
    if regime is not None:
        values["regime"] = regime
    session.execute(update(Job).where(Job.id == job.id).values(**values))
    session.commit()


# Planning by phase.


def test_planning_by_phase(engine, session, store):
    job = new_job(session, store)
    assert plan_names(session, job) == [(StepName.IDEA, 0, None)]
    force_status(session, job, S.RENDER_QUEUED, JobRegime.MULTISHOT)
    assert plan_names(session, job) == [
        (StepName.SCENARIO, 0, None),
        (StepName.CLIP, 0, None),
        (StepName.CHECK, 0, "last_frame"),
    ]
    force_status(session, job, S.RENDERING, JobRegime.KEYFRAMES)
    assert plan_names(session, job) == [
        (StepName.SCENARIO, 0, None),
        (StepName.KEYFRAME, 0, None),
        (StepName.KEYFRAME, 1, None),
        (StepName.CHECK, 0, "keyframes"),
        (StepName.CLIP, 0, None),
    ]
    session.execute(update(Job).where(Job.id == job.id).values(length=JobLength.LONG))
    session.commit()
    names = plan_names(session, job)
    assert names[-3:] == [
        (StepName.CLIP, 0, None),
        (StepName.CLIP, 1, None),
        (StepName.JOIN, 0, None),
    ]
    force_status(session, job, S.RELEASE_QUEUED)
    assert plan_names(session, job) == [(StepName.RELEASE, 0, None)]


def test_critic_off_plans_no_check(engine, session, store):
    session.add(Setting(key="active_critic_engine", value=None))
    session.commit()
    job = new_job(session, store, photos=2)
    force_status(session, job, S.RENDER_QUEUED, JobRegime.KEYFRAMES)
    assert StepName.CHECK not in [step for step, *_ in plan_names(session, job)]


def test_the_planned_models_group_render_jobs(engine, session, store, fakes):
    job = new_job(session, store)
    force_status(session, job, S.RENDER_QUEUED, JobRegime.MULTISHOT)
    me = worker(engine, fakes, store)
    with Session(engine) as other:
        assert me.next_model(other, other.get(Job, job.id)) == "qwen3.8"


# The regimes.


def test_short_one_photo_without_idea_to_release(engine, session, store, fakes):
    job = new_job(session, store, idea=None)
    me = worker(engine, fakes, store)
    me.drain()
    assert status_of(session, job) is S.AWAITING_APPROVAL
    idea = session.scalar(select(Idea).where(Idea.job_id == job.id))
    assert idea.text == "a cat steals the show" and idea.approved_at is None
    assert len(assets(session, job, AssetKind.GEMMA_COPY)) == 1

    approve(session, job)
    me.drain()
    assert status_of(session, job) is S.PREVIEW_READY
    assert steps_of(session, job) == [
        (StepName.IDEA, 0),
        (StepName.SCENARIO, 0),
        (StepName.CLIP, 0),
        (StepName.CHECK, 0),
    ]
    row = session.get(Job, job.id)
    assert row.regime is JobRegime.MULTISHOT and row.recipe_id and row.critic_engine_id
    (request,) = fakes.adapters.clips
    assert request.frames == 241 and request.first_strength == 0.7 and request.guides == ()
    assert request.prompt.endswith("Single take, no cuts. No music.")
    assert request.folder == folder_of(job.id)
    (preview,) = assets(session, job, AssetKind.PREVIEW)
    assert preview.store is AssetStore.SHARED
    now = session.scalar(select(func.now()))
    assert 23.9 * 3600 < (preview.expires_at - now).total_seconds() <= 24 * 3600
    assert media.probe(store.path_for(preview)).frames == 241
    assert {a.store for a in assets(session, job, AssetKind.LATENT)} == {AssetStore.PRIVATE}
    assert output(session, job, StepName.CHECK)["kind"] == "last_frame"
    assert folder_of(job.id) in fakes.comfy.cleaned
    assert not (fakes.deps.work_dir / folder_of(job.id)).exists()

    release(session, job)
    me.drain()
    assert status_of(session, job) is S.RELEASED
    (final,) = assets(session, job, AssetKind.RELEASE)
    assert final.store is AssetStore.SHARED
    (released,) = fakes.adapters.releases
    assert released[0].prompt == request.prompt and released[0].seed == request.seed
    assert released[1].startswith(b"latent video")
    assert released[2] is JobQuality.HD720
    assert media.probe(store.path_for(final)).frames == 241


def test_short_two_photos_with_idea(engine, session, store, fakes):
    job = new_job(session, store, photos=2, idea="they row across a fountain")
    worker(engine, fakes, store).drain()
    assert status_of(session, job) is S.PREVIEW_READY
    idea = session.scalar(select(Idea).where(Idea.job_id == job.id))
    assert idea.text == "they row across a fountain" and idea.approved_at is not None
    assert session.get(Job, job.id).regime is JobRegime.KEYFRAMES
    start, end = fakes.adapters.keyframes  # E16: no middle keyframe in a short video
    assert len(start.photos) == 2 and start.extra_references == ()
    assert end.extra_references
    assert "The place: " in end.prompt and "The light: " in end.prompt
    keyframes = assets(session, job, AssetKind.KEYFRAME)
    (clip,) = fakes.adapters.clips
    assert clip.first_strength == 1.0 and clip.frames == 241
    assert [guide.frame for guide in clip.guides] == [-1]
    by_ordinal = {a.ordinal: store.path_for(a) for a in keyframes}
    assert clip.first_frame == by_ordinal[0]
    assert [g.image for g in clip.guides] == [by_ordinal[1]]
    assert output(session, job, StepName.KEYFRAME, 1)["colour_matched"] is True
    assert output(session, job, StepName.KEYFRAME, 0)["colour_matched"] is False
    assert by_ordinal[100]  # the last frame of run 0
    check = output(session, job, StepName.CHECK)
    assert check["kind"] == "keyframes" and check["faces_ok"] is True
    assert [entry["face"] for entry in check["keyframes"]] == [True, True]
    assert fakes.gemma.calls.count("critic") == 2


def test_asset_pictures_come_first_and_replace_the_photo(engine, session, store, fakes):
    # E16: the man's photo holds a log; he is drawn empty-handed and the log alone first.
    fakes.director.extra = (
        "PROPS: a giant oak log with rough bark\nCLEAN: <image1>\n"
        "PROP PICTURES: a giant oak log with rough bark from <image1>\n"
    )
    job = new_job(session, store, photos=2, idea="he swings a log")
    worker(engine, fakes, store).drain()
    assert status_of(session, job) is S.PREVIEW_READY
    clean, prop, start, end = fakes.adapters.keyframes
    photos = sorted(assets(session, job, AssetKind.PHOTO), key=lambda a: a.ordinal)
    first, second = (store.path_for(a) for a in photos)
    drawn = {a.ordinal: store.path_for(a) for a in assets(session, job, AssetKind.KEYFRAME)}
    assert clean.photos == (first,) and clean.prompt.startswith("Edit <image1>: exactly the same ")
    assert prop.photos == (first,) and prop.prompt.startswith(
        "Show only the object from <image1>: a giant"
    )
    assert start.photos == (drawn[200], second, drawn[201]) and start.extra_references == ()
    assert "Take a giant oak log with rough bark from <image3>." in start.prompt
    assert end.photos == (drawn[0], drawn[200], second) and end.extra_references == ()
    assert end.prompt.startswith("Edit <image1>: keep the place")
    (clip,) = fakes.adapters.clips
    assert clip.first_frame == drawn[0] and [g.image for g in clip.guides] == [drawn[1]]
    check = output(session, job, StepName.CHECK)
    assert len(check["keyframes"]) == 2  # the critic judges keyframes, not asset pictures
    assert output(session, job, StepName.KEYFRAME, 200)["at"] == "asset"


def test_three_people_draw_middle_and_end_from_the_photos(engine, session, store, fakes):
    job = new_job(session, store, photos=3, idea="three friends and a kite")
    worker(engine, fakes, store).drain()
    assert status_of(session, job) is S.PREVIEW_READY
    start, end = fakes.adapters.keyframes
    assert [len(k.photos) for k in (start, end)] == [3, 3]
    assert start.extra_references == end.extra_references == ()
    assert end.prompt.startswith("New place, replace every photo background completely")
    assert "same place, light, camera position and framing" in end.prompt.lower()


def test_long_one_photo_joins_two_runs(engine, session, store, fakes):
    job = new_job(session, store, length=JobLength.LONG)
    worker(engine, fakes, store).drain()
    assert status_of(session, job) is S.PREVIEW_READY
    first, second = fakes.adapters.clips
    assert [first.frames, second.frames] == [241, 241]
    clip0 = output(session, job, StepName.CLIP, 0)
    last = session.get(Asset, uuid.UUID(clip0["last_frame"]))
    assert second.first_frame == store.path_for(last) and second.first_strength == 1.0
    assert [g.frame for g in first.guides] == [120]  # run 1 leaves its end free
    assert [g.frame for g in second.guides] == [-1]  # the plan's end guide
    joined = output(session, job, StepName.JOIN)
    assert joined["frames"] == 481 and joined["run_frames"] == [241, 241]
    (preview,) = assets(session, job, AssetKind.PREVIEW)
    assert media.probe(store.path_for(preview)).frames == 481

    release(session, job)
    worker(engine, fakes, store).drain()
    (final,) = assets(session, job, AssetKind.RELEASE)
    assert media.probe(store.path_for(final)).frames == 481
    assert len(fakes.adapters.releases) == 2


# The critic.


def test_a_failed_face_is_only_a_warning(engine, session, store, fakes, caplog):
    fakes.gemma.verdicts = [
        verdict(False, "The man lost his beard."),  # start
        verdict(False, "The woman is someone else."),  # end (a short video has no middle)
    ]
    job = new_job(session, store, photos=2, idea="rowing")
    with caplog.at_level("WARNING", logger="memcraft.pipeline.steps"):
        worker(engine, fakes, store).drain()
    assert status_of(session, job) is S.PREVIEW_READY
    # Nothing is drawn again, and the clip uses the keyframes as drawn.
    assert len(fakes.adapters.keyframes) == 2
    assert fakes.gemma.calls.count("critic") == 2
    check = output(session, job, StepName.CHECK)
    start, end = check["keyframes"]
    assert (start["face"], start["face_why"]) == (False, "The man lost his beard.")
    assert end["face"] is False
    assert "retry" not in start and "story" not in start and "artefacts" not in start
    assert check["faces_ok"] is False
    assert start["asset_id"] == output(session, job, StepName.KEYFRAME)["asset_id"]
    (clip,) = fakes.adapters.clips
    kept = session.get(Asset, uuid.UUID(start["asset_id"]))
    assert clip.first_frame == store.path_for(kept)
    assert len([a for a in assets(session, job, AssetKind.KEYFRAME) if a.ordinal < 100]) == 2
    assert "The man lost his beard." in caplog.text


def test_a_critic_that_cannot_answer_never_blocks_the_job(engine, session, store, fakes):
    fakes.gemma.verdicts = [{"face_why": "unsure", "face": "maybe"}]
    job = new_job(session, store)
    worker(engine, fakes, store).drain()
    assert status_of(session, job) is S.PREVIEW_READY
    check = output(session, job, StepName.CHECK)
    assert check["kind"] == "last_frame" and check["face"] is None and check["faces_ok"] is None
    assert check["error"]


# Checkpoints.


def test_a_restart_repeats_only_the_unfinished_step(engine, session, store, fakes):
    job = new_job(session, store, length=JobLength.LONG)
    me = worker(engine, fakes, store)
    fakes.adapters.fail_clip = Crash
    with pytest.raises(Crash):
        me.drain()
    me.lease.release()  # a dead process loses its advisory lock with its connection
    done = steps_of(session, job)
    assert (StepName.SCENARIO, 0) in done and (StepName.CLIP, 0) not in done
    keyframes_before = len(fakes.adapters.keyframes)
    director_before = fakes.director.calls
    # The dead worker's claim runs out; another worker takes the job.
    session.execute(
        update(Job)
        .where(Job.id == job.id)
        .values(status=S.RENDER_QUEUED, claimed_by=None, claimed_at=None)
    )
    session.commit()
    worker(engine, fakes, store).drain()
    assert status_of(session, job) is S.PREVIEW_READY
    assert fakes.director.calls == director_before
    assert len(fakes.adapters.keyframes) == keyframes_before
    attempts = session.scalars(
        select(JobStep.attempt).where(JobStep.job_id == job.id, JobStep.step == StepName.CLIP)
    ).all()
    assert sorted(attempts) == [1, 1, 2]  # run 0 was tried twice, run 1 once
    assert len(assets(session, job, AssetKind.CLIP)) == 3  # two runs and the join


def test_a_failed_step_leaves_no_files(engine, session, store, fakes):
    job = new_job(session, store)
    fakes.adapters.fail_clip = RuntimeError
    worker(engine, fakes, store).drain()
    worker(engine, fakes, store).drain()
    assert status_of(session, job) is S.PREVIEW_READY
    assert len(assets(session, job, AssetKind.CLIP)) == 1
    files = [p for p in store._root(AssetStore.PRIVATE).rglob("*.mp4")]
    assert len(files) == 1


def test_release_uses_the_pinned_recipe(engine, session, store, fakes):
    job = new_job(session, store)
    worker(engine, fakes, store).drain()
    pinned = session.get(Job, job.id).recipe_id
    # A new recipe version activated later does not change the job.
    register = register_defaults(session, ROOT / "workflows", activate=True)
    session.commit()
    assert register.recipes["multishot"].id == pinned
    release(session, job, JobQuality.FHD1080)
    worker(engine, fakes, store).drain()
    assert status_of(session, job) is S.RELEASED
    assert fakes.adapters.releases[-1][2] is JobQuality.FHD1080
    assert output(session, job, StepName.RELEASE)["quality"] == "fhd1080"


@pytest.mark.compose
def test_compose_draws_the_pieces_and_never_reuses_the_start_keyframe(
    engine, session, store, fakes
):
    # E16, Denys's layout: background, both people cut out, the log cut out, then keyframes.
    fakes.director.extra = (
        "PROPS: a giant oak log with rough bark\n"
        "PROP PICTURES: a giant oak log with rough bark from <image1>\n"
    )
    job = new_job(session, store, photos=2, idea="he swings a log")
    worker(engine, fakes, store).drain()
    assert status_of(session, job) is S.PREVIEW_READY
    background, first, second, log, start, end = fakes.adapters.keyframes
    assert background.prompt.startswith("The plain grey <image1>") and not background.cutout
    assert "people" in background.negative_prompt and "people" not in start.negative_prompt
    # The photo holding the log is redrawn empty-handed; the other one is cut out as it is.
    assert first.cutout and not first.cut_only and log.cutout
    assert second.cut_only and not second.cutout
    drawn = {a.ordinal: store.path_for(a) for a in assets(session, job, AssetKind.KEYFRAME)}
    pieces = (drawn[200], drawn[201], drawn[202], drawn[203])
    assert start.photos == pieces and end.photos == pieces
    assert start.extra_references == end.extra_references == ()
    assert end.prompt.startswith("Put the people and objects into the empty place of <image1>")
    stored = {a.ordinal: a.store for a in assets(session, job, AssetKind.KEYFRAME)}
    assert stored[200] is AssetStore.SHARED and stored[0] is AssetStore.SHARED
    (clip,) = fakes.adapters.clips
    assert clip.first_frame == drawn[0] and [g.image for g in clip.guides] == [drawn[1]]
