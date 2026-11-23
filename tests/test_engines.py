import json
import uuid
from pathlib import Path

import httpx
import pytest
from sqlalchemy.orm import Session

from memcraft.comfy import ComfyClient
from memcraft.engines import (
    LAST,
    RULES,
    ClipRequest,
    First,
    Guide,
    KeyframeRequest,
    LimitError,
    LtxClips,
    QwenImageKeyframes,
    adapter_for,
    frames_for,
    is_8n_plus_1,
    matching_prompt,
    pinned_adapter,
    plan,
    register_defaults,
)
from memcraft.engines.base import EngineError
from memcraft.models import (
    EngineSlot,
    Job,
    JobLength,
    JobQuality,
    JobRegime,
    User,
    VersionStatus,
)
from memcraft.versions import pin_job
from memcraft.workflows import is_link, load_workflow

WORKFLOWS = Path(__file__).parents[1] / "workflows"


class FakeComfy:
    """Runs nothing: every requested output node writes one file at once."""

    def __init__(self):
        self.prompts = []
        self.uploads = {}

    def handler(self, request):
        path = request.url.path
        if path == "/upload/image":
            name = request.content.split(b'filename="')[1].split(b'"')[0].decode()
            self.uploads[name] = request.content
            return httpx.Response(200, json={"name": name, "subfolder": "", "type": "input"})
        if path == "/prompt":
            body = json.loads(request.content)
            self.prompts.append(body)
            return httpx.Response(200, json={"prompt_id": body["prompt_id"]})
        if path.startswith("/history/"):
            pid = path.removeprefix("/history/")
            body = next(b for b in self.prompts if b["prompt_id"] == pid)
            outputs = {
                node: {"files": [{"filename": f"{node}.bin", "subfolder": "s", "type": "output"}]}
                for node in body["partial_execution_targets"]
            }
            status = {"status_str": "success", "completed": True, "messages": []}
            return httpx.Response(200, json={pid: {"outputs": outputs, "status": status}})
        if path == "/view":
            return httpx.Response(200, content=request.url.params["filename"].encode())
        return httpx.Response(404)

    @property
    def graph(self):
        return self.prompts[-1]["prompt"]

    @property
    def targets(self):
        return self.prompts[-1]["partial_execution_targets"]


@pytest.fixture
def fake():
    return FakeComfy()


@pytest.fixture
def client(fake):
    with ComfyClient(transport=httpx.MockTransport(fake.handler), sleep=lambda _: None) as c:
        yield c


@pytest.fixture
def pictures(tmp_path):
    paths = []
    for i in range(5):
        path = tmp_path / f"p{i}.png"
        path.write_bytes(f"picture {i}".encode())
        paths.append(path)
    return paths


def adapter(kind):
    package = load_workflow(
        WORKFLOWS
        / {
            QwenImageKeyframes: "keyframe-qwen-image-2.1-v1",
            LtxClips: "clip-ltx-2.5-v1",
        }[kind]
    )
    return kind(package.api_json, package.manifest)


def ancestors(graph, targets):
    seen, todo = set(), list(targets)
    while todo:
        node = todo.pop()
        if node in seen:
            continue
        seen.add(node)
        todo += [v[0] for v in graph[node]["inputs"].values() if is_link(v)]
    return seen


def classes(graph, nodes):
    return {graph[n]["class_type"] for n in nodes}


# Keyframes


def test_keyframe_binds_references_in_order(client, fake, pictures, tmp_path):
    qwen = adapter(QwenImageKeyframes)
    request = KeyframeRequest(
        prompt="they dance",
        photos=tuple(pictures[:2]),
        seed=5,
        extra_references=(pictures[2],),
        folder="job1",
    )
    result = qwen.render(client, request, tmp_path / "out" / "kf.png")
    assert result.image.read_bytes().endswith(b".bin")
    assert (result.width, result.height) == (768, 1344)
    graph, manifest = fake.graph, qwen.manifest
    names = [
        graph[manifest.inputs[f"image_{i}"].targets[0].node]["inputs"]["image"] for i in range(1, 5)
    ]
    assert all(name.startswith("job1/") for name in names)
    assert names[3] == names[0]  # a fourth reference was not given
    assert len(set(names[:3])) == 3
    toggles = [
        graph[manifest.inputs[f"image_{i}"].toggle.node]["inputs"]["value"] for i in range(2, 5)
    ]
    assert toggles == [True, True, False]
    assert len(fake.uploads) == 3
    assert fake.targets == [manifest.outputs["image"].node]
    sampler = next(n for n in graph.values() if n["class_type"] == "KSampler")["inputs"]
    assert (sampler["seed"], sampler["steps"], sampler["cfg"]) == (5, 16, 3.0)


def test_keyframe_upload_names_come_from_the_content(client, fake, pictures, tmp_path):
    qwen = adapter(QwenImageKeyframes)
    request = KeyframeRequest(prompt="x", photos=(pictures[0],), seed=1)
    qwen.render(client, request, tmp_path / "a.png")
    qwen.render(client, request, tmp_path / "b.png")
    assert len(fake.uploads) == 1


@pytest.mark.parametrize(
    "change, message",
    [
        ({"photos": ()}, "at least one photo"),
        ({"extra_references": ("a", "b", "c", "d")}, "at most 4"),
        ({"width": 770}, "divide by 32"),
        ({"prompt": " "}, "needs a prompt"),
    ],
)
def test_keyframe_limits(change, message, pictures):
    fields = {"prompt": "x", "photos": (pictures[0],), "seed": 1} | change
    with pytest.raises(LimitError, match=message):
        adapter(QwenImageKeyframes).check(KeyframeRequest(**fields))


def test_matching_prompt():
    start = matching_prompt(
        "They wave.", place="A beach.", light="Noon sun.", photos=2, start_reference=False
    )
    assert start.startswith("New place, replace every photo background completely: A beach.")
    assert "<image1>, <image2>" in start and start.endswith("They wave.")
    later = matching_prompt(
        "They jump.", place="A beach.", light="", photos=2, start_reference=True
    )
    assert later.startswith(
        "Same scene as <image3>: same place, light, camera position and framing"
    )
    with pytest.raises(LimitError):
        matching_prompt("x", place="", light="", photos=4, start_reference=False)


# Clips


def test_clip_stage_one(client, fake, pictures, tmp_path):
    ltx = adapter(LtxClips)
    request = ClipRequest(
        prompt="a dog drives",
        first_frame=pictures[0],
        frames=97,
        seed=9,
        guides=(Guide(pictures[1], 48), Guide(pictures[2], LAST)),
        folder="job2",
    )
    result = ltx.render(client, request, tmp_path / "clip")
    assert {
        p.name for p in (result.video, result.last_frame, result.latent_video, result.latent_audio)
    } == {"preview.mp4", "last.png", "latent_v.latent", "latent_a.latent"}
    graph, manifest = fake.graph, ltx.manifest
    out = manifest.outputs
    assert set(fake.targets) == {
        out[n].node for n in ("preview", "last_frame", "latent_video", "latent_audio")
    }
    value = lambda name: graph[manifest.inputs[name].targets[0].node]["inputs"][  # noqa: E731
        manifest.inputs[name].targets[0].input
    ]
    assert value("frames") == 97 and value("fps") == 30.0
    assert (value("width"), value("height")) == (544, 960)
    assert (value("reference_width"), value("reference_height")) == (1088, 1920)
    assert value("first_strength") == 1.0
    assert (value("guide_1_frame"), value("guide_2_frame")) == (48, -1)
    for i in (1, 2):
        assert graph[manifest.inputs[f"guide_{i}"].toggle.node]["inputs"]["value"] is True
    # Stage 1 never reads a saved latent.
    assert "LoadLatent" not in classes(graph, ancestors(graph, fake.targets))


def test_clip_without_guides_switches_them_off(client, fake, pictures, tmp_path):
    ltx = adapter(LtxClips)
    request = ClipRequest(prompt="x", first_frame=pictures[0], frames=241, seed=1)
    ltx.render(client, request, tmp_path)
    for i in (1, 2):
        toggle = ltx.manifest.inputs[f"guide_{i}"].toggle
        assert fake.graph[toggle.node]["inputs"]["value"] is False


def test_hd_release_reads_only_the_latents(client, fake, pictures, tmp_path):
    ltx = adapter(LtxClips)
    request = ClipRequest(prompt="x", first_frame=pictures[0], frames=97, seed=1, folder="j")
    latent_v, latent_a = pictures[3], pictures[4]
    result = ltx.release(client, request, latent_v, latent_a, JobQuality.HD720, tmp_path / "hd.mp4")
    assert (result.width, result.height) == (720, 1280)
    graph = fake.graph
    assert fake.targets == [ltx.manifest.outputs["hd"].node]
    used = ancestors(graph, fake.targets)
    assert "LoadImage" not in classes(graph, used)
    loads = sorted(
        graph[n]["inputs"]["latent"] for n in used if graph[n]["class_type"] == "LoadLatent"
    )
    assert len(loads) == 2 and all(name.startswith("j/") for name in loads)
    rtx = next(graph[n] for n in used if graph[n]["class_type"] == "RTXVideoSuperResolution")
    assert rtx["inputs"]["resize_type.scale"] == pytest.approx(1280 / 960)
    assert "SamplerCustomAdvanced" not in classes(graph, used)


def test_fhd_release_refines_with_the_same_frames(client, fake, pictures, tmp_path):
    ltx = adapter(LtxClips)
    request = ClipRequest(
        prompt="x", first_frame=pictures[0], frames=97, seed=1, guides=(Guide(pictures[1]),)
    )
    result = ltx.release(
        client,
        request,
        pictures[3],
        pictures[4],
        JobQuality.FHD1080,
        tmp_path / "fhd.mp4",
        refine_seed=7,
    )
    assert (result.width, result.height) == (1080, 1920)
    graph = fake.graph
    used = ancestors(graph, fake.targets)
    assert {"LoadLatent", "LoadImage", "LTXVLatentUpsampler"} <= classes(graph, used)
    assert "EmptyLTXVLatentVideo" not in classes(graph, used)
    seeds = [
        graph[n]["inputs"]["noise_seed"] for n in used if graph[n]["class_type"] == "RandomNoise"
    ]
    assert seeds == [7]


@pytest.mark.parametrize(
    "change, message",
    [
        ({"frames": 96}, "8n\\+1"),
        ({"frames": 305}, "at most 297"),
        ({"fps": 25}, "30 fps"),
        ({"width": 540}, "divide by 32"),
        ({"guides": (Guide(Path("a")),) * 3}, "at most 2 guides"),
        ({"guides": (Guide(Path("a"), 97),)}, "outside the clip"),
        ({"guides": (Guide(Path("a"), 0),)}, "outside the clip"),
    ],
)
def test_clip_limits(change, message, pictures):
    fields = {"prompt": "x", "first_frame": pictures[0], "frames": 97, "seed": 1} | change
    with pytest.raises(LimitError, match=message):
        adapter(LtxClips).check(ClipRequest(**fields))


def test_limits_as_json():
    limits = LtxClips.limits.as_json()
    assert limits["frame_rule"] == "8n+1" and limits["max_frames"] == 297
    assert limits["side_divisor"] == 32 and limits["sound"] is True
    assert QwenImageKeyframes.limits.as_json()["references"] == 4


def test_frames():
    assert [is_8n_plus_1(n) for n in (1, 9, 96, 97, 297)] == [False, True, False, True, True]
    assert frames_for(8, 30, 297) == 241
    assert frames_for(60, 30, 297) == 297
    assert frames_for(0.1, 30, 297) == 9


def test_an_adapter_refuses_another_workflow():
    package = load_workflow(WORKFLOWS / "clip-ltx-2.5-v1")
    with pytest.raises(EngineError):
        QwenImageKeyframes(package.api_json, package.manifest)


# Recipes


def test_multishot_plan():
    short = plan(RULES["multishot"], JobLength.SHORT)
    assert [(r.first, r.first_strength, r.shots, r.max_shots) for r in short.runs] == [
        (First.PHOTO, 0.7, 1, 1)
    ]
    long = plan(RULES["multishot"], JobLength.LONG)
    assert [r.first for r in long.runs] == [First.PHOTO, First.PREVIOUS_LAST_FRAME]
    assert long.runs[1].first_strength == 1.0
    assert long.frames == 241 * 2 - 1


def test_keyframes_plan():
    short = plan(RULES["keyframes"], JobLength.SHORT, frames=97)
    assert short.keyframes == ("start", "middle", "end")
    (run,) = short.runs
    assert (run.first, run.first_keyframe, run.guides) == (First.KEYFRAME, 0, ((1, 48), (2, -1)))
    long = plan(RULES["keyframes"], JobLength.LONG)
    first, second = long.runs
    # Run 1 keeps its end free; run 2 continues from its real last frame.
    assert first.guides == ((1, 120),)
    assert (second.first, second.guides) == (First.PREVIOUS_LAST_FRAME, ((2, -1),))
    assert all(r.first_strength == 1.0 and r.max_shots == 1 for r in long.runs)


@pytest.fixture
def session(engine):
    with Session(engine) as session:
        yield session


def make_job(session, regime):
    user = User(tailscale_login=uuid.uuid4().hex)
    session.add(user)
    session.flush()
    row = Job(
        owner_id=user.id,
        photo_count=1,
        length=JobLength.SHORT,
        quality=JobQuality.HD720,
        regime=regime,
    )
    session.add(row)
    session.flush()
    return row


def test_register_defaults_and_pinned_adapters(session):
    registered = register_defaults(session, WORKFLOWS, activate=True)
    again = register_defaults(session, WORKFLOWS, activate=True)
    assert again.clip_engine.id == registered.clip_engine.id
    assert again.recipes["keyframes"].id == registered.recipes["keyframes"].id
    assert registered.clip_engine.limits["max_frames"] == 297
    for recipe in registered.recipes.values():
        assert recipe.status is VersionStatus.ACTIVE
    job = make_job(session, JobRegime.KEYFRAMES)
    pin_job(session, job, recipe=registered.recipes["keyframes"])
    assert isinstance(pinned_adapter(session, job, EngineSlot.CLIP), LtxClips)
    assert isinstance(pinned_adapter(session, job, EngineSlot.KEYFRAME), QwenImageKeyframes)
    multishot = make_job(session, JobRegime.MULTISHOT)
    pin_job(session, multishot, recipe=registered.recipes["multishot"])
    assert registered.recipes["multishot"].keyframe_engine_id is None
    workflow = session.get(type(registered.clip_engine), registered.clip_engine.id)
    assert workflow is not None


def test_adapter_for_checks_the_stored_content(session):
    registered = register_defaults(session, WORKFLOWS)
    from memcraft.models import WorkflowVersion

    stored = session.get(WorkflowVersion, registered.clip_engine.workflow_version_id)
    assert isinstance(adapter_for(stored), LtxClips)
    stored.api_json = {**stored.api_json, "1": {"class_type": "X", "inputs": {}}}
    with pytest.raises(Exception, match="sha256"):
        adapter_for(stored)


def test_plan_with_one_length_per_run():
    long = plan(RULES["keyframes"], JobLength.LONG, frames=[161, 153])
    assert [r.frames for r in long.runs] == [161, 153]
    assert long.runs[0].guides == ((1, 80),)
    assert long.frames == 313
    with pytest.raises(EngineError):
        plan(RULES["multishot"], JobLength.SHORT, frames=[97, 97])
