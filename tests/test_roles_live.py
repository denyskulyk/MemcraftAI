"""Live runs of the roles against real models. Skipped unless MEMCRAFT_LIVE=1.

Gemma: a llama-server on MEMCRAFT_GEMMA_URL (default http://127.0.0.1:8098) must be running.
Qwen: also MEMCRAFT_LIVE_QWEN=1; needs the app's ComfyUI on 8199 and takes the GPU lock.
Photos come from e0-spike/photos-ai (AI stand-ins, not real people). With MEMCRAFT_LIVE_OUT
set to a file, the answers and times are written there as JSON.
"""

import io
import json
import os
import time
from pathlib import Path

import pytest
from PIL import Image

from memcraft.models import JobLength, PromptRole, PromptVersion
from memcraft.roles.chat import run_chat
from memcraft.roles.critic import check_with_gemma
from memcraft.roles.director import direct_keyframes, direct_multishot
from memcraft.roles.gemma import GemmaClient
from memcraft.roles.seed import GEMMA, QWEN, read_prompt

pytestmark = pytest.mark.skipif(
    os.environ.get("MEMCRAFT_LIVE") != "1", reason="live models: set MEMCRAFT_LIVE=1"
)

ROOT = Path(__file__).parents[1]
PHOTOS = ROOT / "e0-spike" / "photos-ai"
LOCK = ROOT / "data" / "gpu.lock"
CASES = [
    ("p01", "He proudly opens an umbrella and it pulls him up into the air"),
    ("p02", "The toaster launches the toast so hard that it sticks to the ceiling"),
    ("p05", "A pigeon steals a sandwich and they chase it in dramatic slow motion"),
]
RESULTS: list[dict] = []


def small(name, size):
    with Image.open(PHOTOS / f"{name}.png") as image:
        image = image.convert("RGB")
        image.thumbnail(size, Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, "JPEG", quality=90)
    return buffer.getvalue()


def stored(role, model):
    return PromptVersion(role=role, model=model, text=read_prompt(role, model))


def save(row):
    RESULTS.append(row)
    out = os.environ.get("MEMCRAFT_LIVE_OUT")
    if out:
        Path(out).write_text(json.dumps(RESULTS, indent=1, ensure_ascii=False), encoding="utf-8")


@pytest.fixture(scope="module")
def gemma():
    url = os.environ.get("MEMCRAFT_GEMMA_URL", "http://127.0.0.1:8098")
    return GemmaClient(url, timeout=300)


@pytest.mark.parametrize(("photo", "idea"), CASES)
@pytest.mark.parametrize("with_idea", [True, False])
def test_live_chat(gemma, photo, idea, with_idea):
    result = run_chat(
        gemma,
        stored(PromptRole.CHAT, GEMMA),
        [small(photo, (360, 640))],
        idea if with_idea else None,
    )
    save(
        {"role": "chat", "photo": photo, "idea_in": idea if with_idea else ""} | result.as_output()
    )
    assert 1 <= result.score <= 10 and result.reason
    if with_idea:
        assert result.idea == idea


def test_live_chat_three_photos(gemma):
    photos = [small(name, (540, 960)) for name, _ in CASES]
    result = run_chat(gemma, stored(PromptRole.CHAT, GEMMA), photos, None)
    save({"role": "chat", "photo": "p01+p02+p05", "idea_in": ""} | result.as_output())
    assert 1 <= result.score <= 10


def test_live_gemma_critic_on_the_photo_itself(gemma):
    """The photo as its own frame: the face must pass."""
    photo = small("p01", (540, 960))
    verdict = check_with_gemma(
        gemma,
        stored(PromptRole.CRITIC, GEMMA),
        [photo],
        photo,
        "A man in a red rain jacket stands in a park and looks at the camera.",
    )
    save({"role": "critic", "photo": "p01"} | verdict.as_output())
    assert verdict.face


@pytest.fixture(scope="module")
def qwen():
    if os.environ.get("MEMCRAFT_LIVE_QWEN") != "1":
        pytest.skip("Qwen: set MEMCRAFT_LIVE_QWEN=1")
    from memcraft.comfy import ComfyClient
    from memcraft.roles.qwen import ComfyTextRunner

    while True:
        try:
            LOCK.parent.mkdir(exist_ok=True)
            with open(LOCK, "x") as lock:
                lock.write(f"roles live test {os.getpid()}\n")
            break
        except FileExistsError:
            time.sleep(30)
    comfy = ComfyClient()
    try:
        yield ComfyTextRunner(comfy)
    finally:
        try:
            comfy.free()
        finally:
            comfy.close()
            LOCK.unlink(missing_ok=True)


def test_live_multishot(qwen):
    plan = direct_multishot(
        qwen, stored(PromptRole.DIRECTOR_MULTISHOT, QWEN), small("p02", (768, 1344)), CASES[1][1]
    )
    save({"role": "director_multishot"} | plan.as_output())
    assert plan.runs[0].prompt


def test_live_keyframes(qwen):
    photos = [small("p01", (768, 1344)), small("p02", (768, 1344))]
    plan = direct_keyframes(
        qwen,
        stored(PromptRole.DIRECTOR_KEYFRAMES, QWEN),
        photos,
        "They sit in a tiny rowing boat in a fountain, very seriously, as if they cross the ocean",
        length=JobLength.LONG,
    )
    save({"role": "director_keyframes"} | plan.as_output())
    assert len(plan.keyframes) == 3
