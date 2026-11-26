"""Regime 3 (admin only, 2026-11-26): the upload rule, the plan, the director's pictures and
the streaming Flash-Next runner."""

import contextlib
import json
import uuid

import pytest
from sqlalchemy.orm import Session
from test_upload import ALICE, FULL, as_user, picture, post
from test_upload import client as client
from test_upload import store as store

from memcraft.models import Job, JobRegime, User, UserRole, UserStatus
from memcraft.roles import director
from memcraft.roles.llama_text import LlamaTextError, ThinkingTextRunner, answer_text
from memcraft.web.upload import REGIME3_ADMINS_ONLY, REGIME3_NEEDS_IDEA, REGIME_NOTES

ADMIN = "admin@example.com"


def add(engine, login, role=UserRole.USER):
    with Session(engine) as session:
        session.add(User(tailscale_login=login, status=UserStatus.APPROVED, role=role))
        session.commit()


@pytest.mark.parametrize("count", [1, 2, 3])
def test_an_admin_makes_a_regime3_job(client, engine, count):
    add(engine, ADMIN, UserRole.ADMIN)
    answer = post(
        client, [picture()] * count, [FULL] * count, login=ADMIN, idea="a dance", regime="regime3"
    )
    assert answer.status_code == 201, answer.text
    with Session(engine) as session:
        job = session.get(Job, uuid.UUID(answer.json()["id"]))
        assert job.regime is JobRegime.REGIME3 and job.photo_count == count


def test_regime3_needs_an_idea(client, engine):
    add(engine, ADMIN, UserRole.ADMIN)
    for idea in (None, "   "):
        answer = post(client, [picture()], [FULL], login=ADMIN, idea=idea, regime="regime3")
        assert answer.status_code == 422 and answer.json()["detail"] == REGIME3_NEEDS_IDEA


def test_regime3_takes_at_most_three_photos(client, engine):
    add(engine, ADMIN, UserRole.ADMIN)
    answer = post(client, [picture()] * 4, [FULL] * 4, login=ADMIN, idea="x", regime="regime3")
    assert answer.status_code == 422


def test_only_admins_get_regime3(client, engine):
    add(engine, ALICE)
    answer = post(client, [picture()], [FULL], idea="a dance", regime="regime3")
    assert answer.status_code == 403 and answer.json()["detail"] == REGIME3_ADMINS_ONLY
    note = client.get(
        "/api/regime-note?photos=1&length=short&regime=regime3", headers=as_user(ALICE)
    )
    assert note.status_code == 403


def test_other_regimes_cannot_be_asked_for(client, engine):
    add(engine, ADMIN, UserRole.ADMIN)
    answer = post(client, [picture()], [FULL], login=ADMIN, idea="x", regime="keyframes")
    assert answer.status_code == 422


def test_the_regime3_note(client, engine):
    add(engine, ADMIN, UserRole.ADMIN)
    note = client.get(
        "/api/regime-note?photos=1&length=short&regime=regime3", headers=as_user(ADMIN)
    )
    assert note.json() == {"regime": "regime3", "note": REGIME_NOTES[JobRegime.REGIME3]}


def test_regime3_uses_keyframes_recipes():
    assert JobRegime.REGIME3.recipe_regime is JobRegime.KEYFRAMES
    assert JobRegime.MULTISHOT.recipe_regime is JobRegime.MULTISHOT


BIBLE = {
    "people": [{"tag": "the man", "image": 1}, {"tag": "the troll", "image": 2}],
    "props": ["a giant log"],
}


def test_cut_people_cuts_everyone_and_draws_no_props():
    pictures, notes = director.asset_pictures(
        "",
        "a giant log from <image2>",
        BIBLE,
        2,
        7,
        background="the empty place",
        faces={1: [0.1, 0.1, 0.2, 0.2]},
        cut_people=True,
    )
    assert [(p.kind, p.photo) for p in pictures] == [
        ("background", None),
        ("cut", 1),
        ("cut", 2),
        ("face", 1),
    ]
    assert any("regime 3 draws no prop pictures" in note for note in notes)


def test_without_cut_people_the_holder_is_redrawn():
    pictures, _ = director.asset_pictures(
        "", "a giant log from <image2>", BIBLE, 2, 7, background="the empty place"
    )
    assert [p.kind for p in pictures] == ["background", "cut", "clean", "prop"]


def sse(*chunks):
    return [f"data: {json.dumps(chunk)}" for chunk in chunks] + ["data: [DONE]"]


class FakeStream:
    def __init__(self, lines, status=200):
        self.lines, self.status = lines, status
        self.calls = []

    @contextlib.contextmanager
    def __call__(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        response = type(
            "Response",
            (),
            {
                "status_code": self.status,
                "text": "boom",
                "read": lambda self: b"",
                "iter_lines": lambda _self: iter(self.lines),
            },
        )()
        yield response


def delta(content=None, reasoning=None, finish=None):
    fields = {k: v for k, v in (("content", content), ("reasoning_content", reasoning)) if v}
    return {"choices": [{"delta": fields, "finish_reason": finish}]}


def test_the_thinking_runner_streams_only_the_answer():
    stream = FakeStream(
        sse(
            delta(reasoning="Loading model..."),
            delta(reasoning="hmm"),
            delta("TITLE: "),
            delta("A"),
            {"choices": [], "timings": {"predicted_n": 3}},
        )
    )
    pieces, before = [], []
    runner = ThinkingTextRunner(
        "http://127.0.0.1:8080",
        model="qwen-flash-next-strata",
        api_key="k",
        before_call=lambda: before.append(1),
        on_piece=pieces.append,
        stream=stream,
    )
    assert runner("sys", "user", [b"\x89PNGxx"], 5) == "TITLE: A"
    assert before == [1] and pieces == [1, 2, 3, 4, 5]
    assert runner.last_thinking == "Loading model...hmm"
    assert runner.last_timings == {"predicted_n": 3}
    _method, url, kwargs = stream.calls[0]
    assert url == "http://127.0.0.1:8080/v1/chat/completions"
    body = kwargs["json"]
    assert body["model"] == "qwen-flash-next-strata" and body["stream"] is True
    assert body["chat_template_kwargs"] == {"enable_thinking": True}
    assert body["max_tokens"] == 16384 and body["temperature"] == 0.6
    assert kwargs["headers"] == {"Authorization": "Bearer k"}
    assert body["messages"][1]["content"][0]["image_url"]["url"].startswith("data:image/png")


def test_the_thinking_runner_says_when_thinking_used_every_token():
    stream = FakeStream(sse(delta(reasoning="..."), delta(finish="length")))
    runner = ThinkingTextRunner("http://x:1", model="m", stream=stream)
    with pytest.raises(LlamaTextError, match="ran out of tokens while thinking"):
        runner("s", "u", [], 1)


def test_the_thinking_runner_reports_a_bad_answer():
    runner = ThinkingTextRunner("http://x:1", model="m", stream=FakeStream([], status=503))
    with pytest.raises(LlamaTextError, match="503"):
        runner("s", "u", [], 1)


def test_answer_text_drops_thinking_left_in_the_content():
    assert answer_text("<think>a\nb</think>\nTITLE: x") == "TITLE: x"
    assert answer_text("plan...</think>TITLE: y") == "TITLE: y"
