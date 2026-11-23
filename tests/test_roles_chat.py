import uuid

import pytest
from sqlalchemy.orm import Session
from test_roles_gemma import JPEG, FakeServer, reply

from memcraft.models import (
    IdeaSource,
    Job,
    JobLength,
    JobQuality,
    PromptRole,
    PromptVersion,
    User,
)
from memcraft.roles.chat import build, record_idea, run_chat, schema
from memcraft.roles.gemma import GemmaClient, GemmaError
from memcraft.roles.seed import read_prompt, seed_prompts


def prompt():
    return PromptVersion(
        role=PromptRole.CHAT, model="gemma", text=read_prompt(PromptRole.CHAT, "gemma")
    )


def client(*answers):
    server = FakeServer(*answers)
    return GemmaClient("http://127.0.0.1:8098", post=server), server


def test_with_an_idea_there_is_no_idea_field():
    assert list(schema(True)["properties"]) == ["understood", "weakness", "score"]
    assert list(schema(False)["properties"]) == ["idea", "weakness", "score"]


def test_the_prompt_asks_for_what_the_mode_needs():
    system, user = build(prompt(), "He juggles teacups", JobLength.SHORT)
    assert "understood:" in system and "propose your own" not in system
    assert "5 to 10 seconds" in system and "weakness" in system and '"He waves" = 1' in system
    assert user == "The user's idea: He juggles teacups"
    system, user = build(prompt(), "  ", JobLength.LONG)
    assert "propose your own" in system and "10 to 20 seconds" in system
    assert user == "No idea given."


def test_a_user_idea_is_kept_word_for_word():
    gemma, server = client(
        reply({"understood": "He juggles\nteacups.", "weakness": "Seen it.", "score": 4})
    )
    result = run_chat(gemma, prompt(), [JPEG], " He juggles teacups ")
    assert result.idea == "He juggles teacups"
    assert result.understood == "He juggles teacups."
    assert (result.score, result.reason, result.proposed_by) == (4, "Seen it.", IdeaSource.USER)
    assert result.prompt_tokens == 123
    assert server.calls[0][1]["max_tokens"] == 300


def test_without_an_idea_gemma_proposes_one():
    gemma, _ = client(
        reply({"idea": "The toaster files a complaint.", "weakness": "Odd.", "score": 6})
    )
    result = run_chat(gemma, prompt(), [JPEG, JPEG], None)
    assert result.idea == result.understood == "The toaster files a complaint."
    assert result.proposed_by is IdeaSource.GEMMA
    assert result.as_output()["proposed_by"] == "gemma"


@pytest.mark.parametrize(
    "data",
    [
        {"understood": "x", "weakness": "y", "score": 11},
        {"understood": "x", "weakness": "y", "score": "5"},
        {"understood": "", "weakness": "y", "score": 5},
        {"idea": "x", "weakness": "y", "score": 5},
    ],
)
def test_answers_outside_the_schema_are_refused(data):
    gemma, _ = client(reply(data))
    with pytest.raises(GemmaError):
        run_chat(gemma, prompt(), [JPEG], "an idea")


def test_photo_count():
    gemma, _ = client()
    with pytest.raises(ValueError):
        run_chat(gemma, prompt(), [], "idea")
    with pytest.raises(ValueError):
        run_chat(gemma, prompt(), [JPEG] * 4, "idea")


@pytest.fixture
def session(engine):
    with Session(engine) as session:
        yield session


def test_the_idea_records_its_prompt(session):
    stored = {row.role: row for row in seed_prompts(session)}[PromptRole.CHAT]
    user = User(tailscale_login=uuid.uuid4().hex)
    session.add(user)
    session.flush()
    job = Job(owner_id=user.id, photo_count=1, length=JobLength.SHORT, quality=JobQuality.HD720)
    session.add(job)
    session.flush()
    gemma, _ = client(reply({"idea": "A pigeon audits the bench.", "weakness": "w", "score": 5}))
    result = run_chat(gemma, stored, [JPEG])
    idea = record_idea(session, job.id, result, stored)
    assert idea.prompt_version_id == stored.id
    assert (idea.text, idea.score, idea.proposed_by) == ("A pigeon audits the bench.", 5, "gemma")


def test_long_text_is_cut_after_a_sentence_or_word():
    from memcraft.roles.chat import one_line

    assert one_line("Short one.", 50) == "Short one."
    text = "First sentence is here. Second one runs on and on without end"
    assert one_line(text, 40) == "First sentence is here."
    assert one_line("word " * 20, 22) == "word word word word…"
