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
from memcraft.roles.chat import build, record_idea, run_chat, schema, user_idea
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
    assert list(schema(True)["properties"]) == ["understood", "weakness"]
    assert list(schema(False)["properties"]) == ["idea", "weakness"]


def test_the_prompt_asks_for_what_the_mode_needs():
    system, user = build(prompt(), "He juggles teacups", JobLength.SHORT)
    assert "understood:" in system and "propose your own" not in system
    assert "5 to 10 seconds" in system and "weakness" in system and "score" not in system
    assert user == "The user's idea: He juggles teacups"
    system, user = build(prompt(), "  ", JobLength.LONG)
    assert "propose your own" in system and "10 to 20 seconds" in system
    assert user == "No idea given."


def test_a_user_idea_is_kept_word_for_word():
    gemma, server = client(reply({"understood": "He juggles\nteacups.", "weakness": "Seen it."}))
    result = run_chat(gemma, prompt(), [JPEG], " He juggles teacups ")
    assert result.idea == "He juggles teacups"
    assert result.understood == "He juggles teacups."
    assert (result.reason, result.proposed_by) == ("Seen it.", IdeaSource.USER)
    assert result.prompt_tokens == 123
    assert server.calls[0][1]["max_tokens"] == 400


def test_without_an_idea_gemma_proposes_one():
    gemma, _ = client(reply({"idea": "The toaster files a complaint.", "weakness": "Odd."}))
    result = run_chat(gemma, prompt(), [JPEG, JPEG], None)
    assert result.idea == result.understood == "The toaster files a complaint."
    assert result.proposed_by is IdeaSource.GEMMA
    assert result.as_output()["proposed_by"] == "gemma"


@pytest.mark.parametrize(
    "data",
    [
        {"understood": "x", "weakness": ""},
        {"understood": "", "weakness": "y"},
        {"idea": "x", "weakness": "y"},
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
    gemma, _ = client(reply({"idea": "A pigeon audits the bench.", "weakness": "w"}))
    result = run_chat(gemma, stored, [JPEG])
    idea = record_idea(session, job.id, result, stored)
    assert idea.prompt_version_id == stored.id
    assert (idea.text, idea.score, idea.proposed_by) == (
        "A pigeon audits the bench.",
        None,
        "gemma",
    )


def test_a_user_idea_is_kept_as_typed_without_gemma(session):
    user = User(tailscale_login=uuid.uuid4().hex)
    session.add(user)
    session.flush()
    job = Job(owner_id=user.id, photo_count=1, length=JobLength.SHORT, quality=JobQuality.HD720)
    session.add(job)
    session.flush()
    idea = record_idea(session, job.id, user_idea("  He juggles teacups "), None)
    assert (idea.text, idea.understood, idea.score, idea.reason) == (
        "He juggles teacups",
        None,
        None,
        "",
    )
    assert idea.prompt_version_id is None
    assert idea.proposed_by == "user"
    with pytest.raises(ValueError):
        user_idea("   ")


def test_long_text_is_cut_after_a_sentence_or_word():
    from memcraft.roles.chat import one_line

    assert one_line("Short one.", 50) == "Short one."
    text = "First sentence is here. Second one runs on and on without end"
    assert one_line(text, 40) == "First sentence is here."
    assert one_line("word " * 20, 22) == "word word word word…"


def long_idea(end="and at the very end a purple giraffe sneezes.", size=1000):
    """An idea of exactly `size` characters whose last words are `end`."""
    start = "<denys> walks into the kitchen and says Hi to <maksym>. "
    filler = "They argue about the kettle, the toaster and the fridge. "
    body = (start + filler * 40)[: size - len(end) - 1].rstrip()
    idea = f"{body} {end}"
    return idea + "x" * (size - len(idea))


CAST = [
    {"tag": "denys", "photo": 1, "position": 1},
    {"tag": "maksym", "photo": 1, "position": 2},
]


def test_a_full_length_idea_reaches_gemma_whole():
    from memcraft.web.upload import MAX_IDEA

    idea = long_idea(size=MAX_IDEA)
    assert len(idea) == MAX_IDEA == 1000
    gemma, server = client(reply({"understood": "u", "weakness": "w"}))
    run_chat(gemma, prompt(), [JPEG], idea, cast=CAST)
    body = server.calls[0][1]
    text = body["messages"][1]["content"][-1]["text"]
    assert idea in text and text.count("purple giraffe sneezes") == 1
    assert "<denys> = the person on the left in photo 1" in text
    assert "<maksym> = the person on the right in photo 1" in text
    understood = body["response_format"]["json_schema"]["schema"]["properties"]["understood"]
    assert understood["maxLength"] == 400


def test_the_cast_goes_after_the_idea_and_only_with_tags():
    _, user = build(prompt(), "<maksym> trips", JobLength.SHORT, CAST)
    assert user == (
        "The user's idea: <maksym> trips\n\n"
        "The idea names people of the photos by tags in angle brackets:\n"
        "<maksym> = the person on the right in photo 1\n"
        "In understood and weakness, write about these people by their tags, exactly as "
        "written, like <tag>."
    )
    # Without tags the call is the one it was before the cast: same system, same user text.
    assert build(prompt(), "He juggles teacups", JobLength.SHORT, CAST) == build(
        prompt(), "He juggles teacups", JobLength.SHORT
    )
    assert build(prompt(), "He juggles teacups")[1] == "The user's idea: He juggles teacups"


def test_an_older_prompt_without_the_cast_part_takes_the_shipped_one():
    old = prompt().text.split("=== cast ===")[0]
    older = PromptVersion(role=PromptRole.CHAT, model="gemma", text=old)
    assert build(older, "<denys> waves", cast=CAST) == build(prompt(), "<denys> waves", cast=CAST)
