import re

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from memcraft.models import PromptRole, PromptVersion, VersionStatus
from memcraft.roles.seed import (
    FILES,
    PROMPTS,
    PromptError,
    active_prompt,
    fill,
    parts,
    read_prompt,
    seed_prompts,
)
from memcraft.versions import create_prompt_version, get_active, promote


@pytest.fixture
def session(engine):
    with Session(engine) as session:
        yield session


def test_parts_split_on_their_headers():
    text = "ignored\n=== system ===\nHello $name.\n\n=== user ===\nIdea: $idea\n"
    assert parts(text) == {"system": "Hello $name.", "user": "Idea: $idea"}
    with pytest.raises(PromptError, match="no part score"):
        parts(text, ["system", "score"])
    with pytest.raises(PromptError, match="twice"):
        parts(text + "=== user ===\nagain\n")


def test_fill_needs_every_placeholder():
    assert fill("Idea: $idea", {"idea": "a cat"}) == "Idea: a cat"
    with pytest.raises(PromptError, match="idea"):
        fill("Idea: $idea")


@pytest.mark.parametrize("entry", FILES, ids=lambda entry: entry.file)
def test_every_shipped_prompt_has_its_parts_and_is_clean(entry):
    text = read_prompt(entry.role, entry.model)
    assert set(parts(text)) == entry.parts
    raw = (PROMPTS / entry.file).read_text(encoding="utf-8")
    assert not re.search(r"[A-Za-z]:[\\/]|\\Users|/home/|\.(png|jpe?g)\b", raw)
    # Placeholders are plain words; a stray "$" would break fill().
    assert all(re.fullmatch(r"[a-z_]+", name) for name in re.findall(r"\$(\S+?)\b", raw))


def test_seed_stores_each_prompt_once_and_activates_only_the_first(session):
    rows = seed_prompts(session)
    assert len(rows) == len(FILES)
    assert all(row.status is VersionStatus.ACTIVE and row.version == 1 for row in rows)
    assert [row.id for row in seed_prompts(session)] == [row.id for row in rows]

    # An admin's own version stays active when the shipped text is new.
    chosen = create_prompt_version(session, role=PromptRole.CHAT, model="gemma", text="x")
    promote(session, chosen, VersionStatus.CANDIDATE)
    promote(session, chosen, VersionStatus.ACTIVE)
    old = rows[0]
    assert old.status is VersionStatus.RETIRED
    again = seed_prompts(session)
    assert again[0].id == old.id  # the same text is the same version, even retired
    assert get_active(session, PromptVersion, role=PromptRole.CHAT, model="gemma") == chosen


def test_a_changed_file_becomes_a_candidate(session, monkeypatch):
    seed_prompts(session)
    real = read_prompt

    def changed(role, model):
        text = real(role, model)
        return text.replace("tired comedy editor", "strict editor")

    monkeypatch.setattr("memcraft.roles.seed.read_prompt", changed)
    rows = seed_prompts(session)
    chat = rows[0]
    assert (chat.version, chat.status) == (2, VersionStatus.CANDIDATE)
    versions = session.scalars(
        select(PromptVersion.version).where(PromptVersion.role == PromptRole.CHAT)
    ).all()
    assert sorted(versions) == [1, 2]


def test_active_prompt_checks_the_parts(session):
    with pytest.raises(PromptError, match="seed_prompts"):
        active_prompt(session, PromptRole.CRITIC, "gemma")
    broken = create_prompt_version(session, role=PromptRole.CRITIC, model="gemma", text="hi")
    promote(session, broken, VersionStatus.CANDIDATE)
    promote(session, broken, VersionStatus.ACTIVE)
    with pytest.raises(PromptError, match="no part"):
        active_prompt(session, PromptRole.CRITIC, "gemma")
