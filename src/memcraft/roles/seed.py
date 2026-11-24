"""Prompt texts: the files in prompts/ become prompt_versions, and a stored text splits into parts.

A prompt text is plain text in parts, each opened by a line "=== name ===". A part may hold
placeholders such as $idea, which fill() replaces. Every role call reads its parts from the
PromptVersion it records, so a job always shows which words produced its output.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from string import Template

from sqlalchemy import select
from sqlalchemy.orm import Session

from memcraft.models import PromptRole, PromptVersion, VersionStatus
from memcraft.versions import create_prompt_version, get_active, promote

GEMMA = "gemma"
QWEN = "qwen3.8"

PROMPTS = Path(__file__).parent / "prompts"

_PART = re.compile(r"^=== ([a-z][a-z0-9_]*) ===[ \t]*$", re.M)


class PromptError(Exception):
    """The prompt text lacks a part or a placeholder value."""


@dataclass(frozen=True)
class PromptFile:
    role: PromptRole
    model: str
    file: str
    parts: frozenset[str]
    # Parts the shipped file has but older stored versions may lack; the code falls back to
    # the shipped file's part for those.
    optional: frozenset[str] = frozenset()


# The parts each role reads. A stored text without one of them is refused when seeded.
FILES: tuple[PromptFile, ...] = (
    PromptFile(
        PromptRole.CHAT,
        GEMMA,
        "chat-gemma.txt",
        frozenset(
            {"system", "with_idea", "without_idea", "score", "user_with_idea", "user_without_idea"}
        ),
        frozenset({"cast"}),
    ),
    PromptFile(
        PromptRole.DIRECTOR_MULTISHOT,
        QWEN,
        "director-multishot-qwen3.8.txt",
        frozenset(
            {
                "system",
                "runs_one",
                "runs_two",
                "labels_one",
                "labels_two",
                "user",
            }
        ),
        frozenset({"cast"}),
    ),
    PromptFile(
        PromptRole.DIRECTOR_KEYFRAMES,
        QWEN,
        "director-keyframes-qwen3.8.txt",
        frozenset(
            {
                "system",
                "kind_short",
                "kind_long",
                "labels_one",
                "labels_two",
                "suffix",
                "user",
            }
        ),
        # background: E16, the empty place drawn first; a version with it composes keyframes.
        frozenset({"cast", "background"}),
    ),
    PromptFile(PromptRole.CRITIC, GEMMA, "critic-gemma.txt", frozenset({"system", "user"})),
    PromptFile(PromptRole.CRITIC, QWEN, "critic-qwen3.8.txt", frozenset({"system", "user"})),
)


def parts(text: str, required: Iterable[str] = ()) -> dict[str, str]:
    """The named parts of a prompt text, each stripped. Text before the first part is ignored."""
    pieces = _PART.split(text)
    found: dict[str, str] = {}
    for index in range(1, len(pieces) - 1, 2):
        name = pieces[index]
        if name in found:
            raise PromptError(f"the part {name} appears twice")
        found[name] = pieces[index + 1].strip()
    missing = sorted(set(required) - set(found))
    if missing:
        raise PromptError(f"the prompt has no part {', '.join(missing)}")
    return found


def fill(part: str, values: Mapping[str, object] | None = None) -> str:
    """The part with its $placeholders replaced. A placeholder without a value is an error."""
    try:
        return Template(part).substitute({k: str(v) for k, v in (values or {}).items()})
    except (KeyError, ValueError) as error:
        raise PromptError(f"no value for the placeholder {error}") from None


def prompt_file(role: PromptRole, model: str) -> PromptFile:
    for entry in FILES:
        if (entry.role, entry.model) == (role, model):
            return entry
    raise PromptError(f"there is no prompt file for {role.value} {model}")


def read_prompt(role: PromptRole, model: str) -> str:
    """The text of the file as it ships, checked for its parts."""
    entry = prompt_file(role, model)
    text = (PROMPTS / entry.file).read_text(encoding="utf-8").replace("\r\n", "\n").strip() + "\n"
    parts(text, entry.parts | entry.optional)
    return text


def seed_prompts(session: Session) -> list[PromptVersion]:
    """Store every shipped prompt that is not stored yet. The caller commits.

    A text equal to a stored version (of any status but draft) is that version. A new text
    becomes a candidate; it is activated only when nothing of its role and model is active,
    so a seed never replaces the prompt an admin chose.
    """
    rows = []
    for entry in FILES:
        text = read_prompt(entry.role, entry.model)
        row = session.scalar(
            select(PromptVersion)
            .where(
                PromptVersion.role == entry.role,
                PromptVersion.model == entry.model,
                PromptVersion.text == text,
                PromptVersion.status != VersionStatus.DRAFT,
            )
            .order_by(PromptVersion.version.desc())
            .limit(1)
        )
        if row is None:
            row = create_prompt_version(session, role=entry.role, model=entry.model, text=text)
            promote(session, row, VersionStatus.CANDIDATE)
            if get_active(session, PromptVersion, role=entry.role, model=entry.model) is None:
                promote(session, row, VersionStatus.ACTIVE)
        rows.append(row)
    return rows


def active_prompt(session: Session, role: PromptRole, model: str) -> PromptVersion:
    """The active prompt for a role and model, checked for the parts the code reads."""
    row = get_active(session, PromptVersion, role=role, model=model)
    if row is None:
        raise PromptError(f"no active prompt for {role.value} {model}; run seed_prompts()")
    parts(row.text, prompt_file(role, model).parts)
    return row
