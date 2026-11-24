"""The chat agent: Gemma reads the photos and proposes an idea when the friend gave none.

Since 2026-11-24 a friend's own idea skips Gemma (memcraft.pipeline.steps.IdeaStep keeps it as
typed, see user_idea), and ideas are no longer scored: the answer is the idea and its weakness.
run_chat still handles a user idea ("understood", never a rewrite of it) for older callers.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from sqlalchemy.orm import Session

from memcraft import cast as casting
from memcraft.models import Idea, IdeaSource, JobLength, PromptRole, PromptVersion
from memcraft.roles.gemma import GemmaClient, GemmaError, object_schema
from memcraft.roles.seed import GEMMA, fill, parts, prompt_file, read_prompt
from memcraft.versions import pin_idea

SECONDS = {JobLength.SHORT: "5 to 10", JobLength.LONG: "10 to 20"}
# What Gemma writes: an idea of its own, or how it understood the friend's (up to 1000
# characters, memcraft.web.upload.MAX_IDEA, which goes to Gemma whole).
MAX_IDEA = 300
MAX_UNDERSTOOD = 400
MAX_REASON = 160
# Enough for the longest answer: about 140 tokens of text and the JSON around it.
MAX_TOKENS = 400


@dataclass(frozen=True)
class ChatResult:
    idea: str
    # How Gemma understood a user idea; the idea itself when Gemma proposed it.
    understood: str
    reason: str
    proposed_by: IdeaSource
    seconds: float
    prompt_tokens: int | None = None
    output_tokens: int | None = None

    def as_output(self) -> dict[str, Any]:
        """For job_steps.output."""
        output = asdict(self)
        output["proposed_by"] = self.proposed_by.value
        return output


def schema(has_idea: bool) -> dict[str, Any]:
    text = {"type": "string", "maxLength": MAX_UNDERSTOOD if has_idea else MAX_IDEA}
    return object_schema(
        {
            ("understood" if has_idea else "idea"): text,
            "weakness": {"type": "string", "maxLength": MAX_REASON},
        }
    )


def build(
    prompt: PromptVersion,
    idea: str | None,
    length: JobLength = JobLength.SHORT,
    cast: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[str, str]:
    """System prompt and user text for one call. The idea goes in whole; when it names people
    (<denys>, <person2>), a line per person says who they are in the photos."""
    found = parts(prompt.text, prompt_file(PromptRole.CHAT, GEMMA).parts)
    idea = (idea or "").strip()
    task = found["with_idea"] if idea else found["without_idea"]
    system = "\n".join(
        (fill(found["system"], {"seconds": SECONDS[length]}), fill(task), fill(found["score"]))
    )
    if not idea:
        return system, fill(found["user_without_idea"])
    user = fill(found["user_with_idea"], {"idea": idea})
    lines = casting.chat_lines(cast, idea)
    if lines:
        # A prompt version from before the cast takes the part from the shipped file.
        part = found.get("cast") or parts(read_prompt(PromptRole.CHAT, GEMMA))["cast"]
        user = f"{user}\n\n" + fill(part, {"lines": "\n".join(lines)})
    return system, user


def one_line(text: str, limit: int) -> str:
    """One line of at most `limit` characters, cut after a whole sentence or word if needed."""
    line = " ".join(str(text).split())
    if len(line) <= limit:
        return line
    cut = line[:limit]
    end = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
    if end >= limit // 2:
        return cut[: end + 1]
    return cut.rsplit(" ", 1)[0].rstrip(",;:") + "…"


def run_chat(
    client: GemmaClient,
    prompt: PromptVersion,
    photos: Sequence[bytes],
    idea: str | None = None,
    *,
    length: JobLength = JobLength.SHORT,
    seed: int = 1,
    cast: Sequence[Mapping[str, Any]] | None = None,
) -> ChatResult:
    """`photos` are the small JPEG copies for Gemma, in the user's order."""
    if not 1 <= len(photos) <= 3:
        raise ValueError("the chat agent takes 1 to 3 photos")
    has_idea = bool((idea or "").strip())
    system, user = build(prompt, idea, length, cast)
    answer = client.ask(system, user, photos, schema(has_idea), max_tokens=MAX_TOKENS, seed=seed)
    data = answer.data
    try:
        reason = one_line(data["weakness"], MAX_REASON)
        said = one_line(
            data["understood" if has_idea else "idea"], MAX_UNDERSTOOD if has_idea else MAX_IDEA
        )
    except KeyError as error:
        raise GemmaError(f"Gemma's answer lacks {error}") from error
    if not said or not reason:
        raise GemmaError("Gemma's answer does not fit the schema")
    return ChatResult(
        idea=(idea or "").strip() if has_idea else said,
        understood=said,
        reason=reason,
        proposed_by=IdeaSource.USER if has_idea else IdeaSource.GEMMA,
        seconds=answer.seconds,
        prompt_tokens=answer.prompt_tokens,
        output_tokens=answer.output_tokens,
    )


def user_idea(idea: str) -> ChatResult:
    """The friend's own idea as typed, without Gemma: no understanding, no weakness."""
    text = idea.strip()
    if not text:
        raise ValueError("user_idea needs the friend's idea")
    return ChatResult(idea=text, understood="", reason="", proposed_by=IdeaSource.USER, seconds=0.0)


def record_idea(
    session: Session, job_id: uuid.UUID, result: ChatResult, prompt: PromptVersion | None
) -> Idea:
    """The Idea row, with the prompt it came from pinned (none when Gemma did not run).
    The caller commits."""
    idea = Idea(
        job_id=job_id,
        text=result.idea,
        understood=(result.understood or None) if result.proposed_by == IdeaSource.USER else None,
        reason=result.reason,
        proposed_by=result.proposed_by,
    )
    session.add(idea)
    session.flush()
    if prompt is not None:
        pin_idea(session, idea, prompt)
    return idea
