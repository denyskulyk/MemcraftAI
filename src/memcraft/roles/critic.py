"""The critic: one look at a made frame (a keyframe, or the last frame of a clip) next to the
source photos. Pass or fail for face, story and artefacts, a one-line reason each, and one
correction sentence for the next try.

Which critic a job uses is a setting: Gemma (CPU, the default), Qwen 3.8 (GPU, in E0 too
lenient) or off. The settings row "active_critic_engine" holds the id of a critic engine, or
null for off. A job pins the engine it started with (versions.pin_job), and every check
records its prompt (versions.pin_step).
"""

from __future__ import annotations

import json
import re
import time
import uuid
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

from sqlalchemy.orm import Session

from memcraft.models import (
    EngineSlot,
    EngineVersion,
    Job,
    JobStep,
    PromptRole,
    PromptVersion,
    Setting,
    VersionStatus,
)
from memcraft.roles.director import TextRunner
from memcraft.roles.gemma import GemmaClient, GemmaError, object_schema
from memcraft.roles.seed import GEMMA, QWEN, PromptError, active_prompt, fill, parts
from memcraft.versions import pin_step

SETTING = "active_critic_engine"
CRITICS = (GEMMA, QWEN)
CHECKS = ("face", "story", "artefacts")
MAX_WHY = 160
MAX_CORRECTION = 200
MAX_TOKENS = 450


class CriticError(Exception):
    pass


@dataclass(frozen=True)
class Verdict:
    face: bool
    story: bool
    artefacts: bool
    face_why: str
    story_why: str
    artefacts_why: str
    correction: str
    model: str
    seconds: float
    attempts: int = 1

    @property
    def passed(self) -> bool:
        return self.face and self.story and self.artefacts

    def as_output(self) -> dict[str, Any]:
        output = asdict(self)
        output["passed"] = self.passed
        return output


def schema() -> dict[str, Any]:
    verdict = {"type": "string", "enum": ["pass", "fail"]}
    why = {"type": "string", "maxLength": MAX_WHY}
    properties: dict[str, Any] = {}
    for check in CHECKS:
        properties[f"{check}_why"] = why
        properties[check] = verdict
    properties["correction"] = {"type": "string", "maxLength": MAX_CORRECTION}
    return object_schema(properties)


def build(prompt: PromptVersion, photos: int, story: str) -> tuple[str, str]:
    found = parts(prompt.text, ("system", "user"))
    return fill(found["system"]), fill(found["user"], {"count": photos, "story": story.strip()})


def _one_line(value: Any, limit: int) -> str:
    return " ".join(str(value or "").split())[:limit]


def read_verdict(data: Any, model: str, seconds: float, attempts: int = 1) -> Verdict:
    """A Verdict from the critic's JSON. Anything but a clear "pass" counts as a fail."""
    if not isinstance(data, dict):
        raise CriticError("the critic gave no JSON object")
    missing = [check for check in CHECKS if check not in data]
    if missing:
        raise CriticError(f"the critic did not judge {', '.join(missing)}")

    def passed(check: str) -> bool:
        value = str(data[check]).strip().lower()
        if value not in ("pass", "fail"):
            raise CriticError(f"the critic's {check} is {value!r}, not pass or fail")
        return value == "pass"

    verdicts = {check: passed(check) for check in CHECKS}
    correction = _one_line(data.get("correction"), MAX_CORRECTION)
    return Verdict(
        **verdicts,
        **{f"{c}_why": _one_line(data.get(f"{c}_why"), MAX_WHY) for c in CHECKS},
        correction="" if all(verdicts.values()) else correction,
        model=model,
        seconds=seconds,
        attempts=attempts,
    )


def loose_json(text: str) -> Any:
    """The JSON object in an answer that may carry a code fence, a think block or text around."""
    clean = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I).strip()
    try:
        return json.loads(clean)
    except ValueError:
        pass
    start, end = clean.find("{"), clean.rfind("}")
    if start < 0 or end < start:
        return None
    try:
        return json.loads(clean[start : end + 1], strict=False)
    except ValueError:
        return None


def check_with_gemma(
    client: GemmaClient,
    prompt: PromptVersion,
    photos: Sequence[bytes],
    frame: bytes,
    story: str,
    *,
    seed: int = 1,
) -> Verdict:
    system, user = build(prompt, len(photos), story)
    answer = client.ask(
        system,
        user,
        [*photos, frame],
        schema(),
        max_tokens=MAX_TOKENS,
        temperature=0.2,
        seed=seed,
    )
    try:
        return read_verdict(answer.data, GEMMA, answer.seconds)
    except CriticError as error:
        raise GemmaError(str(error)) from error


def check_with_qwen(
    run_text: TextRunner,
    prompt: PromptVersion,
    photos: Sequence[bytes],
    frame: bytes,
    story: str,
    *,
    seed: int = 3,
) -> Verdict:
    """One retry with a new seed when the answer is not a full verdict."""
    system, user = build(prompt, len(photos), story)
    started = time.monotonic()
    try:
        data = loose_json(run_text(system, user, [*photos, frame], seed))
        return read_verdict(data, QWEN, round(time.monotonic() - started, 1))
    except CriticError:
        data = loose_json(run_text(system, user, [*photos, frame], seed + 1))
        return read_verdict(data, QWEN, round(time.monotonic() - started, 1), attempts=2)


class Critic:
    """The critic a job pinned, ready to check frames."""

    def __init__(
        self,
        engine: EngineVersion,
        prompt: PromptVersion,
        *,
        gemma: GemmaClient | None = None,
        run_text: TextRunner | None = None,
    ) -> None:
        if engine.name not in CRITICS:
            raise CriticError(f"unknown critic {engine.name!r}")
        if engine.name == GEMMA and gemma is None:
            raise CriticError("the Gemma critic needs a Gemma client")
        if engine.name == QWEN and run_text is None:
            raise CriticError("the Qwen critic needs a text runner")
        self.engine = engine
        self.prompt = prompt
        self.gemma = gemma
        self.run_text = run_text

    @property
    def needs_gpu(self) -> bool:
        return self.engine.name == QWEN

    def check(self, photos: Sequence[bytes], frame: bytes, story: str, *, seed: int = 1) -> Verdict:
        if self.engine.name == GEMMA and self.gemma is not None:
            return check_with_gemma(self.gemma, self.prompt, photos, frame, story, seed=seed)
        if self.engine.name == QWEN and self.run_text is not None:
            return check_with_qwen(self.run_text, self.prompt, photos, frame, story, seed=seed)
        raise CriticError(f"no way to run the {self.engine.name} critic")

    def check_step(
        self,
        session: Session,
        step: JobStep,
        photos: Sequence[bytes],
        frame: bytes,
        story: str,
        *,
        seed: int = 1,
    ) -> Verdict:
        """Pins the prompt on the check step first, so even a failed call shows its words."""
        pin_step(session, step, prompt=self.prompt)
        return self.check(photos, frame, story, seed=seed)


def chosen_engine(session: Session) -> EngineVersion | None:
    """The critic new jobs get: the engine the setting names, or None when it is off."""
    row = session.get(Setting, SETTING)
    value = None if row is None else row.value
    if value in (None, "", "off"):
        return None
    try:
        engine_id = uuid.UUID(str(value))
    except ValueError:
        raise CriticError(f"the setting {SETTING} is not an engine id") from None
    engine = session.get(EngineVersion, engine_id)
    if engine is None or engine.slot is not EngineSlot.CRITIC:
        raise CriticError(f"the setting {SETTING} names no critic engine")
    if engine.status not in (VersionStatus.CANDIDATE, VersionStatus.ACTIVE):
        raise CriticError(f"the chosen critic is {engine.status.value}")
    return engine


def choose(session: Session, engine: EngineVersion | None) -> None:
    """Set the critic for new jobs; None turns it off. Jobs that started keep theirs."""
    if engine is not None and engine.slot is not EngineSlot.CRITIC:
        raise CriticError("that engine is not a critic")
    row = session.get(Setting, SETTING)
    value = None if engine is None else str(engine.id)
    if row is None:
        session.add(Setting(key=SETTING, value=value))
    else:
        row.value = value
    session.flush()


def pinned_critic(
    session: Session,
    job: Job,
    *,
    gemma: GemmaClient | None = None,
    run_text: TextRunner | None = None,
) -> Critic | None:
    """The critic the job pinned, or None when it was off. Its prompt is the engine's own
    prompt version, or else the active critic prompt of that model."""
    if job.critic_engine_id is None:
        return None
    engine = session.get_one(EngineVersion, job.critic_engine_id)
    if engine.prompt_version_id is not None:
        prompt = session.get_one(PromptVersion, engine.prompt_version_id)
        parts(prompt.text, ("system", "user"))
    else:
        try:
            prompt = active_prompt(session, PromptRole.CRITIC, engine.name)
        except PromptError as error:
            raise CriticError(str(error)) from error
    return Critic(engine, prompt, gemma=gemma, run_text=run_text)
