"""The director: Qwen 3.8 turns the approved idea into LTX-2.5 prompts.

Qwen answers in labelled sections ("TITLE: ...") that this code turns into JSON; in E0 its
nested JSON broke on quoted speech and line breaks, the sections never did. A missing section
costs one retry with a new seed. Qwen writes "image1" where the keyframe model needs
"<image1>", so the code rewrites the tags.

Two regimes:
- multishot: one photo is the first frame; one prompt per LTX run, always a single take (with
  2 shots the faces held in 2 of 4 E0 videos).
- keyframes: a bible (people, place, light, camera, props), three keyframe prompts with their
  frame positions, and one prompt per LTX run between them. Middle and end take the start
  keyframe as an extra reference only for 1 or 2 people; with 3 it lost a face, so they come
  from the photos only. The CAMERA sentence (shot size and position) and the PROPS with their
  colours go word for word into every keyframe: in the first real runs a waist-up start was
  followed by a full-body middle, and a black umbrella turned red.

Never music: every run prompt loses any sentence that mentions music and ends on "No music."
(no_music()), whatever the prompt text asked for.

A long video is two runs; the second starts from the last frame of the first, so the runs
share that frame. Frame counts are 8n+1: one run has 153 to 297, a long video 305 to 593.
"""

from __future__ import annotations

import random
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from memcraft import cast as casting
from memcraft.engines.keyframes import (
    CLEAN_PERSON,
    PROP_FROM_PHOTO,
    PROP_NEW,
    compose_prompt,
    edit_prompt,
    matching_prompt,
)
from memcraft.models import JobLength, PromptRole, PromptVersion
from memcraft.roles.seed import QWEN, fill, parts, prompt_file, read_prompt

# (system prompt, user text, images in order, seed) -> Qwen's raw answer
type TextRunner = Callable[[str, str, Sequence[bytes], int], str]

FPS = 30
FRAME_RANGE = {JobLength.SHORT: (153, 297), JobLength.LONG: (305, 593)}
DEFAULT_FRAMES = {JobLength.SHORT: 241, JobLength.LONG: 481}
MAX_SHOTS = 1
# Middle and end keyframes take the start keyframe as a reference up to this many people.
CHAINED_PEOPLE = 2
# Said in middle and end keyframes drawn from the photos only, so they match the start.
SAME_FRAMING = (
    "Same place, light, camera position and framing as the start of this scene; only the "
    "action and expressions change."
)
# Said in every keyframe after the action when the bible lists props.
PROPS_SENTENCE = (
    "Each of these objects is there only once and keeps exactly this look in every picture: "
    "{props}."
)
NO_MUSIC = "No music."
MAX_RUN_WORDS = 220
_SEED_LIMIT = 2**31 - 1


class DirectorError(Exception):
    pass


class SectionsMissingError(DirectorError):
    def __init__(self, missing: Sequence[str], raw: str) -> None:
        super().__init__("missing " + ", ".join(missing))
        self.missing = list(missing)
        self.raw = raw


class FrameCountError(DirectorError, ValueError):
    pass


# Frames


def check_frames(frames: int, length: JobLength) -> int:
    low, high = FRAME_RANGE[length]
    if type(frames) is not int or frames % 8 != 1 or not low <= frames <= high:
        raise FrameCountError(f"a {length.value} video has 8n+1 frames from {low} to {high}")
    return frames


def split_runs(frames: int, length: JobLength) -> list[int]:
    """Frames of each LTX run. Two runs share one frame, so they add up to frames + 1."""
    check_frames(frames, length)
    if length is JobLength.SHORT:
        return [frames]
    steps = (frames - 1) // 8
    first = (steps + 1) // 2
    return [8 * first + 1, 8 * (steps - first) + 1]


def run_starts(runs: Sequence[int]) -> list[int]:
    """The frame of the whole video at which each run begins."""
    starts, at = [], 0
    for count in runs:
        starts.append(at)
        at += count - 1
    return starts


def render_seeds(seed: int, count: int) -> list[int]:
    """`count` seeds that follow from one job seed, so a retry of the plan draws the same."""
    rng = random.Random(seed)  # noqa: S311 - render seeds, not secrets
    return [rng.randrange(1, _SEED_LIMIT) for _ in range(count)]


# Parsing


_THINK = re.compile(r"<think>.*?</think>", re.S | re.I)
_FENCE = re.compile(r"^\s*```\w*\s*$", re.M)
_TAG = re.compile(r"<?\bimage\s*([1-9])\b>?", re.I)


_WIDE_COLON = "\N{FULLWIDTH COLON}"


def _label_pattern(labels: Sequence[str]) -> re.Pattern[str]:
    names = sorted({*labels}, key=len, reverse=True)
    # A RUN or KEYFRAME label Qwen invented still ends the section before it.
    alternatives = [re.escape(name).replace(r"\ ", r"\s+") for name in names]
    alternatives += [r"RUN\s+\d+", r"KEYFRAME\s+[A-Z]+"]
    return re.compile(
        r"^[ \t>#*_-]*(" + "|".join(alternatives) + r")[ \t*_]*[:" + _WIDE_COLON + r"][ \t*_]*",
        re.M | re.I,
    )


def _clean_value(value: str) -> str:
    value = " ".join(value.split()).strip("*_ ").strip()
    for quote in ('"', "“”"):
        opening, closing = quote[0], quote[-1]
        inner = value[1:-1]
        if (
            len(value) > 1
            and value[0] == opening
            and value[-1] == closing
            and opening not in inner
            and closing not in inner
        ):
            value = inner.strip()
    return value


def parse_sections(
    text: str, labels: Sequence[str], optional: Sequence[str] = ()
) -> tuple[dict[str, str], list[str]]:
    """Labelled sections by their label in capitals, and problems that did not stop parsing.

    Raises SectionsMissingError if a label is missing or empty; an ``optional`` label (one
    that older prompt versions do not ask for) may be missing.
    """
    clean = _FENCE.sub("", _THINK.sub("", text))
    wanted = {" ".join(label.upper().split()) for label in (*labels, *optional)}
    pieces = _label_pattern([*labels, *optional]).split(clean)
    found: dict[str, str] = {}
    problems: list[str] = []
    for index in range(1, len(pieces) - 1, 2):
        label = " ".join(pieces[index].upper().split())
        value = _clean_value(pieces[index + 1])
        if label not in wanted:
            problems.append(f"unexpected section {label}")
        elif label in found and value:
            problems.append(f"{label} appears twice; the first is kept")
        elif value:
            found[label] = value
    missing = [label for label in labels if not found.get(" ".join(label.upper().split()))]
    if missing:
        raise SectionsMissingError(missing, text)
    return found, problems


def image_tags(text: str) -> str:
    """Rewrite "image1", "Image 1" and "<image1>" all as "<image1>"."""
    return _TAG.sub(lambda match: f"<image{match.group(1)}>", text)


def for_ltx(text: str) -> str:
    """A run prompt without picture tags, which only the keyframe model understands:
    "the man from image1 waves" becomes "the man waves"."""
    text = re.sub(r"\s*\b(?:from|in|of)\s+<?\bimage\s*[1-9]\b>?", "", text, flags=re.I)
    text = _TAG.sub("", text)
    return re.sub(r"\s+([,.;:!?])", r"\1", " ".join(text.split()))


_MUSIC = re.compile(r"\b(?:music\w*|soundtracks?|melod\w*)\b", re.I)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+|(?<=[.!?][\"”])\s+")


def no_music(text: str) -> str:
    """A run prompt without any sentence that mentions music. Since E16 it no longer ends on
    "No music.": naming music seemed to call it up; the clip's negative prompt keeps it out."""
    sentences = [part for part in _SENTENCE_END.split(text.strip()) if part.strip()]
    return " ".join(part for part in sentences if not _MUSIC.search(part))


def run_prompt(text: str) -> str:
    """The final LTX prompt of a run: no picture tags, no music."""
    return no_music(for_ltx(text))


def name_people(body: str, people: Sequence[dict[str, Any]], count: int) -> tuple[str, list[str]]:
    """A keyframe that names every photo as <imageN>, and notes on what was repaired.

    Qwen sometimes writes only the tag ("the red-jacket man"). The keyframe model needs the
    picture, so " from <imageN>" goes after the first mention of the tag, or, if the tag is
    not there either, a sentence naming the person is added.
    """
    notes = []
    for number in range(1, count + 1):
        if number in tags_in(body):
            continue
        person = next((p for p in people if p.get("image") == number), None)
        tag = (person or {}).get("tag", "")
        found = re.search(re.escape(tag), body, re.I) if tag else None
        if found:
            body = f"{body[: found.end()]} from <image{number}>{body[found.end() :]}"
            notes.append(f"<image{number}> added after {tag!r}")
        else:
            look = (person or {}).get("look", "")
            who = tag or "the person"
            body = f"{body} {who[0].upper()}{who[1:]} from <image{number}>" + (
                f": {look}." if look else " is in the picture."
            )
            notes.append(f"<image{number}> not named; a sentence was added")
    return body, notes


def tags_in(text: str) -> set[int]:
    return {int(number) for number in re.findall(r"<image([1-9])>", text)}


def _people(value: str, with_images: bool) -> list[dict[str, Any]]:
    people = []
    for chunk in value.split("|"):
        chunk = chunk.strip(" ;")
        if not chunk:
            continue
        person: dict[str, Any] = {}
        if with_images:
            match = re.match(r"<image([1-9])>\s*[:\-]?\s*", chunk)
            if match:
                person["image"] = int(match.group(1))
                chunk = chunk[match.end() :]
            tag, _, look = chunk.partition(",")
        else:
            tag, _, look = chunk.partition(":")
            if not look:
                tag, _, look = chunk.partition(",")
        person["tag"] = tag.strip()
        person["look"] = look.strip()
        people.append(person)
    return people


def _cut_words(prompt: str) -> int:
    return len(re.findall(r"\bcuts?\b|\bdissolves?\b|\btransitions?\s+to\b", prompt, re.I))


# The cast: people the idea names as <denys> or <person2>.


def cast_text(
    found: dict[str, str],
    role: PromptRole,
    cast: Sequence[Mapping[str, Any]] | None,
    idea: str,
) -> str:
    """The CAST section for the user text, or "" when the idea names no one. A prompt version
    from before the cast takes the part from the shipped prompt file."""
    lines = casting.director_lines(cast, idea)
    if not lines:
        return ""
    part = found.get("cast") or parts(read_prompt(role, QWEN))["cast"]
    return fill(part, {"lines": "\n".join(lines)})


def _user_text(
    found: dict[str, str],
    role: PromptRole,
    cast: Sequence[Mapping[str, Any]] | None,
    idea: str,
) -> str:
    """The idea, whole, and the CAST section when the idea names people."""
    user = fill(found["user"], {"idea": idea.strip()})
    extra = cast_text(found, role, cast, idea)
    return f"{user}\n\n{extra}" if extra else user


def distinct_people(used: Sequence[Mapping[str, Any]]) -> list[list[str]]:
    """The tags of each person of the cast once, in cast order: <person2> and <maksym> may be
    the same person."""
    people: dict[tuple[Any, ...], list[str]] = {}
    for entry in used:
        photo = entry.get("photo")
        key = (photo, entry["position"]) if photo is not None else ("tag", entry["tag"])
        people.setdefault(key, []).append(entry["tag"])
    return list(people.values())


def cast_people(
    cast: Sequence[Mapping[str, Any]] | None, idea: str, people: Sequence[dict[str, Any]]
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Each tag of the idea (casefolded) with its entry of PEOPLE, and problems. The director
    lists the people of the cast first, in cast order."""
    used = casting.used(cast, idea)
    found: dict[str, dict[str, Any]] = {}
    problems = []
    for index, tags in enumerate(distinct_people(used)):
        if index >= len(people):
            problems.append(f"PEOPLE does not list <{tags[0]}>")
            continue
        for tag in tags:
            found[tag.casefold()] = people[index]
    return found, problems


def replace_cast_tags(
    text: str, mapping: Mapping[str, Mapping[str, Any]], *, with_image: bool
) -> tuple[str, int]:
    """The text with every <denys> the director left in it replaced by that person's visual
    tag from PEOPLE ("the man in the red jacket"), with " from <imageN>" for a keyframe.
    Returns the text and how many tags were replaced."""
    count = 0

    def visual(match: re.Match[str]) -> str:
        nonlocal count
        person = mapping.get(match.group(1).casefold())
        if person is None or not person.get("tag"):
            return match.group(0)
        count += 1
        tag = person["tag"]
        if with_image and person.get("image"):
            tag = f"{tag} from <image{person['image']}>"
        before = text[: match.start()].rstrip()
        if not before or before[-1] in ".!?":
            tag = tag[0].upper() + tag[1:]
        return tag

    return casting.TAG.sub(visual, text), count


# Results


@dataclass(frozen=True)
class Run:
    prompt: str
    frames: int
    seed: int
    # The frame of the whole video where it begins.
    start: int
    # What its first frame is: "photo", "keyframe" (the start keyframe) or
    # "previous_last_frame" (the real last frame of the run before).
    first: str
    # Keyframes this run is guided by: (keyframe "at", frame index inside the run).
    keyframes: list[tuple[str, int]] = field(default_factory=list)


@dataclass(frozen=True)
class Keyframe:
    """`action` is the director's part of the keyframe prompt. The full prompt is
    memcraft.engines.matching_prompt(action, place=, light=, photos=, start_reference=), which
    puts the new place first for the start keyframe and "same scene as" the start keyframe
    for the others; prompt() does that."""

    at: str
    action: str
    # The frame of the whole video it stands for.
    frame: int
    seed: int
    # References for the keyframe model in order: photo numbers from 1, "start" for the start
    # keyframe once it is drawn, and "asset:K" for picture K of KeyframePlan.assets (E16: a
    # clean picture in place of a photo, or a prop picture after the photos). The prompt calls
    # them <image1>, <image2>, ... Starting with "start", the keyframe is an edit of it.
    references: list[int | str]
    # The words of the prop pictures among the references, in order (start keyframe only).
    props: list[str] = field(default_factory=list)
    # "compose": put together from the background, people and prop pictures (E16).
    layout: str = "photos"
    # Photo numbers whose face crop follows the people pictures (compose layout).
    faces: list[int] = field(default_factory=list)
    # The director moved the camera for this keyframe (compose layout).
    camera_moves: bool = False

    def prompt(self, bible: dict[str, Any]) -> str:
        if self.layout == "compose":
            people = len(self.references) - 1 - len(self.props) - len(self.faces)
            return compose_prompt(
                self.action,
                people=people,
                props=self.props,
                camera_moves=self.camera_moves,
                faces=self.faces,
            )
        people = len(self.references) - len(self.props) - ("start" in self.references)
        if self.references and self.references[0] == "start":
            return edit_prompt(
                self.action, place=bible["place"], light=bible["light"], photos=people
            )
        return matching_prompt(
            self.action,
            place=bible["place"],
            light=bible["light"],
            photos=people,
            start_reference="start" in self.references,
            props=self.props,
        )


@dataclass(frozen=True)
class AssetPicture:
    """A picture drawn before the keyframes (E16): ``kind`` "cut" is photo ``photo`` cut out as it
    is (nothing drawn), "clean" photo ``photo`` redrawn with its people alone and empty-handed,
    "prop" is one prop alone, from photo ``photo`` or, with
    None, drawn on a plain grey canvas, "background" the empty place on that canvas. With
    ``cutout`` it is cut out of its background and kept as RGBA."""

    kind: str
    photo: int | None
    words: str
    seed: int
    prompt: str
    cutout: bool = False
    # kind "face": the face box in the photo as fractions [x, y, w, h] (E16).
    box: list[float] | None = None


@dataclass(frozen=True)
class MultishotPlan:
    title: str
    people: list[dict[str, Any]]
    shots: int
    frames: int
    runs: list[Run]
    seconds: float
    attempts: int
    seed: int
    problems: list[str]
    raw: str

    def as_output(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class KeyframePlan:
    title: str
    bible: dict[str, Any]
    keyframes: list[Keyframe]
    frames: int
    runs: list[Run]
    seconds: float
    attempts: int
    seed: int
    problems: list[str]
    raw: str
    assets: list[AssetPicture] = field(default_factory=list)

    def as_output(self) -> dict[str, Any]:
        """For job_steps.output; each keyframe also carries its full prompt."""
        output = asdict(self)
        for keyframe, row in zip(self.keyframes, output["keyframes"], strict=True):
            row["prompt"] = keyframe.prompt(self.bible)
        return output


# Calls


def _ask(
    run_text: TextRunner,
    system: str,
    user: str,
    images: Sequence[bytes],
    labels: Sequence[str],
    seed: int,
    clock: Callable[[], float],
    optional: Sequence[str] = (),
) -> tuple[dict[str, str], list[str], str, float, int, int]:
    """Sections, problems, raw text, seconds, attempts and the seed that worked."""
    started = clock()
    try:
        raw = run_text(system, user, images, seed)
        found, problems = parse_sections(raw, labels, optional)
        return found, problems, raw, round(clock() - started, 1), 1, seed
    except SectionsMissingError as first:
        # One retry with a new seed; a second miss goes to the caller with its raw text.
        raw = run_text(system, user, images, seed + 1)
        found, problems = parse_sections(raw, labels, optional)
        problems.insert(0, f"first answer: {first}")
        return found, problems, raw, round(clock() - started, 1), 2, seed + 1


def direct_multishot(
    run_text: TextRunner,
    prompt: PromptVersion,
    photo: bytes,
    idea: str,
    *,
    length: JobLength = JobLength.SHORT,
    shots: int = 1,
    frames: int | None = None,
    seed: int = 7,
    clock: Callable[[], float] = time.monotonic,
    cast: Sequence[Mapping[str, Any]] | None = None,
) -> MultishotPlan:
    if not 1 <= shots <= MAX_SHOTS:
        raise DirectorError("a multishot video is a single take")
    if not idea.strip():
        raise DirectorError("the director needs the approved idea")
    frames = check_frames(DEFAULT_FRAMES[length] if frames is None else frames, length)
    counts = split_runs(frames, length)
    found = parts(prompt.text, prompt_file(PromptRole.DIRECTOR_MULTISHOT, QWEN).parts)
    two = len(counts) == 2
    run_seconds = round((counts[0] - 1) / FPS)
    system = fill(
        found["system"],
        {
            "seconds": round((frames - 1) / FPS),
            "runs": fill(found["runs_two" if two else "runs_one"], {"run_seconds": run_seconds}),
            "labels": fill(found["labels_two" if two else "labels_one"]),
            "shots": shots,
            # Older prompt versions keep the single-take rule in a part of its own.
            "shot_rule": fill(found.get("shot_rule_one", "")),
            "words": MAX_RUN_WORDS,
        },
    )
    user = _user_text(found, PromptRole.DIRECTOR_MULTISHOT, cast, idea)
    labels = ["TITLE", "PEOPLE"] + [f"RUN {i + 1}" for i in range(len(counts))]
    sections, problems, raw, seconds, attempts, used = _ask(
        run_text, system, user, [photo], labels, seed, clock
    )
    people = _people(sections["PEOPLE"], with_images=False)
    mapping, notes = cast_people(cast, idea, people)
    problems += notes
    seeds = render_seeds(seed, len(counts))
    runs = []
    for index, (count, start) in enumerate(zip(counts, run_starts(counts), strict=True)):
        text, replaced = replace_cast_tags(sections[f"RUN {index + 1}"], mapping, with_image=False)
        if replaced:
            problems.append(f"RUN {index + 1}: {replaced} name tags replaced by visual tags")
        cuts = _cut_words(text)
        if cuts and not re.search(r"\bno cuts?\b", text, re.I):
            problems.append(f"RUN {index + 1}: {cuts} cut words in a single take")
        if len(text.split()) > MAX_RUN_WORDS + 40:
            problems.append(f"RUN {index + 1}: {len(text.split())} words")
        runs.append(
            Run(
                prompt=run_prompt(text),
                frames=count,
                seed=seeds[index],
                start=start,
                first="photo" if index == 0 else "previous_last_frame",
            )
        )
    return MultishotPlan(
        title=sections["TITLE"],
        people=people,
        shots=shots,
        frames=frames,
        runs=runs,
        seconds=seconds,
        attempts=attempts,
        seed=used,
        problems=problems,
        raw=raw,
    )


KEYFRAME_LABELS = (
    "TITLE",
    "PEOPLE",
    "PLACE",
    "LIGHT",
    "CAMERA",
    "KEYFRAME START",
    "KEYFRAME MIDDLE",
    "KEYFRAME END",
)
# Asked for since director-keyframes v2; a job directed with an older prompt has no props.
OPTIONAL_KEYFRAME_LABELS = ("PROPS", "CLEAN", "PROP PICTURES", "CAMERA MIDDLE", "CAMERA END")
# At most this many prop pictures; with the photos they must fit the keyframe model's pictures:
# 4 with the start keyframe layout, 6 when keyframes are put together from asset pictures.
MAX_PROP_PICTURES = 2
MAX_REFERENCES = 4
MAX_COMPOSE_REFERENCES = 6
# A face crop is cut from the photo in the worker; nothing is drawn for it.
FACE_CROP = "The face of <image1>, cropped."
# The prompt of a photo cut out as it is; the workflow draws nothing for it.
CUT_ONLY = "Cut out the people of <image1> as they are."
_SAME_CAMERA = re.compile(r"^(?:same|unchanged|no change|static|none|-)\b", re.I)


_INVENTORY_LINE = re.compile(r"<image([1-9])>\s*:\s*(.*?)\|\s*holds?\s*:\s*(.*)", re.I)
_NOTHING = re.compile(r"^(?:nothing|none|no(?:thing)? objects?|empty|-|n/?a)\.?$", re.I)


def parse_inventory(text: str, photos: int) -> list[dict[str, Any]]:
    """The inventory answer as one entry per photo: who is in it and what they hold."""
    found: dict[int, dict[str, Any]] = {}
    clean = _FENCE.sub("", _THINK.sub("", text))
    for line in clean.splitlines():
        match = _INVENTORY_LINE.search(image_tags(line))
        if not match:
            continue
        number = int(match.group(1))
        if not 1 <= number <= photos or number in found:
            continue
        items = [
            " ".join(item.split()).strip(" .;\"'*").lower()
            for item in re.split(r"[,;]| and ", match.group(3))
        ]
        items = [re.sub(r"^(?:a|an|the|some|two|his|her|their)\s+", "", item) for item in items]
        holds = [item for item in items if item and not _NOTHING.match(item)][:3]
        found[number] = {"photo": number, "who": match.group(2).strip(" *"), "holds": holds}
    return [found.get(n, {"photo": n, "who": "", "holds": []}) for n in range(1, photos + 1)]


def take_inventory(
    run_text: TextRunner, prompt: PromptVersion, photos: Sequence[bytes], seed: int = 7
) -> tuple[list[dict[str, Any]], str]:
    """E16: before the director, Qwen 3.8 lists what each photo's people hold; the pipeline
    then checks every object with SAM. Returns the entries and the raw answer."""
    found = parts(prompt.text, prompt_file(PromptRole.DIRECTOR_KEYFRAMES, QWEN).parts)
    raw = run_text(found["inventory"], f"Photos: {len(photos)}", list(photos), seed)
    return parse_inventory(raw, len(photos)), raw


def held_text(found: dict[str, str], inventory: Sequence[Mapping[str, Any]]) -> str:
    """The checked inventory for the director's user prompt; only confirmed objects count."""
    lines = []
    for entry in inventory:
        held = [item for item, ok in entry.get("checked", {}).items() if ok]
        what = ", ".join(f"a {item}" for item in held) if held else "nothing"
        lines.append(f"<image{entry['photo']}>: holds {what}")
    return fill(found["held"], {"lines": "\n".join(lines)})


def confirmed_holds(inventory: Sequence[Mapping[str, Any]] | None) -> dict[int, list[str]] | None:
    if inventory is None:
        return None
    return {
        int(entry["photo"]): [item for item, ok in entry.get("checked", {}).items() if ok]
        for entry in inventory
    }


def _held(
    photo: int,
    words: str,
    held: Mapping[int, Sequence[str]],
    confirm: Callable[[int, str], bool] | None,
) -> bool:
    """Whether a prop the director says comes from ``photo`` is held there. With ``confirm``
    (SAM on the photo, E16) its own words are looked for, since Qwen may call the same log a
    stick in the inventory; without it the words must share one with a checked object."""
    if not held.get(photo):
        return False
    if confirm is not None:
        return confirm(photo, words)
    return _matches(words, held[photo])


def _matches(words: str, items: Sequence[str]) -> bool:
    tokens = set(re.findall(r"[a-z]+", words.lower()))
    return any(set(re.findall(r"[a-z]+", item.lower())) & tokens for item in items)


def camera_for(value: str, start: str) -> tuple[str, bool]:
    """A keyframe's camera sentence from CAMERA MIDDLE or CAMERA END, and whether it moved:
    "same" (or nothing) repeats the start CAMERA word for word."""
    value = value.strip()
    if not value or _SAME_CAMERA.match(value) or value == start:
        return start, False
    return value, True


_NO_PROPS = re.compile(r"^(?:none|no props|n/?a|-)\.?$", re.I)


def props_of(value: str) -> list[str]:
    """The props of the PROPS section: split on "|" (or ";"), "none" is no props."""
    parts = [" ".join(part.split()).strip(" .;") for part in re.split(r"[|;]", value)]
    return [part for part in parts if part and not _NO_PROPS.match(part)]


def asset_pictures(
    clean: str,
    prop_pictures: str,
    bible: dict[str, Any],
    photos: int,
    seed: int,
    *,
    background: str | None = None,
    held: Mapping[int, Sequence[str]] | None = None,
    confirm: Callable[[int, str], bool] | None = None,
    faces: Mapping[int, Sequence[float]] | None = None,
) -> tuple[list[AssetPicture], list[str]]:
    """The pictures to draw before the keyframes, from the CLEAN and PROP PICTURES sections,
    and notes on what was left out.

    With ``background`` (the prompt of the empty place) the keyframes are put together from
    asset pictures (E16, Denys's layout): the background first, then every photo's people
    alone and cut out, then the props cut out."""
    notes: list[str] = []
    numbers = sorted(n for n in tags_in(clean) if 1 <= n <= photos)
    if background is not None:
        numbers = list(range(1, photos + 1))
    props: list[tuple[str, int | None]] = []
    for part in re.split(r"[|;]", prop_pictures):
        part = " ".join(part.split()).strip(" .")
        if not part or _NO_PROPS.match(part):
            continue
        found = re.search(r"\s*\bfrom\s+<image([1-9])>", part, re.I)
        photo = int(found.group(1)) if found else None
        words = (part[: found.start()] + part[found.end() :]).strip(" ,.") if found else part
        if photo is not None and not 1 <= photo <= photos:
            photo = None
        if photo is not None and held is not None and not _held(photo, words, held, confirm):
            # E16: the director said "from <image1>" for a cup Vlad does not hold.
            notes.append(f"PROP PICTURES: {words!r} is not held in <image{photo}>; from its words")
            photo = None
        lower = words.lower()
        known = next((p for p in bible["props"] if p.lower() == lower), None) or next(
            (p for p in bible["props"] if lower in p.lower() or p.lower() in lower), None
        )
        if known is None:
            notes.append(f"PROP PICTURES: {words!r} is not in PROPS; left out")
            continue
        if background is not None and photo is None:
            # E16: a prop no photo shows has no look to keep, and its picture was pasted as
            # drawn (a cup held by its side); the keyframe prompt's words draw it better.
            notes.append(f"PROP PICTURES: {words!r} is in no photo; drawn from its words")
            continue
        if all(known != w for w, _ in props):
            props.append((known, photo))
    face_boxes = dict(faces or {}) if background is not None else {}
    limit = (
        MAX_COMPOSE_REFERENCES - 1 - len(face_boxes) if background is not None else MAX_REFERENCES
    )
    room = max(0, min(MAX_PROP_PICTURES, limit - photos))
    if len(props) > room:
        notes.append(f"PROP PICTURES: {len(props) - room} left out (at most {room})")
        props = props[:room]
    # A prop drawn from a photo is in someone's hands there (the troll's log): that photo is
    # drawn clean too, or its grip comes back next to the prop picture (E16, real run).
    holding = {photo for _, photo in props if photo is not None}
    # Whoever really holds something is drawn empty-handed, or it stays in their cut-out.
    holding |= {photo for photo, items in (held or {}).items() if items}
    for _, photo in props:
        if photo is not None and photo not in numbers:
            numbers.append(photo)
            notes.append(f"CLEAN: <image{photo}> added, a prop picture comes from it")
    numbers.sort()
    seeds = render_seeds(seed * 31 + 17, len(numbers) + len(props) + 1)
    cut = background is not None
    pictures: list[AssetPicture] = []
    if background is not None:
        pictures.append(AssetPicture("background", None, "", seeds.pop(), background))
    for number in numbers:
        looks = "; ".join(
            f"{p['tag']}: {p['look']}" if p.get("look") else p["tag"]
            for p in bible["people"]
            if p.get("image") == number and p.get("tag")
        )
        if background is not None and number not in holding:
            # E16, Denys: the photo itself, cut out by BiRefNet, so the face stays 1:1. Only a
            # photo that holds a prop with its own picture is redrawn empty-handed by Qwen.
            pictures.append(AssetPicture("cut", number, looks, seeds.pop(0), CUT_ONLY))
            continue
        text = CLEAN_PERSON.format(looks=looks or "people or animals")
        pictures.append(AssetPicture("clean", number, looks, seeds.pop(0), text, cut))
    for photo, box in sorted(face_boxes.items()):
        pictures.append(AssetPicture("face", photo, "", 0, FACE_CROP, False, list(box)))
    for words, photo in props:
        text = (PROP_FROM_PHOTO if photo else PROP_NEW).format(words=words)
        pictures.append(AssetPicture("prop", photo, words, seeds.pop(0), text, cut))
    return pictures, notes


def framed(
    action: str, bible: dict[str, Any], *, same_framing: bool = False, tail: str = ""
) -> str:
    """A keyframe's action with the bible's camera before it and the props after it, word for
    word the same in every keyframe, so the shot size and the objects cannot drift."""
    pieces = [bible.get("camera", "")]
    if same_framing:
        pieces.append(SAME_FRAMING)
    pieces.append(action)
    if bible.get("props"):
        pieces.append(PROPS_SENTENCE.format(props="; ".join(bible["props"])))
    body = " ".join(piece.strip() for piece in pieces if piece and piece.strip())
    if tail and tail.lower() not in body.lower():
        body = f"{body} {tail}"
    return body


_QUOTE = re.compile(r'"([^"]{2,})"|“([^”]{2,})”|«([^»]{2,})»')


def _words(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.lower()))


def spoken_lines(
    idea: str, runs: Sequence[str], mapping: Mapping[str, Mapping[str, Any]]
) -> tuple[list[str], list[str]]:
    """Every line the idea puts in quotes, spoken in some run (E16: the director dropped the
    troll's line and cut the soldier's). A missing line goes into the run at the same place in
    the story, said by the person the idea names just before it."""
    runs = list(runs)
    notes: list[str] = []
    spoken = " ".join(_words(run) for run in runs)
    for match in _QUOTE.finditer(idea):
        line = next(group for group in match.groups() if group).strip()
        if not _words(line) or _words(line) in spoken:
            continue
        before = idea[: match.start()]
        tags = list(re.finditer(r"<([^<>]+)>", before))
        person = mapping.get(tags[-1].group(1).casefold()) if tags else None
        who = person["tag"] if person and person.get("tag") else "someone"
        index = min(len(runs) - 1, int(len(runs) * match.start() / max(1, len(idea))))
        runs[index] = f'{runs[index].rstrip()} {who[0].upper()}{who[1:]} says "{line}"'
        notes.append(f"RUN {index + 1}: the idea's line {line!r} was missing; added for {who}")
    return runs, notes


def keyframe_frames(counts: Sequence[int]) -> dict[str, int]:
    """Where start, middle and end sit in the whole video, as memcraft.engines.plan() guides
    them. With two runs the middle is the last frame of the first (E16: a free end let the
    camera drift into a close-up); with one run it would sit half way, on a multiple of 8. The
    end is the last frame. A second run starts from the real last frame of the first."""
    middle = counts[0] - 1 if len(counts) > 1 else 8 * round(counts[0] / 16)
    return {"start": 0, "middle": middle, "end": sum(counts) - len(counts)}


def direct_keyframes(
    run_text: TextRunner,
    prompt: PromptVersion,
    photos: Sequence[bytes],
    idea: str,
    *,
    length: JobLength = JobLength.SHORT,
    frames: int | None = None,
    seed: int = 7,
    clock: Callable[[], float] = time.monotonic,
    cast: Sequence[Mapping[str, Any]] | None = None,
    inventory: Sequence[Mapping[str, Any]] | None = None,
    confirm: Callable[[int, str], bool] | None = None,
    faces: Mapping[int, Sequence[float]] | None = None,
) -> KeyframePlan:
    """``inventory``: the checked photo inventory (E16), each entry with ``checked``: object ->
    whether SAM found it; it goes to the director and decides who is drawn empty-handed."""
    if not 1 <= len(photos) <= 3:
        raise DirectorError("the keyframes regime takes 1 to 3 photos")
    if not idea.strip():
        raise DirectorError("the director needs the approved idea")
    frames = check_frames(DEFAULT_FRAMES[length] if frames is None else frames, length)
    counts = split_runs(frames, length)
    two = len(counts) == 2
    found = parts(prompt.text, prompt_file(PromptRole.DIRECTOR_KEYFRAMES, QWEN).parts)
    system = fill(
        found["system"],
        {
            "kind": fill(
                found["kind_long" if two else "kind_short"], {"seconds": round((frames - 1) / FPS)}
            ),
            "labels": fill(found["labels_two" if two else "labels_one"]),
        },
    )
    user = _user_text(found, PromptRole.DIRECTOR_KEYFRAMES, cast, idea)
    if inventory is not None and "held" in found:
        user = f"{user}\n\n{held_text(found, inventory)}"
    labels = [*KEYFRAME_LABELS, *(f"RUN {i + 1}" for i in range(len(counts)))]
    sections, problems, raw, seconds, attempts, used = _ask(
        run_text, system, user, list(photos), labels, seed, clock, OPTIONAL_KEYFRAME_LABELS
    )
    sections = {label: image_tags(value) for label, value in sections.items()}
    count = len(photos)
    for label, value in sections.items():
        extra = sorted(tags_in(value) - set(range(1, count + 1)))
        if extra:
            problems.append(f"{label}: names <image{extra[0]}> but there are {count} photos")
    bible = {
        "people": _people(sections["PEOPLE"], with_images=True),
        "place": sections["PLACE"],
        "light": sections["LIGHT"],
        "camera": sections["CAMERA"],
        "props": props_of(sections.get("PROPS", "")),
    }
    used_cast = casting.used(cast, idea)
    if used_cast:
        # A photo may hold several people: PEOPLE is checked against the cast instead.
        mapping, notes = cast_people(cast, idea, bible["people"])
        problems += notes
        for entry in used_cast:
            person = mapping.get(entry["tag"].casefold())
            photo = entry.get("photo")
            if person is not None and photo is not None and person.get("image") != photo:
                problems.append(f"PEOPLE puts <{entry['tag']}> in the wrong photo")
    else:
        mapping = {}
        if len(bible["people"]) != count:
            problems.append(f"PEOPLE lists {len(bible['people'])} people for {count} photos")
    # With 3 people the start keyframe as a reference lost a face and added noise (E0).
    chained = (len(bible["people"]) or count) <= CHAINED_PEOPLE
    seeds = render_seeds(seed, 3 + len(counts))
    compose = "background" in found
    background = (
        fill(
            found["background"],
            {"place": bible["place"], "light": bible["light"], "camera": bible["camera"]},
        )
        if compose
        else None
    )
    assets, notes = asset_pictures(
        sections.get("CLEAN", ""),
        sections.get("PROP PICTURES", ""),
        bible,
        count,
        seeds[0],
        background=background,
        held=confirmed_holds(inventory) if compose else None,
        confirm=confirm,
        faces=faces,
    )
    problems += notes
    clean = {a.photo: f"asset:{i}" for i, a in enumerate(assets) if a.kind == "clean"}
    people_refs: list[int | str] = [clean.get(n, n) for n in range(1, count + 1)]
    prop_refs: list[int | str] = [f"asset:{i}" for i, a in enumerate(assets) if a.kind == "prop"]
    prop_words = [a.words for a in assets if a.kind == "prop"]
    face_photos = [a.photo or 0 for a in assets if a.kind == "face"]
    positions = keyframe_frames(counts)
    tail = fill(found["suffix"])
    keyframes = []
    for index, at in enumerate(("start", "middle", "end")):
        if at == "middle" and not two:
            # E16: one run is guided by the start and end keyframes only (memcraft.engines.plan);
            # the director still writes KEYFRAME MIDDLE, which helps it plan the action.
            positions.pop("middle")
            continue
        body, replaced = replace_cast_tags(
            sections[f"KEYFRAME {at.upper()}"], mapping, with_image=True
        )
        if replaced:
            problems.append(f"KEYFRAME {at.upper()}: {replaced} name tags replaced")
        body, notes = name_people(body, bible["people"], count)
        problems += [f"KEYFRAME {at.upper()}: {note}" for note in notes]
        if compose:
            # E16: every keyframe from the background, people and prop pictures; the start
            # keyframe is no reference. The director may move the camera per keyframe.
            camera, moves = camera_for(sections.get(f"CAMERA {at.upper()}", ""), bible["camera"])
            body = framed(body, {**bible, "camera": camera}, tail=tail)
            keyframes.append(
                Keyframe(
                    at=at,
                    action=body,
                    frame=positions[at],
                    seed=seeds[index],
                    references=[f"asset:{i}" for i in range(len(assets))],
                    props=prop_words,
                    layout="compose",
                    camera_moves=moves,
                    faces=face_photos,
                )
            )
            continue
        body = framed(body, bible, same_framing=at != "start" and not chained, tail=tail)
        references: list[int | str]
        if not assets:
            references = [*range(1, count + 1), *(["start"] if at != "start" and chained else [])]
        elif at == "start":
            references = [*people_refs, *prop_refs]
        elif chained:
            # E16: with asset pictures, later keyframes are edits of the start keyframe.
            references = ["start", *people_refs]
        else:
            references = list(people_refs)
        keyframes.append(
            Keyframe(
                at=at,
                action=body,
                frame=positions[at],
                seed=seeds[index],
                references=references,
                props=prop_words if at == "start" else [],
            )
        )
    starts = run_starts(counts)
    runs = []
    texts: list[str] = []
    for index, _run_count in enumerate(counts):
        text, replaced = replace_cast_tags(sections[f"RUN {index + 1}"], mapping, with_image=False)
        if replaced:
            problems.append(f"RUN {index + 1}: {replaced} name tags replaced by visual tags")
        texts.append(text)
    texts, notes = spoken_lines(idea, texts, mapping)
    problems += notes
    for index, run_count in enumerate(counts):
        runs.append(
            Run(
                prompt=run_prompt(texts[index]),
                frames=run_count,
                seed=seeds[3 + index],
                start=starts[index],
                first="keyframe" if index == 0 else "previous_last_frame",
                keyframes=[
                    (at, frame - starts[index])
                    for at, frame in positions.items()
                    if starts[index] <= frame <= starts[index] + run_count - 1
                    and not (index > 0 and frame == starts[index])
                ],
            )
        )
    return KeyframePlan(
        title=sections["TITLE"],
        bible=bible,
        keyframes=keyframes,
        frames=frames,
        runs=runs,
        seconds=seconds,
        attempts=attempts,
        seed=used,
        problems=problems,
        raw=raw,
        assets=assets,
    )
