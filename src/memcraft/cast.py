"""The people an idea names: short names under each photo and <name> tags in the idea.

Under each photo the friend may type who is in it, left to right ("denys, maksym"), and then
write <denys> and <maksym> in the idea. <person1>, <person2>, ... work without names: the
people in reading order, photo 1 from left to right, then photo 2 and so on.

A job stores its cast as a list of {"tag", "photo", "position"}: photo and position count from
1, the position from the left inside the crop. A <personN> tag is stored with photo and
position when every photo has names (so the reading order is known), else with photo None and
the reading-order number as its position. Ideas without tags get no cast in the prompts, so
they reach the models exactly as before.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

MAX_NAME = 20
MAX_PER_PHOTO = 5
MAX_PEOPLE = 15  # <person1> to <person15>: 3 photos of 5 people
# A tag in the idea: <name>, the name as letters, digits, "-" and "_".
TAG = re.compile(r"<([^\W_][\w-]{0,40})>")
_GENERIC = re.compile(r"person(\d+)", re.I)
# Names the app keeps for itself: <personN> and the director's picture tags <imageN>.
_RESERVED = re.compile(r"(?:person|image)\s*\d+", re.I)

type Cast = list[dict[str, Any]]


class CastError(ValueError):
    """A name or tag the friend can fix; the message says how."""


def _is_name(name: str) -> bool:
    return (
        1 <= len(name) <= MAX_NAME
        and name[0].isalnum()
        and all(c.isalnum() or c in "-_" for c in name)
    )


def parse_names(raw: Any, photo_count: int) -> list[list[str]]:
    """The names per photo from the form: a list (one entry per photo) of lists of names, or
    of comma-separated strings. Empty entries mean no names for that photo."""
    if raw is None:
        return [[] for _ in range(photo_count)]
    if not isinstance(raw, list) or len(raw) != photo_count:
        raise CastError("Send the names for each photo once.")
    result: list[list[str]] = []
    seen: dict[str, int] = {}
    for number, entry in enumerate(raw, 1):
        if isinstance(entry, str):
            entry = entry.split(",")
        if not isinstance(entry, list) or not all(isinstance(name, str) for name in entry):
            raise CastError(f"Photo {number}: the names could not be read.")
        names = [name.strip().lstrip("<").rstrip(">").strip() for name in entry]
        names = [name for name in names if name]
        if len(names) > MAX_PER_PHOTO:
            raise CastError(f"Photo {number}: name at most {MAX_PER_PHOTO} people.")
        for name in names:
            if not _is_name(name):
                raise CastError(
                    f'Photo {number}: "{name[:40]}" cannot be a name. Use up to {MAX_NAME} '
                    "letters or digits, and - or _ (not first), without spaces."
                )
            if _RESERVED.fullmatch(name):
                raise CastError(
                    f'Photo {number}: "{name}" is kept for the app. Please choose another name.'
                )
            key = name.casefold()
            if key in seen:
                raise CastError(f'The name "{name}" is used twice. Each name must be different.')
            seen[key] = number
        result.append(names)
    return result


def tags_in(idea: str | None) -> list[str]:
    """The tags of an idea in order of first use, as written."""
    found: dict[str, str] = {}
    for match in TAG.finditer(idea or ""):
        found.setdefault(match.group(1).casefold(), match.group(1))
    return list(found.values())


def build_cast(names: Sequence[Sequence[str]], idea: str | None) -> Cast | None:
    """The cast of a job: every named person, then every <personN> the idea uses.

    Raises CastError for a tag in the idea that names no one.
    """
    cast: Cast = [
        {"tag": name, "photo": photo, "position": position}
        for photo, photo_names in enumerate(names, 1)
        for position, name in enumerate(photo_names, 1)
    ]
    named = list(cast)
    known = {entry["tag"].casefold() for entry in cast}
    every_photo_named = bool(names) and all(names)
    unknown = []
    for tag in tags_in(idea):
        if tag.casefold() in known:
            continue
        generic = _GENERIC.fullmatch(tag)
        number = int(generic.group(1)) if generic else 0
        if not 1 <= number <= (len(named) if every_photo_named else MAX_PEOPLE):
            unknown.append(tag)
            continue
        if every_photo_named:
            person = named[number - 1]
            cast.append({"tag": tag, "photo": person["photo"], "position": person["position"]})
        else:
            cast.append({"tag": tag, "photo": None, "position": number})
        known.add(tag.casefold())
    if unknown:
        raise CastError(unknown_message(unknown, cast))
    return cast or None


def unknown_message(unknown: Sequence[str], cast: Cast) -> str:
    listed = ", ".join(f"<{tag}>" for tag in unknown)
    names = [e["tag"] for e in cast if e["photo"] is not None and not _GENERIC.fullmatch(e["tag"])]
    if names:
        known = ", ".join(f"<{name}>" for name in dict.fromkeys(names))
        return (
            f"The idea names {listed}, but no one in the photos has that name. "
            f"The names you gave are {known}; you can also write <person1>, <person2>, …"
        )
    return (
        f"The idea names {listed}, but no one in the photos has that name. Type the names "
        "under the photos, or write <person1>, <person2>, … for the people from left to right."
    )


def used(cast: Sequence[Mapping[str, Any]] | None, idea: str | None) -> Cast:
    """The members of the cast that the idea names, in cast order. Empty without tags."""
    tags = {tag.casefold() for tag in tags_in(idea)}
    return [dict(entry) for entry in cast or () if str(entry.get("tag", "")).casefold() in tags]


def _ordinal(number: int) -> str:
    if 10 <= number % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(number % 10, "th")
    return f"{number}{suffix}"


def _in_photo(cast: Sequence[Mapping[str, Any]], photo: int) -> int:
    """How many people of a photo the cast places: the most a position goes up to."""
    return max((e["position"] for e in cast if e.get("photo") == photo), default=0)


_READING = "counting the people of photo 1 from left to right, then photo 2, then photo 3"


def chat_lines(cast: Sequence[Mapping[str, Any]] | None, idea: str | None) -> list[str]:
    """For Gemma: a line per person the idea names, like
    "<denys> = the person on the left in photo 1". Empty without tags."""
    cast = list(cast or ())
    lines = []
    for entry in used(cast, idea):
        tag, photo, position = entry["tag"], entry.get("photo"), entry["position"]
        if photo is None:
            lines.append(f"<{tag}> = the {_ordinal(position)} person, {_READING}")
            continue
        count = _in_photo(cast, photo)
        if count <= 1:
            where = f"the person in photo {photo}"
        elif count == 2:
            where = f"the person on the {('left', 'right')[position - 1]} in photo {photo}"
        elif count == 3:
            side = ("on the left", "in the middle", "on the right")[position - 1]
            where = f"the person {side} in photo {photo}"
        else:
            where = f"the {_ordinal(position)} person from the left in photo {photo}"
        lines.append(f"<{tag}> = {where}")
    return lines


def director_lines(cast: Sequence[Mapping[str, Any]] | None, idea: str | None) -> list[str]:
    """For the Qwen director, a line per person the idea names, like
    "<denys> = <image1>, the 1st of 2 people from the left"."""
    cast = list(cast or ())
    lines = []
    for entry in used(cast, idea):
        tag, photo, position = entry["tag"], entry.get("photo"), entry["position"]
        if photo is None:
            lines.append(f"<{tag}> = the {_ordinal(position)} person, {_READING}")
            continue
        count = _in_photo(cast, photo)
        if count <= 1:
            where = f"<image{photo}>, the person in it (the leftmost if there are several)"
        else:
            where = f"<image{photo}>, the {_ordinal(position)} of {count} people from the left"
        lines.append(f"<{tag}> = {where}")
    return lines
