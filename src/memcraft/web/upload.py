"""Upload and crop (E6.3): 1 to 3 photos with a 9:16 frame each become a new job.

The endpoint is async only so that it can read the body itself, after the caller is known
and with its own limits. Everything that touches the database or decodes pixels runs in a
thread, so the event loop never waits on it.
"""

from __future__ import annotations

import json
import math
import uuid
from collections.abc import AsyncGenerator
from contextlib import aclosing
from typing import Annotated, Any

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import FormData, Headers, UploadFile
from starlette.formparsers import MultiPartException, MultiPartParser

from memcraft import cast as casting
from memcraft import photos, queue
from memcraft.models import Job, JobLane, JobLength, JobQuality, JobRegime, JobStatus, User
from memcraft.photos import CropBox, FileSizeError, PhotoCountError, PhotoError
from memcraft.pipeline.regime import regime_for
from memcraft.storage import FileStore, StorageError
from memcraft.web.auth import ApprovedUser, SessionDep, StoreDep

router = APIRouter(prefix="/api")

# The idea goes whole to Gemma (llama-server context 16384 tokens; an idea call used 425 to
# 818 prompt tokens before, and 1000 characters are at most about 500 more) and to the Qwen
# director (no input limit in its text graph; only its answer has max_length 1800 tokens).
# On the CPU each 1000 characters cost Gemma about 10 s.
MAX_IDEA = 1000
# Jobs of one person that are waiting or being worked on, so that one friend cannot fill
# the queue. A preview or a closed job no longer counts.
MAX_OPEN_JOBS = 3
OPEN_STATUSES = (
    JobStatus.IDEA_QUEUED,
    JobStatus.IDEA_RUNNING,
    JobStatus.AWAITING_APPROVAL,
    JobStatus.RENDER_QUEUED,
    JobStatus.RENDERING,
)
# Every photo at its largest, plus room for the other fields and the multipart framing.
MAX_BODY = photos.MAX_PHOTOS * photos.MAX_BYTES + 1024 * 1024
MAX_FIELD_BYTES = 16 * 1024
FIELDS = ("photos", "crops", "length", "quality", "idea", "names")

REGIME_NOTES = {
    JobRegime.MULTISHOT: (
        "One photo and a short video: your photo starts the video and the model invents the rest."
    ),
    JobRegime.KEYFRAMES: (
        "Two or three photos, or a long video: key pictures are made first and then "
        "animated. It takes about 2 to 4 times longer, and look and quality can differ."
    ),
}

TOO_LARGE = f"The upload is too large. Each photo can be up to {photos.MAX_BYTES // 1024**2} MB."
TOO_MANY_JOBS = (
    f"You already have {MAX_OPEN_JOBS} videos on the way. "
    "Please wait until one of them is ready, then start the next."
)


class RegimeNote(BaseModel):
    regime: JobRegime
    note: str


class JobCreated(BaseModel):
    id: uuid.UUID
    # How many jobs of the same lane are ahead of this one: 0 means it is next.
    queue_position: int | None


@router.get("/regime-note")
def regime_note(
    user: ApprovedUser,
    photo_count: Annotated[int, Query(alias="photos", ge=1, le=photos.MAX_PHOTOS)],
    length: JobLength,
) -> RegimeNote:
    regime = regime_for(photo_count, length)
    return RegimeNote(regime=regime, note=REGIME_NOTES[regime])


@router.post("/jobs", status_code=201)
async def create_job(
    request: Request, user: ApprovedUser, session: SessionDep, store: StoreDep
) -> JobCreated:
    """Multipart: photos (1 to 3 files), crops (a JSON list of {x, y, w, h}, one per photo,
    as fractions of the photo the way the browser shows it), length, quality, idea, and
    optionally names (a JSON list, one entry per photo, of the names of its people from left
    to right, which the idea may use as <name>; see memcraft.cast)."""
    _check_content_length(request)
    user_id = await run_in_threadpool(_check_open_jobs, session, user)
    form = await _read_form(request)
    try:
        fields = _fields(form)
        return await run_in_threadpool(_create, session, store, user_id, *fields)
    finally:
        await form.close()


def _check_content_length(request: Request) -> None:
    if request.headers.get("content-type", "").split(";")[0].strip().lower() != (
        "multipart/form-data"
    ):
        raise HTTPException(400, "Send the photos as a multipart form.")
    declared = request.headers.get("content-length")
    if declared is not None:
        try:
            size = int(declared)
        except ValueError:
            raise HTTPException(400, "The upload could not be read.") from None
        if size > MAX_BODY:
            raise HTTPException(413, TOO_LARGE)


def _check_open_jobs(session: Session, user: User) -> uuid.UUID:
    """A quick answer before the upload is read. _create() checks again under a lock."""
    user_id = user.id
    too_many = _open_jobs(session, user_id) >= MAX_OPEN_JOBS
    session.rollback()  # no transaction stays open while the photos arrive
    if too_many:
        raise HTTPException(429, TOO_MANY_JOBS)
    return user_id


def _open_jobs(session: Session, user_id: uuid.UUID) -> int:
    return (
        session.scalar(
            select(func.count())
            .select_from(Job)
            .where(Job.owner_id == user_id, Job.status.in_(OPEN_STATUSES))
        )
        or 0
    )


class _BodyTooLargeError(Exception):
    pass


class _Parser(MultiPartParser):
    """Starlette's parser, which also stops at the first photo too many or too large."""

    def __init__(self, headers: Headers, stream: AsyncGenerator[bytes]) -> None:
        super().__init__(
            headers,
            stream,
            max_files=photos.MAX_PHOTOS + 1,
            max_fields=len(FIELDS),
            max_part_size=MAX_FIELD_BYTES,
        )
        self._files = 0
        self._part_bytes = 0

    def on_part_begin(self) -> None:
        super().on_part_begin()
        self._part_bytes = 0

    def on_headers_finished(self) -> None:
        super().on_headers_finished()
        if self._current_part.file is not None:
            self._files += 1
            if self._files > photos.MAX_PHOTOS:
                raise PhotoCountError

    def on_part_data(self, data: bytes, start: int, end: int) -> None:
        if self._current_part.file is not None:
            self._part_bytes += end - start
            if self._part_bytes > photos.MAX_BYTES:
                error = FileSizeError()
                error.photo = self._files
                raise error
        super().on_part_data(data, start, end)


async def _limited(request: Request) -> AsyncGenerator[bytes]:
    """The body, but never more than MAX_BODY bytes of it, whatever Content-Length said."""
    total = 0
    async with aclosing(request.stream()) as stream:
        async for chunk in stream:
            total += len(chunk)
            if total > MAX_BODY:
                raise _BodyTooLargeError
            yield chunk


async def _read_form(request: Request) -> FormData:
    """The parsed form. On any error the parser has closed its temporary files already."""
    try:
        async with aclosing(_limited(request)) as stream:
            return await _Parser(request.headers, stream).parse()
    except _BodyTooLargeError:
        raise HTTPException(413, TOO_LARGE) from None
    except FileSizeError as error:
        raise HTTPException(413, _photo_message(error)) from None
    except PhotoError as error:
        raise HTTPException(422, _photo_message(error)) from None
    except MultiPartException:
        raise HTTPException(400, "The upload could not be read.") from None


type _Fields = tuple[
    list[UploadFile], list[CropBox], JobLength, JobQuality, str | None, casting.Cast | None
]


def _fields(form: FormData) -> _Fields:
    unknown = set(form.keys()) - set(FIELDS)
    if unknown:
        raise HTTPException(422, f"Unexpected field: {sorted(unknown)[0]}.")
    files = form.getlist("photos")
    if not files or not all(isinstance(file, UploadFile) for file in files):
        raise HTTPException(422, PhotoCountError.message)
    crops = _crops(_one(form, "crops", "the crop frames"), len(files))
    length = _choice(form, "length", JobLength, "short or long")
    quality = _choice(form, "quality", JobQuality, "720p or 1080p")
    idea = _one(form, "idea", "the idea", required=False)
    # A browser sends the line breaks of a text box as CRLF; its maxlength counted them as one.
    idea = (idea or "").replace("\r\n", "\n").replace("\r", "\n").strip() or None
    if idea is not None and len(idea) > MAX_IDEA:
        raise HTTPException(422, f"Please keep the idea to {MAX_IDEA} characters at most.")
    return files, crops, length, quality, idea, _cast(form, len(files), idea)


def _cast(form: FormData, count: int, idea: str | None) -> casting.Cast | None:
    raw = _one(form, "names", "the names", required=False)
    try:
        names = json.loads(raw) if raw else None
    except ValueError:
        raise HTTPException(422, "The names could not be read.") from None
    try:
        return casting.build_cast(casting.parse_names(names, count), idea)
    except casting.CastError as error:
        raise HTTPException(422, str(error)) from None


def _one(form: FormData, name: str, what: str, *, required: bool = True) -> str | None:
    values = form.getlist(name)
    if not values and not required:
        return None
    if len(values) != 1 or not isinstance(values[0], str):
        raise HTTPException(422, f"Send {what} once.")
    return values[0]


def _choice[T: (JobLength, JobQuality)](
    form: FormData, name: str, kind: type[T], allowed: str
) -> T:
    value = _one(form, name, f"the {name}")
    try:
        return kind(value)
    except ValueError:
        raise HTTPException(422, f"The {name} must be {allowed}.") from None


def _crops(raw: str | None, count: int) -> list[CropBox]:
    try:
        items: Any = json.loads(raw or "")
    except ValueError:
        raise HTTPException(422, "The crop frames could not be read.") from None
    if not isinstance(items, list) or len(items) != count:
        raise HTTPException(422, "Send one crop frame for each photo.")
    crops = []
    for number, item in enumerate(items, 1):
        values = [item.get(key) for key in "xywh"] if isinstance(item, dict) else []
        if len(values) != 4 or not all(_is_number(value) for value in values):
            raise HTTPException(422, f"Photo {number}: the crop frame could not be read.")
        crops.append(CropBox(*map(float, values)))
    return crops


def _is_number(value: object) -> bool:
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _photo_message(error: PhotoError) -> str:
    return f"Photo {error.photo}: {error}" if error.photo else str(error)


def _create(
    session: Session,
    store: FileStore,
    user_id: uuid.UUID,
    files: list[UploadFile],
    crops: list[CropBox],
    length: JobLength,
    quality: JobQuality,
    idea: str | None,
    cast: casting.Cast | None,
) -> JobCreated:
    """The job and its photos in one transaction: all of it or nothing."""
    try:
        # One upload at a time per person, so two at once cannot both pass the limit.
        # FOR NO KEY UPDATE: it does not stop the worker saving files of this person.
        session.execute(
            select(User.id).where(User.id == user_id).with_for_update(key_share=True)
        ).one()
        if _open_jobs(session, user_id) >= MAX_OPEN_JOBS:
            raise HTTPException(429, TOO_MANY_JOBS)
        job = Job(
            owner_id=user_id,
            idea_text=idea,
            cast=cast,
            photo_count=len(files),
            length=length,
            quality=quality,
            lane=JobLane.IDEA,
            status=JobStatus.IDEA_QUEUED,
            regime=regime_for(len(files), length),
        )
        session.add(job)
        session.flush()
        job_id = job.id
        uploads = [(file.file, crop) for file, crop in zip(files, crops, strict=True)]
        photos.intake(session, store, owner_id=user_id, job_id=job_id, uploads=uploads)
        session.commit()
    except PhotoError as error:
        session.rollback()
        raise HTTPException(422, _photo_message(error)) from None
    except StorageError:
        session.rollback()
        raise HTTPException(503, "Photos cannot be saved right now. Please try later.") from None
    except BaseException:
        session.rollback()
        raise
    return JobCreated(id=job_id, queue_position=queue.queue_position(session, job_id))
