"""Face boxes for the crop warning on the New video page.

A friend who crops a landscape group photo to 9:16 can cut someone off without noticing.
The page asks for the faces once per photo and then checks them against the frame on every
move. Browsers with the FaceDetector API do it themselves; the others send the photo here.

The photo is the raw request body (image/png or image/jpeg), read into memory with the same
limits as an upload, never written anywhere and dropped when the answer is sent. Faces come
from three of OpenCV's bundled Haar cascades, and a face counts only when all three find it:
one or two alone also find collars, buckets and dogs. A face that is missed only means a
warning that does not appear.
"""

from __future__ import annotations

import io
import threading
from collections.abc import Sequence
from functools import cache
from typing import Any, NamedTuple

from fastapi import APIRouter, HTTPException, Request
from PIL import Image
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from memcraft import photos
from memcraft.photos import (
    DamagedPhotoError,
    FileSizeError,
    NotAPhotoError,
    PhotoError,
    PixelCountError,
)
from memcraft.web.auth import ApprovedUser

router = APIRouter(prefix="/api")

TYPES = {"image/png": "PNG", "image/jpeg": "JPEG"}
# The detectors see the photo at most this large: enough for a face in a group photo.
DETECT_SIDE = 1024
CASCADES = (
    "haarcascade_frontalface_alt.xml",
    "haarcascade_frontalface_alt2.xml",
    "haarcascade_frontalface_default.xml",
)
VOTES = 3
SAME_FACE = 0.3  # intersection over union of two boxes of one face
MIN_FACE = 0.03  # of the shorter side, and never under 24 pixels
UNAVAILABLE = "The face check is not available right now."
# One detection at a time: decoding and the cascades take memory and CPU, and a cascade
# object is not safe to share between threads.
_busy = threading.Lock()


class FaceBox(BaseModel):
    """A face as fractions of the upright photo, the way the browser shows it."""

    x: float
    y: float
    w: float
    h: float


class Faces(BaseModel):
    faces: list[FaceBox]


class _Box(NamedTuple):
    x: float
    y: float
    w: float
    h: float


class DetectorUnavailableError(Exception):
    pass


@router.post("/faces")
async def find_faces(request: Request, user: ApprovedUser) -> Faces:
    """Body: one PNG or JPEG photo. Answer: its faces, left to right."""
    kind = request.headers.get("content-type", "").split(";")[0].strip().lower()
    if kind not in TYPES:
        raise HTTPException(415, "Send one PNG or JPEG photo.")
    declared = request.headers.get("content-length")
    if declared is not None:
        try:
            size = int(declared)
        except ValueError:
            raise HTTPException(400, "The photo could not be read.") from None
        if size > photos.MAX_BYTES:
            raise HTTPException(413, FileSizeError.message)
    data = bytearray()
    async for chunk in request.stream():
        data += chunk
        if len(data) > photos.MAX_BYTES:
            raise HTTPException(413, FileSizeError.message)
    try:
        boxes = await run_in_threadpool(faces_in, bytes(data))
    except PhotoError as error:
        raise HTTPException(422, str(error)) from None
    except DetectorUnavailableError:
        raise HTTPException(503, UNAVAILABLE) from None
    return Faces(faces=[FaceBox(**box._asdict()) for box in boxes])


def faces_in(data: bytes) -> list[_Box]:
    """The faces in one photo as fractions of the upright photo, left to right."""
    with _busy:
        image = _upright_gray(data)
        found = _detect(image)
    width, height = image.size
    boxes = [
        _Box(
            round(box.x / width, 4),
            round(box.y / height, 4),
            round(box.w / width, 4),
            round(box.h / height, 4),
        )
        for box in found
    ]
    return sorted(boxes)


def _upright_gray(data: bytes) -> Image.Image:
    """The photo in grey, at most DETECT_SIDE pixels on its long side, turned upright."""
    formats = [name for name, magic in photos._MAGIC.items() if data.startswith(magic)]
    if not formats:
        raise NotAPhotoError
    try:
        image = Image.open(io.BytesIO(data), formats=formats)
        if image.width * image.height > photos.MAX_PIXELS:
            raise PixelCountError
        orientation = photos._orientation(image)
        if image.format == "JPEG":
            image.draft("L", (DETECT_SIDE, DETECT_SIDE))  # decodes a large JPEG smaller
        image = image.convert("L")
        image.thumbnail((DETECT_SIDE, DETECT_SIDE), Image.Resampling.BILINEAR)
    except (PhotoError, MemoryError):
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise PixelCountError from None
    except Exception as error:
        raise DamagedPhotoError from error
    turn = photos._TURNS.get(orientation)
    return image if turn is None else image.transpose(turn)


@cache
def _cascades() -> tuple[Any, ...]:
    try:
        import cv2  # optional until the image has it
    except ImportError as error:
        raise DetectorUnavailableError from error
    loaded = tuple(cv2.CascadeClassifier(cv2.data.haarcascades + name) for name in CASCADES)
    if any(cascade.empty() for cascade in loaded):
        raise DetectorUnavailableError
    return loaded


def _detect(image: Image.Image) -> list[_Box]:
    import numpy  # comes with OpenCV

    cascades = _cascades()
    pixels = numpy.asarray(image)
    smallest = max(24, round(MIN_FACE * min(image.size)))
    found: list[tuple[int, _Box]] = []
    for number, cascade in enumerate(cascades):
        boxes = cascade.detectMultiScale(
            pixels, scaleFactor=1.08, minNeighbors=3, minSize=(smallest, smallest)
        )
        found += [(number, _Box(*map(float, box))) for box in boxes]
    return _agreed(found, len(cascades))


def _agreed(found: Sequence[tuple[int, _Box]], cascades: int) -> list[_Box]:
    """Boxes of one face grouped together; a group counts when VOTES cascades found it."""
    groups: list[list[tuple[int, _Box]]] = []
    for item in found:
        for group in groups:
            if any(_overlap(item[1], other) >= SAME_FACE for _, other in group):
                group.append(item)
                break
        else:
            groups.append([item])
    faces = []
    for group in groups:
        if len({number for number, _ in group}) >= min(VOTES, cascades):
            boxes = [box for _, box in group]
            faces.append(_Box(*(sum(values) / len(boxes) for values in zip(*boxes, strict=True))))
    return faces


def _overlap(a: _Box, b: _Box) -> float:
    width = min(a.x + a.w, b.x + b.w) - max(a.x, b.x)
    height = min(a.y + a.h, b.y + b.h) - max(a.y, b.y)
    if width <= 0 or height <= 0:
        return 0.0
    both = width * height
    return both / (a.w * a.h + b.w * b.h - both)
