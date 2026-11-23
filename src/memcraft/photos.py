"""Photo intake: an untrusted upload becomes a clean 9:16 PNG, and later a small copy for Gemma.

intake() runs in the web process and make_gemma_copies() in the worker, because the copies
go to the private folder. Both leave the commit to the caller, as FileStore.save() does.

The saved photo is a PNG: the upload was usually lossy once already, the generators should
not see a second loss, and the file does not live long. It is written from bare pixels, so
nothing the camera or an editor put into the upload survives.
"""

from __future__ import annotations

import io
import math
import uuid
from collections.abc import Mapping, Sequence
from contextlib import suppress
from datetime import datetime
from typing import BinaryIO, NamedTuple

from PIL import Image, ImageCms
from sqlalchemy import select
from sqlalchemy.orm import Session

from memcraft.models import Asset, AssetKind, Job
from memcraft.storage import FileStore

MAX_PHOTOS = 3
MAX_BYTES = 20 * 1024 * 1024
MAX_PIXELS = 60_000_000

# A crop is always 9k x 16k pixels.
MIN_CROP = (288, 512)
MAX_OUTPUT = (2160, 3840)
ASPECT_TOLERANCE = 0.01

# E0.13: 360x640 keeps one photo near 10 s; 2 or 3 photos need 540x960 with a token cap.
GEMMA_COPY_SIZES: Mapping[int, tuple[int, int]] = {1: (360, 640), 2: (540, 960), 3: (540, 960)}
GEMMA_COPY_QUALITY = 90

PHOTO_MIME = "image/png"
GEMMA_COPY_MIME = "image/jpeg"

_MAGIC = {"PNG": b"\x89PNG\r\n\x1a\n", "JPEG": b"\xff\xd8\xff"}
_READ = 1024 * 1024
_ORIENTATION = 0x0112
_ROUNDING = 1e-6  # for fractions the browser computed in floating point
_WHITE = (255, 255, 255)
_TURNS = {
    2: Image.Transpose.FLIP_LEFT_RIGHT,
    3: Image.Transpose.ROTATE_180,
    4: Image.Transpose.FLIP_TOP_BOTTOM,
    5: Image.Transpose.TRANSPOSE,
    6: Image.Transpose.ROTATE_270,
    7: Image.Transpose.TRANSVERSE,
    8: Image.Transpose.ROTATE_90,
}

type _Box = tuple[int, int, int, int]


class PhotoError(Exception):
    """The user's photos cannot be used. str() of it is safe to show to the user."""

    message = "This photo cannot be used."
    # Which photo of the upload it was, starting at 1; set by intake().
    photo: int | None = None

    def __init__(self, message: str | None = None) -> None:
        super().__init__(message or self.message)


class PhotoCountError(PhotoError):
    message = f"Add 1 to {MAX_PHOTOS} photos."


class FileSizeError(PhotoError):
    message = f"This photo is larger than {MAX_BYTES // 1024**2} MB."


class PixelCountError(PhotoError):
    message = f"This photo has more than {MAX_PIXELS // 1_000_000} megapixels."


class NotAPhotoError(PhotoError):
    message = "Only PNG and JPEG photos are accepted."


class AnimatedPhotoError(PhotoError):
    message = "Animated pictures are not accepted."


class DamagedPhotoError(PhotoError):
    message = "This photo is damaged and cannot be read."


class CropError(PhotoError):
    message = "The crop must lie inside the photo and have the shape 9:16."


class CropTooSmallError(CropError):
    message = f"The crop must be at least {MIN_CROP[0]} x {MIN_CROP[1]} pixels."


class CropBox(NamedTuple):
    """Left, top, width and height as fractions of the photo the way the browser shows it,
    which is after its EXIF orientation."""

    x: float
    y: float
    w: float
    h: float


class PreparedPhoto(NamedTuple):
    data: bytes
    mime: str
    width: int
    height: int


def crop_pixels(crop: CropBox, width: int, height: int) -> _Box:
    """Left, top, right and bottom of the largest exact 9:16 box in the middle of `crop`."""
    try:
        x, y, w, h = (float(value) for value in crop)
    except (TypeError, ValueError, OverflowError):
        raise CropError from None
    if not all(map(math.isfinite, (x, y, w, h))) or min(x, y) < 0 or min(w, h) <= 0:
        raise CropError
    if max(x + w, y + h) > 1 + _ROUNDING:
        raise CropError
    pixels_wide, pixels_high = w * width, h * height
    if abs(pixels_wide * 16 / (pixels_high * 9) - 1) > ASPECT_TOLERANCE + _ROUNDING:
        raise CropError
    unit = math.floor(min(pixels_wide / 9, pixels_high / 16) + _ROUNDING)
    unit = min(unit, width // 9, height // 16)
    if unit * 9 < MIN_CROP[0]:
        raise CropTooSmallError
    wide, high = unit * 9, unit * 16
    left = min(max(round(x * width + (pixels_wide - wide) / 2), 0), width - wide)
    top = min(max(round(y * height + (pixels_high - high) / 2), 0), height - high)
    return left, top, left + wide, top + high


def prepare_photo(source: BinaryIO, crop: CropBox) -> PreparedPhoto:
    """Check, crop and re-encode one upload. Touches neither the disk nor the database.

    The type comes from the bytes alone. Of a JPEG with several pictures inside (MPO, which
    many phones write) only the first, ordinary one is used. An animated PNG is refused.
    Slow for a large photo: call it from a thread, not from an event loop.
    """
    image = _open(_read(source))
    image = _cropped(image, crop)
    image = _to_srgb(image)
    target = io.BytesIO()
    _bare(image).save(target, "PNG")
    return PreparedPhoto(target.getvalue(), PHOTO_MIME, *image.size)


def intake(
    session: Session,
    store: FileStore,
    *,
    owner_id: uuid.UUID,
    job_id: uuid.UUID,
    uploads: Sequence[tuple[BinaryIO, CropBox]],
    expires_at: datetime | None = None,
) -> list[Asset]:
    """Save the photos of a new job, in their order. All of them or none; the caller commits.

    Nothing is written before every photo has passed, so a PhotoError leaves no trace.
    """
    if not 1 <= len(uploads) <= MAX_PHOTOS:
        raise PhotoCountError
    session.flush()  # the job may still be pending
    job = session.get(Job, job_id)
    if job is None or job.owner_id != owner_id or job.photo_count != len(uploads):
        raise ValueError("the job does not match these photos")
    if list_photos(session, job_id):
        raise ValueError("the job already has its photos")
    prepared = []
    for number, (source, crop) in enumerate(uploads, 1):
        try:
            prepared.append(prepare_photo(source, crop))
        except PhotoError as error:
            error.photo = number
            raise
    assets = []
    with session.begin_nested():  # so that a caller who goes on keeps no half of them
        for index, photo in enumerate(prepared):
            asset = store.save(
                session,
                owner_id=owner_id,
                job_id=job_id,
                kind=AssetKind.PHOTO,
                ordinal=index,
                mime=photo.mime,
                source=io.BytesIO(photo.data),
                expires_at=expires_at,
            )
            assets.append(asset)
    return assets


def list_photos(session: Session, job_id: uuid.UUID) -> list[Asset]:
    """The full-size photos of a job, in the order the user gave them."""
    return _assets(session, job_id, AssetKind.PHOTO)


def make_gemma_copies(
    session: Session,
    store: FileStore,
    job_id: uuid.UUID,
    *,
    sizes: Mapping[int, tuple[int, int]] = GEMMA_COPY_SIZES,
    quality: int = GEMMA_COPY_QUALITY,
) -> list[Asset]:
    """One small JPEG for every photo of the job, in the same order. The caller commits.

    `sizes` is by the number of photos. A copy that exists already is kept, whatever its
    size. Only the worker that has claimed the job may call this: two at once would both
    make copies.
    """
    photos = list_photos(session, job_id)
    if not photos:
        raise ValueError("the job has no photos")
    size = sizes[len(photos)]
    copies = {copy.ordinal: copy for copy in _assets(session, job_id, AssetKind.GEMMA_COPY)}
    result = []
    with session.begin_nested():
        for photo in photos:
            copy = copies.get(photo.ordinal)
            if copy is None:
                copy = store.save(
                    session,
                    owner_id=photo.owner_id,
                    job_id=job_id,
                    kind=AssetKind.GEMMA_COPY,
                    ordinal=photo.ordinal,
                    mime=GEMMA_COPY_MIME,
                    source=_small_jpeg(store, photo, size, quality),
                    expires_at=photo.expires_at,
                )
            result.append(copy)
    return result


def _assets(session: Session, job_id: uuid.UUID, kind: AssetKind) -> list[Asset]:
    return list(
        session.scalars(
            select(Asset)
            .where(Asset.job_id == job_id, Asset.kind == kind)
            .order_by(Asset.ordinal, Asset.id)
        )
    )


def _small_jpeg(store: FileStore, photo: Asset, size: tuple[int, int], quality: int) -> BinaryIO:
    with Image.open(store.path_for(photo), formats=["PNG"]) as image:
        image.thumbnail(size, Image.Resampling.LANCZOS, reducing_gap=None)  # never enlarges
        target = io.BytesIO()
        _bare(image).save(target, "JPEG", quality=quality)
    target.seek(0)
    return target


def _read(source: BinaryIO) -> bytes:
    data = bytearray()
    while chunk := source.read(_READ):
        data += chunk
        if len(data) > MAX_BYTES:
            raise FileSizeError
    return bytes(data)


def _open(data: bytes) -> Image.Image:
    """A fully decoded image whose size was checked before the decoding."""
    formats = [name for name, magic in _MAGIC.items() if data.startswith(magic)]
    if not formats:
        raise NotAPhotoError
    # Any error in here comes from the bytes, and Pillow raises a dozen kinds of them.
    try:
        image = Image.open(io.BytesIO(data), formats=formats)
        if image.width * image.height > MAX_PIXELS:
            raise PixelCountError
        if image.format == "PNG" and getattr(image, "is_animated", False):
            raise AnimatedPhotoError
        if str(image.tile[0].args).endswith("16B"):
            # A 16-bit colour key means nothing once the samples are 8-bit.
            image.info.pop("transparency", None)
        image.load()
    except (PhotoError, MemoryError):
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise PixelCountError from None
    except Exception as error:
        raise DamagedPhotoError from error
    return image


def _orientation(image: Image.Image) -> int:
    # Not Image.getexif(): that falls back to XMP, which browsers ignore, and the crop box
    # was drawn on what the browser showed.
    exif = Image.Exif()
    with suppress(Exception):
        exif.load(image.info.get("exif", b""))
        value = exif.get(_ORIENTATION, 1)
        if value in _TURNS:
            return int(value)
    return 1


def _cropped(image: Image.Image, crop: CropBox) -> Image.Image:
    """The upright crop. Only the crop is turned, so a large photo is not copied whole."""
    orientation = _orientation(image)
    width, height = image.size
    if orientation >= 5:
        width, height = height, width
    box = crop_pixels(crop, width, height)
    image = image.crop(_stored_box(box, orientation, width, height))
    turn = _TURNS.get(orientation)
    return image if turn is None else image.transpose(turn)


def _stored_box(box: _Box, orientation: int, width: int, height: int) -> _Box:
    """Where a box of the upright photo of width x height lies in the stored pixels."""
    left, top, right, bottom = box
    from_right, from_bottom = (width - right, width - left), (height - bottom, height - top)
    (x0, x1), (y0, y1) = {
        1: ((left, right), (top, bottom)),
        2: (from_right, (top, bottom)),
        3: (from_right, from_bottom),
        4: ((left, right), from_bottom),
        5: ((top, bottom), (left, right)),
        6: ((top, bottom), from_right),
        7: (from_bottom, from_right),
        8: (from_bottom, (left, right)),
    }[orientation]
    return x0, y0, x1, y1


def _to_srgb(image: Image.Image) -> Image.Image:
    """RGB in sRGB, not above MAX_OUTPUT, with transparent parts on white."""
    profile = image.info.get("icc_profile")
    if image.mode.startswith("I"):  # 16-bit grey
        image = image.convert("I").point(lambda value: value * (1 / 257) + 0.5).convert("L")
    if image.has_transparency_data:
        image = image.convert("LA" if image.mode in ("1", "L", "LA") else "RGBA")
    elif image.mode not in ("L", "RGB", "CMYK"):
        image = image.convert("L" if image.mode == "1" else "RGB")
    if image.width > MAX_OUTPUT[0]:
        # Before the rest, so that the rest works on a bounded image.
        image = image.resize(MAX_OUTPUT, Image.Resampling.LANCZOS)
    alpha = None
    if image.mode in ("LA", "RGBA"):
        *bands, alpha = image.split()
        image = bands[0] if len(bands) == 1 else Image.merge("RGB", bands)
    if profile:
        image = _apply_profile(image, profile)
    if image.mode != "RGB":
        image = image.convert("RGB")
    if alpha is not None:
        image = Image.composite(image, Image.new("RGB", image.size, _WHITE), alpha)
    return image


def _apply_profile(image: Image.Image, profile: bytes) -> Image.Image:
    """Unchanged if the profile is broken or is not for this kind of image."""
    try:
        source = ImageCms.ImageCmsProfile(io.BytesIO(profile))
        space = source.profile.xcolor_space.strip()
        if space == "RGB" and image.mode == "L":
            image = image.convert("RGB")
        if {"RGB": "RGB", "GRAY": "L", "CMYK": "CMYK"}.get(space) != image.mode:
            return image
        target = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB"))
        return ImageCms.profileToProfile(image, source, target, outputMode="RGB")
    except (ImageCms.PyCMSError, OSError, TypeError, ValueError):
        return image


def _bare(image: Image.Image) -> Image.Image:
    """The pixels alone: Pillow writes some of image.info back when it saves."""
    bare = Image.new("RGB", image.size)
    bare.paste(image)
    return bare
