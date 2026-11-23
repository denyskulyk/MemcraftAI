import io
import math
import random
import struct
import uuid
import zlib
from datetime import UTC, datetime, timedelta

import pytest
from PIL import Image, ImageChops, ImageCms, ImageOps, PngImagePlugin
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from memcraft import photos
from memcraft.models import Asset, AssetKind, AssetStore, Job, JobLength, JobQuality, User
from memcraft.photos import (
    AnimatedPhotoError,
    CropBox,
    CropError,
    CropTooSmallError,
    DamagedPhotoError,
    FileSizeError,
    NotAPhotoError,
    PhotoCountError,
    PhotoError,
    PixelCountError,
    crop_pixels,
    intake,
    list_photos,
    make_gemma_copies,
    prepare_photo,
)
from memcraft.storage import FileStore, StoreUnavailableError

FULL = CropBox(0, 0, 1, 1)
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


@pytest.fixture
def roots(tmp_path):
    shared, private = tmp_path / "shared", tmp_path / "private"
    shared.mkdir()
    private.mkdir()
    return shared, private


@pytest.fixture
def store(roots):
    return FileStore(*roots)


@pytest.fixture
def web(roots):
    return FileStore(roots[0], None)


@pytest.fixture
def session(engine):
    with Session(engine) as session:
        yield session


def files(*folders):
    return [path for folder in folders for path in folder.rglob("*") if path.is_file()]


def new_job(session, photo_count=1):
    user = User(tailscale_login=uuid.uuid4().hex)
    session.add(user)
    session.flush()
    job = Job(
        owner_id=user.id,
        photo_count=photo_count,
        length=JobLength.SHORT,
        quality=JobQuality.HD720,
    )
    session.add(job)
    session.commit()
    return job


def take(session, store, job, uploads, **kw):
    return intake(session, store, owner_id=job.owner_id, job_id=job.id, uploads=list(uploads), **kw)


def count(session, kind):
    return session.scalar(select(func.count()).select_from(Asset).where(Asset.kind == kind))


def scene(width=360, height=640):
    """Every place has its own colour, so a wrong crop or turn shows."""
    across = Image.linear_gradient("L").transpose(Image.Transpose.ROTATE_90)
    down = Image.linear_gradient("L")
    corner = Image.new("L", (width, height), 40)
    corner.paste(220, (0, 0, width // 3, height // 5))
    return Image.merge(
        "RGB", [across.resize((width, height)), down.resize((width, height)), corner]
    )


def encoded(image, format="PNG", **kw):
    target = io.BytesIO()
    image.save(target, format, **kw)
    return target.getvalue()


def plain(color, mode="RGB", size=(360, 640)):
    return Image.new(mode, size, color)


def prepare(data, crop=FULL):
    return prepare_photo(io.BytesIO(data), crop)


def opened(photo):
    image = Image.open(io.BytesIO(photo.data))
    image.load()
    return image


def prepared_pixel(data):
    return opened(prepare(data)).getpixel((200, 300))


def near(pixel, expected, tolerance=3):
    return all(abs(a - b) <= tolerance for a, b in zip(pixel, expected, strict=True))


def chunk(kind, data):
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))


def raw_png(width, height, depth, color_type, rows, extra=b""):
    header = struct.pack(">IIBBBBB", width, height, depth, color_type, 0, 0, 0)
    return (
        PNG_SIGNATURE
        + chunk(b"IHDR", header)
        + extra
        + chunk(b"IDAT", zlib.compress(rows))
        + chunk(b"IEND", b"")
    )


def png16(color_type, samples, extra=b""):
    row = b"\0" + struct.pack(f">{len(samples)}H", *samples) * 360
    return raw_png(360, 640, 16, color_type, row * 640, extra)


def png_chunks(data):
    assert data.startswith(PNG_SIGNATURE)
    position, kinds = len(PNG_SIGNATURE), []
    while position < len(data):
        length = int.from_bytes(data[position : position + 4])
        kinds.append(data[position + 4 : position + 8])
        position += length + 12
    return kinds


def jpeg_markers(data):
    """Segment markers up to the scan."""
    assert data.startswith(b"\xff\xd8")
    position, markers = 2, []
    while True:
        marker = data[position + 1]
        markers.append(marker)
        if marker == 0xDA:
            return markers
        position += 2 + int.from_bytes(data[position + 2 : position + 4])


# The crop box.


def test_exact_crop():
    assert crop_pixels(FULL, 1080, 1920) == (0, 0, 1080, 1920)
    assert crop_pixels(CropBox(0.25, 0, 0.5, 1), 2160, 1920) == (540, 0, 1620, 1920)


def test_every_crop_is_exactly_9_by_16_and_inside():
    rng = random.Random(7)  # noqa: S311
    tried = 0
    while tried < 5000:
        width, height = rng.randint(288, 9000), rng.randint(512, 9000)
        pixels_high = rng.uniform(530, 9000)
        pixels_wide = pixels_high * 9 / 16 * rng.uniform(0.991, 1.009)
        if pixels_wide > width or pixels_high > height:
            continue
        tried += 1
        w, h = pixels_wide / width, pixels_high / height
        # The ends too, where rounding could push the box out.
        x = rng.choice([0.0, 1 - w, rng.uniform(0, 1 - w)])
        y = rng.choice([0.0, 1 - h, rng.uniform(0, 1 - h)])
        left, top, right, bottom = crop_pixels(CropBox(x, y, w, h), width, height)
        assert 0 <= left < right <= width
        assert 0 <= top < bottom <= height
        assert (right - left) * 16 == (bottom - top) * 9
        assert (right - left) % 9 == 0
        # As large as fits, and where it was asked for.
        assert right - left > min(pixels_wide, pixels_high * 9 / 16) - 9.001
        assert left >= math.floor(x * width) - 1
        assert right <= math.ceil((x + w) * width) + 1
        assert top >= math.floor(y * height) - 1
        assert bottom <= math.ceil((y + h) * height) + 1


def test_whole_pixel_boxes_survive_the_fractions():
    rng = random.Random(8)  # noqa: S311
    for _ in range(5000):
        unit = rng.randint(32, 240)
        width, height = rng.randint(unit * 9, 5000), rng.randint(unit * 16, 5000)
        left, top = rng.randint(0, width - unit * 9), rng.randint(0, height - unit * 16)
        crop = CropBox(left / width, top / height, unit * 9 / width, unit * 16 / height)
        expected = (left, top, left + unit * 9, top + unit * 16)
        assert crop_pixels(crop, width, height) == expected


@pytest.mark.parametrize(
    "crop",
    [
        CropBox(-0.001, 0, 0.9, 0.9),
        CropBox(0, -0.001, 0.9, 0.9),
        CropBox(0.2, 0, 0.9, 0.9),
        CropBox(0, 0.2, 0.9, 0.9),
        CropBox(0, 0, 1.01, 1.01),
        CropBox(0, 0, 0, 1),
        CropBox(0, 0, 1, 0),
        CropBox(0, 0, -1, -1),
        CropBox(1.5, 1.5, 0.5, 0.5),
        CropBox(math.nan, 0, 1, 1),
        CropBox(0, math.nan, 1, 1),
        CropBox(0, 0, math.nan, 1),
        CropBox(0, 0, 1, math.nan),
        CropBox(math.inf, 0, 1, 1),
        CropBox(0, -math.inf, 1, 1),
        CropBox(0, 0, math.inf, math.inf),
        CropBox(0, 0, 10**400, 1),
        CropBox(None, 0, 1, 1),
        CropBox("left", 0, 1, 1),
        CropBox(0, 0, 1, 0.98),
        CropBox(0, 0, 0.98, 1),
        CropBox(0, 0, 1, 0.5),
    ],
)
def test_bad_crop(crop):
    with pytest.raises(CropError) as caught:
        crop_pixels(crop, 1080, 1920)
    assert not isinstance(caught.value, CropTooSmallError)


def test_crop_of_the_wrong_shape_on_a_wide_photo():
    with pytest.raises(CropError):
        crop_pixels(FULL, 1920, 1080)


def test_smallest_crop():
    assert crop_pixels(FULL, 288, 512) == (0, 0, 288, 512)
    for size in ((287, 512), (288, 511), (279, 496)):
        with pytest.raises(CropTooSmallError, match="288 x 512"):
            crop_pixels(FULL, *size)
    with pytest.raises(CropTooSmallError):
        crop_pixels(CropBox(0, 0, 0.1, 0.1), 2160, 3840)
    with pytest.raises(CropTooSmallError):
        crop_pixels(CropBox(0, 0, 1e-300, 1e-300), 2160, 3840)


# The output.


def test_crop_takes_the_right_pixels():
    image = scene(900, 1000)
    photo = prepare(encoded(image), CropBox(0.3, 0.2, 360 / 900, 640 / 1000))
    assert photo.mime == "image/png"
    assert (photo.width, photo.height) == opened(photo).size == (360, 640)
    assert opened(photo).tobytes() == image.crop((270, 200, 630, 840)).tobytes()


def test_small_photo_is_not_enlarged():
    assert opened(prepare(encoded(scene(288, 512)))).size == (288, 512)


def test_large_photo_is_reduced():
    photo = prepare(encoded(scene(2250, 4000)))
    assert opened(photo).size == (2160, 3840)
    assert near(opened(photo).getpixel((2100, 100)), scene(2160, 3840).getpixel((2100, 100)))


@pytest.mark.parametrize("mode", ["RGBA", "P", "L", "1", "CMYK"])
def test_large_photo_of_any_mode_is_reduced(mode):
    format = "JPEG" if mode == "CMYK" else "PNG"
    result = opened(prepare(encoded(Image.new(mode, (2250, 4000)), format)))
    assert (result.mode, result.size) == ("RGB", (2160, 3840))


def test_largest_size_is_kept():
    image = scene(2160, 3840)
    assert opened(prepare(encoded(image))).tobytes() == image.tobytes()


@pytest.mark.parametrize("orientation", range(1, 9))
@pytest.mark.parametrize("format", ["PNG", "JPEG"])
def test_orientation(orientation, format):
    upright = scene(700, 900)
    undo = {
        2: Image.Transpose.FLIP_LEFT_RIGHT,
        3: Image.Transpose.ROTATE_180,
        4: Image.Transpose.FLIP_TOP_BOTTOM,
        5: Image.Transpose.TRANSPOSE,
        6: Image.Transpose.ROTATE_90,
        7: Image.Transpose.TRANSVERSE,
        8: Image.Transpose.ROTATE_270,
    }
    stored = upright.transpose(undo[orientation]) if orientation in undo else upright
    exif = Image.Exif()
    exif[0x0112] = orientation
    options = {"quality": 100, "subsampling": 0} if format == "JPEG" else {}
    data = encoded(stored, format, exif=exif, **options)
    whole = ImageOps.exif_transpose(Image.open(io.BytesIO(data)))
    assert whole.size == upright.size
    crop = CropBox(50 / 700, 200 / 900, 360 / 700, 640 / 900)
    result = opened(prepare(data, crop))
    assert result.tobytes() == whole.crop((50, 200, 410, 840)).tobytes()
    difference = ImageChops.difference(result, upright.crop((50, 200, 410, 840)))
    assert max(high for _low, high in difference.getextrema()) <= (6 if format == "JPEG" else 0)


@pytest.mark.parametrize("value", [0, 9, 60000])
def test_meaningless_orientation_is_ignored(value):
    exif = Image.Exif()
    exif[0x0112] = value
    assert prepare(encoded(scene(), exif=exif)).data == prepare(encoded(scene())).data


def test_broken_exif_is_ignored():
    data = encoded(scene(), "JPEG", exif=b"Exif\0\0MM\0*\xff\xff\xff\xff")
    assert prepare(data).width == 360


def test_xmp_orientation_is_ignored_like_in_browsers():
    xmp = b'<x:xmpmeta xmlns:x="adobe:ns:meta/"><a tiff:Orientation="6"/></x:xmpmeta>'
    data = encoded(scene(), "JPEG", xmp=xmp)
    assert Image.open(io.BytesIO(data)).getexif().get(0x0112) == 6  # Pillow alone would turn it
    assert prepare(data).width == 360


# Metadata.

SECRETS = [b"SECRET", b"Exif", b"GPS", b"http://ns.adobe.com", b"Photoshop", b"ICC_PROFILE"]


def secret_exif():
    exif = Image.Exif()
    exif[0x010F] = "SECRETMAKER"
    exif[0x0112] = 1
    exif[0x8769] = {0x9286: b"ASCII\0\0\0SECRETCOMMENT"}
    exif[0x8825] = {
        1: "N",
        2: (50.0, 27.0, 1.0),
        3: "E",
        4: (30.0, 31.0, 1.0),
        29: "SECRETDATE",
    }
    return exif


def srgb_profile():
    return ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()


def secret_jpeg():
    data = encoded(
        scene(),
        "JPEG",
        quality=95,
        exif=secret_exif(),
        xmp=b"<x:xmpmeta>SECRETXMP</x:xmpmeta>",
        comment=b"SECRETNOTE",
        icc_profile=srgb_profile(),
        dpi=(300, 300),
    )
    iptc = b"Photoshop 3.0\x008BIM\x04\x04\0\0\0\0\0\x0fSECRETIPTCBLOCK\0"
    segment = b"\xff\xed" + struct.pack(">H", len(iptc) + 2) + iptc
    return data[:2] + segment + data[2:] + b"SECRETTRAILER"


def secret_png():
    info = PngImagePlugin.PngInfo()
    info.add_text("Comment", "SECRETTEXT")
    info.add_text("Zip", "SECRETZIP" * 20, zip=True)
    info.add_itxt("XML:com.adobe.xmp", "<x:xmpmeta>SECRETXMP</x:xmpmeta>")
    data = encoded(
        scene(),
        exif=secret_exif(),
        pnginfo=info,
        icc_profile=srgb_profile(),
        dpi=(300, 300),
    )
    return data + b"SECRETTRAILER"


@pytest.mark.parametrize("make", [secret_jpeg, secret_png])
def test_all_metadata_is_gone(make):
    data = make()
    assert data.count(b"SECRET") >= 4
    assert Image.open(io.BytesIO(data)).getexif().get_ifd(0x8825)[29] == "SECRETDATE"
    photo = prepare(data)
    assert not [secret for secret in SECRETS if secret in photo.data]
    assert set(png_chunks(photo.data)) == {b"IHDR", b"IDAT", b"IEND"}
    image = opened(photo)
    assert image.info == {}
    assert not image.getexif()


# Formats and modes.


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        (encoded(plain((200, 100, 50))), (200, 100, 50)),
        (encoded(plain((200, 100, 50)), "JPEG", quality=95), (200, 100, 50)),
        (encoded(plain((200, 100, 50)), "JPEG", quality=95, progressive=True), (200, 100, 50)),
        (encoded(plain(120, "L")), (120, 120, 120)),
        (encoded(plain(120, "L"), "JPEG"), (120, 120, 120)),
        (encoded(plain(1, "1")), (255, 255, 255)),
        (encoded(plain(0, "1")), (0, 0, 0)),
        (encoded(plain((200, 100, 50, 255), "RGBA")), (200, 100, 50)),
        (encoded(plain((200, 100, 50, 0), "RGBA")), (255, 255, 255)),
        (encoded(plain((200, 100, 50, 128), "RGBA")), (227, 177, 152)),
        (encoded(plain((100, 0), "LA")), (255, 255, 255)),
        (encoded(plain((100, 128), "LA")), (177, 177, 177)),
        (encoded(plain((200, 100, 50)).quantize(4)), (200, 100, 50)),
        (encoded(plain((200, 100, 50)), transparency=(200, 100, 50)), (255, 255, 255)),
        (encoded(plain((200, 100, 50)), transparency=(1, 1, 1)), (200, 100, 50)),
        (encoded(plain(120, "L"), transparency=120), (255, 255, 255)),
        (encoded(plain(0, "1"), transparency=0), (255, 255, 255)),
        (png16(0, [40000]), (156, 156, 156)),
        (png16(0, [65535]), (255, 255, 255)),
        (png16(0, [0]), (0, 0, 0)),
        (png16(2, [51400, 25700, 12850]), (200, 100, 50)),
        (png16(4, [25700, 65535]), (100, 100, 100)),
        (png16(4, [25700, 0]), (255, 255, 255)),
        (png16(6, [51400, 25700, 12850, 65535]), (200, 100, 50)),
        (png16(6, [51400, 25700, 12850, 0]), (255, 255, 255)),
    ],
)
def test_modes(data, expected):
    assert near(prepared_pixel(data), expected)


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        (png16(0, [40000], chunk(b"tRNS", struct.pack(">H", 40000))), (156, 156, 156)),
        (png16(2, [0] * 3, chunk(b"tRNS", struct.pack(">3H", 0, 0, 0))), (0, 0, 0)),
    ],
)
def test_16_bit_color_key_is_ignored(data, expected):
    assert near(prepared_pixel(data), expected)


def test_palette_with_transparency():
    image = plain(0, "P")
    image.putpalette([200, 100, 50, 10, 20, 30])
    image.paste(1, (0, 0, 360, 100))
    data = encoded(image, transparency=bytes([0, 255]))
    result = opened(prepare(data))
    assert result.getpixel((200, 300)) == (255, 255, 255)
    assert result.getpixel((200, 50)) == (10, 20, 30)


def test_cmyk_jpeg():
    data = encoded(plain((0, 255, 255, 0), "CMYK"), "JPEG", quality=95)
    assert Image.open(io.BytesIO(data)).mode == "CMYK"
    assert near(prepared_pixel(data), (255, 0, 0), 6)


def wide_profile():
    """An sRGB matrix profile with its red and green primaries swapped, so a use of it shows."""
    data = bytearray(srgb_profile())
    for index in range(int.from_bytes(data[128:132])):
        entry = 132 + index * 12
        tag = bytes(data[entry : entry + 4])
        data[entry : entry + 4] = {b"rXYZ": b"gXYZ", b"gXYZ": b"rXYZ"}.get(tag, tag)
    return bytes(data)


@pytest.mark.parametrize("format", ["PNG", "JPEG"])
def test_profile_is_applied(format):
    options = {"quality": 95} if format == "JPEG" else {}
    data = encoded(plain((220, 40, 40)), format, icc_profile=wide_profile(), **options)
    assert near(prepared_pixel(data), (40, 220, 40), 8)


def test_profile_is_applied_under_transparency():
    data = encoded(plain((220, 40, 40, 255), "RGBA"), icc_profile=wide_profile())
    assert near(prepared_pixel(data), (40, 220, 40), 8)
    data = encoded(plain((220, 40, 40, 0), "RGBA"), icc_profile=wide_profile())
    assert prepared_pixel(data) == (255, 255, 255)


def test_rgb_profile_on_a_grey_photo():
    data = encoded(plain(120, "L"), icc_profile=srgb_profile())
    assert near(prepared_pixel(data), (120, 120, 120))


@pytest.mark.parametrize("profile", [b"junk", srgb_profile()[:200], b"\0" * 4000])
def test_broken_profile_is_ignored(profile):
    data = encoded(plain((200, 100, 50)), icc_profile=profile)
    assert near(prepared_pixel(data), (200, 100, 50))


def test_profile_of_another_kind_is_ignored():
    lab = ImageCms.ImageCmsProfile(ImageCms.createProfile("LAB")).tobytes()
    data = encoded(plain((200, 100, 50)), icc_profile=lab)
    assert near(prepared_pixel(data), (200, 100, 50))


def test_first_picture_of_an_mpo():
    target = io.BytesIO()
    first, second = plain((200, 100, 50)), plain((0, 0, 255), size=(100, 100))
    first.save(target, "MPO", save_all=True, append_images=[second], quality=95)
    data = target.getvalue()
    assert Image.open(io.BytesIO(data)).format == "MPO"
    assert near(prepared_pixel(data), (200, 100, 50))
    assert set(png_chunks(prepare(data).data)) == {b"IHDR", b"IDAT", b"IEND"}


# Refused files.


def test_animated_png():
    data = encoded(plain((1, 2, 3)), save_all=True, append_images=[plain((4, 5, 6))])
    with pytest.raises(AnimatedPhotoError):
        prepare(data)


@pytest.mark.parametrize("format", ["GIF", "WEBP", "BMP", "TIFF", "ICO", "PPM"])
def test_other_formats(format):
    data = encoded(scene(), format)
    with pytest.raises(NotAPhotoError):
        prepare(data)


def test_name_of_the_upload_does_not_matter():
    class Upload(io.BytesIO):
        name = "photo.png"
        content_type = "image/png"

    with pytest.raises(NotAPhotoError):
        prepare_photo(Upload(encoded(scene(), "GIF")), FULL)
    disguised = Upload(encoded(scene(), "JPEG"))
    disguised.name, disguised.content_type = "a.gif", "image/gif"
    assert prepare_photo(disguised, FULL).mime == "image/png"


@pytest.mark.parametrize("data", [b"", b"hello", b"<svg/>", b"\xff\xd8", b" " + PNG_SIGNATURE])
def test_not_an_image(data):
    with pytest.raises(NotAPhotoError):
        prepare(data)


@pytest.mark.parametrize(
    "data",
    [
        PNG_SIGNATURE,
        PNG_SIGNATURE + b"\0" * 200,
        PNG_SIGNATURE + encoded(plain(1, "L"), "JPEG"),
        b"\xff\xd8\xff" + b"\0" * 200,
        b"\xff\xd8\xff\xe0",
        encoded(scene())[:-2000],
        encoded(scene())[:60],
        encoded(scene())[:20],
        encoded(scene(), "JPEG")[:-2000],
        encoded(scene(), "JPEG")[:300],
        raw_png(360, 640, 8, 2, b"\0" * 1000),
        raw_png(0, 0, 8, 2, b""),
        raw_png(360, 640, 7, 2, b""),
        PNG_SIGNATURE + chunk(b"IEND", b""),
    ],
)
def test_damaged(data):
    with pytest.raises(DamagedPhotoError) as caught:
        prepare(data)
    assert str(caught.value) == DamagedPhotoError.message


def test_random_damage_gives_only_photo_errors():
    rng = random.Random(9)  # noqa: S311
    for original in (secret_png(), secret_jpeg(), encoded(scene().quantize(16))):
        for _ in range(150):
            data = bytearray(original)
            for _ in range(rng.randint(1, 8)):
                data[rng.randrange(min(len(data), 4000))] = rng.randrange(256)
            try:
                photo = prepare(bytes(data))
            except PhotoError:
                continue
            assert photo.width * 16 == photo.height * 9


@pytest.mark.filterwarnings("ignore::PIL.Image.DecompressionBombWarning")
@pytest.mark.parametrize("size", [(6000, 10667), (10000, 10000), (60000, 60000)])
@pytest.mark.parametrize("pillow_limit", [None, 1_000_000_000_000, Image.MAX_IMAGE_PIXELS])
def test_bomb_is_refused_before_decoding(size, pillow_limit, monkeypatch):
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", pillow_limit)
    monkeypatch.setattr(PngImagePlugin.PngImageFile, "load", lambda _self: pytest.fail("decoded"))
    with pytest.raises(PixelCountError, match="60 megapixels"):
        prepare(raw_png(*size, 8, 2, b"\0" * 100_000))


def test_bomb_warning_as_an_error():
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(PixelCountError):
            prepare(raw_png(10000, 10000, 8, 2, b"\0" * 1000))


def test_jpeg_bomb():
    data = bytearray(encoded(scene(), "JPEG"))
    start = data.index(b"\xff\xc0")
    data[start + 5 : start + 9] = struct.pack(">HH", 10000, 7000)
    with pytest.raises(PixelCountError):
        prepare(bytes(data))


def test_size_limit(monkeypatch):
    data = encoded(scene())
    monkeypatch.setattr(photos, "MAX_BYTES", len(data))
    assert prepare(data).width == 360
    monkeypatch.setattr(photos, "MAX_BYTES", len(data) - 1)
    with pytest.raises(FileSizeError, match="20 MB"):
        prepare(data)


def test_endless_upload_is_cut_off():
    class Endless(io.RawIOBase):
        served = 0

        def read(self, size=-1):
            assert 0 < size <= 1024 * 1024
            self.served += size
            return b"\0" * size

    source = Endless()
    with pytest.raises(FileSizeError):
        prepare_photo(source, FULL)
    assert source.served <= photos.MAX_BYTES + 1024 * 1024


def test_short_reads():
    class Slow(io.BytesIO):
        def read(self, size=-1):
            return super().read(min(size, 100))

    assert prepare_photo(Slow(encoded(scene())), FULL).width == 360


def test_failed_read_is_not_a_photo_error():
    class Broken(io.BytesIO):
        def read(self, size=-1):
            raise ConnectionError

    with pytest.raises(ConnectionError):
        prepare_photo(Broken(), FULL)


def test_crop_error_reaches_the_caller():
    with pytest.raises(CropError):
        prepare(encoded(scene()), CropBox(0, 0, 1, 0.5))


# Saving.


def uploads(*images):
    return [(io.BytesIO(encoded(image)), FULL) for image in images]


COLORS = [(200, 0, 0), (0, 200, 0), (0, 0, 200)]


def color_of(store, asset):
    with Image.open(store.path_for(asset)) as image:
        return image.convert("RGB").getpixel((100, 100))


def test_web_process_saves_photos(web, session, roots):
    job = new_job(session)
    data = secret_jpeg()
    expires = datetime.now(UTC) + timedelta(days=1)
    (asset,) = take(session, web, job, [(io.BytesIO(data), FULL)], expires_at=expires)
    session.commit()
    assert asset.kind is AssetKind.PHOTO
    assert asset.store is AssetStore.SHARED
    assert asset.mime == "image/png"
    assert (asset.owner_id, asset.job_id) == (job.owner_id, job.id)
    assert asset.expires_at is not None
    (path,) = files(*roots)
    assert path == web.path_for(asset)
    assert path.suffix == ".png"
    assert path.read_bytes() == prepare(data).data
    assert b"SECRET" not in path.read_bytes()


def test_web_process_cannot_make_copies(web, session, roots):
    job = new_job(session)
    take(session, web, job, uploads(scene()))
    session.commit()
    with pytest.raises(StoreUnavailableError):
        make_gemma_copies(session, web, job.id)
    session.commit()
    assert count(session, AssetKind.GEMMA_COPY) == 0
    assert files(roots[1]) == []


def test_pending_job(web, session):
    owner_id = new_job(session).owner_id
    job = Job(
        id=uuid.uuid4(),
        owner_id=owner_id,
        photo_count=1,
        length=JobLength.SHORT,
        quality=JobQuality.HD720,
    )
    session.add(job)
    assert len(take(session, web, job, uploads(scene()))) == 1
    session.commit()


def test_order_is_kept(store, session):
    for _ in range(5):
        job = new_job(session, 3)
        job_id = job.id
        saved = take(session, store, job, uploads(*map(plain, COLORS)))
        copies = make_gemma_copies(session, store, job.id)  # same transaction
        assert [color_of(store, asset) for asset in saved] == COLORS
        assert all(
            near(color_of(store, copy), color) for copy, color in zip(copies, COLORS, strict=True)
        )
        session.commit()
        session.expunge_all()
        found = list_photos(session, job_id)
        assert [color_of(store, asset) for asset in found] == COLORS


@pytest.mark.parametrize("number", [0, 4])
def test_photo_count(web, session, roots, number):
    job = new_job(session)
    with pytest.raises(PhotoCountError, match="1 to 3"):
        take(session, web, job, uploads(*[scene()] * number))
    assert files(*roots) == []


def test_job_must_match(web, session, roots):
    job = new_job(session)
    with pytest.raises(ValueError, match="match"):
        take(session, web, job, uploads(scene(), scene()))
    stranger = new_job(session)
    with pytest.raises(ValueError, match="match"):
        intake(session, web, owner_id=stranger.owner_id, job_id=job.id, uploads=uploads(scene()))
    with pytest.raises(ValueError, match="match"):
        intake(session, web, owner_id=job.owner_id, job_id=uuid.uuid4(), uploads=uploads(scene()))
    assert files(*roots) == []


def test_second_intake_is_refused(web, session, roots):
    job = new_job(session)
    take(session, web, job, uploads(scene()))
    with pytest.raises(ValueError, match="already"):
        take(session, web, job, uploads(scene()))
    session.commit()
    assert len(files(*roots)) == 1


def test_bad_second_photo_leaves_nothing(web, session, roots):
    job = new_job(session, 2)
    good = (io.BytesIO(encoded(scene())), FULL)
    with pytest.raises(NotAPhotoError) as caught:
        take(session, web, job, [good, (io.BytesIO(b"no"), FULL)])
    assert caught.value.photo == 2
    session.commit()
    assert files(*roots) == []
    assert count(session, AssetKind.PHOTO) == 0


def failing_save(store, monkeypatch, at):
    real, calls = store.save, []

    def save(*args, **kw):
        calls.append(1)
        if len(calls) == at:
            raise OSError("disk full")
        return real(*args, **kw)

    monkeypatch.setattr(store, "save", save)


def test_failed_second_save_leaves_nothing_even_if_committed(web, session, roots, monkeypatch):
    job = new_job(session, 2)
    failing_save(web, monkeypatch, at=2)
    with pytest.raises(OSError, match="disk full"):
        take(session, web, job, uploads(scene(), scene()))
    session.commit()
    assert files(*roots) == []
    assert count(session, AssetKind.PHOTO) == 0
    monkeypatch.undo()
    assert len(take(session, web, job, uploads(scene(), scene()))) == 2


def test_rollback_removes_the_photos(web, session, roots):
    job = new_job(session, 2)
    take(session, web, job, uploads(scene(), scene()))
    assert len(files(*roots)) == 2
    session.rollback()
    assert files(*roots) == []
    assert count(session, AssetKind.PHOTO) == 0


# Copies for Gemma.


@pytest.mark.parametrize(("number", "size"), [(1, (540, 960)), (2, (360, 640)), (3, (360, 640))])
def test_copy_size_by_photo_count(web, store, session, roots, number, size):
    job = new_job(session, number)
    saved = take(session, web, job, uploads(*[scene(1080, 1920)] * number))
    session.commit()
    before = [store.path_for(asset).read_bytes() for asset in saved]
    copies = make_gemma_copies(session, store, job.id)
    session.commit()
    assert len(copies) == number
    assert len(files(roots[1])) == number
    for copy in copies:
        assert copy.kind is AssetKind.GEMMA_COPY
        assert copy.store is AssetStore.PRIVATE
        assert copy.mime == "image/jpeg"
        assert (copy.owner_id, copy.job_id) == (job.owner_id, job.id)
        with Image.open(store.path_for(copy)) as image:
            assert (image.format, image.mode, image.size) == ("JPEG", "RGB", size)
    # ComfyUI's photo stays as it was.
    assert [store.path_for(asset).read_bytes() for asset in saved] == before


def test_copy_looks_like_its_photo_and_has_no_metadata(store, session):
    job = new_job(session)
    take(session, store, job, [(io.BytesIO(secret_jpeg()), FULL)])
    (copy,) = make_gemma_copies(session, store, job.id, sizes={1: (180, 320)})
    data = store.path_for(copy).read_bytes()
    assert not [secret for secret in SECRETS if secret in data]
    # Only JFIF, tables, frame and scan: no APP1 to APP15 and no comment.
    assert set(jpeg_markers(data)) <= {0xE0, 0xDB, 0xC0, 0xC4, 0xDD, 0xDA}
    with Image.open(store.path_for(copy)) as image:
        assert not image.getexif()
        assert near(image.getpixel((150, 250)), scene(180, 320).getpixel((150, 250)), 8)


def test_copy_is_not_enlarged(store, session):
    job = new_job(session)
    take(session, store, job, uploads(scene(288, 512)))
    (copy,) = make_gemma_copies(session, store, job.id)
    with Image.open(store.path_for(copy)) as image:
        assert image.size == (288, 512)


def test_copy_sizes_and_quality_can_change(store, session):
    sizes = []
    for quality in (30, 95):
        job = new_job(session)
        take(session, store, job, uploads(scene(1080, 1920)))
        (copy,) = make_gemma_copies(session, store, job.id, sizes={1: (720, 1280)}, quality=quality)
        with Image.open(store.path_for(copy)) as image:
            assert image.size == (720, 1280)
        sizes.append(copy.size_bytes)
    assert sizes[0] < sizes[1]


def test_copies_are_made_once(store, session, roots):
    job = new_job(session, 2)
    job_id = job.id
    take(session, store, job, uploads(scene(), scene()))
    session.commit()
    first = [copy.id for copy in make_gemma_copies(session, store, job.id)]
    session.commit()
    session.expunge_all()
    again = [copy.id for copy in make_gemma_copies(session, store, job_id)]
    session.commit()
    assert again == first
    assert len(files(roots[1])) == 2
    assert count(session, AssetKind.GEMMA_COPY) == 2


def test_only_the_missing_copy_is_made_again(store, session):
    job = new_job(session, 3)
    take(session, store, job, uploads(*map(plain, COLORS)))
    first = [copy.id for copy in make_gemma_copies(session, store, job.id)]
    session.commit()
    store.delete_assets(session, [first[1]])
    copies = make_gemma_copies(session, store, job.id)
    session.commit()
    assert [copies[0].id, copies[2].id] == [first[0], first[2]]
    assert copies[1].id != first[1]
    for copy, color in zip(copies, COLORS, strict=True):
        assert near(color_of(store, copy), color)


def test_failed_copy_leaves_none_and_can_be_retried(store, session, roots, monkeypatch):
    job = new_job(session, 2)
    take(session, store, job, uploads(scene(), scene()))
    session.commit()
    failing_save(store, monkeypatch, at=2)
    with pytest.raises(OSError, match="disk full"):
        make_gemma_copies(session, store, job.id)
    session.commit()
    assert files(roots[1]) == []
    assert count(session, AssetKind.GEMMA_COPY) == 0
    monkeypatch.undo()
    assert len(make_gemma_copies(session, store, job.id)) == 2


def test_copies_expire_with_their_photo(store, session):
    job = new_job(session)
    expires = datetime.now(UTC) + timedelta(days=1)
    (photo,) = take(session, store, job, uploads(scene()), expires_at=expires)
    session.commit()
    (copy,) = make_gemma_copies(session, store, job.id)
    session.commit()
    assert copy.expires_at == photo.expires_at is not None


def test_job_without_photos(store, session):
    with pytest.raises(ValueError, match="no photos"):
        make_gemma_copies(session, store, new_job(session).id)


def test_jobs_do_not_mix(store, session):
    job, other = new_job(session), new_job(session)
    take(session, store, job, uploads(plain(COLORS[0])))
    take(session, store, other, uploads(plain(COLORS[1])))
    (copy,) = make_gemma_copies(session, store, other.id)
    assert near(color_of(store, copy), COLORS[1])
    assert len(list_photos(session, job.id)) == 1


def test_job_delete_takes_photos_and_copies(store, session, roots):
    job = new_job(session)
    take(session, store, job, uploads(scene()))
    make_gemma_copies(session, store, job.id)
    session.commit()
    assert store.delete_job(session, job.id) == 2
    assert files(*roots) == []


def test_messages_are_for_the_user():
    errors = PhotoError.__subclasses__()
    while errors:
        error = errors.pop()
        errors += error.__subclasses__()
        assert str(error()) == error.message != PhotoError.message
