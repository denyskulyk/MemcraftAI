import importlib.util
import io
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy.orm import Session

from memcraft import photos
from memcraft.config import Settings, get_settings
from memcraft.models import User, UserStatus
from memcraft.web import faces
from memcraft.web.auth import get_session

ALICE, BOB = "alice@example.com", "bob@example.com"
STAND_INS = Path(__file__).resolve().parents[1] / "e0-spike" / "photos-ai"


def stand_in(name):
    path = STAND_INS / f"{name}.png"
    if not path.is_file():
        pytest.skip(f"{path} is not there")
    return Image.open(path).convert("RGB")


def encoded(image, kind="PNG", **options):
    data = io.BytesIO()
    image.save(data, kind, **options)
    return data.getvalue()


needs_opencv = pytest.mark.skipif(
    importlib.util.find_spec("cv2") is None, reason="opencv is not installed"
)


@pytest.fixture
def client(engine):
    settings = Settings(_env_file=None)
    app = FastAPI()
    app.include_router(faces.router)

    def session():
        with Session(engine) as session:
            yield session

    app.dependency_overrides[get_session] = session
    app.dependency_overrides[get_settings] = lambda: settings
    return TestClient(app, base_url="http://localhost")


def add_user(engine, login, status=UserStatus.APPROVED):
    with Session(engine) as session:
        session.add(User(tailscale_login=login, status=status))
        session.commit()


def ask(client, body, kind="image/png", login=ALICE, **headers):
    headers = {"Content-Type": kind, "Tailscale-User-Login": login, **headers}
    return client.post("/api/faces", content=body, headers=headers)


def centre(box):
    return box["x"] + box["w"] / 2, box["y"] + box["h"] / 2


@needs_opencv
def test_finds_each_face_of_a_group_photo(client, engine):
    add_user(engine, ALICE)
    answer = ask(client, encoded(stand_in("p09")))
    assert answer.status_code == 200, answer.text
    found = answer.json()["faces"]
    assert len(found) == 3
    assert [box["x"] for box in found] == sorted(box["x"] for box in found)
    for box in found:
        assert 0 <= box["x"] < box["x"] + box["w"] <= 1
        assert 0 <= box["y"] < box["y"] + box["h"] <= 1
        assert 0.1 < box["w"] < 0.5


@needs_opencv
@pytest.mark.parametrize(("name", "count"), [("p02", 1), ("p06", 2), ("p07", 2), ("p08", 2)])
def test_no_collars_or_dogs_count_as_faces(client, engine, name, count):
    add_user(engine, ALICE)
    answer = ask(client, encoded(stand_in(name), "JPEG", quality=90), "image/jpeg")
    assert len(answer.json()["faces"]) == count


@needs_opencv
def test_a_landscape_group_photo_keeps_its_people_apart(client, engine):
    add_user(engine, ALICE)
    left, right = stand_in("p09"), stand_in("p07")
    group = Image.new("RGB", (left.width + right.width + 1200, left.height), (90, 90, 90))
    group.paste(left, (0, 0))
    group.paste(right, (left.width + 1200, 0))
    found = ask(client, encoded(group, "JPEG", quality=90), "image/jpeg").json()["faces"]
    assert len(found) == 5
    # Three on the left photo, two on the right one, nobody in the grey gap.
    gap = (left.width / group.width, (left.width + 1200) / group.width)
    assert sum(centre(box)[0] < gap[0] for box in found) == 3
    assert sum(centre(box)[0] > gap[1] for box in found) == 2


@needs_opencv
def test_boxes_are_fractions_of_the_upright_photo(client, engine):
    add_user(engine, ALICE)
    upright = stand_in("p07")
    plain = ask(client, encoded(upright, "JPEG", quality=90), "image/jpeg").json()["faces"]
    # Stored turned on its side, with EXIF orientation 6: a browser shows it upright.
    exif = Image.Exif()
    exif[0x0112] = 6
    sideways = encoded(upright.rotate(90, expand=True), "JPEG", quality=90, exif=exif.tobytes())
    turned = ask(client, sideways, "image/jpeg").json()["faces"]
    assert len(turned) == len(plain) == 2
    for a, b in zip(plain, turned, strict=True):
        assert all(abs(p - q) < 0.02 for p, q in zip(centre(a), centre(b), strict=True))


@needs_opencv
def test_a_photo_without_people_has_no_faces(client, engine):
    add_user(engine, ALICE)
    answer = ask(client, encoded(Image.new("RGB", (1600, 900), (40, 140, 200))))
    assert answer.json() == {"faces": []}


def test_only_png_and_jpeg_bodies(client, engine):
    add_user(engine, ALICE)
    assert ask(client, b"GIF89a", "image/gif").status_code == 415
    assert ask(client, b"<html>", "text/html").status_code == 415
    answer = ask(client, b"not a photo at all")
    assert answer.status_code == 422
    assert answer.json()["detail"] == photos.NotAPhotoError.message
    broken = encoded(Image.new("RGB", (600, 400)))[:200]
    assert ask(client, broken).json()["detail"] == photos.DamagedPhotoError.message


def test_same_size_limits_as_an_upload(client, engine, monkeypatch):
    add_user(engine, ALICE)
    monkeypatch.setattr(photos, "MAX_BYTES", 1000)
    answer = ask(client, b"\x89PNG\r\n\x1a\n" + bytes(2000))
    assert answer.status_code == 413
    assert answer.json()["detail"] == photos.FileSizeError.message

    def chunks():  # no Content-Length: the limit holds while the body arrives
        for _ in range(4):
            yield bytes(400)

    assert ask(client, chunks()).status_code == 413
    monkeypatch.setattr(photos, "MAX_BYTES", 20 * 1024 * 1024)
    monkeypatch.setattr(photos, "MAX_PIXELS", 1000)
    answer = ask(client, encoded(Image.new("RGB", (100, 100))))
    assert answer.status_code == 422
    assert answer.json()["detail"] == photos.PixelCountError.message


def test_only_approved_friends_from_this_site(client, engine):
    add_user(engine, BOB, UserStatus.PENDING)
    body = encoded(Image.new("RGB", (64, 64)))
    assert ask(client, body, login=BOB).status_code == 403
    assert (
        client.post("/api/faces", content=body, headers={"Content-Type": "image/png"}).status_code
        == 401
    )
    add_user(engine, ALICE)
    assert ask(client, body, **{"Sec-Fetch-Site": "cross-site"}).status_code == 403


def test_without_a_detector_the_check_is_unavailable(client, engine, monkeypatch):
    add_user(engine, ALICE)

    def missing():
        raise faces.DetectorUnavailableError

    monkeypatch.setattr(faces, "_cascades", missing)
    answer = ask(client, encoded(Image.new("RGB", (64, 64))))
    assert answer.status_code == 503
    assert answer.json()["detail"] == faces.UNAVAILABLE


def test_a_face_counts_only_when_every_cascade_found_it():
    face = faces._Box(10, 10, 100, 100)
    near = faces._Box(14, 12, 96, 98)
    collar = faces._Box(300, 300, 40, 40)
    found = [(0, face), (1, near), (2, face), (0, collar), (1, collar)]
    (only,) = faces._agreed(found, 3)
    assert only.x == pytest.approx((10 + 14 + 10) / 3)
    assert faces._overlap(face, collar) == 0
    assert faces._overlap(face, face) == 1
