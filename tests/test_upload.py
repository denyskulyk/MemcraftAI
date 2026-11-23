import io
import json
import uuid
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from memcraft import photos
from memcraft.config import Settings, get_settings
from memcraft.models import (
    Asset,
    AssetKind,
    Job,
    JobLane,
    JobLength,
    JobQuality,
    JobRegime,
    JobStatus,
    User,
    UserStatus,
)
from memcraft.storage import FileStore
from memcraft.web import upload
from memcraft.web.app import create_app
from memcraft.web.auth import get_session, get_store
from memcraft.web.upload import REGIME_NOTES, regime_for

ALICE, BOB = "alice@example.com", "bob@example.com"
BASE = "http://localhost"
FULL = {"x": 0, "y": 0, "w": 1, "h": 1}


@pytest.fixture
def store(tmp_path):
    (tmp_path / "shared").mkdir()
    (tmp_path / "private").mkdir()
    return FileStore(tmp_path / "shared", tmp_path / "private")


@pytest.fixture
def client(engine, store):
    settings = Settings(_env_file=None)
    app = create_app(settings)

    def session():
        with Session(engine) as session:
            yield session

    app.dependency_overrides[get_session] = session
    app.dependency_overrides[get_store] = lambda: store
    app.dependency_overrides[get_settings] = lambda: settings
    return TestClient(app, base_url=BASE)


def as_user(login, **extra):
    return {"Tailscale-User-Login": login, **extra}


def add_user(engine, login, status=UserStatus.APPROVED):
    with Session(engine) as session:
        user = User(tailscale_login=login, status=status)
        session.add(user)
        session.commit()
        return user.id


def picture(size=(900, 1600), kind="PNG", color=(200, 80, 40)):
    data = io.BytesIO()
    Image.new("RGB", size, color).save(data, kind)
    return data.getvalue()


def landscape_crop(width=1920, height=1080):
    w = height * 9 / 16 / width
    return {"x": (1 - w) / 2, "y": 0, "w": w, "h": 1}


def post(client, files, crops, login=ALICE, headers=None, **fields):
    data = {"crops": json.dumps(crops), "length": "short", "quality": "hd720", **fields}
    data = {key: value for key, value in data.items() if value is not None}
    parts = [("photos", (f"photo{number}.png", content)) for number, content in enumerate(files)]
    return client.post(
        "/api/jobs", data=data, files=parts or None, headers={**as_user(login), **(headers or {})}
    )


def counts(engine):
    with Session(engine) as session:
        return tuple(
            session.scalar(select(func.count()).select_from(table)) for table in (Job, Asset)
        )


def stored_files(tmp_path):
    """Every file left in either store, and any temporary file the store wrote."""
    return sorted(path for path in tmp_path.rglob("*") if path.is_file())


def test_one_photo_makes_a_queued_job(client, engine, store):
    add_user(engine, ALICE)
    answer = post(client, [picture()], [FULL], idea="  ")
    assert answer.status_code == 201, answer.text
    body = answer.json()
    assert body["queue_position"] == 0
    with Session(engine) as session:
        job = session.get(Job, uuid.UUID(body["id"]))
        assert (job.status, job.lane, job.photo_count) == (
            JobStatus.IDEA_QUEUED,
            JobLane.IDEA,
            1,
        )
        assert (job.length, job.quality, job.regime) == (
            JobLength.SHORT,
            JobQuality.HD720,
            JobRegime.MULTISHOT,
        )
        assert job.idea_text is None
        (asset,) = session.scalars(select(Asset).where(Asset.job_id == job.id))
        assert (asset.kind, asset.ordinal, asset.mime, asset.owner_id) == (
            AssetKind.PHOTO,
            0,
            "image/png",
            job.owner_id,
        )
        assert asset.expires_at is None
        with Image.open(store.path_for(asset)) as saved:
            assert saved.size == (900, 1600)


def test_three_photos_keep_their_order(client, engine, store):
    add_user(engine, ALICE)
    add_user(engine, BOB)
    assert post(client, [picture()], [FULL], login=BOB).status_code == 201
    with Session(engine) as session:  # SQLite's clock counts whole seconds
        session.execute(update(Job).values(queued_at=datetime.now(UTC) - timedelta(minutes=1)))
        session.commit()
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    files = [
        picture((1920, 1080), "JPEG", colors[0]),
        picture(color=colors[1]),
        picture((1920, 1080), "PNG", colors[2]),
    ]
    crops = [landscape_crop(), FULL, landscape_crop()]
    answer = post(
        client, files, crops, length="long", quality="fhd1080", idea="  a cat runs away  "
    )
    assert answer.status_code == 201, answer.text
    assert answer.json()["queue_position"] == 1  # Bob's job is ahead
    with Session(engine) as session:
        job = session.get(Job, uuid.UUID(answer.json()["id"]))
        assert (job.photo_count, job.idea_text, job.regime) == (
            3,
            "a cat runs away",
            JobRegime.KEYFRAMES,
        )
        assert (job.length, job.quality) == (JobLength.LONG, JobQuality.FHD1080)
        assets = session.scalars(
            select(Asset).where(Asset.job_id == job.id).order_by(Asset.ordinal)
        ).all()
        assert [asset.ordinal for asset in assets] == [0, 1, 2]
        for asset, color in zip(assets, colors, strict=True):
            with Image.open(store.path_for(asset)) as saved:
                assert saved.width * 16 == saved.height * 9
                red, green, blue = saved.convert("RGB").getpixel((10, 10))
                assert max(range(3), key=lambda i: (red, green, blue)[i]) == color.index(255)


@pytest.mark.parametrize(
    ("files", "crops", "fields", "status", "detail"),
    [
        (1, [FULL, FULL], {}, 422, "Send one crop frame for each photo."),
        (2, [FULL], {}, 422, "Send one crop frame for each photo."),
        (1, "not json", {}, 422, "The crop frames could not be read."),
        (1, [{"x": 0, "y": 0, "w": 1}], {}, 422, "Photo 1: the crop frame could not be read."),
        (1, [{"x": True, "y": 0, "w": 1, "h": 1}], {}, 422, "Photo 1: the crop"),
        (4, [FULL] * 4, {}, 422, "Add 1 to 3 photos."),
        (0, [], {}, 400, "Send the photos as a multipart form."),
        (1, [FULL], {"length": None}, 422, "Send the length once."),
        (1, [FULL], {"quality": None}, 422, "Send the quality once."),
        (1, [FULL], {"crops": None}, 422, "Send the crop frames once."),
        (1, [FULL], {"length": "medium"}, 422, "The length must be short or long."),
        (1, [FULL], {"quality": "4k"}, 422, "The quality must be 720p or 1080p."),
        (1, [FULL], {"idea": "x" * 501}, 422, "Please keep the idea under 500 characters."),
        (1, [FULL], {"extra": "1"}, 422, "Unexpected field: extra."),
        (1, [{"x": 0.5, "y": 0, "w": 1, "h": 1}], {}, 422, "Photo 1: The crop must lie inside"),
    ],
)
def test_bad_forms_are_refused(
    client, engine, store, files, crops, fields, status, detail, tmp_path
):
    add_user(engine, ALICE)
    if crops != "not json":
        fields.setdefault("crops", json.dumps(crops))
    else:
        fields["crops"] = crops
    data = {"length": "short", "quality": "hd720", **fields}
    data = {key: value for key, value in data.items() if value is not None}
    parts = [("photos", (f"p{number}.png", picture())) for number in range(files)]
    answer = client.post("/api/jobs", data=data, files=parts or None, headers=as_user(ALICE))
    assert answer.status_code == status, answer.text
    assert answer.json()["detail"].startswith(detail)
    assert counts(engine) == (0, 0)
    assert stored_files(tmp_path) == []


def test_not_a_photo_names_the_photo(client, engine, store, tmp_path):
    add_user(engine, ALICE)
    gif = io.BytesIO()
    Image.new("RGB", (900, 1600)).save(gif, "GIF")
    answer = post(client, [picture(), gif.getvalue()], [FULL, FULL])
    assert answer.status_code == 422
    assert answer.json()["detail"] == "Photo 2: Only PNG and JPEG photos are accepted."
    assert counts(engine) == (0, 0)
    assert stored_files(tmp_path) == []


def test_failure_after_good_photos_leaves_nothing(client, engine, store, tmp_path):
    add_user(engine, ALICE)
    small = picture((180, 320))
    answer = post(client, [picture(), picture(), small], [FULL, FULL, FULL])
    assert answer.status_code == 422
    assert answer.json()["detail"] == "Photo 3: The crop must be at least 288 x 512 pixels."
    assert counts(engine) == (0, 0)
    assert stored_files(tmp_path) == []


def test_oversize_photo_is_refused_while_it_arrives(client, engine, store, tmp_path):
    add_user(engine, ALICE)
    big = b"\x89PNG\r\n\x1a\n" + bytes(photos.MAX_BYTES)
    answer = post(client, [picture(), big], [FULL, FULL])
    assert answer.status_code == 413
    assert answer.json()["detail"] == "Photo 2: This photo is larger than 20 MB."
    assert counts(engine) == (0, 0)
    assert stored_files(tmp_path) == []


def test_body_over_the_cap_is_refused(client, engine, store, monkeypatch):
    add_user(engine, ALICE)
    monkeypatch.setattr(upload, "MAX_BODY", 50_000)
    # With a Content-Length, before anything is read.
    answer = post(client, [picture((900, 1600), "BMP")], [FULL])
    assert answer.status_code == 413
    assert answer.json()["detail"] == upload.TOO_LARGE

    # Without one, while the body arrives.
    boundary = "b0undary"
    head = (
        f'--{boundary}\r\nContent-Disposition: form-data; name="photos"; filename="a.png"\r\n'
        "Content-Type: image/png\r\n\r\n"
    ).encode()

    def body():
        yield head
        for _ in range(100):
            yield bytes(1000)

    answer = client.post(
        "/api/jobs",
        content=body(),
        headers={**as_user(ALICE), "Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    assert answer.status_code == 413
    assert counts(engine) == (0, 0)


def test_other_body_types_are_refused(client, engine):
    add_user(engine, ALICE)
    answer = client.post("/api/jobs", json={"photos": []}, headers=as_user(ALICE))
    assert answer.status_code == 400


def test_pending_and_unknown_users_are_refused(client, engine):
    add_user(engine, BOB, UserStatus.PENDING)
    assert post(client, [picture()], [FULL], login=BOB).status_code == 403
    answer = client.post("/api/jobs", files=[("photos", ("a.png", picture()))])
    assert answer.status_code == 401
    assert client.get("/api/regime-note?photos=1&length=short").status_code == 401
    assert counts(engine) == (0, 0)


@pytest.mark.parametrize(
    "headers", [{"Sec-Fetch-Site": "cross-site"}, {"Origin": "https://evil.example"}]
)
def test_upload_from_another_site_is_refused(client, engine, headers):
    add_user(engine, ALICE)
    answer = post(client, [picture()], [FULL], headers=headers)
    assert answer.status_code == 403
    assert counts(engine) == (0, 0)


def test_open_jobs_are_limited_per_user(client, engine, store):
    alice = add_user(engine, ALICE)
    add_user(engine, BOB)
    with Session(engine) as session:
        for status in (*upload.OPEN_STATUSES[:2], JobStatus.FAILED, JobStatus.PREVIEW_READY):
            session.add(
                Job(
                    owner_id=alice,
                    photo_count=1,
                    length=JobLength.SHORT,
                    quality=JobQuality.HD720,
                    status=status,
                )
            )
        session.commit()
    assert post(client, [picture()], [FULL]).status_code == 201  # the third open job
    answer = post(client, [picture()], [FULL])
    assert answer.status_code == 429
    assert answer.json()["detail"] == upload.TOO_MANY_JOBS
    assert post(client, [picture()], [FULL], login=BOB).status_code == 201
    assert counts(engine) == (6, 2)


@pytest.mark.parametrize(
    ("count", "length", "regime"),
    [
        (1, JobLength.SHORT, JobRegime.MULTISHOT),
        (2, JobLength.SHORT, JobRegime.KEYFRAMES),
        (3, JobLength.SHORT, JobRegime.KEYFRAMES),
        (1, JobLength.LONG, JobRegime.KEYFRAMES),
        (2, JobLength.LONG, JobRegime.KEYFRAMES),
        (3, JobLength.LONG, JobRegime.KEYFRAMES),
    ],
)
def test_regime_rule(client, engine, count, length, regime):
    assert regime_for(count, length) == regime
    add_user(engine, ALICE)
    answer = client.get(
        "/api/regime-note", params={"photos": count, "length": length.value}, headers=as_user(ALICE)
    )
    assert answer.json() == {"regime": regime.value, "note": REGIME_NOTES[regime]}


@pytest.mark.parametrize("query", ["photos=0&length=short", "photos=4&length=long", "photos=1"])
def test_regime_note_checks_its_input(client, engine, query):
    add_user(engine, ALICE)
    assert client.get(f"/api/regime-note?{query}", headers=as_user(ALICE)).status_code == 422


def test_regime_rule_refuses_other_counts():
    for count in (0, 4):
        with pytest.raises(ValueError, match="photo_count"):
            regime_for(count, JobLength.SHORT)
