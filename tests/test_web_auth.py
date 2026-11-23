import io
import uuid
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from memcraft.config import Settings, get_settings
from memcraft.models import AssetKind, Job, JobLength, JobQuality, User, UserRole, UserStatus
from memcraft.storage import FileStore
from memcraft.web import api, upload
from memcraft.web.app import create_app
from memcraft.web.auth import get_session, get_store

ALICE, BOB = "alice@example.com", "bob@example.com"
BASE = "http://localhost"


@pytest.fixture
def store(tmp_path):
    (tmp_path / "shared").mkdir()
    (tmp_path / "private").mkdir()
    return FileStore(tmp_path / "shared", tmp_path / "private")


@pytest.fixture
def client(engine, store):
    settings = Settings(_env_file=None, admin_login="Owner@example.com")
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


def add_job(engine, store, owner_id, kind=AssetKind.RELEASE, expires_at=None):
    with Session(engine) as session:
        job = Job(
            owner_id=owner_id,
            photo_count=1,
            length=JobLength.SHORT,
            quality=JobQuality.HD720,
            idea_text="a private idea",
        )
        session.add(job)
        session.flush()
        mime = "video/mp4" if kind != AssetKind.PHOTO else "image/png"
        if kind == AssetKind.KEYFRAME:
            mime = "image/png"
        asset = store.save(
            session,
            owner_id=owner_id,
            job_id=job.id,
            kind=kind,
            mime=mime,
            source=io.BytesIO(b"0123456789"),
            expires_at=expires_at,
        )
        session.commit()
        return job.id, asset.id


def test_no_tailscale_login_is_refused(client):
    assert client.get("/api/me").status_code == 401


def test_two_login_headers_are_refused(client):
    headers = [("Tailscale-User-Login", ALICE), ("Tailscale-User-Login", BOB)]
    assert client.get("/api/me", headers=headers).status_code == 401


def test_unknown_user_waits_for_approval(client, engine):
    me = client.get("/api/me", headers=as_user("Alice@Example.com")).json()
    assert me["status"] == "pending"
    assert me["tailscale_login"] == ALICE
    answer = client.get("/api/jobs", headers=as_user(ALICE))
    assert answer.status_code == 403
    assert answer.json()["detail"] == "waiting for approval"
    with Session(engine) as session:
        assert session.scalar(select(func.count()).select_from(User)) == 1


def test_blocked_user_is_refused(client, engine):
    add_user(engine, ALICE, UserStatus.BLOCKED)
    assert client.get("/api/jobs", headers=as_user(ALICE)).json()["detail"] == "blocked"


def test_only_the_configured_login_becomes_admin(client, engine):
    me = client.get("/api/me", headers=as_user("owner@example.com")).json()
    assert (me["role"], me["status"]) == ("admin", "approved")
    assert client.get("/api/me", headers=as_user(ALICE)).json()["role"] == UserRole.USER


def test_empty_admin_setting_makes_nobody_admin(client):
    client.app.dependency_overrides[get_settings] = lambda: Settings(_env_file=None)
    me = client.get("/api/me", headers=as_user("owner@example.com")).json()
    assert (me["role"], me["status"]) == ("user", "pending")


def test_names_outside_ascii_are_decoded(client):
    name = "=?utf-8?q?Ren=C3=A9?="
    me = client.get("/api/me", headers=as_user(ALICE, **{"Tailscale-User-Name": name})).json()
    assert me["display_name"] == "René"


@pytest.mark.parametrize("host", ["evil.example", "localhost.evil.example", "100.79.0.1"])
def test_other_host_names_are_refused(client, host):
    assert client.get("/api/me", headers=as_user(ALICE, Host=host)).status_code == 400


def test_wildcard_hosts_are_refused():
    with pytest.raises(ValueError, match="wildcards"):
        create_app(Settings(_env_file=None, allowed_hosts="*"))


def test_owner_sees_job_and_file(client, engine, store):
    job_id, asset_id = add_job(engine, store, add_user(engine, ALICE))
    headers = as_user(ALICE)
    assert [job["id"] for job in client.get("/api/jobs", headers=headers).json()] == [str(job_id)]
    detail = client.get(f"/api/jobs/{job_id}", headers=headers).json()
    assert [asset["id"] for asset in detail["assets"]] == [str(asset_id)]
    file = client.get(f"/api/files/{asset_id}", headers=headers)
    assert file.content == b"0123456789"
    assert file.headers["cache-control"] == "private, no-store"
    part = client.get(f"/api/files/{asset_id}", headers={**headers, "Range": "bytes=2-4"})
    assert (part.status_code, part.content) == (206, b"234")


def test_nobody_reaches_another_users_job_or_file(client, engine, store):
    job_id, asset_id = add_job(engine, store, add_user(engine, ALICE))
    add_user(engine, BOB)
    client.get("/api/me", headers=as_user("owner@example.com"))
    for login in (BOB, "owner@example.com"):
        headers = as_user(login)
        assert client.get("/api/jobs", headers=headers).json() == []
        for path, missing in (
            (f"/api/jobs/{job_id}", f"/api/jobs/{uuid.uuid4()}"),
            (f"/api/files/{asset_id}", f"/api/files/{uuid.uuid4()}"),
        ):
            theirs, unknown = (
                client.get(path, headers=headers),
                client.get(missing, headers=headers),
            )
            assert theirs.status_code == unknown.status_code == 404
            assert theirs.content == unknown.content


def test_file_under_another_users_job_is_hidden(client, engine, store):
    alice, bob = add_user(engine, ALICE), add_user(engine, BOB)
    job_id, _ = add_job(engine, store, alice)
    with Session(engine) as session:
        stray = store.save(
            session,
            owner_id=bob,
            job_id=job_id,
            kind=AssetKind.RELEASE,
            mime="video/mp4",
            source=io.BytesIO(b"x"),
        )
        session.commit()
        stray_id = stray.id
    for login in (ALICE, BOB):
        answer = client.get(f"/api/files/{stray_id}", headers=as_user(login))
        assert answer.status_code == 404


def test_worker_files_and_expired_files_are_hidden(client, engine, store):
    owner = add_user(engine, ALICE)
    _, keyframe = add_job(engine, store, owner, AssetKind.KEYFRAME)
    past = datetime.now(UTC) - timedelta(minutes=1)
    _, preview = add_job(engine, store, owner, AssetKind.PREVIEW, past)
    for asset_id in (keyframe, preview):
        answer = client.get(f"/api/files/{asset_id}", headers=as_user(ALICE))
        assert answer.status_code == 404


def test_every_api_route_needs_an_approved_user(client, engine):
    """A new route that forgets the check fails here. /api/me is the one exception."""
    add_user(engine, BOB, UserStatus.PENDING)
    checked = 0
    for route in [*api.router.routes, *upload.router.routes]:
        if route.path == "/api/me":
            continue
        path = route.path.replace("{job_id}", str(uuid.uuid4()))
        path = path.replace("{asset_id}", str(uuid.uuid4()))
        assert "{" not in path, f"teach this test about {route.path}"
        for method in route.methods:
            assert client.request(method, path).status_code == 401, route.path
            answer = client.request(method, path, headers=as_user(BOB))
            assert answer.status_code == 403, route.path
            checked += 1
    assert checked >= 3


@pytest.mark.parametrize(
    ("headers", "allowed"),
    [
        ({"Sec-Fetch-Site": "same-origin"}, True),
        ({"Sec-Fetch-Site": "cross-site"}, False),
        ({"Sec-Fetch-Site": "same-site"}, False),
        ({"Origin": BASE}, True),
        ({"Origin": "https://evil.example"}, False),
        ({"Origin": "null"}, False),
        ({}, True),
    ],
)
def test_changes_from_other_sites_are_refused(client, engine, headers, allowed):
    from memcraft.web.auth import ApprovedUser

    add_user(engine, ALICE)

    @client.app.post("/probe")
    def probe(user: ApprovedUser) -> str:
        return user.tailscale_login

    # The static files are mounted at "/", so the probe has to come first.
    client.app.router.routes.insert(0, client.app.router.routes.pop())
    answer = client.post("/probe", headers=as_user(ALICE, **headers))
    assert answer.status_code == (200 if allowed else 403)
