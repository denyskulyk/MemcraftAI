import base64
import hashlib
import io
import json
import struct
import uuid

import cbor2
import pytest
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from memcraft.config import Settings, get_settings
from memcraft.models import (
    AdminSession,
    Asset,
    AssetKind,
    AuthChallenge,
    FileDeletion,
    Job,
    JobLength,
    JobQuality,
    Passkey,
    User,
    UserStatus,
)
from memcraft.storage import FileStore
from memcraft.web.admin import router
from memcraft.web.app import create_app
from memcraft.web.auth import get_session, get_store

ORIGIN = "https://pc.example.ts.net"
RP_ID = "pc.example.ts.net"
OWNER, ALICE = "owner@example.com", "alice@example.com"


def b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


class SoftKey:
    """A passkey in memory that answers like a browser with user verification."""

    def __init__(self, rp_id=RP_ID, origin=ORIGIN):
        self.key = ec.generate_private_key(ec.SECP256R1())
        self.credential_id = uuid.uuid4().bytes
        self.rp_id, self.origin, self.count = rp_id, origin, 0

    def _client_data(self, kind, options):
        data = {"type": kind, "challenge": options["challenge"], "origin": self.origin}
        return json.dumps(data).encode()

    def _auth_data(self, extra=b"", flags=0x05):
        self.count += 1
        rp = hashlib.sha256(self.rp_id.encode()).digest()
        return rp + bytes([flags | (0x40 if extra else 0)]) + struct.pack(">I", self.count) + extra

    def create(self, options):
        numbers = self.key.public_key().public_numbers()
        cose = cbor2.dumps(
            {1: 2, 3: -7, -1: 1, -2: numbers.x.to_bytes(32), -3: numbers.y.to_bytes(32)}
        )
        attested = bytes(16) + struct.pack(">H", len(self.credential_id)) + self.credential_id
        auth_data = self._auth_data(attested + cose)
        attestation = cbor2.dumps({"fmt": "none", "attStmt": {}, "authData": auth_data})
        return {
            "id": b64(self.credential_id),
            "rawId": b64(self.credential_id),
            "type": "public-key",
            "response": {
                "clientDataJSON": b64(self._client_data("webauthn.create", options)),
                "attestationObject": b64(attestation),
            },
        }

    def get(self, options, flags=0x05):
        client_data = self._client_data("webauthn.get", options)
        auth_data = self._auth_data(flags=flags)
        signature = self.key.sign(
            auth_data + hashlib.sha256(client_data).digest(), ec.ECDSA(hashes.SHA256())
        )
        return {
            "id": b64(self.credential_id),
            "rawId": b64(self.credential_id),
            "type": "public-key",
            "response": {
                "clientDataJSON": b64(client_data),
                "authenticatorData": b64(auth_data),
                "signature": b64(signature),
            },
        }


@pytest.fixture
def store(tmp_path):
    (tmp_path / "shared").mkdir()
    return FileStore(tmp_path / "shared", None)  # as in the web container


@pytest.fixture
def client(engine, store):
    settings = Settings(
        _env_file=None,
        admin_login=OWNER,
        public_origin=ORIGIN,
        allowed_hosts=f"localhost,{RP_ID}",
    )
    app = create_app(settings)

    def session():
        with Session(engine) as session:
            yield session

    app.dependency_overrides[get_session] = session
    app.dependency_overrides[get_store] = lambda: store
    app.dependency_overrides[get_settings] = lambda: settings
    client = TestClient(app, base_url=ORIGIN)
    client.headers["Tailscale-User-Login"] = OWNER
    client.headers["Sec-Fetch-Site"] = "same-origin"
    return client


def enrol(client, key):
    options = client.post("/api/admin/passkeys/options").json()
    return client.post("/api/admin/passkeys", json=key.create(options))


def sign_in(client, key, **kwargs):
    options = client.post("/api/admin/sign-in/options").json()
    return client.post("/api/admin/sign-in", json=key.get(options, **kwargs))


def count(engine, model):
    with Session(engine) as session:
        return session.scalar(select(func.count()).select_from(model))


def add_user(engine, login, status=UserStatus.PENDING):
    with Session(engine) as session:
        user = User(tailscale_login=login, status=status)
        session.add(user)
        session.commit()
        return user.id


def test_first_passkey_signs_the_admin_in(client, engine):
    assert client.get("/api/admin/status").json() == {"has_passkey": False, "signed_in": False}
    assert client.get("/api/admin/users").status_code == 401
    answer = enrol(client, SoftKey())
    assert answer.status_code == 201
    cookie = answer.headers["set-cookie"]
    for part in ("HttpOnly", "Secure", "SameSite=strict", "Path=/"):
        assert part.lower() in cookie.lower()
    assert client.get("/api/admin/status").json() == {"has_passkey": True, "signed_in": True}
    assert client.get("/api/admin/users").status_code == 200


def test_sign_in_with_the_passkey(client, engine):
    key = SoftKey()
    enrol(client, key)
    client.post("/api/admin/sign-out")
    assert client.get("/api/admin/users").status_code == 401
    assert sign_in(client, key).status_code == 200
    assert client.get("/api/admin/users").status_code == 200
    with Session(engine) as session:
        assert session.scalar(select(Passkey.sign_count)) == 2


def test_second_passkey_needs_a_signed_in_admin(client):
    first = SoftKey()
    enrol(client, first)
    client.post("/api/admin/sign-out")
    assert client.post("/api/admin/passkeys/options").status_code == 401
    sign_in(client, first)
    assert enrol(client, SoftKey()).status_code == 201


def test_only_the_admin_sees_the_admin_pages(client, engine):
    add_user(engine, ALICE, UserStatus.APPROVED)
    for path in ("/api/admin/status", "/api/admin/users"):
        assert client.get(path, headers={"Tailscale-User-Login": ALICE}).status_code == 404
    answer = client.post("/api/admin/passkeys/options", headers={"Tailscale-User-Login": ALICE})
    assert answer.status_code == 404


def test_the_cookie_alone_is_not_enough(client, engine):
    """A stolen cookie fails without the admin's own Tailscale login."""
    enrol(client, SoftKey())
    add_user(engine, ALICE, UserStatus.APPROVED)
    with Session(engine) as session:
        session.execute(
            User.__table__.update().where(User.tailscale_login == ALICE).values(role="admin")
        )
        session.commit()
    answer = client.get("/api/admin/users", headers={"Tailscale-User-Login": ALICE})
    assert answer.status_code == 401


@pytest.mark.parametrize(
    "key",
    [
        SoftKey(origin="https://evil.example"),
        SoftKey(rp_id="evil.example"),
    ],
    ids=["other origin", "other site"],
)
def test_passkeys_for_another_site_are_refused(client, key):
    assert enrol(client, key).status_code == 400


def test_a_passkey_without_user_verification_is_refused(client):
    key = SoftKey()
    enrol(client, key)
    client.post("/api/admin/sign-out")
    assert sign_in(client, key, flags=0x01).status_code == 400


def test_a_challenge_works_once(client):
    key = SoftKey()
    options = client.post("/api/admin/passkeys/options").json()
    credential = key.create(options)
    assert client.post("/api/admin/passkeys", json=credential).status_code == 201
    assert client.post("/api/admin/passkeys", json=credential).status_code == 400


def test_an_unknown_passkey_is_refused(client):
    enrol(client, SoftKey())
    client.post("/api/admin/sign-out")
    assert sign_in(client, SoftKey()).status_code == 400


def test_sign_out_ends_the_session(client, engine):
    enrol(client, SoftKey())
    assert count(engine, AdminSession) == 1
    assert client.post("/api/admin/sign-out").status_code == 204
    assert count(engine, AdminSession) == 0
    assert count(engine, AuthChallenge) == 0


def test_approve_and_block(client, engine):
    enrol(client, SoftKey())
    alice = add_user(engine, ALICE)
    me = {"Tailscale-User-Login": ALICE}
    assert client.get("/api/jobs", headers=me).status_code == 403
    users = client.get("/api/admin/users").json()
    assert users[0]["tailscale_login"] == ALICE  # waiting people first
    assert client.post(f"/api/admin/users/{alice}/approve").json()["status"] == "approved"
    assert client.get("/api/jobs", headers=me).status_code == 200
    assert client.post(f"/api/admin/users/{alice}/block").json()["status"] == "blocked"
    assert client.get("/api/jobs", headers=me).json()["detail"] == "blocked"


def test_the_admin_cannot_lock_themselves_out(client, engine):
    enrol(client, SoftKey())
    with Session(engine) as session:
        me = session.scalar(select(User.id).where(User.tailscale_login == OWNER))
    assert client.post(f"/api/admin/users/{me}/block").status_code == 409
    assert client.delete(f"/api/admin/users/{me}").status_code == 409


def test_delete_removes_the_person_and_every_file(client, engine, store):
    enrol(client, SoftKey())
    alice = add_user(engine, ALICE, UserStatus.APPROVED)
    with Session(engine) as session:
        job = Job(owner_id=alice, photo_count=1, length=JobLength.SHORT, quality=JobQuality.HD720)
        session.add(job)
        session.flush()
        photo = store.save(
            session,
            owner_id=alice,
            job_id=job.id,
            kind=AssetKind.PHOTO,
            mime="image/png",
            source=io.BytesIO(b"png"),
        )
        session.commit()
        path = store.path_for(photo)
        # A keyframe the web container cannot see: its row goes, its file waits for the worker.
        session.execute(
            Asset.__table__.insert().values(
                id=uuid.uuid4(),
                owner_id=alice,
                job_id=job.id,
                kind="keyframe",
                store="private",
                path="ab/" + "ab" + "0" * 30 + ".png",
                mime="image/png",
                size_bytes=1,
                sha256="x",
            )
        )
        session.commit()
    assert client.delete(f"/api/admin/users/{alice}").status_code == 204
    assert not path.exists()
    assert count(engine, User) == 1
    assert count(engine, Job) == count(engine, Asset) == 0
    assert count(engine, FileDeletion) == 1
    assert client.delete(f"/api/admin/users/{alice}").status_code == 404


def test_changes_from_other_sites_are_refused(client, engine):
    enrol(client, SoftKey())
    alice = add_user(engine, ALICE)
    answer = client.post(
        f"/api/admin/users/{alice}/approve", headers={"Sec-Fetch-Site": "cross-site"}
    )
    assert answer.status_code == 403


def test_every_admin_route_needs_the_admin(client, engine):
    """A new admin route that forgets its check fails here."""
    add_user(engine, ALICE, UserStatus.APPROVED)
    open_before_sign_in = {
        "/api/admin/status",
        "/api/admin/passkeys/options",
        "/api/admin/passkeys",
        "/api/admin/sign-in/options",
        "/api/admin/sign-in",
        "/api/admin/sign-out",
    }
    checked = 0
    for route in router.routes:
        path = route.path.replace("{user_id}", str(uuid.uuid4()))
        assert "{" not in path, f"teach this test about {route.path}"
        for method in route.methods:
            alice = client.request(method, path, headers={"Tailscale-User-Login": ALICE})
            assert alice.status_code == 404, route.path
            if route.path not in open_before_sign_in:
                assert client.request(method, path).status_code == 401, route.path
            checked += 1
    assert checked >= 10
