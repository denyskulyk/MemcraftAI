import io

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select, update
from sqlalchemy.orm import Session
from test_admin import ORIGIN, RP_ID, SoftKey, enrol

from memcraft.config import Settings, get_settings
from memcraft.models import (
    AssetKind,
    Job,
    JobLane,
    JobLength,
    JobQuality,
    JobStatus,
    JobStep,
    Setting,
    StepName,
    StepStatus,
    User,
    UserStatus,
)
from memcraft.storage import FileStore
from memcraft.web.app import create_app
from memcraft.web.auth import get_session, get_store

OWNER, ALICE = "owner@example.com", "alice@example.com"


@pytest.fixture
def store(tmp_path):
    (tmp_path / "shared").mkdir()
    return FileStore(tmp_path / "shared", None)


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
    enrol(client, SoftKey())
    return client


@pytest.fixture
def alice(engine):
    with Session(engine) as session:
        user = User(tailscale_login=ALICE, status=UserStatus.APPROVED)
        session.add(user)
        session.commit()
        return user.id


def add_job(engine, store, owner_id, status, **values):
    with Session(engine) as session:
        job = Job(
            owner_id=owner_id,
            photo_count=1,
            length=JobLength.SHORT,
            quality=JobQuality.HD720,
            lane=JobLane.RENDER,
            status=status,
            idea_text="private idea",
            **values,
        )
        session.add(job)
        session.flush()
        store.save(
            session,
            owner_id=owner_id,
            job_id=job.id,
            kind=AssetKind.PHOTO,
            mime="image/png",
            source=io.BytesIO(b"png"),
        )
        session.commit()
        return job.id


def test_the_admin_sees_every_job_with_its_steps(client, engine, store, alice):
    job_id = add_job(engine, store, alice, JobStatus.RENDERING, claimed_by="worker-1")
    with Session(engine) as session:
        session.add_all(
            [
                JobStep(job_id=job_id, step=StepName.SCENARIO, status=StepStatus.DONE),
                JobStep(
                    job_id=job_id,
                    step=StepName.CLIP,
                    attempt=1,
                    status=StepStatus.FAILED,
                    error="out of memory",
                ),
                JobStep(job_id=job_id, step=StepName.CLIP, attempt=2, status=StepStatus.RUNNING),
            ]
        )
        session.commit()
    other = add_job(engine, store, alice, JobStatus.FAILED, error="ComfyUI did not answer")
    answer = client.get("/api/admin/jobs")
    assert answer.status_code == 200
    jobs = {job["id"]: job for job in answer.json()}
    assert set(jobs) == {str(job_id), str(other)}
    job = jobs[str(job_id)]
    assert (job["owner_login"], job["status"], job["regime"], job["claimed_by"]) == (
        ALICE,
        "rendering",
        None,
        "worker-1",
    )
    assert job["steps"] == {
        "done": 1,
        "total": 2,  # one row per step: the retry replaces the failed try
        "current": "clip 1",
        "last_error": "out of memory",
    }
    assert jobs[str(other)]["error"] == "ComfyUI did not answer"
    assert jobs[str(other)]["steps"]["total"] == 0
    assert "idea_text" not in job  # the friend's idea is none of the admin's business


def test_a_job_the_friend_deleted_is_gone_from_the_admin_page(client, engine, store, alice):
    job_id = add_job(engine, store, alice, JobStatus.PREVIEW_READY)
    friend = {"Tailscale-User-Login": ALICE}
    assert client.delete(f"/api/jobs/{job_id}", headers=friend).status_code == 204
    assert client.get("/api/admin/jobs").json() == []


def test_cancel_takes_the_job_from_the_worker(client, engine, store, alice):
    job_id = add_job(engine, store, alice, JobStatus.RENDERING, claimed_by="worker-1")
    answer = client.post(f"/api/admin/jobs/{job_id}/cancel", json={"expected": "rendering"})
    assert answer.status_code == 200, answer.text
    assert (answer.json()["status"], answer.json()["claimed_by"]) == ("cancelled", None)
    with Session(engine) as session:
        job = session.get(Job, job_id)
        assert (job.status, job.claimed_by) == (JobStatus.CANCELLED, None)
    again = client.post(f"/api/admin/jobs/{job_id}/cancel", json={"expected": "cancelled"})
    assert again.status_code == 409
    # The friend can now delete it.
    friend = {"Tailscale-User-Login": ALICE}
    assert client.delete(f"/api/jobs/{job_id}", headers=friend).status_code == 204


def test_cancel_refuses_a_job_that_moved_on(client, engine, store, alice):
    job_id = add_job(engine, store, alice, JobStatus.PREVIEW_READY)
    answer = client.post(f"/api/admin/jobs/{job_id}/cancel", json={"expected": "rendering"})
    assert answer.status_code == 409
    assert "changed" in answer.json()["detail"]
    with Session(engine) as session:
        assert session.get(Job, job_id).status == JobStatus.PREVIEW_READY


def test_cancel_refuses_a_finished_job(client, engine, store, alice):
    job_id = add_job(engine, store, alice, JobStatus.RELEASED)
    answer = client.post(f"/api/admin/jobs/{job_id}/cancel", json={"expected": "released"})
    assert answer.status_code == 409
    assert "finished" in answer.json()["detail"]


def test_cancel_needs_the_expected_status(client, engine, store, alice):
    job_id = add_job(engine, store, alice, JobStatus.RENDER_QUEUED)
    assert client.post(f"/api/admin/jobs/{job_id}/cancel").status_code == 422
    unknown = "00000000-0000-0000-0000-000000000000"
    answer = client.post(f"/api/admin/jobs/{unknown}/cancel", json={"expected": "rendering"})
    assert answer.status_code == 404


def test_pause_and_stop(client, engine, store, alice):
    assert client.get("/api/admin/pause").json() == {
        "paused": False,
        "running": 0,
        "stopped": False,
    }
    job_id = add_job(engine, store, alice, JobStatus.RENDERING, claimed_by="worker-1")
    answer = client.post("/api/admin/pause", json={"paused": True})
    assert answer.json() == {"paused": True, "running": 1, "stopped": False}
    with Session(engine) as session:
        assert session.scalar(select(Setting.value).where(Setting.key == "paused")) is True
        # The worker finishes its step and lets go.
        session.execute(
            update(Job)
            .where(Job.id == job_id)
            .values(status=JobStatus.PREVIEW_READY, claimed_by=None)
        )
        session.commit()
    assert client.get("/api/admin/pause").json() == {"paused": True, "running": 0, "stopped": True}
    answer = client.post("/api/admin/pause", json={"paused": False})
    assert answer.json() == {"paused": False, "running": 0, "stopped": False}
    with Session(engine) as session:
        assert session.scalar(select(Setting.value).where(Setting.key == "paused")) is False


def test_pause_keeps_the_seeded_row(client, engine):
    with Session(engine) as session:
        session.add(Setting(key="paused", value=False))
        session.commit()
    client.post("/api/admin/pause", json={"paused": True})
    with Session(engine) as session:
        assert session.scalars(select(Setting.value).where(Setting.key == "paused")).all() == [True]


def test_pause_from_another_site_is_refused(client):
    answer = client.post(
        "/api/admin/pause", json={"paused": True}, headers={"Sec-Fetch-Site": "cross-site"}
    )
    assert answer.status_code == 403
    assert client.get("/api/admin/pause").json()["paused"] is False


@pytest.mark.parametrize("body", [{}, {"paused": "yes please"}, {"paused": None}])
def test_pause_needs_a_boolean(client, body):
    assert client.post("/api/admin/pause", json=body).status_code == 422


def test_a_retry_that_worked_hides_the_failed_try(engine):
    from memcraft.web.admin import _step_summaries

    with Session(engine) as session:
        owner = User(tailscale_login="friend@example.com")
        session.add(owner)
        session.flush()
        job = Job(
            owner_id=owner.id, photo_count=1, length=JobLength.SHORT, quality=JobQuality.HD720
        )
        session.add(job)
        session.flush()
        session.add_all(
            [
                JobStep(
                    job_id=job.id,
                    step=StepName.IDEA,
                    attempt=1,
                    status=StepStatus.FAILED,
                    error="model unloaded",
                ),
                JobStep(job_id=job.id, step=StepName.IDEA, attempt=2, status=StepStatus.DONE),
            ]
        )
        session.flush()
        summary = _step_summaries(session, [job.id])[job.id]
    assert (summary.done, summary.total, summary.last_error) == (1, 1, None)


def test_the_debug_drawer_lists_pictures_with_labels_and_prompts(client, engine, store):
    with Session(engine) as session:
        owner = session.scalar(select(User.id).where(User.tailscale_login == OWNER))
    job_id = add_job(engine, store, owner, JobStatus.PREVIEW_READY)
    scenario = {
        "title": "Log day",
        "bible": {"place": "a yard"},
        "problems": ["a note"],
        "assets": [
            {"kind": "background", "words": "", "prompt": "empty yard"},
            {"kind": "clean", "photo": 1, "words": "the man", "prompt": "the man alone"},
            {"kind": "prop", "photo": None, "words": "a log", "prompt": "a log alone"},
        ],
        "keyframes": [{"at": "start", "prompt": "start prompt"}, {"at": "end", "prompt": "end"}],
    }
    with Session(engine) as session:
        job = session.get(Job, job_id)
        session.add(
            JobStep(
                job_id=job_id,
                step=StepName.SCENARIO,
                inputs={},
                inputs_hash="h",
                status=StepStatus.DONE,
                output=scenario,
            )
        )
        session.add(
            JobStep(
                job_id=job_id,
                step=StepName.CHECK,
                inputs={},
                inputs_hash="c",
                status=StepStatus.DONE,
                output={
                    "kind": "keyframes",
                    "keyframes": [
                        {"ordinal": 0, "face": True, "face_why": "same man"},
                        {"ordinal": 1, "face": False, "face_why": "lost his beard"},
                    ],
                },
            )
        )
        for ordinal in (200, 201, 202, 0, 1, 100):
            store.save(
                session,
                owner_id=job.owner_id,
                job_id=job_id,
                kind=AssetKind.KEYFRAME,
                mime="image/png",
                source=io.BytesIO(b"png"),
                ordinal=ordinal,
            )
        session.commit()
    answer = client.get(f"/api/admin/jobs/{job_id}/debug")
    assert answer.status_code == 200
    data = answer.json()
    assert data["title"] == "Log day" and data["problems"] == ["a note"]
    labels = [p["label"] for p in data["pictures"]]
    assert labels == [
        "background",
        "character: photo 1",
        "prop: a log",
        "keyframe start",
        "keyframe end",
        "last frame of run 1",
    ]
    assert data["pictures"][3]["prompt"] == "start prompt"
    verdicts = [(p["verdict"], p["reason"]) for p in data["pictures"]]
    assert verdicts[3:5] == [("pass", "same man"), ("fail", "lost his beard")]
    assert verdicts[0] == (None, None)
    picture = client.get(f"/api/admin/files/{data['pictures'][0]['asset_id']}")
    assert picture.status_code == 200 and picture.content == b"png"
    # Only keyframe pictures: a photo is not served here.
    assert client.get(f"/api/admin/jobs/{job_id}/debug").status_code == 200
    assert client.get("/api/admin/files/00000000-0000-0000-0000-000000000000").status_code == 404
