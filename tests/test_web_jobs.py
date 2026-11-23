import asyncio
import io
import json
import uuid
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from memcraft.config import Settings, get_settings
from memcraft.models import (
    Asset,
    AssetKind,
    FileDeletion,
    Idea,
    IdeaSource,
    Job,
    JobLane,
    JobLength,
    JobQuality,
    JobStatus,
    JobStep,
    StepName,
    User,
    UserStatus,
)
from memcraft.storage import FileStore
from memcraft.web import api
from memcraft.web.app import create_app
from memcraft.web.auth import get_session, get_store

ALICE, BOB = "alice@example.com", "bob@example.com"
BASE = "http://localhost"


@pytest.fixture
def store(tmp_path):
    (tmp_path / "shared").mkdir()
    return FileStore(tmp_path / "shared", None)  # as in the web container


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
    client = TestClient(app, base_url=BASE)
    client.headers["Tailscale-User-Login"] = ALICE
    client.headers["Sec-Fetch-Site"] = "same-origin"
    return client


@pytest.fixture
def alice(engine):
    return add_user(engine, ALICE)


def add_user(engine, login, status=UserStatus.APPROVED):
    with Session(engine) as session:
        user = User(tailscale_login=login, status=status)
        session.add(user)
        session.commit()
        return user.id


def add_job(engine, store, owner_id, status, *, idea=None, preview=False, **values):
    """A job as the worker would leave it, with a photo, maybe an idea and a preview."""
    lane = JobLane.IDEA if status in (JobStatus.IDEA_QUEUED, JobStatus.IDEA_RUNNING) else None
    with Session(engine) as session:
        job = Job(
            owner_id=owner_id,
            photo_count=1,
            length=JobLength.SHORT,
            quality=JobQuality.HD720,
            status=status,
            lane=lane or JobLane.RENDER,
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
        if preview:
            store.save(
                session,
                owner_id=owner_id,
                job_id=job.id,
                kind=AssetKind.PREVIEW,
                mime="video/mp4",
                source=io.BytesIO(b"mp4 preview"),
                expires_at=preview if isinstance(preview, datetime) else None,
            )
        if idea is not None:
            session.add(
                Idea(
                    job_id=job.id,
                    text=idea,
                    score=7,
                    reason="a cat that thinks it is a lion",
                    proposed_by=IdeaSource.GEMMA,
                )
            )
        session.commit()
        return job.id


def job_row(engine, job_id):
    with Session(engine) as session:
        return session.get(Job, job_id)


def count(engine, model, *where):
    with Session(engine) as session:
        return session.scalar(select(func.count()).select_from(model).where(*where))


def test_the_apps_idea_waits_for_approval(client, engine, store, alice):
    job_id = add_job(engine, store, alice, JobStatus.AWAITING_APPROVAL, idea="The cat roars")
    detail = client.get(f"/api/jobs/{job_id}").json()
    assert detail["idea"]["text"] == "The cat roars"
    assert (detail["idea"]["score"], detail["idea"]["proposed_by"]) == (7, "gemma")
    assert detail["idea"]["approved_at"] is None
    answer = client.post(f"/api/jobs/{job_id}/approve")
    assert answer.status_code == 200, answer.text
    body = answer.json()
    assert (body["status"], body["queue_position"]) == ("render_queued", 0)
    assert body["idea"]["approved_at"] is not None
    job = job_row(engine, job_id)
    assert (job.status, job.lane) == (JobStatus.RENDER_QUEUED, JobLane.RENDER)
    # The second press, or a press in another tab, finds the job moved on.
    assert client.post(f"/api/jobs/{job_id}/approve").status_code == 409
    assert client.post(f"/api/jobs/{job_id}/reject").status_code == 409


def test_reject_closes_the_job(client, engine, store, alice):
    job_id = add_job(engine, store, alice, JobStatus.AWAITING_APPROVAL, idea="The cat roars")
    answer = client.post(f"/api/jobs/{job_id}/reject")
    assert answer.json()["status"] == "closed"
    assert job_row(engine, job_id).status == JobStatus.CLOSED
    assert count(engine, Idea, Idea.approved_at.is_not(None)) == 0
    assert client.post(f"/api/jobs/{job_id}/approve").status_code == 409


def test_a_conflict_leaves_the_idea_unapproved(client, engine, store, alice):
    job_id = add_job(engine, store, alice, JobStatus.RENDER_QUEUED, idea="The cat roars")
    assert client.post(f"/api/jobs/{job_id}/approve").status_code == 409
    assert count(engine, Idea, Idea.approved_at.is_not(None)) == 0


def test_the_friends_own_idea_shows_its_score(client, engine, store, alice):
    job_id = add_job(engine, store, alice, JobStatus.RENDER_QUEUED, idea_text="My idea")
    with Session(engine) as session:
        session.add(
            Idea(
                job_id=job_id,
                text="My idea",
                score=9,
                reason="understood",
                proposed_by=IdeaSource.USER,
            )
        )
        session.commit()
    idea = client.get(f"/api/jobs/{job_id}").json()["idea"]
    assert (idea["proposed_by"], idea["score"], idea["reason"]) == ("user", 9, "understood")


def test_without_an_idea_row_approve_is_refused(client, engine, store, alice):
    job_id = add_job(engine, store, alice, JobStatus.AWAITING_APPROVAL)
    assert client.post(f"/api/jobs/{job_id}/approve").status_code == 409
    assert job_row(engine, job_id).status == JobStatus.AWAITING_APPROVAL


@pytest.mark.parametrize("action", ["approve", "reject", "release"])
def test_nobody_acts_on_another_users_job(client, engine, store, alice, action):
    bob = add_user(engine, BOB)
    job_id = add_job(
        engine, store, bob, JobStatus.AWAITING_APPROVAL, idea="Bob's idea", preview=True
    )
    assert client.post(f"/api/jobs/{job_id}/{action}").status_code == 404
    assert client.delete(f"/api/jobs/{job_id}").status_code == 404
    assert client.get(f"/api/jobs/{job_id}/events").status_code == 404
    assert job_row(engine, job_id).status == JobStatus.AWAITING_APPROVAL


def test_release_keeps_the_quality_or_takes_a_new_one(client, engine, store, alice):
    kept = add_job(engine, store, alice, JobStatus.PREVIEW_READY, preview=True)
    changed = add_job(engine, store, alice, JobStatus.PREVIEW_READY, preview=True)
    assert client.post(f"/api/jobs/{kept}/release").json()["quality"] == "hd720"
    answer = client.post(f"/api/jobs/{changed}/release", json={"quality": "fhd1080"})
    assert answer.status_code == 200, answer.text
    assert (answer.json()["status"], answer.json()["quality"]) == ("release_queued", "fhd1080")
    job = job_row(engine, changed)
    assert (job.status, job.lane, job.quality) == (
        JobStatus.RELEASE_QUEUED,
        JobLane.RENDER,
        JobQuality.FHD1080,
    )
    assert client.post(f"/api/jobs/{changed}/release").status_code == 409


def test_release_refuses_a_bad_quality(client, engine, store, alice):
    job_id = add_job(engine, store, alice, JobStatus.PREVIEW_READY, preview=True)
    assert client.post(f"/api/jobs/{job_id}/release", json={"quality": "4k"}).status_code == 422
    assert job_row(engine, job_id).status == JobStatus.PREVIEW_READY


def test_an_expired_preview_cannot_be_released(client, engine, store, alice):
    past = datetime.now(UTC) - timedelta(minutes=1)
    job_id = add_job(engine, store, alice, JobStatus.PREVIEW_READY, preview=past)
    answer = client.post(f"/api/jobs/{job_id}/release", json={"quality": "fhd1080"})
    assert answer.status_code == 409
    job = job_row(engine, job_id)
    assert (job.status, job.quality) == (JobStatus.PREVIEW_READY, JobQuality.HD720)


def test_release_needs_a_preview_waiting(client, engine, store, alice):
    job_id = add_job(engine, store, alice, JobStatus.RENDERING)
    assert client.post(f"/api/jobs/{job_id}/release").status_code == 409


def test_delete_removes_the_job_and_every_file(client, engine, store, alice):
    job_id = add_job(engine, store, alice, JobStatus.PREVIEW_READY, idea="x", preview=True)
    with Session(engine) as session:
        paths = [store.path_for(a) for a in session.scalars(select(Asset))]
        session.add(JobStep(job_id=job_id, step=StepName.CLIP))
        # A keyframe the web container cannot see: its row goes, its file waits for the worker.
        session.execute(
            Asset.__table__.insert().values(
                id=uuid.uuid4(),
                owner_id=alice,
                job_id=job_id,
                kind="keyframe",
                store="private",
                path="cd/cd" + "0" * 30 + ".png",
                mime="image/png",
                size_bytes=1,
                sha256="x",
            )
        )
        session.commit()
    assert len(paths) == 2 and all(path.exists() for path in paths)
    assert client.delete(f"/api/jobs/{job_id}").status_code == 204
    assert not any(path.exists() for path in paths)
    for model in (Job, Asset, Idea, JobStep):
        assert count(engine, model) == 0, model
    assert count(engine, FileDeletion) == 1
    assert client.delete(f"/api/jobs/{job_id}").status_code == 404
    assert client.get(f"/api/jobs/{job_id}").status_code == 404


@pytest.mark.parametrize(
    "status", [JobStatus.IDEA_RUNNING, JobStatus.RENDERING, JobStatus.RELEASING]
)
def test_a_running_job_cannot_be_deleted(client, engine, store, alice, status):
    job_id = add_job(engine, store, alice, status, claimed_by="worker-1")
    answer = client.delete(f"/api/jobs/{job_id}")
    assert answer.status_code == 409
    assert "wait" in answer.json()["detail"]
    assert (count(engine, Job), count(engine, Asset)) == (1, 1)


def test_a_waiting_job_can_be_deleted(client, engine, store, alice):
    job_id = add_job(engine, store, alice, JobStatus.RENDER_QUEUED)
    assert client.delete(f"/api/jobs/{job_id}").status_code == 204
    assert count(engine, Job) == 0


def test_history_links_the_released_video(client, engine, store, alice):
    released = add_job(engine, store, alice, JobStatus.RELEASED)
    waiting = add_job(engine, store, alice, JobStatus.RENDER_QUEUED)
    with Session(engine) as session:
        video = store.save(
            session,
            owner_id=alice,
            job_id=released,
            kind=AssetKind.RELEASE,
            mime="video/mp4",
            source=io.BytesIO(b"final mp4"),
        )
        session.commit()
        video_id = video.id
    jobs = {job["id"]: job for job in client.get("/api/jobs").json()}
    assert jobs[str(released)]["video_id"] == str(video_id)
    assert jobs[str(waiting)]["video_id"] is None
    inline = client.get(f"/api/files/{video_id}")
    assert inline.headers["content-disposition"].startswith("inline")
    download = client.get(f"/api/files/{video_id}?download=1")
    assert download.content == b"final mp4"
    assert download.headers["content-disposition"].startswith("attachment")
    assert "memcraft-release.mp4" in download.headers["content-disposition"]


def events(text):
    """Server-sent events as (event, data) pairs, and the comment lines."""
    found, comments = [], []
    for block in text.split("\n\n"):
        name, data = "message", None
        for line in block.splitlines():
            if line.startswith(":"):
                comments.append(line)
            elif line.startswith("event: "):
                name = line.removeprefix("event: ")
            elif line.startswith("data: "):
                data = json.loads(line.removeprefix("data: "))
        if data is not None:
            found.append((name, data))
    return found, comments


@pytest.mark.parametrize(
    "status", [JobStatus.RELEASED, JobStatus.PREVIEW_READY, JobStatus.AWAITING_APPROVAL]
)
def test_the_stream_ends_when_nothing_moves(client, engine, store, alice, status):
    job_id = add_job(engine, store, alice, status, progress=1)
    answer = client.get(f"/api/jobs/{job_id}/events")
    assert answer.status_code == 200
    assert answer.headers["content-type"].startswith("text/event-stream")
    assert answer.headers["cache-control"] == "no-store"
    assert answer.text.startswith("retry: ")
    found, _ = events(answer.text)
    state = {"status": status.value, "progress": 1.0, "queue_position": None, "error": None}
    assert found == [("message", state), ("end", state)]


def run_stream(engine, owner_id, job_id, steps):
    """Drive the stream; before each next event, steps[i] changes the database."""

    async def drive():
        with Session(engine) as session:
            first = api._state(session, owner_id, job_id)
        stream = api._events(engine, owner_id, job_id, first)
        out = []
        try:
            async for chunk in stream:
                out.append(chunk)
                if chunk.startswith("data:") or chunk.startswith("event:"):
                    if steps:
                        steps.pop(0)()
        finally:
            await stream.aclose()
        return "".join(out)

    return asyncio.run(asyncio.wait_for(drive(), timeout=20))


def change(engine, job_id, **values):
    def run():
        with Session(engine) as session:
            session.execute(update(Job).where(Job.id == job_id).values(**values))
            session.commit()

    return run


def test_the_stream_follows_progress_to_the_preview(engine, store, alice, monkeypatch):
    monkeypatch.setattr(api, "POLL_SECONDS", 0.01)
    job_id = add_job(engine, store, alice, JobStatus.RENDERING, progress=0.2)
    text = run_stream(
        engine,
        alice,
        job_id,
        [
            change(engine, job_id, progress=0.5),
            change(engine, job_id, status=JobStatus.PREVIEW_READY, progress=1),
        ],
    )
    found, _ = events(text)
    assert [(name, data["status"], data["progress"]) for name, data in found] == [
        ("message", "rendering", 0.2),
        ("message", "rendering", 0.5),
        ("message", "preview_ready", 1.0),
        ("end", "preview_ready", 1.0),
    ]


def test_the_stream_shows_the_queue_place(engine, store, alice, monkeypatch):
    monkeypatch.setattr(api, "POLL_SECONDS", 0.01)
    ahead = add_job(engine, store, alice, JobStatus.RENDER_QUEUED)
    with Session(engine) as session:  # SQLite's clock counts whole seconds
        session.execute(update(Job).values(queued_at=datetime.now(UTC) - timedelta(minutes=1)))
        session.commit()
    job_id = add_job(engine, store, alice, JobStatus.RENDER_QUEUED)
    text = run_stream(
        engine,
        alice,
        job_id,
        [
            change(engine, ahead, status=JobStatus.CANCELLED),
            change(engine, job_id, status="failed", error="out of memory"),
        ],
    )
    found, _ = events(text)
    assert [(data["status"], data["queue_position"]) for _, data in found] == [
        ("render_queued", 1),
        ("render_queued", 0),
        ("failed", None),
        ("failed", None),
    ]
    assert found[-1] == ("end", {**found[-1][1], "error": "out of memory"})


def test_the_stream_sends_keep_alives_and_ends_in_time(engine, store, alice, monkeypatch):
    monkeypatch.setattr(api, "POLL_SECONDS", 0.01)
    monkeypatch.setattr(api, "HEARTBEAT_SECONDS", 0.05)
    monkeypatch.setattr(api, "STREAM_SECONDS", 0.3)
    job_id = add_job(engine, store, alice, JobStatus.RENDERING, progress=0.2)
    text = run_stream(engine, alice, job_id, [])
    found, comments = events(text)
    assert [name for name, _ in found] == ["message"]  # no "end": the browser reconnects
    assert comments and set(comments) == {": keep-alive"}


def test_the_stream_says_when_the_job_is_gone(engine, store, alice, monkeypatch):
    monkeypatch.setattr(api, "POLL_SECONDS", 0.01)
    job_id = add_job(engine, store, alice, JobStatus.RENDER_QUEUED)

    def delete():
        with Session(engine) as session:
            store.delete_job(session, job_id)

    text = run_stream(engine, alice, job_id, [delete])
    found, _ = events(text)
    assert [name for name, _ in found] == ["message", "gone"]
