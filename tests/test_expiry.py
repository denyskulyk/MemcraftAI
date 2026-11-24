import io
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta, timezone

import pytest
from sqlalchemy import event, func, select, update
from sqlalchemy.orm import Session

from memcraft import expiry
from memcraft.expiry import FINISHED, expire_finished, expire_previews
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
)
from memcraft.queue import StatusConflictError, transition
from memcraft.storage import FileStore

WAIT = 30
S = JobStatus
NOW = datetime(2026, 1, 10, 12, tzinfo=UTC)
PAST, FUTURE = NOW - timedelta(minutes=1), NOW + timedelta(minutes=1)
DAY = timedelta(hours=24)
MIMES = {
    AssetKind.PHOTO: "image/png",
    AssetKind.KEYFRAME: "image/png",
    AssetKind.LATENT: "application/x-safetensors",
    AssetKind.CLIP: "video/mp4",
    AssetKind.PREVIEW: "video/mp4",
}
on_postgres = pytest.mark.parametrize(
    "engine", [pytest.param("postgres", marks=pytest.mark.postgres)], indirect=True
)


@pytest.fixture
def roots(tmp_path):
    shared, private = tmp_path / "shared", tmp_path / "private"
    shared.mkdir()
    private.mkdir()
    return shared, private


@pytest.fixture
def store(roots):
    return FileStore(*roots)


def limit_lock_waits(_dialect, _record, _args, params):
    params["options"] = "-c lock_timeout=15s"


@pytest.fixture
def session(engine):
    if engine.dialect.name == "postgresql":  # a broken lock must fail, not hang
        engine.dispose()
        event.listen(engine, "do_connect", limit_lock_waits)
    with Session(engine) as session:
        yield session


def files(*folders):
    return [path for folder in folders for path in folder.rglob("*") if path.is_file()]


def new_job(store, session, status=S.PREVIEW_READY, expires_at=PAST, kinds=MIMES):
    owner_id = session.scalar(select(User.id))
    if owner_id is None:
        user = User(tailscale_login=uuid.uuid4().hex)
        session.add(user)
        session.flush()
        owner_id = user.id
    job = Job(
        owner_id=owner_id,
        idea_text="a cat",
        photo_count=1,
        length=JobLength.SHORT,
        quality=JobQuality.HD720,
        lane=JobLane.RENDER,
        status=status,
        error="a cat failed",
    )
    session.add(job)
    session.flush()
    session.add(Idea(job_id=job.id, text="cat", score=9, reason="r", proposed_by=IdeaSource.USER))
    session.add(
        JobStep(
            job_id=job.id,
            step=StepName.CLIP,
            inputs={"prompt": "cat"},
            output={"text": "cat"},
            inputs_hash="h",
            error="cat",
        )
    )
    for kind in kinds:
        preview = kind is AssetKind.PREVIEW
        store.save(
            session,
            owner_id=owner_id,
            job_id=job.id,
            kind=kind,
            mime=MIMES[kind],
            source=io.BytesIO(b"data"),
            expires_at=expires_at if preview else None,
        )
    session.commit()
    return job


def age(session, job, **values):
    session.execute(
        update(Asset)
        .where(Asset.job_id == job.id, Asset.kind == AssetKind.PREVIEW)
        .values(**values)
    )
    session.commit()


def count(session, model, *where):
    return session.scalar(select(func.count()).select_from(model).where(*where))


def unharmed(session, job):
    session.expire_all()
    return job.status is S.PREVIEW_READY and count(session, Asset, Asset.job_id == job.id) == 5


def test_expiry(store, session, roots):
    job = new_job(store, session)
    kept = new_job(store, session, expires_at=FUTURE)
    assert len(files(*roots)) == 10
    assert expire_previews(session, store, NOW) == 1
    assert job.status is S.EXPIRED
    assert job.idea_text is None
    assert job.error is None
    assert count(session, Asset, Asset.job_id == job.id) == 0
    assert count(session, Idea, Idea.job_id == job.id) == 0
    assert count(session, FileDeletion) == 0
    (step,) = session.scalars(select(JobStep).where(JobStep.job_id == job.id))
    assert (step.inputs, step.output, step.inputs_hash, step.error) == (None,) * 4
    # SQL NULL, not the JSON value null.
    assert count(session, JobStep, JobStep.inputs.is_(None), JobStep.output.is_(None)) == 1

    assert unharmed(session, kept)
    assert kept.idea_text == "a cat"
    assert count(session, Idea, Idea.job_id == kept.id) == 1
    paths = {store.path_for(asset) for asset in session.scalars(select(Asset))}
    assert len(paths) == 5
    assert set(files(*roots)) == paths
    assert expire_previews(session, store, NOW) == 0


def test_expires_exactly_at_the_time(store, session):
    job = new_job(store, session, expires_at=NOW)
    assert expire_previews(session, store, NOW - timedelta(seconds=1)) == 0
    assert expire_previews(session, store, NOW) == 1
    assert job.status is S.EXPIRED


def test_without_expires_at_a_preview_lives_a_day(store, session):
    old = new_job(store, session, expires_at=None)
    young = new_job(store, session, expires_at=None)
    age(session, old, created_at=NOW - DAY)
    age(session, young, created_at=NOW - DAY + timedelta(seconds=1))
    assert expire_previews(session, store, NOW) == 1
    assert old.status is S.EXPIRED
    assert unharmed(session, young)


def test_expires_at_wins_over_created_at(store, session):
    job = new_job(store, session, expires_at=FUTURE)
    age(session, job, created_at=NOW - 9 * DAY)
    assert expire_previews(session, store, NOW) == 0


def test_now_is_the_database_clock(store, session):
    old = new_job(store, session, expires_at=datetime.now(UTC) - timedelta(minutes=1))
    young = new_job(store, session, expires_at=None)
    assert expire_previews(session, store) == 1
    assert old.status is S.EXPIRED
    assert unharmed(session, young)


def test_now_in_another_time_zone(store, session):
    job = new_job(store, session, expires_at=NOW)
    ahead = timezone(timedelta(hours=5))
    early = (NOW - timedelta(minutes=1)).astimezone(ahead)
    assert expire_previews(session, store, early) == 0
    assert expire_previews(session, store, NOW.astimezone(ahead)) == 1
    assert job.status is S.EXPIRED


def test_now_needs_a_time_zone(store, session):
    with pytest.raises(ValueError, match="zone"):
        expire_previews(session, store, NOW.replace(tzinfo=None))


@pytest.mark.parametrize("status", sorted(set(JobStatus) - {S.PREVIEW_READY}))
def test_other_statuses_are_left_alone(store, session, status, roots):
    job = new_job(store, session, status)
    assert expire_previews(session, store, NOW) == 0
    assert job.status is status
    assert job.idea_text == "a cat"
    assert len(files(*roots)) == 5


def test_without_a_preview_nothing_expires(store, session, roots):
    """Only an old photo: the job is broken, but there is no preview to count from."""
    job = new_job(store, session, kinds=[AssetKind.PHOTO])
    session.execute(update(Asset).values(created_at=NOW - 9 * DAY))
    assert expire_previews(session, store, NOW) == 0
    assert job.status is S.PREVIEW_READY


def test_a_second_preview_keeps_the_job(store, session):
    job = new_job(store, session)
    store.save(
        session,
        owner_id=job.owner_id,
        job_id=job.id,
        kind=AssetKind.PREVIEW,
        mime="video/mp4",
        source=io.BytesIO(b"data"),
        expires_at=FUTURE,
    )
    session.commit()
    assert expire_previews(session, store, NOW) == 0
    assert expire_previews(session, store, FUTURE) == 1


def test_open_work_of_the_caller_is_committed(store, session):
    session.add(User(tailscale_login="kept"))
    expire_previews(session, store, NOW)
    session.rollback()
    assert count(session, User) == 1


def test_crash_before_the_commit_changes_nothing(store, session, roots):
    job = new_job(store, session)

    def boom(session):
        raise RuntimeError("crash")

    def arm(*_args):
        event.listen(session, "before_commit", boom)

    # Inside delete_assets(), after everything was sent to the database.
    event.listen(FileDeletion, "after_insert", arm)
    try:
        assert expire_previews(session, store, NOW) == 0
    finally:
        event.remove(FileDeletion, "after_insert", arm)
        event.remove(session, "before_commit", boom)
    assert unharmed(session, job)
    assert job.idea_text == "a cat"
    assert count(session, Idea) == 1
    assert count(session, FileDeletion) == 0
    assert len(files(*roots)) == 5
    assert expire_previews(session, store, NOW) == 1
    assert files(*roots) == []


def test_crash_after_the_commit_leaves_the_files_pending(store, session, roots, monkeypatch):
    job = new_job(store, session)

    class CrashError(BaseException):
        pass

    def crash(*_args):
        raise CrashError

    monkeypatch.setattr(store, "_purge", crash)
    with pytest.raises(CrashError):
        expire_previews(session, store, NOW)
    monkeypatch.undo()
    session.rollback()
    session.expire_all()
    assert job.status is S.EXPIRED
    assert count(session, Asset) == 0
    assert len(files(*roots)) == count(session, FileDeletion) == 5
    assert store.purge_deleted_files(session) == (5, 0)
    assert files(*roots) == []


def test_web_store_leaves_private_files_to_the_worker(store, session, roots):
    new_job(store, session)
    assert expire_previews(session, FileStore(roots[0]), NOW) == 1
    assert files(roots[0]) == []
    # Keyframes are shared since E16; Gemma copies, latents and clips stay with the worker.
    assert len(files(roots[1])) == 2
    assert store.purge_deleted_files(session) == (2, 0)


@pytest.mark.parametrize("bad", range(3))
def test_one_failing_job_does_not_stop_the_others(store, session, roots, monkeypatch, caplog, bad):
    jobs = sorted((new_job(store, session) for _ in range(3)), key=lambda job: job.id)
    failing = jobs[bad].id
    real = expiry.transition

    def broken(session, job, *args, **kw):
        result = real(session, job, *args, **kw)
        if job.id == failing:
            raise RuntimeError("broken")
        return result

    monkeypatch.setattr(expiry, "transition", broken)
    assert expire_previews(session, store, NOW) == 2
    monkeypatch.undo()
    session.expire_all()
    for n, job in enumerate(jobs):
        assert unharmed(session, job) if n == bad else job.status is S.EXPIRED
    assert len(files(*roots)) == 5
    assert str(failing) in caplog.text
    assert "a cat" not in caplog.text


# Closed, cancelled and failed jobs.


def finish(session, job, last_change):
    session.execute(update(Job).where(Job.id == job.id).values(updated_at=last_change))
    session.commit()


def stripped(session, job, status):
    session.expire_all()
    steps = session.scalars(select(JobStep).where(JobStep.job_id == job.id)).all()
    return (
        job.status is status
        and job.idea_text is None
        and job.error is None
        and count(session, Asset, Asset.job_id == job.id) == 0
        and count(session, Idea, Idea.job_id == job.id) == 0
        and len(steps) == 1
        and (steps[0].inputs, steps[0].output, steps[0].inputs_hash, steps[0].error) == (None,) * 4
    )


@pytest.mark.parametrize("status", FINISHED)
def test_finished_jobs_are_stripped_after_a_day(store, session, roots, status):
    old = new_job(store, session, status)
    young = new_job(store, session, status)
    finish(session, old, NOW - DAY)
    finish(session, young, NOW - DAY + timedelta(seconds=1))
    assert expire_finished(session, store, NOW) == 1
    assert stripped(session, old, status)  # the row stays, with its status
    assert young.idea_text == "a cat"
    assert count(session, Asset, Asset.job_id == young.id) == 5
    assert len(files(*roots)) == 5
    assert count(session, FileDeletion) == 0
    assert expire_finished(session, store, NOW + 9 * DAY) == 1  # the young one; old is done
    assert expire_finished(session, store, NOW + 9 * DAY) == 0
    assert files(*roots) == []


def test_a_finished_job_with_only_text_left_is_stripped(store, session):
    job = new_job(store, session, S.FAILED, kinds=[])
    finish(session, job, NOW - DAY)
    assert expire_finished(session, store, NOW) == 1
    assert stripped(session, job, S.FAILED)


@pytest.mark.parametrize("status", sorted(set(JobStatus) - set(FINISHED)))
def test_other_jobs_are_not_stripped(store, session, status):
    job = new_job(store, session, status)
    finish(session, job, NOW - 9 * DAY)
    assert expire_finished(session, store, NOW) == 0
    session.expire_all()
    assert job.status is status
    assert job.idea_text == "a cat"


def test_expire_previews_leaves_finished_jobs_alone(store, session):
    job = new_job(store, session, S.CLOSED)
    finish(session, job, NOW - 9 * DAY)
    assert expire_previews(session, store, NOW) == 0
    assert count(session, Asset, Asset.job_id == job.id) == 5


def test_finished_now_is_the_database_clock(store, session):
    job = new_job(store, session, S.CANCELLED)
    assert expire_finished(session, store) == 0
    finish(session, job, datetime.now(UTC) - DAY - timedelta(minutes=1))
    assert expire_finished(session, store) == 1
    assert stripped(session, job, S.CANCELLED)


def test_finished_needs_a_time_zone(store, session):
    with pytest.raises(ValueError, match="zone"):
        expire_finished(session, store, NOW.replace(tzinfo=None))


@on_postgres
def test_a_locked_finished_job_is_skipped(store, engine, session):
    job = new_job(store, session, S.FAILED)
    finish(session, job, NOW - DAY)
    session.execute(select(Job).where(Job.id == job.id).with_for_update()).all()
    with Session(engine) as sweeper:
        assert expire_finished(sweeper, store, NOW) == 0
    session.commit()
    with Session(engine) as sweeper:
        assert expire_finished(sweeper, store, NOW) == 1


# Threads, each with its own session and connection.


@on_postgres
def test_an_open_release_wins(store, engine, session, roots, caplog):
    job = new_job(store, session)
    other = new_job(store, session)
    transition(session, job, S.RELEASE_QUEUED, expected=S.PREVIEW_READY)
    with Session(engine) as sweeper:
        assert expire_previews(sweeper, store, NOW) == 1  # the other one
    assert caplog.text == ""  # skipped, not failed on a lock timeout
    session.commit()
    assert job.status is S.RELEASE_QUEUED
    assert other.status is S.EXPIRED
    with Session(engine) as sweeper:
        assert expire_previews(sweeper, store, NOW) == 0
    assert len(files(*roots)) == 5


@on_postgres
def test_a_release_committed_after_the_list_was_read_wins(
    store, engine, session, roots, monkeypatch, caplog
):
    job = new_job(store, session)
    real = expiry._expire

    def late(*args):
        transition(session, job, S.RELEASE_QUEUED, expected=S.PREVIEW_READY)
        session.commit()
        return real(*args)

    monkeypatch.setattr(expiry, "_expire", late)
    with Session(engine) as sweeper:
        assert expire_previews(sweeper, store, NOW) == 0
    assert caplog.text == ""  # not due any more, so not even tried
    assert job.status is S.RELEASE_QUEUED
    assert len(files(*roots)) == 5


@on_postgres
def test_a_release_during_the_deletion_loses(store, engine, session, roots):
    """The sweep holds the lock when the release arrives."""
    job_id = new_job(store, session).id
    session.commit()
    holding, go_on = threading.Event(), threading.Event()

    with Session(engine) as sweeper, ThreadPoolExecutor(2) as pool:

        def pause(_session):
            if sweeper.deleted_here:
                holding.set()
                go_on.wait(WAIT)

        sweeper.deleted_here = False
        event.listen(
            sweeper,
            "after_flush",
            lambda session, _context: setattr(
                session,
                "deleted_here",
                session.deleted_here or any(isinstance(o, FileDeletion) for o in session.new),
            ),
        )
        event.listen(sweeper, "before_commit", pause)

        def release():
            with Session(engine) as mine:
                job = mine.get(Job, job_id)
                try:
                    transition(mine, job, S.RELEASE_QUEUED, expected=S.PREVIEW_READY)
                except StatusConflictError:
                    return "lost"
                mine.commit()
                return "won"

        sweep = pool.submit(expire_previews, sweeper, store, NOW)
        assert holding.wait(WAIT)
        releasing = pool.submit(release)
        with pytest.raises(TimeoutError):
            releasing.result(0.3)  # it waits for the lock
        go_on.set()
        assert sweep.result(WAIT) == 1
        assert releasing.result(WAIT) == "lost"
    session.expire_all()
    assert session.get(Job, job_id).status is S.EXPIRED
    assert files(*roots) == []


@on_postgres
def test_release_and_expiry_at_once(store, engine, session, roots):
    for _ in range(15):
        job_id = new_job(store, session).id
        session.commit()
        barrier = threading.Barrier(2)

        def release(job_id=job_id, barrier=barrier):
            with Session(engine) as mine:
                job = mine.get(Job, job_id)
                barrier.wait(WAIT)
                try:
                    transition(mine, job, S.RELEASE_QUEUED, expected=S.PREVIEW_READY)
                except StatusConflictError:
                    return False
                mine.commit()
                return True

        def sweep(barrier=barrier):
            with Session(engine) as mine:
                mine.connection()
                barrier.wait(WAIT)
                return expire_previews(mine, store, NOW)

        with ThreadPoolExecutor(2) as pool:
            released, expired = pool.submit(release), pool.submit(sweep)
            released, expired = released.result(WAIT), expired.result(WAIT)
        session.expire_all()
        job = session.get(Job, job_id)
        left = count(session, Asset, Asset.job_id == job_id)
        assert (released, expired, job.status, left) in [
            (True, 0, S.RELEASE_QUEUED, 5),
            (False, 1, S.EXPIRED, 0),
        ]
        assert len(files(*roots)) == count(session, Asset)
        session.commit()


@on_postgres
def test_a_save_in_progress_puts_expiry_off(store, engine, session, roots):
    """Otherwise the expired job would keep the file that was being saved."""
    job = new_job(store, session)
    store.save(
        session,
        owner_id=job.owner_id,
        job_id=job.id,
        kind=AssetKind.KEYFRAME,
        mime="image/png",
        source=io.BytesIO(b"data"),
    )
    with Session(engine) as sweeper:
        assert expire_previews(sweeper, store, NOW) == 0
        session.commit()
        assert expire_previews(sweeper, store, NOW) == 1
    assert files(*roots) == []


@on_postgres
def test_two_sweeps_at_once(store, engine, session, roots):
    for _ in range(20):
        new_job(store, session)
    barrier = threading.Barrier(2)

    def sweep():
        with Session(engine) as mine:
            mine.connection()
            barrier.wait(WAIT)
            return expire_previews(mine, store, NOW)

    with ThreadPoolExecutor(2) as pool:
        results = [future.result(WAIT) for future in [pool.submit(sweep), pool.submit(sweep)]]
    assert sum(results) == 20
    session.expire_all()
    assert set(session.scalars(select(Job.status))) == {S.EXPIRED}
    assert count(session, Asset) == count(session, FileDeletion) == 0
    assert files(*roots) == []
