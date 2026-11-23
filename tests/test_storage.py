import hashlib
import io
import os
import re
import subprocess
import sys
import time
import uuid
from datetime import timedelta

import pytest
from conftest import require_test_database
from sqlalchemy import event, func, select
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.orm import Session

from memcraft import storage
from memcraft.models import (
    Asset,
    AssetKind,
    AssetStore,
    FileDeletion,
    Idea,
    IdeaSource,
    Job,
    JobLength,
    JobQuality,
    JobStep,
    StepName,
    User,
)
from memcraft.storage import (
    FileStore,
    FileTooLargeError,
    StoreUnavailableError,
    UnsafePathError,
    UnsupportedTypeError,
)

HOUR = timedelta(hours=1)
on_windows = pytest.mark.skipif(sys.platform != "win32", reason="Windows file locks")


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


def new_job(session, owner_id=None):
    if owner_id is None:
        user = User(tailscale_login=uuid.uuid4().hex)
        session.add(user)
        session.flush()
        owner_id = user.id
    job = Job(owner_id=owner_id, photo_count=1, length=JobLength.SHORT, quality=JobQuality.HD720)
    session.add(job)
    session.commit()
    return job


def save(store, session, job, kind=AssetKind.PHOTO, mime="image/png", data=b"data", **kw):
    return store.save(
        session,
        owner_id=job.owner_id,
        job_id=job.id,
        kind=kind,
        mime=mime,
        source=io.BytesIO(data),
        **kw,
    )


def count(session, model):
    return session.scalar(select(func.count()).select_from(model))


def age(path, seconds=7200):
    stamp = time.time() - seconds
    os.utime(path, (stamp, stamp))


def fail_commit(session):
    """The database work succeeds, then the commit does not happen."""

    def boom(_session):
        raise RuntimeError("crash")

    event.listen(session, "before_commit", boom)
    return lambda: event.remove(session, "before_commit", boom)


# Files only.


@pytest.mark.parametrize(
    "path",
    [
        "../secret.png",
        "ab/../../secret.png",
        "ab/../" + "ab" * 16 + ".png",
        "/etc/passwd",
        "C:/Windows/win.ini",
        "ab\\" + "ab" * 16 + ".png",
        "ab/" + "ab" * 16 + ".png:stream",
        "ab/" + "ab" * 16 + ".exe",
        "ab/" + "ab" * 16 + ".png\n",
        "ab/" + "cd" * 16 + ".png",
        "ab/" + "ab" * 16 + ".png.part",
        "",
    ],
)
def test_unsafe_path_is_refused(store, path):
    with pytest.raises(UnsafePathError):
        store.path_for(Asset(store=AssetStore.SHARED, path=path))


def test_linked_folder_leaving_the_root_is_refused(store, roots, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    link = roots[0] / "ab"
    if sys.platform == "win32":
        made = subprocess.run(  # noqa: S603
            ["cmd", "/c", "mklink", "/J", str(link), str(outside)],  # noqa: S607
            capture_output=True,
        )
        assert made.returncode == 0
    else:
        link.symlink_to(outside, target_is_directory=True)
    asset = Asset(store=AssetStore.SHARED, path="ab/" + "ab" * 16 + ".png")
    with pytest.raises(UnsafePathError):
        store.path_for(asset)
    link.rmdir() if sys.platform == "win32" else link.unlink()


def test_unavailable_root(web, tmp_path):
    path = "ab/" + "ab" * 16 + ".mp4"
    with pytest.raises(StoreUnavailableError):
        web.path_for(Asset(store=AssetStore.PRIVATE, path=path))
    unmounted = FileStore(tmp_path / "missing")
    with pytest.raises(StoreUnavailableError):
        unmounted.path_for(Asset(store=AssetStore.SHARED, path=path))


def test_roots_must_be_separate(tmp_path):
    with pytest.raises(ValueError, match="separate"):
        FileStore(tmp_path, tmp_path)
    with pytest.raises(ValueError, match="separate"):
        FileStore(tmp_path, tmp_path / "private")


def test_every_kind_has_a_store():
    assert set(storage.STORE_FOR_KIND) == set(AssetKind)


def test_only_a_test_database_may_be_dropped():
    require_test_database("memcraft_test")
    for name in ("memcraft", "test_memcraft", "", None):
        with pytest.raises(RuntimeError):
            require_test_database(name)


# Saving.


def test_save(store, session, roots):
    job = new_job(session)
    data = os.urandom(3 * 1024 * 1024 + 7)
    asset = save(store, session, job, AssetKind.CLIP, "video/mp4", data)
    session.commit()
    assert re.fullmatch(r"([0-9a-f]{2})/\1[0-9a-f]{30}\.mp4", asset.path)
    assert asset.id.hex not in asset.path
    assert asset.store is AssetStore.PRIVATE
    assert asset.size_bytes == len(data)
    assert asset.sha256 == hashlib.sha256(data).hexdigest()
    assert store.path_for(asset).read_bytes() == data
    assert files(*roots) == [store.path_for(asset)]


def test_photo_is_shared(store, web, session, roots):
    asset = save(store, session, new_job(session))
    assert web.path_for(asset).is_relative_to(roots[0])


def test_save_flushes_a_pending_job_first(store, session):
    job = new_job(session)
    other = Job(
        id=uuid.uuid4(),
        owner_id=job.owner_id,
        photo_count=1,
        length=JobLength.SHORT,
        quality=JobQuality.HD720,
    )
    session.add(other)
    assert save(store, session, other).job_id == other.id


def test_web_cannot_save_private_files(web, session):
    with pytest.raises(StoreUnavailableError):
        save(web, session, new_job(session), AssetKind.KEYFRAME)


def test_unknown_type_is_refused(store, session, roots):
    with pytest.raises(UnsupportedTypeError):
        save(store, session, new_job(session), mime="text/html")
    assert files(*roots) == []


def test_size_limit(store, session, roots):
    job = new_job(session)
    save(store, session, job, data=b"x" * 10, max_bytes=10)
    with pytest.raises(FileTooLargeError):
        save(store, session, job, data=b"x" * 11, max_bytes=10)
    assert len(files(*roots)) == 1
    assert count(session, Asset) == 1


def test_failed_read_leaves_nothing(store, session, roots):
    class Broken(io.BytesIO):
        def read(self, size=-1):
            raise ConnectionError

    job = new_job(session)
    with pytest.raises(ConnectionError):
        store.save(
            session,
            owner_id=job.owner_id,
            job_id=job.id,
            kind=AssetKind.PHOTO,
            mime="image/png",
            source=Broken(),
        )
    assert files(*roots) == []


def test_failed_row_removes_the_file(store, session, roots):
    job = new_job(session)
    job = Job(id=uuid.uuid4(), owner_id=job.owner_id)  # never added
    with pytest.raises(IntegrityError):
        save(store, session, job)
    assert files(*roots) == []


def test_rollback_removes_the_file(store, session, roots):
    job = new_job(session)
    kept = save(store, session, job)
    session.commit()
    save(store, session, job)
    session.rollback()
    assert files(*roots) == [store.path_for(kept)]
    session.commit()
    session.rollback()
    assert files(*roots) == [store.path_for(kept)]


def test_close_without_commit_removes_the_file(store, session, roots):
    job = new_job(session)
    save(store, session, job)
    session.close()
    assert files(*roots) == []


def test_savepoints(store, session, roots):
    job = new_job(session)
    outer = save(store, session, job)
    with session.begin_nested() as undone:
        save(store, session, job)
        undone.rollback()
    with session.begin_nested():
        released = save(store, session, job)
    expected = {store.path_for(outer), store.path_for(released)}
    assert set(files(*roots)) == expected
    session.commit()
    assert set(files(*roots)) == expected


def test_outer_rollback_removes_a_released_savepoint(store, session, roots):
    job = new_job(session)
    with session.begin_nested():
        save(store, session, job)
    session.rollback()
    assert files(*roots) == []


def test_crash_before_the_database_commit_removes_the_file(store, session, roots):
    job = new_job(session)
    save(store, session, job)
    restore = fail_commit(session)
    with pytest.raises(RuntimeError):
        session.commit()
    session.rollback()
    restore()
    assert files(*roots) == []


def test_unclear_commit_keeps_the_file_for_the_sweep(store, engine, session, roots, monkeypatch):
    job = new_job(session)
    save(store, session, job)

    def lost(_connection):
        raise ConnectionError

    monkeypatch.setattr(engine.dialect, "do_commit", lost)
    with pytest.raises(ConnectionError):
        session.commit()
    session.rollback()
    monkeypatch.undo()
    (left,) = files(*roots)
    age(left)
    assert store.sweep_orphans(session, HOUR) == 1
    # The next rollback cleans up as usual.
    save(store, session, job)
    session.rollback()
    assert files(*roots) == []


# Deleting.


def test_delete_removes_row_and_file(store, session, roots):
    job = new_job(session)
    gone, kept = save(store, session, job), save(store, session, job)
    session.commit()
    kept_path = store.path_for(kept)
    assert store.delete_assets(session, [gone.id]) == 1
    assert files(*roots) == [kept_path]
    assert count(session, Asset) == 1
    assert count(session, FileDeletion) == 0


def test_delete_by_condition(store, session, roots):
    job = new_job(session)
    save(store, session, job)
    save(store, session, job, AssetKind.LATENT, "application/x-safetensors")
    session.commit()
    assert store.delete_assets(session, where=Asset.kind == AssetKind.LATENT) == 1
    assert files(roots[1]) == []
    assert len(files(roots[0])) == 1


def test_delete_needs_a_selection(store, session):
    with pytest.raises(ValueError, match="either"):
        store.delete_assets(session)
    assert store.delete_assets(session, []) == 0


def test_delete_of_an_uncommitted_asset(store, session, roots):
    asset = save(store, session, new_job(session))
    assert store.delete_assets(session, [asset.id]) == 1
    assert files(*roots) == []


def test_rollback_of_a_delete_keeps_both(store, session, roots):
    asset = save(store, session, new_job(session))
    session.commit()
    path = store.path_for(asset)
    restore = fail_commit(session)
    with pytest.raises(RuntimeError):
        store.delete_job(session, asset.job_id)
    restore()
    assert files(*roots) == [path]
    assert count(session, Asset) == count(session, Job) == 1
    assert count(session, FileDeletion) == 0


def test_crash_after_the_commit_leaves_the_file_pending(store, session, roots, monkeypatch):
    asset = save(store, session, new_job(session))
    session.commit()

    def down(*_args):
        raise OperationalError("", {}, Exception())

    monkeypatch.setattr(store, "_purge", down)
    assert store.delete_assets(session, [asset.id]) == 1
    monkeypatch.undo()
    assert len(files(*roots)) == count(session, FileDeletion) == 1
    assert store.purge_deleted_files(session) == (1, 0)
    assert files(*roots) == []


@on_windows
def test_locked_file_stays_pending(store, session, roots):
    asset = save(store, session, new_job(session))
    session.commit()
    with store.path_for(asset).open("rb"):
        store.delete_assets(session, [asset.id])
        assert count(session, Asset) == 0
        assert count(session, FileDeletion) == 1
        assert store.purge_deleted_files(session) == (0, 1)
        assert len(files(*roots)) == 1
    assert store.purge_deleted_files(session) == (1, 0)
    assert files(*roots) == []


def test_web_leaves_private_files_to_the_worker(store, web, session, roots):
    job = new_job(session)
    save(store, session, job)
    save(store, session, job, AssetKind.CLIP, "video/mp4")
    session.commit()
    assert web.delete_job(session, job.id) == 2
    assert files(roots[0]) == []
    assert len(files(roots[1])) == 1
    assert web.purge_deleted_files(session) == (0, 1)
    assert store.purge_deleted_files(session) == (1, 0)
    assert files(*roots) == []


def test_unmounted_root_is_not_a_missing_file(store, session, roots, tmp_path):
    asset = save(store, session, new_job(session))
    session.commit()
    away = tmp_path / "away"
    roots[0].rename(away)
    store.delete_assets(session, [asset.id])
    assert store.purge_deleted_files(session) == (0, 1)
    away.rename(roots[0])
    assert store.purge_deleted_files(session) == (1, 0)
    assert files(*roots) == []


def test_missing_file_counts_as_done(store, session):
    asset = save(store, session, new_job(session))
    session.commit()
    store.path_for(asset).unlink()
    store.delete_assets(session, [asset.id])
    assert count(session, FileDeletion) == 0


def test_bad_row_does_not_stop_the_purge(store, session, roots, tmp_path):
    (tmp_path / "precious.png").write_bytes(b"x")
    session.add(FileDeletion(store=AssetStore.SHARED, path="../precious.png"))
    session.add(FileDeletion(store=AssetStore.SHARED, path="ab/" + "ab" * 16 + ".png"))
    assert store.purge_deleted_files(session) == (1, 1)
    assert (tmp_path / "precious.png").exists()


def test_purge_passes_pending_rows_across_batches(store, session, roots, monkeypatch):
    monkeypatch.setattr(storage, "_BATCH", 2)
    job = new_job(session)
    assets = [save(store, session, job) for _ in range(5)]
    session.add(FileDeletion(store=AssetStore.SHARED, path="bad"))
    session.add(FileDeletion(store=AssetStore.SHARED, path="worse"))
    session.commit()
    assert store.delete_assets(session, [asset.id for asset in assets]) == 5
    assert store.purge_deleted_files(session) == (0, 2)
    assert files(*roots) == []


def test_job_delete_leaves_other_jobs_alone(store, session, roots):
    job = new_job(session)
    other = new_job(session, job.owner_id)
    for each in (job, other):
        step = JobStep(job_id=each.id, step=StepName.CLIP)
        session.add(step)
        session.add(
            Idea(job_id=each.id, text="t", score=9, reason="r", proposed_by=IdeaSource.USER)
        )
        session.flush()
        save(store, session, each, AssetKind.CLIP, "video/mp4", step_id=step.id)
        save(store, session, each)
    session.commit()
    job_id, other_id = job.id, other.id
    assert store.delete_job(session, job_id) == 2
    assert session.scalars(select(Job.id)).all() == [other_id]
    assert set(session.scalars(select(Asset.job_id))) == {other_id}
    assert session.scalars(select(JobStep.job_id)).all() == [other_id]
    assert session.scalars(select(Idea.job_id)).all() == [other_id]
    left = {store.path_for(asset) for asset in session.scalars(select(Asset))}
    assert len(left) == 2
    assert set(files(*roots)) == left


def test_user_delete(store, session, roots):
    job, other = new_job(session), new_job(session)
    new_job(session, job.owner_id)
    save(store, session, job)
    save(store, session, other)
    session.commit()
    owner_id, other_owner = job.owner_id, other.owner_id
    assert store.delete_user(session, owner_id) == 1
    assert session.scalars(select(User.id)).all() == [other_owner]
    assert session.scalars(select(Job.owner_id)).all() == [other_owner]
    assert len(files(*roots)) == 1


# Sweeping.


def test_sweep_removes_only_old_orphans(store, session, roots, monkeypatch):
    monkeypatch.setattr(storage, "_BATCH", 2)
    job = new_job(session)
    kept = store.path_for(save(store, session, job))
    session.commit()
    folder = roots[0] / "ab"
    folder.mkdir(exist_ok=True)
    orphans = [folder / f"ab{n:030x}.png" for n in range(5)]
    orphans.append(folder / f"ab{9:030x}.mp4.part")
    young = folder / f"ab{10:030x}.png"
    foreign = [folder / "notes.txt", roots[0] / "marker"]
    for path in [*orphans, young, *foreign]:
        path.write_bytes(b"x")
    for path in [*orphans, kept, *foreign]:
        age(path)
    assert store.sweep_orphans(session, HOUR) == len(orphans)
    assert set(files(*roots)) == {kept, young, *foreign}


def test_sweep_checks_the_store_as_well_as_the_path(store, session, roots):
    asset = save(store, session, new_job(session))
    session.commit()
    shared = store.path_for(asset)
    stray = roots[1] / asset.path
    stray.parent.mkdir()
    stray.write_bytes(b"x")
    age(shared)
    age(stray)
    assert store.sweep_orphans(session, HOUR) == 1
    assert files(*roots) == [shared]


def test_sweep_skips_an_unreachable_root(web, session, roots):
    stray = roots[1] / "ab" / ("ab" * 16 + ".png")
    stray.parent.mkdir()
    stray.write_bytes(b"x")
    age(stray)
    assert web.sweep_orphans(session, HOUR) == 0
    assert stray.exists()


def test_sweep_does_not_follow_links(store, session, roots, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    target = outside / ("ab" * 16 + ".png")
    target.write_bytes(b"x")
    age(target)
    link = roots[0] / "ab"
    if sys.platform == "win32":
        subprocess.run(  # noqa: S603
            ["cmd", "/c", "mklink", "/J", str(link), str(outside)],  # noqa: S607
            capture_output=True,
            check=True,
        )
    else:
        link.symlink_to(outside, target_is_directory=True)
    assert store.sweep_orphans(session, HOUR) == 0
    assert target.exists()
    link.rmdir() if sys.platform == "win32" else link.unlink()


def test_sweep_needs_a_positive_age(store, session):
    with pytest.raises(ValueError, match="positive"):
        store.sweep_orphans(session, timedelta(0))


def test_paths_are_absolute(store, session):
    assert store.path_for(save(store, session, new_job(session))).is_absolute()
