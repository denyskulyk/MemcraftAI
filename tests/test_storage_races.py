"""save(), delete_job() and purge_deleted_files() in threads, each with its own connection."""

import io
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import pytest
from sqlalchemy import event, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from memcraft.models import (
    Asset,
    AssetKind,
    AssetStore,
    FileDeletion,
    Job,
    JobLength,
    JobQuality,
    User,
)
from memcraft.storage import FileStore

WAIT = 30
ROUNDS = 30

pytestmark = pytest.mark.parametrize(
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


def new_job(session):
    """The ids of a committed job and its owner."""
    user = User(tailscale_login=uuid.uuid4().hex)
    session.add(user)
    session.flush()
    job = Job(owner_id=user.id, photo_count=1, length=JobLength.SHORT, quality=JobQuality.HD720)
    session.add(job)
    session.flush()
    ids = job.id, user.id
    session.commit()
    return ids


def save(store, session, job_id, owner_id, kind=AssetKind.PHOTO):
    return store.save(
        session,
        owner_id=owner_id,
        job_id=job_id,
        kind=kind,
        mime="image/png",
        source=io.BytesIO(b"data"),
    )


def count(session, model, *where):
    return session.scalar(select(func.count()).select_from(model).where(*where))


def nothing_left(session, roots):
    session.rollback()
    return (
        files(*roots) == []
        and count(session, Job) == 0
        and count(session, Asset) == 0
        and count(session, FileDeletion) == 0
    )


def both(*works):
    with ThreadPoolExecutor(len(works)) as pool:
        return [future.result(WAIT) for future in [pool.submit(work) for work in works]]


def test_open_save_wins_and_the_delete_removes_it(store, engine, session, roots):
    job_id, owner_id = new_job(session)
    save(store, session, job_id, owner_id)
    save(store, session, job_id, owner_id, AssetKind.KEYFRAME)

    def remove():
        with Session(engine) as mine:
            return store.delete_job(mine, job_id)

    with ThreadPoolExecutor(1) as pool:
        removing = pool.submit(remove)
        with pytest.raises(TimeoutError):
            removing.result(0.3)  # it waits for the save
        session.commit()
        assert removing.result(WAIT) == 2
    assert nothing_left(session, roots)


def test_open_save_rolled_back_under_a_waiting_delete(store, engine, session, roots):
    job_id, owner_id = new_job(session)
    save(store, session, job_id, owner_id)

    def remove():
        with Session(engine) as mine:
            return store.delete_job(mine, job_id)

    with ThreadPoolExecutor(1) as pool:
        removing = pool.submit(remove)
        with pytest.raises(TimeoutError):
            removing.result(0.3)
        session.rollback()
        assert removing.result(WAIT) == 0
    assert nothing_left(session, roots)


@pytest.mark.parametrize("delete_user", [False, True])
def test_open_delete_wins_and_the_save_fails(store, engine, session, roots, delete_user):
    job_id, owner_id = new_job(session)
    save(store, session, job_id, owner_id)
    session.commit()
    holding, go_on = threading.Event(), threading.Event()

    def pause(_session):
        holding.set()
        go_on.wait(WAIT)

    def remove():
        with Session(engine) as mine:
            event.listen(mine, "before_commit", pause, once=True)
            if delete_user:
                return store.delete_user(mine, owner_id)
            return store.delete_job(mine, job_id)

    def late_save():
        with Session(engine) as mine:
            save(store, mine, job_id, owner_id)
            mine.commit()

    with ThreadPoolExecutor(2) as pool:
        removing = pool.submit(remove)
        assert holding.wait(WAIT)
        saving = pool.submit(late_save)
        with pytest.raises(TimeoutError):
            saving.result(0.3)  # it waits for the delete
        go_on.set()
        assert removing.result(WAIT) == 1
        with pytest.raises(IntegrityError, match="foreign key"):
            saving.result(WAIT)
    assert nothing_left(session, roots)


def test_save_and_delete_at_once(store, engine, session, roots):
    for n in range(ROUNDS):
        job_id, owner_id = new_job(session)
        barrier = threading.Barrier(3)

        def keep(kind, job_id=job_id, owner_id=owner_id, barrier=barrier):
            with Session(engine) as mine:
                mine.connection()
                barrier.wait(WAIT)
                try:
                    save(store, mine, job_id, owner_id, kind)
                    mine.commit()
                except IntegrityError:
                    return 0
                return 1

        def remove(job_id=job_id, barrier=barrier, n=n):
            with Session(engine) as mine:
                mine.connection()
                barrier.wait(WAIT)
                time.sleep(n / 6000)  # a save writes its file first, so it needs a start
                return store.delete_job(mine, job_id)

        one, two, removed = both(
            lambda keep=keep: keep(AssetKind.PHOTO),
            lambda keep=keep: keep(AssetKind.KEYFRAME),
            remove,
        )
        assert removed == one + two
        assert nothing_left(session, roots)


def test_two_deletes_of_one_job(store, engine, session, roots):
    for _ in range(10):
        job_id, owner_id = new_job(session)
        for _ in range(5):
            save(store, session, job_id, owner_id)
        session.commit()
        barrier = threading.Barrier(2)

        def remove(job_id=job_id, barrier=barrier):
            with Session(engine) as mine:
                mine.connection()
                barrier.wait(WAIT)
                return store.delete_job(mine, job_id)

        assert sorted(both(remove, remove)) == [0, 5]
        assert nothing_left(session, roots)


def put_pending(store, session, roots, number):
    """Files whose rows are gone, as a delete in the web container leaves them."""
    for _ in range(number):
        name = uuid.uuid4().hex
        folder = roots[1] / name[:2]
        folder.mkdir(exist_ok=True)
        (folder / f"{name}.png").write_bytes(b"x")
        session.add(FileDeletion(store=AssetStore.PRIVATE, path=f"{name[:2]}/{name}.png"))
    session.commit()


def test_two_purges_at_once(store, engine, session, roots, monkeypatch):
    monkeypatch.setattr("memcraft.storage._BATCH", 7)
    unlinked = []
    real = store._unlink

    def counted(kind, path):
        unlinked.append(path)
        return real(kind, path)

    monkeypatch.setattr(store, "_unlink", counted)
    put_pending(store, session, roots, 200)
    barrier = threading.Barrier(2)

    def purge():
        with Session(engine) as mine:
            mine.connection()
            barrier.wait(WAIT)
            return store.purge_deleted_files(mine)

    results = both(purge, purge)
    assert sum(result.removed for result in results) == 200
    assert len(unlinked) == len(set(unlinked)) == 200
    assert store.purge_deleted_files(session) == (0, 0)
    assert files(*roots) == []


def test_a_purge_skips_rows_another_purge_holds(store, engine, session, roots, monkeypatch):
    put_pending(store, session, roots, 3)
    holding, go_on = threading.Event(), threading.Event()
    slow = FileStore(*roots)
    real = slow._unlink

    def paused(kind, path):
        holding.set()
        go_on.wait(WAIT)
        return real(kind, path)

    monkeypatch.setattr(slow, "_unlink", paused)

    def purge():
        with Session(engine) as mine:
            return slow.purge_deleted_files(mine)

    with ThreadPoolExecutor(1) as pool:
        purging = pool.submit(purge)
        assert holding.wait(WAIT)
        try:
            assert store.purge_deleted_files(session) == (0, 3)
        finally:
            go_on.set()
        assert purging.result(WAIT) == (3, 0)
    assert files(*roots) == []


def test_delete_and_purge_at_once(store, engine, session, roots):
    for _ in range(10):
        job_id, owner_id = new_job(session)
        for _ in range(5):
            save(store, session, job_id, owner_id)
        session.commit()
        barrier = threading.Barrier(2)

        def remove(job_id=job_id, barrier=barrier):
            with Session(engine) as mine:
                mine.connection()
                barrier.wait(WAIT)
                return store.delete_job(mine, job_id)

        def purge(barrier=barrier):
            with Session(engine) as mine:
                mine.connection()
                barrier.wait(WAIT)
                for _ in range(5):
                    store.purge_deleted_files(mine)

        assert both(remove, purge)[0] == 5
        assert nothing_left(session, roots)
