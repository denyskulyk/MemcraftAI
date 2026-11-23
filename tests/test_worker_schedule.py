import uuid
from datetime import timedelta

import pytest
from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from memcraft.models import Job, JobLane, JobLength, JobQuality, JobStatus, Setting, User
from memcraft.storage import FileStore
from memcraft.worker.pause import is_paused, set_paused
from memcraft.worker.schedule import Schedule, Task, default_tasks

on_postgres = pytest.mark.parametrize(
    "engine", [pytest.param("postgres", marks=pytest.mark.postgres)], indirect=True
)


class Clock:
    now = 0.0

    def __call__(self):
        return self.now


def test_tasks_run_when_due(engine):
    clock, ran = Clock(), []

    def boom(_session):
        raise RuntimeError("boom")

    schedule = Schedule(
        engine,
        [Task("a", 10, lambda s: ran.append("a")), Task("b", 30, boom)],
        clock=clock,
    )
    assert schedule.run_pending() == ["a"]  # b failed, and is logged
    clock.now = 9
    assert schedule.run_pending() == []
    clock.now = 10
    assert schedule.run_pending() == ["a"]
    clock.now = 30
    assert schedule.run_pending() == ["a"]
    assert ran == ["a"] * 3


def test_default_tasks(tmp_path):
    assert [task.name for task in default_tasks(None)] == ["requeue_stale"]
    (tmp_path / "s").mkdir()
    names = [task.name for task in default_tasks(FileStore(tmp_path / "s"))]
    assert names == [
        "requeue_stale",
        "expire_previews",
        "expire_finished",
        "purge_deleted_files",
        "sweep_orphans",
    ]


@on_postgres
def test_the_default_tasks_run(engine, tmp_path):
    (tmp_path / "s").mkdir()
    with Session(engine) as session:
        user = User(tailscale_login=uuid.uuid4().hex)
        session.add(user)
        session.flush()
        job = Job(
            owner_id=user.id,
            photo_count=1,
            length=JobLength.SHORT,
            quality=JobQuality.HD720,
            lane=JobLane.RENDER,
            status=JobStatus.RENDERING,
            claimed_by="dead",
        )
        session.add(job)
        session.flush()
        old = func.now() - timedelta(minutes=5)
        session.execute(update(Job).where(Job.id == job.id).values(claimed_at=old))
        session.commit()
        schedule = Schedule(engine, default_tasks(FileStore(tmp_path / "s")))
        assert len(schedule.run_pending()) == 5
        session.expire_all()
        assert session.scalar(select(Job.status)) is JobStatus.RENDER_QUEUED


def test_pause_switch(engine):
    with Session(engine) as session:
        assert not is_paused(session)  # no row yet
        set_paused(session, True)
        session.commit()
        assert is_paused(session)
        set_paused(session, False)
        session.commit()
        assert not is_paused(session)
        assert session.scalar(select(func.count()).select_from(Setting)) == 1
        assert session.get(Setting, "paused").value is False


@pytest.mark.parametrize("value", [True, False])
def test_pause_leaves_the_commit_to_the_caller(engine, value):
    with Session(engine) as session:
        set_paused(session, value)
        session.rollback()
        assert not is_paused(session)
