"""Housekeeping the worker runs on a timer: stale claims, expiry, file purges and sweeps.

Every task is safe to run in several processes at once. A task that fails is logged and tried
again at its next turn.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta

from sqlalchemy import Engine
from sqlalchemy.orm import Session

from memcraft.expiry import expire_finished, expire_previews
from memcraft.models import JobLane
from memcraft.queue import requeue_stale
from memcraft.storage import FileStore

log = logging.getLogger(__name__)

# The heartbeat renews claims every 10 s, so a claim this old has a dead worker.
STALE_CLAIM = timedelta(seconds=90)
# Longer than any transaction that saves files.
ORPHAN_AGE = timedelta(hours=6)


@dataclass
class Task:
    name: str
    every: float  # seconds
    run: Callable[[Session], object]
    due_at: float = 0.0  # monotonic; 0 runs at the first turn


class Schedule:
    def __init__(
        self, engine: Engine, tasks: list[Task], clock: Callable[[], float] = time.monotonic
    ) -> None:
        self._engine = engine
        self.tasks = tasks
        self._clock = clock

    def run_pending(self) -> list[str]:
        """Run every task that is due. Returns the names of those that ran without error."""
        ran = []
        for task in self.tasks:
            now = self._clock()
            if now < task.due_at:
                continue
            task.due_at = now + task.every
            try:
                with Session(self._engine) as session:
                    result = task.run(session)
                    session.commit()
            except Exception:
                log.exception("scheduled task %s failed", task.name)
                continue
            if result:
                log.info("%s: %s", task.name, result)
            ran.append(task.name)
        return ran

    def run(self, stop: threading.Event, tick: float = 1.0) -> None:
        while not stop.is_set():
            self.run_pending()
            stop.wait(tick)


def default_tasks(store: FileStore | None, stale_claim: timedelta = STALE_CLAIM) -> list[Task]:
    tasks = [
        Task(
            "requeue_stale",
            30,
            lambda s: sum(requeue_stale(s, lane, stale_claim) for lane in JobLane),
        )
    ]
    if store is not None:
        tasks += [
            Task("expire_previews", 300, lambda s: expire_previews(s, store)),
            Task("expire_finished", 300, lambda s: expire_finished(s, store)),
            Task("purge_deleted_files", 60, lambda s: store.purge_deleted_files(s).removed),
            Task("sweep_orphans", 3600, lambda s: store.sweep_orphans(s, ORPHAN_AGE)),
        ]
    return tasks
