"""Job progress: step progress mapped to the job's share, written to jobs.progress and sent on
the Postgres channel `job_progress`, so that the web app can stream it as server-sent events.

A notification carries JSON: {"job_id", "status", "progress", "step"}. The web side uses
job_events() and sse() below; each open stream holds one Postgres connection.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from typing import Any

from sqlalchemy import Engine, text, update
from sqlalchemy.engine import make_url
from sqlalchemy.exc import SQLAlchemyError

from memcraft.models import Job, JobRegime, JobStatus, JobStep, StepName
from memcraft.queue import QUEUED, RUNNING

log = logging.getLogger(__name__)

CHANNEL = "job_progress"
# Fewer writes than this apart are dropped, except a step's end.
MIN_INTERVAL = 0.5
MIN_DELTA = 0.01

# The share of each step in one claim of the job, in the order the steps run. A step with
# several ordinals (keyframes, clips) splits its share evenly between them.
IDEA_WEIGHTS: Mapping[StepName, float] = {StepName.IDEA: 1.0}
RELEASE_WEIGHTS: Mapping[StepName, float] = {StepName.RELEASE: 1.0}
RENDER_WEIGHTS: Mapping[JobRegime, Mapping[StepName, float]] = {
    JobRegime.MULTISHOT: {
        StepName.SCENARIO: 0.05,
        StepName.CLIP: 0.80,
        StepName.CHECK: 0.05,
        StepName.JOIN: 0.10,
    },
    # The keyframes are checked before LTX loads, so CHECK comes before CLIP here.
    JobRegime.KEYFRAMES: {
        StepName.SCENARIO: 0.05,
        StepName.KEYFRAME: 0.25,
        StepName.CHECK: 0.05,
        StepName.CLIP: 0.55,
        StepName.JOIN: 0.10,
    },
    # Regime 3: keyframes without a check.
    JobRegime.REGIME3: {
        StepName.SCENARIO: 0.05,
        StepName.KEYFRAME: 0.30,
        StepName.CLIP: 0.55,
        StepName.JOIN: 0.10,
    },
}


def weights_for(status: JobStatus, regime: JobRegime | None) -> Mapping[StepName, float]:
    if status in (JobStatus.IDEA_QUEUED, JobStatus.IDEA_RUNNING):
        return IDEA_WEIGHTS
    if status in (JobStatus.RELEASE_QUEUED, JobStatus.RELEASING):
        return RELEASE_WEIGHTS
    return RENDER_WEIGHTS[regime or JobRegime.MULTISHOT]


def job_fraction(
    weights: Mapping[StepName, float], step: StepName, ordinal: int, count: int, fraction: float
) -> float:
    """How far the whole claim is when `step` (ordinal of count) is `fraction` done.

    Steps earlier in `weights` count as done. A step missing from `weights` adds nothing.
    """
    total = sum(weights.values())
    if total <= 0:
        return 0.0
    before = 0.0
    for name, weight in weights.items():
        if name == step:
            if ordinal >= 200:
                # Asset pictures (keyframe 200+, steps.ASSET_ORDINAL) come before keyframe 0.
                share = 0.0
            else:
                share = (min(ordinal, count - 1) + min(1.0, max(0.0, fraction))) / max(count, 1)
            return min(1.0, (before + weight * share) / total)
        before += weight
    # Not weighted: the steps before it in the run order are unknown, so keep the last value.
    return -1.0


@dataclass
class ProgressEvent:
    job_id: uuid.UUID
    status: JobStatus
    progress: float
    step: StepName | None = None

    def payload(self) -> str:
        return json.dumps(
            {
                "job_id": str(self.job_id),
                "status": self.status.value,
                "progress": round(self.progress, 4),
                "step": self.step.value if self.step else None,
            }
        )

    @classmethod
    def parse(cls, payload: str) -> ProgressEvent:
        data = json.loads(payload)
        return cls(
            job_id=uuid.UUID(data["job_id"]),
            status=JobStatus(data["status"]),
            progress=float(data["progress"]),
            step=StepName(data["step"]) if data.get("step") else None,
        )


def notify(connection: Any, event: ProgressEvent) -> None:
    """Send the event inside the caller's transaction; Postgres delivers it on commit."""
    if connection.dialect.name == "postgresql":
        connection.execute(
            text("SELECT pg_notify(:channel, :payload)"),
            {"channel": CHANNEL, "payload": event.payload()},
        )


class ProgressReporter:
    """Turns one claim's step progress into job progress, throttled and never going back."""

    def __init__(
        self,
        engine: Engine,
        job_id: uuid.UUID,
        worker_id: str,
        status: JobStatus,
        regime: JobRegime | None,
        *,
        min_interval: float = MIN_INTERVAL,
        on_progress: Any = None,
    ) -> None:
        self._engine = engine
        self._job_id = job_id
        self._worker_id = worker_id
        self._status = status
        self._weights = weights_for(status, regime)
        self._min_interval = min_interval
        self._on_progress = on_progress
        self.value = 0.0
        self._sent_value = -1.0
        self._sent_at = 0.0

    def start(self, value: float = 0.0) -> None:
        """Set the claim's starting point, which may be lower than what a last claim left."""
        self.value = value
        self._write(None, None, value, force=True)

    def step(
        self, step: StepName, ordinal: int, count: int, fraction: float, step_id: uuid.UUID
    ) -> None:
        if self._on_progress is not None:
            self._on_progress()
        value = job_fraction(self._weights, step, ordinal, count, fraction)
        if value < 0:
            value = self.value
        self.value = max(self.value, value)
        self._write(step, step_id, fraction, force=fraction >= 1.0)

    def _write(
        self, step: StepName | None, step_id: uuid.UUID | None, fraction: float, *, force: bool
    ) -> None:
        now = time.monotonic()
        if not force and (
            now - self._sent_at < self._min_interval or self.value - self._sent_value < MIN_DELTA
        ):
            return
        try:
            with self._engine.begin() as connection:
                moved = connection.execute(
                    update(Job)
                    .where(Job.id == self._job_id, Job.claimed_by == self._worker_id)
                    .values(progress=self.value)
                )
                if step_id is not None:
                    connection.execute(
                        update(JobStep)
                        .where(JobStep.id == step_id)
                        .values(progress=min(1.0, max(0.0, fraction)))
                    )
                if moved.rowcount:
                    notify(connection, ProgressEvent(self._job_id, self._status, self.value, step))
        except SQLAlchemyError:
            log.warning("progress of job %s not written", self._job_id, exc_info=True)
            return
        self._sent_value, self._sent_at = self.value, now


def publish_status(connection: Any, job: Job, step: StepName | None = None) -> None:
    """Tell listeners about a status change, inside the transaction that makes it."""
    notify(connection, ProgressEvent(job.id, job.status, float(job.progress or 0), step))


# The web side.

KEEPALIVE = 15.0


def _streaming(status: JobStatus) -> bool:
    return status in QUEUED or status in RUNNING


def psycopg_conninfo(database_url: str) -> str:
    """The SQLAlchemy URL as libpq connection info for psycopg."""
    url = make_url(database_url).set(drivername="postgresql")
    return url.render_as_string(hide_password=False)


async def job_events(
    database_url: str, job_id: uuid.UUID, *, keepalive: float = KEEPALIVE
) -> AsyncIterator[ProgressEvent | None]:
    """Events of one job: its state now, then every change, until it stops moving.

    Yields None every `keepalive` seconds without news, for an SSE comment line. Ends after an
    event whose status is neither queued nor running, or at once for an unknown job. The caller
    checks that the viewer may see the job. On Windows this needs a selector event loop.
    """
    import psycopg

    async with await psycopg.AsyncConnection.connect(
        psycopg_conninfo(database_url), autocommit=True
    ) as connection:
        # LISTEN before reading the row, so that no change falls between the two.
        await connection.execute(f"LISTEN {CHANNEL}")
        cursor = await connection.execute(
            "SELECT status, progress FROM jobs WHERE id = %s", (job_id,)
        )
        row = await cursor.fetchone()
        if row is None:
            return
        current = ProgressEvent(job_id, JobStatus(row[0]), float(row[1]))
        yield current
        if not _streaming(current.status):
            return
        wanted = str(job_id)
        while True:
            news = False
            async for note in connection.notifies(timeout=keepalive):
                if wanted not in note.payload:
                    continue
                try:
                    event = ProgressEvent.parse(note.payload)
                except (ValueError, KeyError):
                    continue
                if event.job_id != job_id:
                    continue
                news = True
                yield event
                if not _streaming(event.status):
                    return
                break
            if not news:
                yield None


def sse(event: ProgressEvent | None) -> str:
    """One server-sent event for job_events() output; None becomes a keepalive comment."""
    if event is None:
        return ": keepalive\n\n"
    return f"event: progress\ndata: {event.payload()}\n\n"
