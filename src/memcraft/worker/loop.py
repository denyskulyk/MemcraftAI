"""The worker: an idea lane (CPU) and a render lane (GPU) running side by side in one process.

For each claimed job the worker asks the planner for the job's steps and runs the first one
that is not done, until none is left; then it moves the job to the planner's outcome. Every
attempt of a step is a job_steps row (attempt, status, inputs hash, started and finished), so
a restarted worker skips done steps and runs only the unfinished one. A failed step is tried
again up to `max_attempts` times, then the job fails with the error. A step the watchdog found
hung is tried once more. A heartbeat renews the claim, so requeue_stale() can use a short limit.

Before every step the worker reads settings.paused. While paused it finishes the step it is
in, unloads models through the `unload` callback, gives up the GPU lease and waits; the claim
stays alive. A stop request lets the running step finish; the job's claim then runs out and
requeue_stale() puts the job back in line.

With a `memory` watch, RAM and VRAM are logged at the start and end of every step and kept under
the "memory" key of job_steps.output; before a GPU step it makes room when free RAM is below
settings.memory_min_free_gb (default: the watch's own limit).
"""

from __future__ import annotations

import logging
import os
import socket
import threading
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import timedelta
from enum import Enum, auto
from typing import Any

from sqlalchemy import Engine, func, select, update
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from memcraft.models import Job, JobLane, JobStatus, JobStep, Setting, StepName, StepStatus
from memcraft.queue import StatusConflictError, claim, transition
from memcraft.storage import FileStore
from memcraft.worker.lease import MAX_WAIT, GpuLease, Lease, LocalLease, acquire, claim_render
from memcraft.worker.memory import SETTING_MIN_FREE, MemoryWatch
from memcraft.worker.pause import is_paused
from memcraft.worker.progress import ProgressReporter, job_fraction, publish_status, weights_for
from memcraft.worker.schedule import Schedule
from memcraft.worker.steps import PlannedStep, StepContext, StepRegistry, StepStopped
from memcraft.worker.watchdog import HUNG, Watch, Watchdog

log = logging.getLogger(__name__)

INTERRUPTED = "interrupted: the worker stopped during the step"
MAX_ERROR = 2000


@dataclass
class WorkerConfig:
    poll: float = 1.0  # seconds between looks at an empty queue or a pause
    heartbeat: float = 10.0
    max_attempts: int = 3
    retry_delay: float = 5.0
    max_wait: timedelta = MAX_WAIT
    progress_interval: float = 0.5


class _Outcome(Enum):
    DONE = auto()  # the step is done
    AGAIN = auto()  # look at the job again: a retry, or a pause came first
    STOP = auto()  # the job is not ours any more: finished, failed, lost or stopping


@dataclass
class _LaneState:
    lane: JobLane
    loaded: str | None = None  # the model the last GPU step of this lane used
    dirty: bool = False  # a model may be loaded since the last unload
    paused: bool = False


class Heartbeat:
    """Renews the claim of one job from its own thread. `lost` is set once it is gone."""

    def __init__(
        self, engine: Engine, job_id: uuid.UUID, worker_id: str, status: JobStatus, every: float
    ) -> None:
        self._engine = engine
        self._job_id = job_id
        self._worker_id = worker_id
        self._status = status
        self._every = every
        self.lost = threading.Event()
        self._done = threading.Event()
        self._callbacks: list[Callable[[], None]] = []
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, name=f"heartbeat-{job_id}", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._done.set()
        if self._thread.is_alive() and self._thread is not threading.current_thread():
            self._thread.join(timeout=self._every + 5)

    def on_lost(self, callback: Callable[[], None]) -> Callable[[], None]:
        """Call `callback` when the claim is lost. Returns a function that forgets it."""
        with self._lock:
            self._callbacks.append(callback)
        if self.lost.is_set():
            callback()

        def forget() -> None:
            with self._lock:
                if callback in self._callbacks:
                    self._callbacks.remove(callback)

        return forget

    def beat(self) -> bool:
        try:
            with self._engine.begin() as connection:
                renewed = connection.execute(
                    update(Job)
                    .where(
                        Job.id == self._job_id,
                        Job.status == self._status,
                        Job.claimed_by == self._worker_id,
                    )
                    .values(claimed_at=func.now())
                ).rowcount
        except SQLAlchemyError:
            log.warning("heartbeat of job %s failed; trying again", self._job_id, exc_info=True)
            return True
        if not renewed:
            log.warning("job %s is not ours any more", self._job_id)
            self.lost.set()
            with self._lock:
                callbacks = list(self._callbacks)
            for callback in callbacks:
                callback()
        return bool(renewed)

    def _run(self) -> None:
        while not self._done.wait(self._every) and not self.lost.is_set():
            self.beat()


class Worker:
    def __init__(
        self,
        engine: Engine,
        registry: StepRegistry,
        *,
        store: FileStore | None = None,
        worker_id: str | None = None,
        lease: Lease | None = None,
        unload: Callable[[JobLane], None] | None = None,
        watchdog: Watchdog | None = None,
        schedule: Schedule | None = None,
        config: WorkerConfig | None = None,
        memory: MemoryWatch | None = None,
    ) -> None:
        self.engine = engine
        self.registry = registry
        self.store = store
        self.worker_id = worker_id or (
            f"{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:6]}"
        )
        if lease is None:
            lease = GpuLease(engine) if engine.dialect.name == "postgresql" else LocalLease()
        self.lease = lease
        self.unload = unload
        self.watchdog = watchdog
        self.schedule = schedule
        self.memory = memory
        self.config = config or WorkerConfig()
        self.stopping = threading.Event()
        self._lanes = {lane: _LaneState(lane) for lane in JobLane}
        self._threads: list[threading.Thread] = []

    # Running.

    def start(self) -> None:
        """Start both lanes, the watchdog and the schedule in threads of their own."""
        targets: list[tuple[str, Callable[[], None]]] = [
            (f"lane-{lane.value}", lambda lane=lane: self.run_lane(lane)) for lane in JobLane
        ]
        if self.watchdog is not None:
            targets.append(("watchdog", lambda: self.watchdog.run(self.stopping)))
        if self.schedule is not None:
            targets.append(("schedule", lambda: self.schedule.run(self.stopping)))
        for name, target in targets:
            thread = threading.Thread(target=target, name=name, daemon=True)
            thread.start()
            self._threads.append(thread)

    def stop(self, timeout: float | None = None) -> bool:
        """Ask every thread to end after its current step. True if all ended in time."""
        self.stopping.set()
        for thread in self._threads:
            thread.join(timeout)
        return not any(thread.is_alive() for thread in self._threads)

    def run_lane(self, lane: JobLane) -> None:
        state = self._lanes[lane]
        try:
            while not self.stopping.is_set():
                try:
                    if self._paused():
                        self._enter_pause(state)
                        self.stopping.wait(self.config.poll)
                        continue
                    state.paused = False
                    if not self.run_one(lane):
                        if lane is JobLane.RENDER:
                            self._release_gpu(state)
                        self.stopping.wait(self.config.poll)
                except Exception:
                    log.exception("the %s lane hit an error", lane.value)
                    self.stopping.wait(self.config.poll)
        finally:
            if lane is JobLane.RENDER:
                self._release_gpu(state)

    def drain(self, limit: int = 1000) -> int:
        """Work in this thread until neither lane has a job or the worker is paused.

        Returns how many claims were worked on. For tests and `--once`.
        """
        count = 0
        while count < limit and not self.stopping.is_set() and not self._paused():
            took = [self.run_one(lane) for lane in JobLane]
            if not any(took):
                break
            count += sum(took)
        self._release_gpu(self._lanes[JobLane.RENDER])
        return count

    def run_one(self, lane: JobLane) -> bool:
        """Claim one job of the lane and work on it. False if there was none."""
        state = self._lanes[lane]
        with Session(self.engine) as session:
            if lane is JobLane.RENDER:
                job = claim_render(
                    session,
                    self.worker_id,
                    loaded=state.loaded,
                    model_of=self.next_model,
                    max_wait=self.config.max_wait,
                )
            else:
                job = claim(session, lane, self.worker_id)
            if job is None:
                return False
            log.info("%s lane took job %s (%s)", lane.value, job.id, job.status.value)
            self._work(session, job, state)
        return True

    def next_model(self, session: Session, job: Job) -> str | None:
        """The model of the job's next GPU step, as the planner sees it now."""
        done = self._done_rows(session, job.id)
        for planned in self._pending(self.registry.planner.plan(session, job, done), done):
            if self.registry.handler(planned.step).needs_gpu:
                return self.registry.model_of(planned)
        return None

    # One job.

    def _work(self, session: Session, job: Job, state: _LaneState) -> None:
        job_id, running = job.id, job.status
        heartbeat = Heartbeat(self.engine, job_id, self.worker_id, running, self.config.heartbeat)
        heartbeat.start()
        reporter = ProgressReporter(
            self.engine,
            job_id,
            self.worker_id,
            running,
            job.regime,
            min_interval=self.config.progress_interval,
        )
        started = False
        try:
            while not heartbeat.lost.is_set() and not self.stopping.is_set():
                if not self._wait_while_paused(state, heartbeat):
                    return
                session.expire_all()
                job = session.get(Job, job_id)
                if job is None or job.status is not running or job.claimed_by != self.worker_id:
                    session.rollback()
                    return
                done = self._done_rows(session, job_id)
                planned = list(self._pending(self.registry.planner.plan(session, job, done), done))
                if not started:
                    reporter.start(_baseline(running, job, planned))
                    started = True
                if not planned:
                    self._finish(session, job_id, running)
                    return
                outputs = {(row.step, row.ordinal): row.output for row in done}
                session.commit()
                result = self._run_step(
                    session, job, planned[0], outputs, state, heartbeat, reporter
                )
                if result is _Outcome.STOP:
                    return
        finally:
            heartbeat.stop()

    def _run_step(
        self,
        session: Session,
        job: Job,
        planned: PlannedStep,
        outputs: Mapping[tuple[StepName, int], Any],
        state: _LaneState,
        heartbeat: Heartbeat,
        reporter: ProgressReporter,
    ) -> _Outcome:
        job_id, running, owner_id, regime = job.id, job.status, job.owner_id, job.regime
        handler = self.registry.handler(planned.step)
        digest = planned.inputs_hash

        if handler.needs_gpu:
            if state.lane is JobLane.IDEA:
                return self._fail(session, job_id, running, "idea lane steps run on the CPU")
            if not self.lease.check():
                state.loaded = None  # another process may have used the GPU meanwhile

                def keep_waiting() -> bool:
                    return not (self.stopping.is_set() or heartbeat.lost.is_set() or self._paused())

                if not acquire(self.lease, keep_waiting, poll=self.config.poll):
                    return _Outcome.AGAIN

        rows = session.scalars(
            select(JobStep)
            .where(
                JobStep.job_id == job_id,
                JobStep.step == planned.step,
                JobStep.ordinal == planned.ordinal,
            )
            .order_by(JobStep.attempt)
        ).all()
        for row in rows:
            if row.status in (StepStatus.RUNNING, StepStatus.PENDING):
                # Left by a worker that died: that attempt failed.
                row.status = StepStatus.FAILED
                row.error = INTERRUPTED
                row.finished_at = _clock(session)
        failed = [r for r in rows if r.status is StepStatus.FAILED and r.inputs_hash == digest]
        if len(failed) >= self.config.max_attempts:
            session.commit()
            return self._fail(session, job_id, running, failed[-1].error or "the step failed")
        hung_before = any((row.error or "").startswith(HUNG) for row in failed)
        row = JobStep(
            job_id=job_id,
            step=planned.step,
            ordinal=planned.ordinal,
            attempt=max((r.attempt for r in rows), default=0) + 1,
            status=StepStatus.RUNNING,
            inputs=dict(planned.inputs),
            inputs_hash=digest,
            started_at=_clock(session),
        )
        session.add(row)
        session.flush()
        step_id, attempt = row.id, row.attempt
        if not self._touch(session, job_id, running):
            session.rollback()
            return _Outcome.STOP
        session.commit()

        if handler.needs_gpu or handler.model or handler.service:
            state.dirty = True
        if handler.needs_gpu:
            state.loaded = self.registry.model_of(planned)

        stop_step = threading.Event()
        forget = heartbeat.on_lost(stop_step.set)
        label = f"job {job_id} step {planned.step.value}/{planned.ordinal}"
        # Before the watchdog's watch: waiting for RAM is not a hung service.
        memory = self._memory_before(handler, label, stop_step)
        watch = (
            self.watchdog.watch(handler.service, stop_step.set)
            if self.watchdog is not None
            else Watch(None, None)
        )

        def report(fraction: float) -> None:
            watch.progress(fraction)
            reporter.step(planned.step, planned.ordinal, planned.count, fraction, step_id)

        step_session = Session(self.engine)
        ctx = StepContext(
            job_id=job_id,
            owner_id=owner_id,
            regime=regime,
            step=planned.step,
            ordinal=planned.ordinal,
            count=planned.count,
            attempt=attempt,
            step_id=step_id,
            inputs=planned.inputs,
            session=step_session,
            store=self.store,
            outputs=outputs,
            _report=report,
            _stop=stop_step,
        )
        error: Exception | None = None
        try:
            try:
                output = handler.run(ctx)
                if memory is not None:
                    output = {**(output or {}), "memory": self._memory_after(memory, label)}
                self._mark_done(step_session, job_id, running, step_id, output)
            except Exception as caught:
                step_session.rollback()
                error = caught
        finally:
            step_session.close()
            forget()
            hung = watch.hung
            watch.close()

        if error is None:
            reporter.step(planned.step, planned.ordinal, planned.count, 1.0, step_id)
            return _Outcome.DONE
        if memory is not None and "end" not in memory:
            self._memory_after(memory, label)
        kept = {"memory": memory} if memory is not None else None
        if heartbeat.lost.is_set() or isinstance(error, _LostClaimError):
            log.warning("job %s lost during step %s", job_id, planned.step.value)
            self._record_failure(session, step_id, INTERRUPTED, kept)
            return _Outcome.STOP
        if hung is not None:
            message = HUNG + hung
        elif isinstance(error, StepStopped):
            message = INTERRUPTED
        else:
            message = _describe(error)
        log.warning(
            "job %s step %s/%d attempt %d failed: %s",
            job_id,
            planned.step.value,
            planned.ordinal,
            attempt,
            message,
            exc_info=error if hung is None else None,
        )
        self._record_failure(session, step_id, message, kept)
        if hung is not None and hung_before:
            return self._fail(session, job_id, running, message)
        if len(failed) + 1 >= self.config.max_attempts:
            return self._fail(session, job_id, running, message)
        if self.stopping.wait(self.config.retry_delay):
            return _Outcome.STOP
        return _Outcome.AGAIN

    def _mark_done(
        self,
        session: Session,
        job_id: uuid.UUID,
        running: JobStatus,
        step_id: uuid.UUID,
        output: Mapping[str, Any] | None,
    ) -> None:
        """Commit the step's files and its done mark together, if the job is still ours."""
        row = session.get(JobStep, step_id)
        if row is None:
            raise _LostClaimError("the step row is gone")
        row.status = StepStatus.DONE
        row.output = dict(output) if output is not None else None
        row.progress = 1.0
        row.error = None
        row.finished_at = _clock(session)
        session.flush()
        if not self._touch(session, job_id, running):
            raise _LostClaimError("the job is not ours any more")
        session.commit()

    def _record_failure(
        self,
        session: Session,
        step_id: uuid.UUID,
        message: str,
        output: Mapping[str, Any] | None = None,
    ) -> None:
        values: dict[str, Any] = {
            "status": StepStatus.FAILED,
            "error": message[:MAX_ERROR],
            "finished_at": _clock(session),
        }
        if output is not None:
            values["output"] = dict(output)
        try:
            session.rollback()
            session.execute(
                update(JobStep)
                .where(JobStep.id == step_id, JobStep.status == StepStatus.RUNNING)
                .values(**values)
            )
            session.commit()
        except SQLAlchemyError:
            session.rollback()
            log.exception("could not record the failure of step %s", step_id)

    def _touch(self, session: Session, job_id: uuid.UUID, running: JobStatus) -> bool:
        """Renew the claim inside the session's transaction. False if it is not ours."""
        return (
            session.execute(
                update(Job)
                .where(Job.id == job_id, Job.status == running, Job.claimed_by == self.worker_id)
                .values(claimed_at=func.now())
                .returning(Job.id)
                .execution_options(synchronize_session=False)
            ).first()
            is not None
        )

    def _finish(self, session: Session, job_id: uuid.UUID, running: JobStatus) -> None:
        job = session.get(Job, job_id)
        outcome = self.registry.planner.outcome(session, job)
        try:
            moved = transition(session, job, outcome, expected=running, worker_id=self.worker_id)
        except StatusConflictError:
            session.rollback()
            log.warning("job %s changed before it could finish", job_id)
            return
        if outcome is not JobStatus.FAILED:
            moved.progress = 1.0
        session.flush()
        publish_status(session.connection(), moved)
        session.commit()
        log.info("job %s is %s", job_id, outcome.value)

    def _fail(
        self, session: Session, job_id: uuid.UUID, running: JobStatus, message: str
    ) -> _Outcome:
        session.rollback()
        job = session.get(Job, job_id)
        if job is None:
            return _Outcome.STOP
        try:
            moved = transition(
                session, job, JobStatus.FAILED, expected=running, worker_id=self.worker_id
            )
        except StatusConflictError:
            session.rollback()
            return _Outcome.STOP
        moved.error = message[:MAX_ERROR]
        session.flush()
        publish_status(session.connection(), moved)
        session.commit()
        log.warning("job %s failed: %s", job_id, message)
        return _Outcome.STOP

    # Memory.

    def _memory_before(
        self, handler: Any, label: str, stop: threading.Event
    ) -> dict[str, Any] | None:
        if self.memory is None:
            return None
        try:
            return self.memory.before_step(
                needs_gpu=handler.needs_gpu,
                service=handler.service,
                label=label,
                min_free_gb=self._memory_limit(),
                stop=stop,
            )
        except Exception:
            log.exception("the memory watch failed before %s", label)
            return {}

    def _memory_after(self, record: dict[str, Any], label: str) -> dict[str, Any]:
        try:
            return self.memory.after_step(record, label)  # type: ignore[union-attr]
        except Exception:
            log.exception("the memory watch failed after %s", label)
            return record

    def _memory_limit(self) -> float | None:
        """settings.memory_min_free_gb, or None for the watch's own limit."""
        try:
            with Session(self.engine) as session:
                value = session.scalar(select(Setting.value).where(Setting.key == SETTING_MIN_FREE))
            return None if value is None else float(value)
        except (SQLAlchemyError, TypeError, ValueError):
            log.warning("cannot read settings.%s", SETTING_MIN_FREE, exc_info=True)
            return None

    # Pause and the GPU.

    def _paused(self) -> bool:
        with Session(self.engine) as session:
            return is_paused(session)

    def _wait_while_paused(self, state: _LaneState, heartbeat: Heartbeat) -> bool:
        while self._paused():
            self._enter_pause(state)
            if self.stopping.wait(self.config.poll) or heartbeat.lost.is_set():
                return False
        state.paused = False
        return True

    def _enter_pause(self, state: _LaneState) -> None:
        if state.paused:
            return
        state.paused = True
        log.info("%s lane paused", state.lane.value)
        if state.dirty and self.unload is not None:
            try:
                self.unload(state.lane)
            except Exception:
                log.exception("unloading the %s lane's models failed", state.lane.value)
        state.dirty = False
        if state.lane is JobLane.RENDER:
            self._release_gpu(state)

    def _release_gpu(self, state: _LaneState) -> None:
        if self.lease.held:
            self.lease.release()
        state.loaded = None

    # Helpers.

    @staticmethod
    def _done_rows(session: Session, job_id: uuid.UUID) -> list[JobStep]:
        return list(
            session.scalars(
                select(JobStep)
                .where(JobStep.job_id == job_id, JobStep.status == StepStatus.DONE)
                .order_by(JobStep.created_at, JobStep.attempt)
            )
        )

    @staticmethod
    def _pending(planned: Sequence[PlannedStep], done: Sequence[JobStep]) -> list[PlannedStep]:
        finished = {(row.step, row.ordinal, row.inputs_hash) for row in done}
        return [p for p in planned if (p.step, p.ordinal, p.inputs_hash) not in finished]


class _LostClaimError(Exception):
    pass


def _baseline(running: JobStatus, job: Job, pending: Sequence[PlannedStep]) -> float:
    if not pending:
        return 1.0
    first = pending[0]
    value = job_fraction(
        weights_for(running, job.regime), first.step, first.ordinal, first.count, 0
    )
    return max(0.0, value)


def _clock(session: Session) -> Any:
    """The time now. Postgres's now() is when the transaction began, which for a step's
    session can be the start of the step."""
    if session.get_bind().dialect.name == "postgresql":
        return func.clock_timestamp()
    return func.now()


def _describe(error: BaseException) -> str:
    text = str(error).strip()
    return f"{type(error).__name__}: {text}" if text else type(error).__name__
