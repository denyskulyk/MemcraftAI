"""The watchdog: notices a hung ComfyUI (8199) or Gemma llama-server while a step uses it.

A service is hung when it stops answering for `answer_limit`, or when the step using it reports
no progress for `stall_limit` while the GPU idles (per nvidia-smi), or for three times that
whatever the GPU does. The watchdog then stops the step, calls the service's injected restart,
and the worker retries the step once.

It restarts only the app's own instances, through the callbacks it is given, and refuses to
watch port 8188 (the owner's ComfyUI) or 8080 (llama-swap).
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import NamedTuple
from urllib.parse import urlsplit

log = logging.getLogger(__name__)

NOT_OURS = frozenset({8188, 8080})
# 120 s since 2026-11-24 (was 300): the longest quiet start in the logs was about 15 s,
# and a hung keyframe cost a job 5 minutes of waiting. A busy GPU still gets 360 s.
STALL_LIMIT = 120.0
ANSWER_LIMIT = 60.0
CHECK_EVERY = 5.0
# Below this GPU utilisation, in percent, the GPU counts as idle.
IDLE_GPU = 5

HUNG = "hung: "


class GpuReading(NamedTuple):
    utilization: int  # percent
    memory_used: int  # MiB
    memory_total: int  # MiB


def read_gpu(
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> GpuReading | None:
    """The first GPU's load from nvidia-smi, or None if it cannot be read."""
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None
    try:
        done = run(
            [
                exe,
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        first = done.stdout.strip().splitlines()[0]
        utilization, used, total = (int(part.strip()) for part in first.split(","))
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        log.warning("nvidia-smi could not be read", exc_info=True)
        return None
    return GpuReading(utilization, used, total)


@dataclass
class Service:
    """One of the app's own servers. `probe` returns True when it answers; it must be quick."""

    name: str
    port: int
    probe: Callable[[], bool]
    restart: Callable[[], None]
    uses_gpu: bool = True
    last_answer: float = field(default=0.0, repr=False)

    def __post_init__(self) -> None:
        if self.port in NOT_OURS:
            raise ValueError(f"port {self.port} is not the app's own instance")


@dataclass
class _Watch:
    service: str
    stop: Callable[[], None]
    last_progress: float
    best: float = -1.0
    hung: str | None = None


class Watch:
    """What a step holds while it runs: report progress, and ask whether it was found hung."""

    def __init__(self, dog: Watchdog | None, token: int | None) -> None:
        self._dog, self._token = dog, token

    def progress(self, fraction: float) -> None:
        if self._dog is not None and self._token is not None:
            self._dog._progress(self._token, fraction)

    @property
    def hung(self) -> str | None:
        if self._dog is None or self._token is None:
            return None
        return self._dog._hung(self._token)

    def close(self) -> None:
        if self._dog is not None and self._token is not None:
            self._dog._finish(self._token)
            self._token = None


class Watchdog:
    def __init__(
        self,
        services: list[Service],
        *,
        gpu: Callable[[], GpuReading | None] = read_gpu,
        stall_limit: float = STALL_LIMIT,
        answer_limit: float = ANSWER_LIMIT,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._services = {service.name: service for service in services}
        self._gpu = gpu
        self._stall_limit = stall_limit
        self._answer_limit = answer_limit
        self._clock = clock
        self._lock = threading.Lock()
        self._watches: dict[int, _Watch] = {}
        self._next = 0
        self.last_gpu: GpuReading | None = None

    def watch(self, service: str | None, stop: Callable[[], None]) -> Watch:
        """Start watching a step that uses `service`. `stop` tells the step to end now."""
        if service is None or service not in self._services:
            return Watch(None, None)
        now = self._clock()
        with self._lock:
            self._next += 1
            token = self._next
            self._watches[token] = _Watch(service, stop, now)
            known = self._services[service]
            # A service counts as answering when a step starts; the probe takes it from here.
            known.last_answer = max(known.last_answer, now)
        return Watch(self, token)

    def check(self) -> list[str]:
        """One round of checks. Returns the services restarted."""
        with self._lock:
            watches = list(self._watches.values())
        in_use = {watch.service for watch in watches if watch.hung is None}
        if not in_use:
            return []
        gpu = self.last_gpu = self._gpu()
        now = self._clock()
        restarted = []
        for name in sorted(in_use):
            service = self._services[name]
            try:
                answers = bool(service.probe())
            except Exception:
                answers = False
            if answers:
                service.last_answer = now
            reason = None
            if now - service.last_answer > self._answer_limit:
                reason = f"{name} did not answer for {now - service.last_answer:.0f} s"
            else:
                for watch in watches:
                    if watch.service != name or watch.hung is not None:
                        continue
                    quiet = now - watch.last_progress
                    idle = gpu is None or gpu.utilization < IDLE_GPU or not service.uses_gpu
                    if quiet > self._stall_limit * 3 or (quiet > self._stall_limit and idle):
                        reason = f"{name} made no progress for {quiet:.0f} s"
                        break
            if reason is None:
                continue
            log.warning("watchdog: %s; restarting it", reason)
            for watch in watches:
                if watch.service == name and watch.hung is None:
                    watch.hung = reason
                    try:
                        watch.stop()
                    except Exception:
                        log.exception("watchdog could not stop the step")
            try:
                service.restart()
            except Exception:
                log.exception("watchdog could not restart %s", name)
            service.last_answer = self._clock()  # give the restart time to come up
            restarted.append(name)
        return restarted

    def run(self, stop: threading.Event, every: float = CHECK_EVERY) -> None:
        while not stop.wait(every):
            try:
                self.check()
            except Exception:
                log.exception("watchdog check failed")

    def _progress(self, token: int, fraction: float) -> None:
        with self._lock:
            watch = self._watches.get(token)
            if watch is not None and fraction > watch.best:
                watch.best = fraction
                watch.last_progress = self._clock()

    def _hung(self, token: int) -> str | None:
        with self._lock:
            watch = self._watches.get(token)
            return None if watch is None else watch.hung

    def _finish(self, token: int) -> None:
        with self._lock:
            self._watches.pop(token, None)


def comfy_service(restart: Callable[[], None], base_url: str | None = None) -> Service:
    """The app's ComfyUI, probed with ComfyClient.is_up(); `restart` is the lead's callback."""
    from memcraft.comfy import DEFAULT_URL, ComfyClient, check_url

    url = check_url(base_url or DEFAULT_URL)
    port = urlsplit(url).port or 80

    def probe() -> bool:
        with ComfyClient(base_url=url) as client:
            return client.is_up()

    return Service("comfyui", port, probe, restart)


def llama_server_service(
    restart: Callable[[], None], base_url: str, *, uses_gpu: bool = False
) -> Service:
    """The app's own Gemma llama-server, probed on its /health; never llama-swap."""
    import httpx

    parts = urlsplit(base_url)
    if parts.hostname not in ("127.0.0.1", "localhost", "::1"):
        raise ValueError("the app's llama-server runs on this PC only")
    url = base_url.rstrip("/")

    def probe() -> bool:
        try:
            return httpx.get(f"{url}/health", timeout=3.0).is_success
        except httpx.HTTPError:
            return False

    return Service("gemma", parts.port or 80, probe, restart, uses_gpu=uses_gpu)
