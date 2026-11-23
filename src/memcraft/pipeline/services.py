"""The two servers the pipeline talks to, both the app's own: ComfyUI on 8199 and a CPU-only
llama-server for Gemma on 8098.

ComfyServer starts ComfyUI through deploy/comfyui.ps1 when it does not answer, frees its
models, and restarts it for the watchdog. GemmaServer starts llama-server with the GPU off when
a step needs Gemma and stops it after some idle minutes, so its 16 GB of RAM come back while
videos render. Neither ever touches Denys's ComfyUI on 8188 or llama-swap on 8080; moving Gemma
into llama-swap (E1.6) is a separate decision.
"""

from __future__ import annotations

import contextlib
import logging
import os
import subprocess
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from memcraft.comfy import DEFAULT_URL, ComfyClient, check_url
from memcraft.roles.gemma import GemmaClient
from memcraft.worker.watchdog import NOT_OURS

log = logging.getLogger(__name__)

PROJECT = Path(__file__).resolve().parents[3]
COMFY_SCRIPT = PROJECT / "deploy" / "comfyui.ps1"
COMFY_DATA = PROJECT / "data" / "comfy"
GEMMA_PORT = 8098
GEMMA_URL = f"http://127.0.0.1:{GEMMA_PORT}"
LLAMA_SERVER = Path(r"C:\Source\LocalAI\llama.cpp\bin\llama-server.exe")
GEMMA_MODELS = Path(r"C:\Source\LocalAI\gemma-26b\models")
GEMMA_MODEL = "gemma-4-26B-A4B-it-ultra-uncensored-heretic.i1-Q4_K_M.gguf"
GEMMA_MMPROJ = "gemma-4-26B-A4B-it-ultra-uncensored-heretic.mmproj-f16.gguf"
IDLE_MINUTES = 5.0
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
_NEW_GROUP = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)


class ServiceError(Exception):
    pass


def _powershell() -> str:
    for name in ("pwsh.exe", "pwsh", "powershell.exe"):
        found = _which(name)
        if found:
            return found
    raise ServiceError("no PowerShell to run deploy scripts")


def _which(name: str) -> str | None:
    import shutil

    return shutil.which(name)


def _own_pids(image: str, pattern: str) -> list[int]:
    """Process ids of `image` whose command line matches the regex `pattern` (Windows)."""
    script = (
        f"Get-CimInstance Win32_Process -Filter \"Name='{image}'\" | "
        f"Where-Object {{ $_.CommandLine -match '{pattern}' }} | "
        "ForEach-Object { $_.ProcessId }"
    )
    try:
        done = subprocess.run(  # noqa: S603 - fixed script
            [_powershell(), "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
            creationflags=_NO_WINDOW,
        )
    except (OSError, subprocess.TimeoutExpired, ServiceError):
        log.warning("cannot list %s processes", image, exc_info=True)
        return []
    return [int(line) for line in done.stdout.split() if line.strip().isdigit()]


def _kill(pid: int) -> None:
    subprocess.run(  # noqa: S603 - taskkill of a process this module started
        ["taskkill", "/PID", str(pid), "/T", "/F"],  # noqa: S607
        capture_output=True,
        timeout=30,
        check=False,
        creationflags=_NO_WINDOW,
    )


@dataclass
class ComfyServer:
    """The app's ComfyUI. `client()` makes a client; the caller closes it."""

    base_url: str = DEFAULT_URL
    script: Path = COMFY_SCRIPT
    data_dir: Path = COMFY_DATA
    start_timeout: int = 300
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self) -> None:
        self.base_url = check_url(self.base_url)

    def client(self) -> ComfyClient:
        return ComfyClient(self.base_url)

    def is_up(self) -> bool:
        with self.client() as client:
            return client.is_up()

    def ensure_up(self) -> None:
        if self.is_up():
            return
        with self._lock:
            if self.is_up():
                return
            log.info("starting ComfyUI on %s", self.base_url)
            self._script("start")

    def free(self) -> None:
        if not self.is_up():
            return
        with self.client() as client:
            client.free()

    def restart(self) -> None:
        with self._lock:
            log.warning("restarting ComfyUI on %s", self.base_url)
            self._script("stop")
            self._script("start")

    def stop(self) -> None:
        with self._lock:
            self._script("stop")

    def clean(self, folder: str) -> None:
        """Remove a job's subfolder from ComfyUI's input and output folders."""
        import shutil

        from memcraft.engines.base import check_folder

        check_folder(folder)
        for kind in ("input", "output"):
            shutil.rmtree(self.data_dir / kind / folder, ignore_errors=True)

    def _script(self, action: str) -> None:
        args = [_powershell(), "-NoProfile", "-ExecutionPolicy", "Bypass", "-File"]
        args += [str(self.script), action, "-TimeoutSeconds", str(self.start_timeout)]
        done = subprocess.run(  # noqa: S603 - the project's own deploy script
            args,
            capture_output=True,
            text=True,
            timeout=self.start_timeout + 60,
            check=False,
            creationflags=_NO_WINDOW,
        )
        if done.returncode != 0:
            raise ServiceError(f"comfyui.ps1 {action} failed: {(done.stdout + done.stderr)[-500:]}")
        log.info("comfyui.ps1 %s: %s", action, done.stdout.strip()[-200:])


@dataclass
class GemmaConfig:
    exe: Path = LLAMA_SERVER
    model: Path = GEMMA_MODELS / GEMMA_MODEL
    mmproj: Path = GEMMA_MODELS / GEMMA_MMPROJ
    port: int = GEMMA_PORT
    threads: int = 8
    context: int = 16384
    # E0.13: a cap on image tokens makes 2 or 3 photos faster; None keeps every token (the
    # critic in E0.14 ran without a cap).
    image_max_tokens: int | None = None
    log_file: Path = PROJECT / "data" / "logs" / "gemma-server.log"
    start_timeout: float = 240.0

    def args(self) -> list[str]:
        args = [
            str(self.exe),
            "--host",
            "127.0.0.1",
            "--port",
            str(self.port),
            "--device",
            "none",
            "-m",
            str(self.model),
            "--mmproj",
            str(self.mmproj),
            "--no-mmproj-offload",
            "--jinja",
            "-t",
            str(self.threads),
            "-c",
            str(self.context),
            "-np",
            "1",
            "--cache-ram",
            "0",
        ]
        if self.image_max_tokens:
            args += ["--image-max-tokens", str(self.image_max_tokens)]
        return args


class GemmaServer:
    """Gemma on the CPU, started on demand and stopped after `idle_minutes` without use."""

    def __init__(
        self,
        config: GemmaConfig | None = None,
        *,
        idle_minutes: float = IDLE_MINUTES,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config or GemmaConfig()
        if self.config.port in NOT_OURS:
            raise ValueError(f"port {self.config.port} is not the app's own llama-server")
        self.url = f"http://127.0.0.1:{self.config.port}"
        self.idle_seconds = idle_minutes * 60
        self._clock = clock
        self._lock = threading.RLock()
        self._busy = 0
        self._last_used = clock()
        self._process: subprocess.Popen[bytes] | None = None
        self._log: object | None = None

    # Use.

    @contextlib.contextmanager
    def client(self, timeout: float = 600.0) -> Iterator[GemmaClient]:
        """A client of a running server; the server is not stopped while a client is out."""
        with self._lock:
            self._busy += 1
        try:
            self.ensure_up()
            yield GemmaClient(self.url, timeout=timeout)
        finally:
            with self._lock:
                self._busy -= 1
                self._last_used = self._clock()

    def is_up(self) -> bool:
        try:
            return httpx.get(f"{self.url}/health", timeout=3.0).is_success
        except httpx.HTTPError:
            return False

    def ensure_up(self) -> None:
        with self._lock:
            if self.is_up():
                return
            if self._process is not None and self._process.poll() is None:
                self._wait_up(self._process)
                return
            self._start()

    # Life cycle.

    def _start(self) -> None:
        config = self.config
        for path in (config.exe, config.model, config.mmproj):
            if not Path(path).is_file():
                raise ServiceError(f"llama-server or a Gemma file is missing: {path}")
        self._kill_strays()
        config.log_file.parent.mkdir(parents=True, exist_ok=True)
        log_handle = config.log_file.open("ab")
        log.info("starting Gemma on %s (CPU)", self.url)
        started = self._clock()
        process = subprocess.Popen(  # noqa: S603 - the configured llama-server
            config.args(),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            creationflags=_NO_WINDOW | _NEW_GROUP,
            cwd=str(Path(config.exe).parent),
        )
        self._process, self._log = process, log_handle
        self._wait_up(process)
        self._last_used = self._clock()
        log.info("Gemma is up after %.0f s", self._clock() - started)

    def _wait_up(self, process: subprocess.Popen[bytes]) -> None:
        deadline = time.monotonic() + self.config.start_timeout
        while not self.is_up():
            if process.poll() is not None:
                self._forget()
                raise ServiceError(
                    f"llama-server exited with {process.returncode}; see {self.config.log_file}"
                )
            if time.monotonic() > deadline:
                raise ServiceError(f"Gemma did not answer within {self.config.start_timeout} s")
            time.sleep(0.5)

    def stop(self) -> None:
        with self._lock:
            process = self._process
            if process is not None and process.poll() is None:
                log.info("stopping Gemma on %s", self.url)
                _kill(process.pid)
                with contextlib.suppress(subprocess.TimeoutExpired):
                    process.wait(30)
            self._forget()
            self._kill_strays()

    def restart(self) -> None:
        with self._lock:
            log.warning("restarting Gemma on %s", self.url)
            self.stop()
            self._start()

    def stop_if_idle(self) -> bool:
        """Stop the server if nobody used it for `idle_seconds`. True if it was stopped."""
        with self._lock:
            if self._busy or self._process is None:
                return False
            if self._clock() - self._last_used < self.idle_seconds:
                return False
            log.info("Gemma idle for %.0f min", self.idle_seconds / 60)
            self.stop()
            return True

    def run_idle_stopper(self, stop: threading.Event, every: float = 15.0) -> None:
        while not stop.wait(every):
            try:
                self.stop_if_idle()
            except Exception:
                log.exception("stopping the idle Gemma failed")

    @property
    def running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def _forget(self) -> None:
        self._process = None
        if self._log is not None:
            with contextlib.suppress(OSError):
                self._log.close()  # type: ignore[attr-defined]
            self._log = None

    def _kill_strays(self) -> None:
        """A llama-server on our port left by a worker that died."""
        if os.name != "nt":
            return
        for pid in _own_pids("llama-server.exe", rf"--port\s+{self.config.port}\b"):
            log.warning("stopping a stray llama-server %d on port %d", pid, self.config.port)
            _kill(pid)


def stop_all(*stoppers: Callable[[], None]) -> None:
    for stop in stoppers:
        try:
            stop()
        except Exception:
            log.exception("stopping a service failed")
