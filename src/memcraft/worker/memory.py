"""Memory watch (E8.2): RAM and VRAM around every step, and room made before a GPU step.

`MemoryProbe.snapshot()` reads free RAM, the RAM of the worker, of the app's ComfyUI (8199) and
of Gemma's llama-server (llama-swap's memcraft-gemma or the worker's own on 8098), and the
VRAM from nvidia-smi. It never raises: what cannot be read is left out.

`MemoryWatch.before_step()` takes the start snapshot. Before a GPU step, when free RAM is
below `min_free_gb`, it first unloads Gemma (unless the step itself talks to Gemma), then frees
ComfyUI's cached models, and waits up to `wait_seconds` for RAM to come back. It never fails
the step: if RAM stays low the step runs anyway and the record says so. `after_step()` takes
the end snapshot. The worker writes both under the "memory" key of job_steps.output.
"""

from __future__ import annotations

import ctypes
import logging
import os
import subprocess
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)

GB = 1024**3
MB = 1024**2
MIN_FREE_GB = 6.0
WAIT_SECONDS = 180.0
SETTING_MIN_FREE = "memory_min_free_gb"
GEMMA_SERVICE = "gemma"
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

# Which processes to look at: a name, the image and a regex on the command line.
WATCHED: dict[str, tuple[str, str]] = {
    "comfyui": ("python.exe", r"\bmain\.py\b.*--port\s+8199\b"),
    "gemma": ("llama-server.exe", r"gemma-4-26B.*--device\s+none|--device\s+none.*gemma-4-26B"),
}


# Reading memory on Windows, with ctypes (no extra dependency).


class _MemoryStatus(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


class _ProcessCounters(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_ulong),
        ("PageFaultCount", ctypes.c_ulong),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
        ("PrivateUsage", ctypes.c_size_t),
    ]


_QUERY_LIMITED = 0x1000
_VM_READ = 0x0010
_STILL_ACTIVE = 259


def system_ram() -> tuple[int, int] | None:
    """(free, total) bytes of physical RAM, or None off Windows."""
    if os.name != "nt":
        return None
    status = _MemoryStatus()
    status.dwLength = ctypes.sizeof(_MemoryStatus)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):  # type: ignore[attr-defined]
        return None
    return int(status.ullAvailPhys), int(status.ullTotalPhys)


def process_ram(pid: int) -> tuple[int, int] | None:
    """(working set, private) bytes of a running process, or None if it is gone."""
    if os.name != "nt":
        return None
    kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
    psapi = ctypes.windll.psapi  # type: ignore[attr-defined]
    kernel32.OpenProcess.restype = ctypes.c_void_p
    handle = kernel32.OpenProcess(_QUERY_LIMITED | _VM_READ, False, pid)
    if not handle:
        handle = kernel32.OpenProcess(_QUERY_LIMITED, False, pid)
    if not handle:
        return None
    try:
        code = ctypes.c_ulong()
        if kernel32.GetExitCodeProcess(ctypes.c_void_p(handle), ctypes.byref(code)) and (
            code.value != _STILL_ACTIVE
        ):
            return None
        counters = _ProcessCounters()
        counters.cb = ctypes.sizeof(_ProcessCounters)
        if not psapi.GetProcessMemoryInfo(
            ctypes.c_void_p(handle), ctypes.byref(counters), counters.cb
        ):
            return None
        return int(counters.WorkingSetSize), int(counters.PrivateUsage)
    finally:
        kernel32.CloseHandle(ctypes.c_void_p(handle))


def find_pids(watched: Mapping[str, tuple[str, str]] = WATCHED) -> dict[str, list[int]]:
    """The process ids of each watched name, by image and command line (one PowerShell)."""
    if os.name != "nt" or not watched:
        return {}
    images = sorted({image for image, _ in watched.values()})
    filter_ = " OR ".join(f"Name='{image}'" for image in images)
    lines = []
    for name, (image, pattern) in watched.items():
        lines.append(
            f"$all | Where-Object {{ $_.Name -eq '{image}' -and $_.CommandLine -match "
            f"'{pattern}' }} | ForEach-Object {{ '{name} ' + $_.ProcessId }}"
        )
    script = f'$all = @(Get-CimInstance Win32_Process -Filter "{filter_}"); ' + "; ".join(lines)
    try:
        done = subprocess.run(  # noqa: S603 - a fixed script
            [_shell(), "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
            creationflags=_NO_WINDOW,
        )
    except (OSError, subprocess.TimeoutExpired):
        log.warning("cannot list processes for the memory watch", exc_info=True)
        return {}
    found: dict[str, list[int]] = {name: [] for name in watched}
    for line in done.stdout.splitlines():
        name, _, pid = line.strip().partition(" ")
        if name in found and pid.isdigit():
            found[name].append(int(pid))
    return found


def _shell() -> str:
    import shutil

    for name in ("pwsh.exe", "pwsh", "powershell.exe"):
        found = shutil.which(name)
        if found:
            return found
    raise OSError("no PowerShell")


def gpu_memory() -> dict[str, int] | None:
    """VRAM of the first GPU in MB: used, total, free. None without nvidia-smi."""
    try:
        done = subprocess.run(
            [  # noqa: S607 - nvidia-smi from PATH
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
            creationflags=_NO_WINDOW,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    first = done.stdout.strip().splitlines()[:1]
    if done.returncode != 0 or not first:
        return None
    try:
        used, total, free = (int(float(part)) for part in first[0].split(","))
    except ValueError:
        return None
    return {"used_mb": used, "total_mb": total, "free_mb": free}


# The snapshot.


@dataclass
class MemoryProbe:
    """Reads memory. Every reader can be replaced, for tests."""

    ram: Callable[[], tuple[int, int] | None] = system_ram
    process: Callable[[int], tuple[int, int] | None] = process_ram
    finder: Callable[[], dict[str, list[int]]] = find_pids
    gpu: Callable[[], dict[str, int] | None] = gpu_memory
    own_pid: int = field(default_factory=os.getpid)
    # A process that is not found is looked for again at most this often (PowerShell is slow).
    rescan_seconds: float = 60.0
    clock: Callable[[], float] = time.monotonic
    _pids: dict[str, list[int]] = field(default_factory=dict, repr=False)
    _scanned: float | None = field(default=None, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def free_gb(self) -> float | None:
        try:
            ram = self.ram()
        except Exception:
            log.warning("reading free RAM failed", exc_info=True)
            return None
        return None if ram is None else round(ram[0] / GB, 2)

    def snapshot(self) -> dict[str, Any]:
        record: dict[str, Any] = {"at": time.strftime("%Y-%m-%dT%H:%M:%S")}
        try:
            ram = self.ram()
            if ram is not None:
                record["ram_free_gb"] = round(ram[0] / GB, 2)
                record["ram_total_gb"] = round(ram[1] / GB, 2)
        except Exception:
            log.warning("reading free RAM failed", exc_info=True)
        processes: dict[str, Any] = {}
        try:
            mine = self.process(self.own_pid)
            if mine is not None:
                processes["worker"] = _mb(mine)
            for name, pids in self._watched().items():
                values = [v for v in (self.process(pid) for pid in pids) if v is not None]
                if values:
                    processes[name] = _mb((sum(v[0] for v in values), sum(v[1] for v in values)))
        except Exception:
            log.warning("reading process RAM failed", exc_info=True)
        record["processes"] = processes
        try:
            vram = self.gpu()
            if vram is not None:
                record["vram"] = vram
        except Exception:
            log.warning("reading VRAM failed", exc_info=True)
        return record

    def forget(self) -> None:
        """Look the processes up again on the next snapshot (after a service was stopped)."""
        with self._lock:
            self._scanned = None

    def _watched(self) -> dict[str, list[int]]:
        with self._lock:
            alive = {
                name: [pid for pid in pids if self.process(pid) is not None]
                for name, pids in self._pids.items()
            }
            lost = any(len(alive[name]) != len(pids) for name, pids in self._pids.items())
            missing = not self._pids or any(not pids for pids in alive.values())
            stale = self._scanned is None or self.clock() - self._scanned >= self.rescan_seconds
            if lost or (missing and stale):
                try:
                    self._pids = {name: list(pids) for name, pids in self.finder().items()}
                except Exception:
                    log.warning("finding the watched processes failed", exc_info=True)
                    self._pids = alive
                self._scanned = self.clock()
                return dict(self._pids)
            self._pids = alive
            return alive


def _mb(values: tuple[int, int]) -> dict[str, int]:
    return {"working_set_mb": round(values[0] / MB), "private_mb": round(values[1] / MB)}


def _size(mb: int) -> str:
    return f"{mb / 1024:.1f} GB" if mb >= 1024 else f"{mb} MB"


def describe(snapshot: Mapping[str, Any]) -> str:
    """One line for the worker log."""
    parts = []
    if "ram_free_gb" in snapshot:
        total = snapshot.get("ram_total_gb", 0)
        parts.append(f"RAM free {snapshot['ram_free_gb']:.1f}/{total:.0f} GB")
    for name, values in (snapshot.get("processes") or {}).items():
        parts.append(f"{name} {_size(values['working_set_mb'])}")
    vram = snapshot.get("vram")
    if vram:
        parts.append(f"VRAM {vram['used_mb'] / 1024:.1f}/{vram['total_mb'] / 1024:.0f} GB")
    return ", ".join(parts) or "no memory readings"


# The watch.


@dataclass
class MemoryWatch:
    """Snapshots around steps, and room made before a GPU step when RAM is tight."""

    probe: MemoryProbe = field(default_factory=MemoryProbe)
    free_gemma: Callable[[], None] | None = None
    free_comfy: Callable[[], None] | None = None
    min_free_gb: float = MIN_FREE_GB
    wait_seconds: float = WAIT_SECONDS
    poll: float = 5.0
    clock: Callable[[], float] = time.monotonic

    def before_step(
        self,
        *,
        needs_gpu: bool,
        service: str | None = None,
        label: str = "",
        min_free_gb: float | None = None,
        stop: threading.Event | None = None,
    ) -> dict[str, Any]:
        """The record of the step's start: {"start": snapshot, "made_room": {...}?}."""
        record: dict[str, Any] = {}
        limit = self.min_free_gb if min_free_gb is None else min_free_gb
        if needs_gpu and limit > 0:
            made = self.make_room(limit, keep_gemma=service == GEMMA_SERVICE, stop=stop)
            if made is not None:
                record["made_room"] = made
        record["start"] = self.probe.snapshot()
        log.info("memory at the start of %s: %s", label or "a step", describe(record["start"]))
        return record

    def after_step(self, record: dict[str, Any], label: str = "") -> dict[str, Any]:
        record["end"] = self.probe.snapshot()
        log.info("memory at the end of %s: %s", label or "a step", describe(record["end"]))
        return record

    def make_room(
        self, limit_gb: float, *, keep_gemma: bool = False, stop: threading.Event | None = None
    ) -> dict[str, Any] | None:
        """Free RAM if below `limit_gb`. None if there was enough; else what was done."""
        free = self.probe.free_gb()
        if free is None or free >= limit_gb:
            return None
        log.warning("only %.1f GB RAM free (want %.1f GB): making room", free, limit_gb)
        made: dict[str, Any] = {"free_gb_before": free, "limit_gb": limit_gb, "actions": []}
        actions: list[tuple[str, Callable[[], None] | None]] = []
        if not keep_gemma:
            actions.append(("unloaded Gemma", self.free_gemma))
        actions.append(("freed ComfyUI models", self.free_comfy))
        started = self.clock()
        for name, action in actions:
            if action is None:
                continue
            try:
                action()
                made["actions"].append(name)
                log.info("memory watch: %s", name)
            except Exception as error:
                made["actions"].append(f"{name} failed: {error}"[:200])
                log.warning("memory watch: %s failed: %s", name, error)
        self.probe.forget()
        free = self.probe.free_gb()
        while free is not None and free < limit_gb:
            if self.clock() - started >= self.wait_seconds:
                break
            if stop is not None:
                if stop.wait(self.poll):
                    break
            else:
                time.sleep(self.poll)
            free = self.probe.free_gb()
        made["free_gb_after"] = free
        made["waited_seconds"] = round(self.clock() - started, 1)
        made["enough"] = free is not None and free >= limit_gb
        if made["enough"]:
            log.info("memory watch: %.1f GB free after %.0f s", free, made["waited_seconds"])
        else:
            log.warning(
                "memory watch: still %s GB free after %.0f s; the step runs anyway",
                free,
                made["waited_seconds"],
            )
        return made
