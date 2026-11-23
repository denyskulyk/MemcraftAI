"""Client for the app's own headless ComfyUI on 127.0.0.1:8199 (started by deploy/comfyui.ps1).

It never talks to the ComfyUI on port 8188: the constructor refuses that port and any host that
is not this PC.

    with ComfyClient() as comfy:
        name = comfy.upload(Path("photo.png"), "a1b2.png", subfolder="job123")
        result = comfy.run(graph, outputs=["43"], timeout=600, on_progress=print)
        for file in result.files("43"):
            comfy.download(file, Path("out") / file.filename)
        comfy.free()

Calls:

- ``is_up()``, ``system_stats()``, ``object_info(class_type=None)``.
- ``upload(source, name, subfolder="")`` puts a file (path or bytes) into ComfyUI's input folder
  and returns the name a LoadImage or LoadLatent node takes ("subfolder/name").
- ``queue(graph, outputs=None, prompt_id=None)`` sends an API-format graph and returns its
  prompt id. ``outputs`` lists the output nodes to run; ComfyUI then neither runs nor checks
  the others. A rejected graph raises GraphRejectedError with ComfyUI's ``node_errors``.
- ``wait(prompt_id, timeout, on_progress=None, should_cancel=None)`` polls /queue and
  /history until the prompt ends and returns a RunResult. ``on_progress`` gets a Progress
  (pending with its place in the queue, or running) at every poll. ``should_cancel()`` returning
  True cancels the prompt and raises CancelledRunError. After ``timeout`` seconds the prompt is
  cancelled and ComfyTimeoutError is raised. A failed run raises ExecutionFailedError.
- ``run(...)`` is queue() and wait() in one call.
- ``status(prompt_id)`` is "pending", "running", "done" or "unknown": a worker that restarted
  can find out what happened to a prompt it queued (give queue() its own prompt_id for that).
- ``result(prompt_id)`` is the RunResult of a finished prompt, or None.
- ``download(file, dest)`` copies an output file over HTTP (/view).
- ``cancel(prompt_id)`` removes a pending prompt or interrupts it while it runs;
  ``interrupt()`` stops whatever runs now.
- ``free(unload_models=True, free_memory=True)`` releases VRAM and RAM.

Progress comes from polling, so it says pending or running and the time spent, not the step of
the sampler; ComfyUI's websocket would need a new dependency.
"""

from __future__ import annotations

import re
import time
import uuid
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

import httpx

DEFAULT_URL = "http://127.0.0.1:8199"
FORBIDDEN_PORT = 8188
_LOOPBACK = frozenset({"127.0.0.1", "localhost", "::1"})
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SUBFOLDER = re.compile(r"^(?:[A-Za-z0-9_-]{1,64}(?:/[A-Za-z0-9_-]{1,64}){0,3})?$")

type State = Literal["pending", "running", "done", "unknown"]


class ComfyError(Exception):
    pass


class ComfyUnavailableError(ComfyError):
    """ComfyUI does not answer."""


class GraphRejectedError(ComfyError):
    """POST /prompt refused the graph. ``node_errors`` maps node ids to ComfyUI's reasons."""

    def __init__(self, message: str, error: Any = None, node_errors: Any = None) -> None:
        super().__init__(message)
        self.error = error
        self.node_errors = node_errors or {}


class ExecutionFailedError(ComfyError):
    """The graph started and a node failed."""

    def __init__(self, message: str, prompt_id: str, details: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.prompt_id = prompt_id
        self.details = dict(details)


class CancelledRunError(ComfyError):
    """The run was cancelled or interrupted before it finished."""


class ComfyTimeoutError(ComfyError):
    """The run took longer than allowed; it has been cancelled."""


@dataclass(frozen=True)
class OutputFile:
    filename: str
    subfolder: str = ""
    type: str = "output"

    @property
    def relpath(self) -> str:
        """Relative to ComfyUI's folder of this type, with forward slashes."""
        return f"{self.subfolder}/{self.filename}" if self.subfolder else self.filename


@dataclass(frozen=True)
class Progress:
    prompt_id: str
    state: State
    # Prompts ahead of this one; 0 while it runs.
    position: int
    elapsed: float


@dataclass(frozen=True)
class RunResult:
    prompt_id: str
    # What /history reports per output node, as ComfyUI sent it.
    outputs: dict[str, Any]
    seconds: float
    status: dict[str, Any] = field(default_factory=dict)

    def files(self, node: str) -> list[OutputFile]:
        """Files one output node wrote: images, videos, latents, whatever the key."""
        found = []
        for value in self.outputs.get(node, {}).values():
            if not isinstance(value, list):
                continue
            for item in value:
                if isinstance(item, Mapping) and isinstance(item.get("filename"), str):
                    found.append(
                        OutputFile(
                            filename=item["filename"],
                            subfolder=str(item.get("subfolder") or ""),
                            type=str(item.get("type") or "output"),
                        )
                    )
        return found


def check_url(base_url: str) -> str:
    """The URL without a trailing slash, if it points at this PC and not at port 8188."""
    parts = urlsplit(base_url)
    if parts.scheme not in ("http", "https") or parts.hostname is None:
        raise ValueError(f"not an http URL: {base_url}")
    if parts.hostname not in _LOOPBACK:
        raise ValueError("the app's ComfyUI runs on this PC only")
    if parts.port == FORBIDDEN_PORT:
        raise ValueError("port 8188 is Denys's own ComfyUI; the app uses 8199")
    return base_url.rstrip("/")


class ComfyClient:
    def __init__(
        self,
        base_url: str = DEFAULT_URL,
        *,
        timeout: float = 30.0,
        poll_interval: float = 1.0,
        # How long the server may stay unreachable during wait() before it counts as gone.
        unreachable_after: float = 60.0,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.base_url = check_url(base_url)
        self.poll_interval = poll_interval
        self.unreachable_after = unreachable_after
        self._sleep = sleep
        self._clock = clock
        self._http = httpx.Client(base_url=self.base_url, timeout=timeout, transport=transport)

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> ComfyClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # --- plain calls -------------------------------------------------------------------

    def is_up(self) -> bool:
        try:
            return self._http.get("/system_stats", timeout=3.0).is_success
        except httpx.HTTPError:
            return False

    def system_stats(self) -> dict[str, Any]:
        return self._json("GET", "/system_stats")

    def object_info(self, class_type: str | None = None) -> dict[str, Any]:
        path = "/object_info" if class_type is None else f"/object_info/{class_type}"
        return self._json("GET", path)

    def upload(self, source: Path | bytes, name: str, *, subfolder: str = "") -> str:
        """Put a file into the input folder, replacing one of the same name."""
        if not _NAME.match(name) or ".." in name:
            raise ValueError(f"unsafe file name: {name!r}")
        if not _SUBFOLDER.match(subfolder):
            raise ValueError(f"unsafe subfolder: {subfolder!r}")
        content = source if isinstance(source, bytes) else Path(source).read_bytes()
        answer = self._json(
            "POST",
            "/upload/image",
            files={"image": (name, content, "application/octet-stream")},
            data={"subfolder": subfolder, "type": "input", "overwrite": "true"},
        )
        stored = answer.get("name")
        if stored != name:
            raise ComfyError(f"ComfyUI stored the upload as {stored!r}, not {name!r}")
        return f"{subfolder}/{name}" if subfolder else name

    def queue(
        self,
        graph: Mapping[str, Any],
        *,
        outputs: Iterable[str] | None = None,
        prompt_id: str | None = None,
    ) -> str:
        prompt_id = str(uuid.UUID(prompt_id)) if prompt_id else str(uuid.uuid4())
        body: dict[str, Any] = {"prompt": graph, "prompt_id": prompt_id, "client_id": "memcraft"}
        if outputs is not None:
            targets = list(outputs)
            unknown = [node for node in targets if node not in graph]
            if not targets or unknown:
                raise ValueError(f"output nodes not in the graph: {unknown or 'none given'}")
            body["partial_execution_targets"] = targets
        response = self._request("POST", "/prompt", json=body)
        try:
            answer = response.json()
        except ValueError:
            answer = {}
        if response.status_code == 400 or answer.get("error"):
            error = answer.get("error")
            message = error.get("message") if isinstance(error, Mapping) else error
            raise GraphRejectedError(
                f"ComfyUI rejected the graph: {message or response.text[:500]}",
                error,
                answer.get("node_errors"),
            )
        _ok(response)
        return str(answer["prompt_id"])

    def status(self, prompt_id: str) -> State:
        if self._history(prompt_id) is not None:
            return "done"
        state, _ = self._queue_place(prompt_id)
        return state

    def result(self, prompt_id: str) -> RunResult | None:
        """The finished run, or None while it is not finished. Raises if it failed."""
        entry = self._history(prompt_id)
        return None if entry is None else _finish(prompt_id, entry, 0.0)

    def wait(
        self,
        prompt_id: str,
        *,
        timeout: float,
        on_progress: Callable[[Progress], None] | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> RunResult:
        start = self._clock()
        last_seen = start
        missing = 0
        while True:
            now = self._clock()
            elapsed = now - start
            if should_cancel is not None and should_cancel():
                self.cancel(prompt_id)
                raise CancelledRunError(f"prompt {prompt_id} was cancelled")
            if elapsed > timeout:
                self.cancel(prompt_id)
                raise ComfyTimeoutError(f"prompt {prompt_id} took longer than {timeout:.0f} s")
            try:
                entry = self._history(prompt_id)
                if entry is not None:
                    return _finish(prompt_id, entry, elapsed)
                state, position = self._queue_place(prompt_id)
            except ComfyUnavailableError:
                if now - last_seen > self.unreachable_after:
                    raise
                self._sleep(self.poll_interval)
                continue
            last_seen = now
            if state == "unknown":
                # It may have finished between the two calls; give history a few more looks.
                missing += 1
                if missing > 3:
                    raise ComfyError(f"prompt {prompt_id} is neither queued nor in the history")
            else:
                missing = 0
                if on_progress is not None:
                    on_progress(Progress(prompt_id, state, position, elapsed))
            self._sleep(self.poll_interval)

    def run(
        self,
        graph: Mapping[str, Any],
        *,
        timeout: float,
        outputs: Iterable[str] | None = None,
        prompt_id: str | None = None,
        on_progress: Callable[[Progress], None] | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> RunResult:
        queued = self.queue(graph, outputs=outputs, prompt_id=prompt_id)
        return self.wait(
            queued, timeout=timeout, on_progress=on_progress, should_cancel=should_cancel
        )

    def download(self, file: OutputFile, dest: Path) -> Path:
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        params = {"filename": file.filename, "subfolder": file.subfolder, "type": file.type}
        part = dest.with_name(dest.name + ".part")
        try:
            with self._http.stream("GET", "/view", params=params, timeout=120.0) as response:
                _ok(response)
                with part.open("wb") as out:
                    for chunk in response.iter_bytes(1024 * 1024):
                        out.write(chunk)
            part.replace(dest)
        except httpx.HTTPError as error:
            raise ComfyUnavailableError(f"cannot download {file.relpath}: {error}") from error
        finally:
            part.unlink(missing_ok=True)
        return dest

    def cancel(self, prompt_id: str) -> None:
        """Remove the prompt from the queue, or interrupt it if it runs. Idempotent."""
        self._request("POST", "/queue", json={"delete": [prompt_id]})
        state, _ = self._queue_place(prompt_id)
        if state == "running":
            # With a prompt id ComfyUI interrupts only if that prompt is the one running.
            self._request("POST", "/interrupt", json={"prompt_id": prompt_id})

    def interrupt(self) -> None:
        self._request("POST", "/interrupt", json={})

    def free(self, *, unload_models: bool = True, free_memory: bool = True) -> None:
        self._request(
            "POST", "/free", json={"unload_models": unload_models, "free_memory": free_memory}
        )

    # --- internals ---------------------------------------------------------------------

    def _history(self, prompt_id: str) -> dict[str, Any] | None:
        answer = self._json("GET", f"/history/{prompt_id}")
        entry = answer.get(prompt_id)
        if not isinstance(entry, Mapping):
            return None
        status = entry.get("status") or {}
        # A prompt is written to the history when it ends; "completed" false with no status
        # string has not been seen, but would mean it is not finished.
        if status and not status.get("completed", True) and not status.get("status_str"):
            return None
        return dict(entry)

    def _queue_place(self, prompt_id: str) -> tuple[State, int]:
        answer = self._json("GET", "/queue")
        running = answer.get("queue_running") or []
        pending = answer.get("queue_pending") or []
        if any(_item_id(item) == prompt_id for item in running):
            return "running", 0
        ordered = sorted(pending, key=lambda item: item[0] if item else 0)
        for place, item in enumerate(ordered):
            if _item_id(item) == prompt_id:
                return "pending", len(running) + place
        return "unknown", 0

    def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        try:
            response = self._http.request(method, path, **kwargs)
        except httpx.HTTPError as error:
            raise ComfyUnavailableError(f"ComfyUI at {self.base_url}: {error}") from error
        return response

    def _json(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        response = self._request(method, path, **kwargs)
        _ok(response)
        try:
            answer = response.json()
        except ValueError as error:
            raise ComfyError(f"{path} did not answer JSON") from error
        if not isinstance(answer, dict):
            raise ComfyError(f"{path} did not answer a JSON object")
        return answer


def _ok(response: httpx.Response) -> None:
    if response.status_code >= 500:
        raise ComfyUnavailableError(f"{response.request.url.path}: HTTP {response.status_code}")
    if not response.is_success:
        raise ComfyError(f"{response.request.url.path}: HTTP {response.status_code}")


def _item_id(item: Any) -> str | None:
    return item[1] if isinstance(item, list | tuple) and len(item) > 1 else None


def _finish(prompt_id: str, entry: Mapping[str, Any], elapsed: float) -> RunResult:
    status = dict(entry.get("status") or {})
    messages = status.get("messages") or []
    events = {m[0]: m[1] for m in messages if isinstance(m, list | tuple) and len(m) == 2}
    if "execution_interrupted" in events:
        raise CancelledRunError(f"prompt {prompt_id} was interrupted")
    if status.get("status_str") == "error" or "execution_error" in events:
        details = events.get("execution_error") or {}
        node = details.get("node_id")
        kind = details.get("node_type")
        text = details.get("exception_message") or "unknown error"
        raise ExecutionFailedError(
            f"node {node} ({kind}) failed: {str(text).strip()[:500]}", prompt_id, details
        )
    outputs = entry.get("outputs") or {}
    return RunResult(prompt_id, dict(outputs), elapsed, status)
