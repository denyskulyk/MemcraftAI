"""What every engine adapter shares: limits, errors, and running one bound graph on ComfyUI."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from memcraft.comfy import ComfyClient, Progress, RunResult
from memcraft.workflows import Manifest, bind, check_workflow, output_node

_FOLDER = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class EngineError(Exception):
    pass


class LimitError(EngineError, ValueError):
    """The request asks for more than the engine can do."""


class OutputMissingError(EngineError):
    """ComfyUI finished but did not write a file the adapter needs."""


@dataclass(frozen=True)
class RunOptions:
    """How one ComfyUI run is waited on. None for timeout means the adapter's default."""

    timeout: float | None = None
    on_progress: Callable[[Progress], None] | None = None
    should_cancel: Callable[[], bool] | None = None
    # A uuid the caller chose, so that a restarted worker can look the run up.
    prompt_id: str | None = None


class Adapter:
    """One engine, bound to one workflow version (its graph and manifest)."""

    slot: str = ""
    name: str = ""
    workflow_name: str = ""

    def __init__(self, api_json: Mapping[str, Any], manifest: Manifest | Mapping[str, Any]):
        self.api_json = dict(api_json)
        self.manifest = check_workflow(api_json, manifest)
        if self.manifest.name != self.workflow_name:
            raise EngineError(
                f"{type(self).__name__} runs {self.workflow_name}, not {self.manifest.name}"
            )

    def bind(self, values: Mapping[str, Any]) -> dict[str, Any]:
        return bind(self.api_json, self.manifest, values)

    def output_nodes(self, names: Iterable[str]) -> list[str]:
        return [output_node(self.manifest, name) for name in names]

    def run(
        self,
        client: ComfyClient,
        values: Mapping[str, Any],
        outputs: Iterable[str],
        options: RunOptions,
        default_timeout: float,
    ) -> RunResult:
        graph = self.bind(values)
        return client.run(
            graph,
            outputs=self.output_nodes(outputs),
            timeout=options.timeout if options.timeout is not None else default_timeout,
            prompt_id=options.prompt_id,
            on_progress=options.on_progress,
            should_cancel=options.should_cancel,
        )

    def fetch(self, client: ComfyClient, result: RunResult, output: str, dest: Path) -> Path:
        """Download the first file one output wrote."""
        files = result.files(output_node(self.manifest, output))
        if not files:
            raise OutputMissingError(f"{self.name}: no file for {output}")
        return client.download(files[0], dest)


def check_folder(folder: str) -> str:
    if not _FOLDER.match(folder):
        raise ValueError(f"unsafe folder name: {folder!r}")
    return folder


def upload_input(client: ComfyClient, path: Path, folder: str) -> str:
    """Put a local file into ComfyUI's input folder under a name taken from its content.

    The same file gets the same name, so a retried step does not pile up copies.
    """
    path = Path(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:24]
    suffix = path.suffix.lower() or ".bin"
    return client.upload(path, f"{digest}{suffix}", subfolder=check_folder(folder))


def is_8n_plus_1(frames: int) -> bool:
    return frames >= 9 and frames % 8 == 1


def frames_for(seconds: float, fps: int, max_frames: int) -> int:
    """The 8n+1 frame count closest to a duration, within the engine's limit."""
    cap = max_frames - (max_frames - 1) % 8
    return max(9, min(8 * round(seconds * fps / 8) + 1, cap))
