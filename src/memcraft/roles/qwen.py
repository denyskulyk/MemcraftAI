"""Qwen 3.8 as a text model inside the app's ComfyUI: the TextRunner the director and critic use.

The graph is workflows/qwen3.8-director. It has four picture slots because ComfyUI checks
every LoadImage; slots the caller does not fill fall back to the first picture, and
text_graph() then takes their links out, so Qwen sees each picture once.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from memcraft.workflows import WorkflowPackage, bind, is_link, load_workflow, output_node

if TYPE_CHECKING:
    from memcraft.comfy import ComfyClient

WORKFLOW = Path(__file__).parents[3] / "workflows" / "qwen3.8-director"
MAX_IMAGES = 4
MAX_LENGTH = 1800
TIMEOUT = 600.0


class QwenError(Exception):
    pass


def text_graph(
    package: WorkflowPackage,
    system: str,
    prompt: str,
    images: Sequence[str],
    seed: int,
    *,
    max_length: int = MAX_LENGTH,
) -> dict[str, Any]:
    """The bound graph for 1 to 4 pictures, named as ComfyUI's input folder knows them."""
    if not 1 <= len(images) <= MAX_IMAGES:
        raise QwenError(f"the text graph takes 1 to {MAX_IMAGES} pictures")
    values: dict[str, Any] = {
        "system": system,
        "prompt": prompt,
        "seed": seed,
        "max_length": max_length,
    }
    values |= {f"image_{index + 1}": name for index, name in enumerate(images)}
    graph = bind(package.api_json, package.manifest, values)
    unused = {f"image_{index + 1}" for index in range(len(images), MAX_IMAGES)}
    loaders = {target.node for name in unused for target in package.manifest.inputs[name].targets}
    doomed = {number for number in graph if _feeds_from(graph, number, loaders)}
    for node in graph.values():
        for field, value in list(node["inputs"].items()):
            if is_link(value) and value[0] in doomed:
                del node["inputs"][field]
    return _reachable(graph, output_node(package.manifest, "text"))


def _feeds_from(graph: Mapping[str, Any], node: str, loaders: set[str]) -> bool:
    """True if this node is one of the loaders or is made only from them."""
    if node in loaders:
        return True
    links = [v for v in graph[node]["inputs"].values() if is_link(v)]
    return bool(links) and all(_feeds_from(graph, link[0], loaders) for link in links)


def _reachable(graph: dict[str, Any], output: str) -> dict[str, Any]:
    keep, todo = set(), [output]
    while todo:
        node = todo.pop()
        if node in keep:
            continue
        keep.add(node)
        todo += [v[0] for v in graph[node]["inputs"].values() if is_link(v)]
    return {number: node for number, node in graph.items() if number in keep}


def answer_text(outputs: Mapping[str, Any], node: str) -> str:
    """The text a PreviewAny node shows, from ComfyUI's history."""
    text = (outputs.get(node) or {}).get("text")
    if isinstance(text, str):
        return text
    if isinstance(text, list) and all(isinstance(item, str) for item in text):
        return "".join(text)
    raise QwenError("the text graph gave no text")


def _picture_name(data: bytes) -> str:
    kind = "png" if data.startswith(b"\x89PNG") else "jpg"
    return f"{hashlib.sha256(data).hexdigest()[:32]}.{kind}"


class ComfyTextRunner:
    """A TextRunner on the app's ComfyUI. Pictures are uploaded under a name made from their
    sha256, so nothing about a person shows in ComfyUI's input folder.

    The caller holds the GPU lock around calls and frees the models afterwards.
    """

    def __init__(
        self,
        comfy: ComfyClient,
        *,
        package: WorkflowPackage | None = None,
        subfolder: str = "memcraft",
        max_length: int = MAX_LENGTH,
        timeout: float = TIMEOUT,
        should_cancel: Callable[[], bool] | None = None,
    ) -> None:
        self.comfy = comfy
        self.package = package or load_workflow(WORKFLOW)
        self.subfolder = subfolder
        self.max_length = max_length
        self.timeout = timeout
        self.should_cancel = should_cancel

    def __call__(self, system: str, prompt: str, images: Sequence[bytes], seed: int) -> str:
        names = [
            self.comfy.upload(data, _picture_name(data), subfolder=self.subfolder)
            for data in images
        ]
        graph = text_graph(self.package, system, prompt, names, seed, max_length=self.max_length)
        node = output_node(self.package.manifest, "text")
        result = self.comfy.run(
            graph, timeout=self.timeout, outputs=[node], should_cancel=self.should_cancel
        )
        return answer_text(result.outputs, node)
