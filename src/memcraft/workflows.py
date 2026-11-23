"""ComfyUI workflows as data: an API-format graph and a manifest that names its inputs.

Code asks for "seed" or "first_frame"; only the manifest knows node numbers.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError

API_FILE = "api.json"
MANIFEST_FILE = "manifest.json"

Name = Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_]{0,63}$")]
Text = Annotated[str, StringConstraints(min_length=1)]


class WorkflowError(Exception):
    pass


class InvalidWorkflowError(WorkflowError):
    """The graph or the manifest is malformed, or they do not fit together."""


class BindError(WorkflowError):
    """The values do not fit the manifest."""


class InputType(StrEnum):
    STRING = "string"
    INT = "int"
    FLOAT = "float"
    BOOL = "bool"
    # The name of a file in ComfyUI's input folder.
    IMAGE = "image"


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class NodeRef(_Model):
    """`class_type` proves that the node is the one the author of the manifest meant."""

    node: Text
    class_type: Text


class Target(NodeRef):
    input: Text


class NamedInput(_Model):
    type: InputType
    targets: Annotated[tuple[Target, ...], Field(min_length=1)]
    # An input that is not required keeps the value stored in the graph.
    required: bool = False
    min: int | float | None = None
    max: int | float | None = None
    # A switch in the graph that bind() sets to whether the caller gave this input.
    toggle: Target | None = None
    # ComfyUI checks every file, even behind a switch that is off, so an image that is not
    # given has to name some file that exists: the one of this other input.
    fallback: Name | None = None


class Manifest(_Model):
    format: Literal[1]
    name: Text
    version: Annotated[int, Field(ge=1)]
    source: str = ""
    inputs: dict[Name, NamedInput]
    outputs: Annotated[dict[Name, NodeRef], Field(min_length=1)]


@dataclass(frozen=True)
class WorkflowPackage:
    """What is on disk: the two JSON documents as they are, and the parsed manifest."""

    api_json: dict[str, Any]
    manifest_json: dict[str, Any]
    manifest: Manifest

    @property
    def sha256(self) -> str:
        return workflow_sha256(self.api_json, self.manifest_json)


def load_workflow(folder: Path) -> WorkflowPackage:
    api_json = _read(Path(folder) / API_FILE)
    manifest_json = _read(Path(folder) / MANIFEST_FILE)
    return WorkflowPackage(api_json, manifest_json, check_workflow(api_json, manifest_json))


def check_workflow(api_json: Any, manifest: Manifest | Mapping[str, Any]) -> Manifest:
    """The parsed manifest, or an InvalidWorkflowError that lists every problem."""
    parsed = parse_manifest(manifest)
    problems = list(_graph_problems(api_json))
    if not problems:
        problems = list(_manifest_problems(api_json, parsed))
    if problems:
        raise InvalidWorkflowError("; ".join(problems))
    return parsed


def parse_manifest(manifest: Manifest | Mapping[str, Any]) -> Manifest:
    if isinstance(manifest, Manifest):
        return manifest
    try:
        # As JSON, so that strict mode still accepts a list for a tuple.
        return Manifest.model_validate_json(json.dumps(manifest), strict=True)
    except (ValidationError, TypeError, ValueError) as error:
        raise InvalidWorkflowError(f"bad manifest: {error}") from error


def bind(
    api_json: Mapping[str, Any],
    manifest: Manifest | Mapping[str, Any],
    values: Mapping[str, Any],
) -> dict[str, Any]:
    """A new graph with the values filled in. The graph passed in is never changed."""
    parsed = check_workflow(api_json, manifest)
    unknown = sorted(set(values) - set(parsed.inputs))
    if unknown:
        raise BindError(f"unknown inputs: {', '.join(unknown)}")
    missing = sorted(name for name, x in parsed.inputs.items() if x.required and name not in values)
    if missing:
        raise BindError(f"missing inputs: {', '.join(missing)}")
    for name, value in values.items():
        _check_value(name, parsed.inputs[name], value)

    graph = copy.deepcopy(dict(api_json))
    for name, spec in parsed.inputs.items():
        given = name in values
        if spec.toggle is not None:
            _put(graph, spec.toggle, given)
        if given or spec.fallback is not None:
            value = values[name if given else spec.fallback]
            for target in spec.targets:
                _put(graph, target, value)
    return graph


def output_node(manifest: Manifest | Mapping[str, Any], name: str) -> str:
    """The node whose results hold this output."""
    try:
        return parse_manifest(manifest).outputs[name].node
    except KeyError:
        raise BindError(f"unknown output: {name}") from None


def workflow_sha256(api_json: Any, manifest: Any) -> str:
    """The same for equal content, whatever the key order or a trip through JSONB."""
    content = _canonical({"api_json": api_json, "manifest": manifest})
    text = json.dumps(
        content, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )
    return hashlib.sha256(text.encode()).hexdigest()


def _canonical(value: Any) -> Any:
    if isinstance(value, float) and value.is_integer():
        return int(value)  # JSONB may hand 1e16 back as an integer, and drops the sign of -0.0
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise InvalidWorkflowError("JSON keys must be strings")
        return {key: _canonical(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_canonical(item) for item in value]
    return value


def _read(path: Path) -> dict[str, Any]:
    try:
        content = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_no_duplicates,
            parse_constant=_no_constant,
        )
    except (OSError, ValueError) as error:
        raise InvalidWorkflowError(f"cannot read {path.name}: {error}") from error
    if not isinstance(content, dict):
        raise InvalidWorkflowError(f"{path.name} is not a JSON object")
    return content


def _no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError("the same key appears twice in one object")
    return result


def _no_constant(name: str) -> Any:
    raise ValueError(f"{name} is not JSON")


def is_link(value: Any) -> bool:
    """A wire from another node, as opposed to a value."""
    return (
        isinstance(value, list)
        and len(value) == 2
        and isinstance(value[0], str)
        and type(value[1]) is int
    )


def _graph_problems(api_json: Any) -> Iterator[str]:
    if not isinstance(api_json, Mapping) or not api_json:
        yield "the graph is not an API-format object"
        return
    for number, node in api_json.items():
        if (
            not isinstance(node, Mapping)
            or not isinstance(node.get("class_type"), str)
            or not isinstance(node.get("inputs"), Mapping)
        ):
            yield f"node {number} is not in API format"
            continue
        for field, value in node["inputs"].items():
            if isinstance(value, list) and not is_link(value):
                yield f"node {number}: {field} is neither a value nor a link"
            elif is_link(value) and (value[0] not in api_json or value[1] < 0):
                yield f"node {number}: {field} is linked to a missing node {value[0]}"


def _manifest_problems(api_json: Mapping[str, Any], manifest: Manifest) -> Iterator[str]:
    owners: dict[tuple[str, str], str] = {}
    for name, spec in manifest.inputs.items():
        for target in spec.targets:
            yield from _target_problems(api_json, name, target, spec.type)
        if spec.toggle is not None:
            yield from _target_problems(api_json, name, spec.toggle, InputType.BOOL)
        for target in (*spec.targets, *((spec.toggle,) if spec.toggle else ())):
            key = (target.node, target.input)
            if key in owners:
                yield f"{name}: node {target.node} {target.input} already belongs to {owners[key]}"
            owners[key] = name
        if spec.fallback is not None:
            other = manifest.inputs.get(spec.fallback)
            if other is None or not other.required or other.type is not spec.type:
                yield f"{name}: the fallback must be a required input of the same type"
            if spec.required:
                yield f"{name}: a required input needs no fallback"
        if spec.min is not None or spec.max is not None:
            if spec.type not in (InputType.INT, InputType.FLOAT):
                yield f"{name}: only numbers have a min and max"
            elif spec.min is not None and spec.max is not None and spec.min > spec.max:
                yield f"{name}: min is above max"
    for name, ref in manifest.outputs.items():
        yield from _node_problems(api_json, name, ref)


def _node_problems(api_json: Mapping[str, Any], name: str, ref: NodeRef) -> Iterator[str]:
    if ref.node not in api_json:
        yield f"{name}: there is no node {ref.node}"
    elif api_json[ref.node]["class_type"] != ref.class_type:
        yield (
            f"{name}: node {ref.node} is {api_json[ref.node]['class_type']}, not {ref.class_type}"
        )


def _target_problems(
    api_json: Mapping[str, Any], name: str, target: Target, kind: InputType
) -> Iterator[str]:
    problems = list(_node_problems(api_json, name, target))
    yield from problems
    if problems:
        return
    inputs = api_json[target.node]["inputs"]
    if target.input not in inputs:
        yield f"{name}: node {target.node} has no input {target.input}"
    elif is_link(inputs[target.input]):
        yield f"{name}: {target.input} of node {target.node} comes from another node"
    elif not _has_type(inputs[target.input], kind):
        yield f"{name}: {target.input} of node {target.node} does not hold a {kind.value}"


def _has_type(value: Any, kind: InputType) -> bool:
    match kind:
        case InputType.BOOL:
            return isinstance(value, bool)
        case InputType.INT:
            return isinstance(value, int) and not isinstance(value, bool)
        case InputType.FLOAT:
            return isinstance(value, int | float) and not isinstance(value, bool)
        case InputType.STRING | InputType.IMAGE:
            return isinstance(value, str)


def _check_value(name: str, spec: NamedInput, value: Any) -> None:
    if not _has_type(value, spec.type):
        raise BindError(f"{name} must be a {spec.type.value}, not {type(value).__name__}")
    if spec.type is InputType.IMAGE and not value:
        raise BindError(f"{name} must name a file")
    if isinstance(value, float) and not -float("inf") < value < float("inf"):
        raise BindError(f"{name} must be a finite number")
    if spec.min is not None and value < spec.min:
        raise BindError(f"{name} must be at least {spec.min}")
    if spec.max is not None and value > spec.max:
        raise BindError(f"{name} must be at most {spec.max}")


def _put(graph: dict[str, Any], target: Target, value: Any) -> None:
    graph[target.node]["inputs"][target.input] = value
