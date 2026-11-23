import copy
import json
import re
from pathlib import Path

import pytest

from memcraft.workflows import (
    BindError,
    InvalidWorkflowError,
    bind,
    check_workflow,
    is_link,
    load_workflow,
    output_node,
    parse_manifest,
    workflow_sha256,
)

WORKFLOWS = Path(__file__).parents[1] / "workflows"
FOLDERS = sorted(path for path in WORKFLOWS.iterdir() if path.is_dir())
REQUIRED = {
    "video-ltx2_5-v8": {"first_frame": "a.png", "positive_prompt": "a cat", "seed": 7},
    "qwen2.1-with-lora-v7": {"image_1": "a.png", "positive_prompt": "a cat", "seed": 7},
}


def api():
    return {
        "1": {"class_type": "LoadImage", "inputs": {"image": "placeholder.png"}},
        "2": {"class_type": "LoadImage", "inputs": {"image": "placeholder.png"}},
        "3": {"class_type": "PrimitiveBoolean", "inputs": {"value": True}},
        "4": {"class_type": "Sampler", "inputs": {"seed": 0, "cfg": 1.5, "image": ["1", 0]}},
        "5": {"class_type": "SaveImage", "inputs": {"images": ["4", 0], "extra": ["2", 0]}},
    }


def target(node, class_type, field):
    return {"node": node, "class_type": class_type, "input": field}


def manifest():
    return {
        "format": 1,
        "name": "demo",
        "version": 1,
        "inputs": {
            "first": {
                "type": "image",
                "required": True,
                "targets": [target("1", "LoadImage", "image")],
            },
            "last": {
                "type": "image",
                "targets": [target("2", "LoadImage", "image")],
                "toggle": target("3", "PrimitiveBoolean", "value"),
                "fallback": "first",
            },
            "seed": {
                "type": "int",
                "min": 0,
                "max": 9,
                "targets": [target("4", "Sampler", "seed")],
            },
            "cfg": {"type": "float", "targets": [target("4", "Sampler", "cfg")]},
        },
        "outputs": {"picture": {"node": "5", "class_type": "SaveImage"}},
    }


def test_bind_fills_a_new_graph():
    graph, names = api(), manifest()
    filled = bind(graph, names, {"first": "a.png", "last": "b.png", "seed": 5, "cfg": 3})
    assert graph == api() and names == manifest()
    assert filled["1"]["inputs"]["image"] == "a.png"
    assert filled["2"]["inputs"]["image"] == "b.png"
    assert filled["3"]["inputs"]["value"] is True
    assert filled["4"]["inputs"] == {"seed": 5, "cfg": 3, "image": ["1", 0]}
    filled["5"]["inputs"]["images"].append("shared?")
    assert graph == api()


def test_bind_without_the_optional_inputs():
    filled = bind(api(), manifest(), {"first": "a.png"})
    assert filled["2"]["inputs"]["image"] == "a.png"  # a file that exists, never the placeholder
    assert filled["3"]["inputs"]["value"] is False
    assert filled["4"]["inputs"]["seed"] == 0


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ({}, "missing inputs: first"),
        ({"first": "a.png", "frist": "b.png"}, "unknown inputs: frist"),
        ({"first": ""}, "first must name a file"),
        ({"first": 1}, "first must be a image, not int"),
        ({"first": "a.png", "seed": True}, "seed must be a int, not bool"),
        ({"first": "a.png", "seed": 1.0}, "seed must be a int, not float"),
        ({"first": "a.png", "seed": -1}, "at least 0"),
        ({"first": "a.png", "seed": 10}, "at most 9"),
        ({"first": "a.png", "cfg": float("nan")}, "finite"),
        ({"first": "a.png", "cfg": ["1", 0]}, "cfg must be a float, not list"),
        ({"first": "a.png", "last": None}, "last must be a image, not NoneType"),
    ],
)
def test_bind_refuses(values, message):
    with pytest.raises(BindError, match=message):
        bind(api(), manifest(), values)


def _change(path, value):
    def change(graph, names):
        place = names if path[0] != "api" else graph
        *parents, last = path if path[0] != "api" else path[1:]
        for key in parents:
            place = place[key]
        place[last] = value

    return change


SEED = ("inputs", "seed", "targets", 0)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (_change((*SEED, "node"), "99"), "seed: there is no node 99"),
        (_change((*SEED, "input"), "noise"), "seed: node 4 has no input noise"),
        (_change((*SEED, "class_type"), "KSampler"), "seed: node 4 is Sampler, not KSampler"),
        (_change((*SEED, "input"), "image"), "seed: image of node 4 comes from another node"),
        (_change((*SEED, "input"), "cfg"), "seed: cfg of node 4 does not hold a int"),
        (_change(("inputs", "seed", "targets"), []), "bad manifest"),
        (_change(("inputs", "seed", "type"), "number"), "bad manifest"),
        (_change(("inputs", "seed", "min"), "0"), "bad manifest"),
        (_change(("inputs", "seed", "min"), 10), "min is above max"),
        (_change(("inputs", "first", "min"), 1), "only numbers"),
        (_change(("inputs", "Bad Name"), {}), "bad manifest"),
        (_change(("inputs", "last", "fallback"), "seed"), "fallback must be a required input"),
        (_change(("inputs", "last", "fallback"), "gone"), "fallback must be a required input"),
        (_change(("inputs", "last", "required"), True), "needs no fallback"),
        (
            _change(("inputs", "last", "toggle"), target("4", "Sampler", "cfg")),
            "last: cfg of node 4 does not hold a bool",
        ),
        (
            _change(("inputs", "last", "targets"), [target("1", "LoadImage", "image")]),
            "node 1 image already belongs to first",
        ),
        (_change(("outputs", "picture", "node"), "6"), "picture: there is no node 6"),
        (_change(("outputs",), {}), "bad manifest"),
        (_change(("format",), 2), "bad manifest"),
        (_change(("version",), 0), "bad manifest"),
        (_change(("typo",), 1), "bad manifest"),
        (_change(("api", "4", "inputs", "image"), ["77", 0]), "linked to a missing node 77"),
        (_change(("api", "4", "inputs", "image"), [1, 2, 3]), "neither a value nor a link"),
        (_change(("api", "nodes"), []), "node nodes is not in API format"),
    ],
)
def test_a_bad_workflow_is_refused(change, message):
    graph, names = api(), manifest()
    check_workflow(graph, names)
    change(graph, names)
    with pytest.raises(InvalidWorkflowError, match=message):
        check_workflow(graph, names)
    with pytest.raises(InvalidWorkflowError):
        bind(graph, names, {"first": "a.png"})


def test_every_problem_is_listed():
    names = manifest()
    names["inputs"]["seed"]["targets"][0]["node"] = "98"
    names["outputs"]["picture"]["node"] = "99"
    with pytest.raises(InvalidWorkflowError, match=r"no node 98; .*no node 99"):
        check_workflow(api(), names)


def test_output_node():
    assert output_node(manifest(), "picture") == "5"
    with pytest.raises(BindError, match="unknown output: video"):
        output_node(manifest(), "video")


def test_sha256_is_canonical():
    expected = workflow_sha256(api(), manifest())
    assert re.fullmatch("[0-9a-f]{64}", expected)
    reordered = dict(reversed(api().items()))
    assert workflow_sha256(reordered, manifest()) == expected
    assert workflow_sha256(json.loads(json.dumps(api(), indent=4)), manifest()) == expected
    # What JSONB does to numbers.
    assert workflow_sha256({"a": [1e16, -0.0, 3.0]}, {}) == workflow_sha256(
        {"a": [10**16, 0, 3]}, {}
    )
    # But true is not 1, and text is not a number.
    assert workflow_sha256({"a": True}, {}) != workflow_sha256({"a": 1}, {})
    assert workflow_sha256({"a": "1"}, {}) != workflow_sha256({"a": 1}, {})


def test_sha256_covers_the_graph_and_the_manifest():
    expected = workflow_sha256(api(), manifest())
    graph = api()
    graph["4"]["inputs"]["seed"] = 1
    names = manifest()
    names["inputs"]["seed"]["max"] = 10
    assert len({expected, workflow_sha256(graph, manifest()), workflow_sha256(api(), names)}) == 3


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ('{"1": {}, "1": {}}', "appears twice"),
        ('{"1": NaN}', "NaN is not JSON"),
        ("[]", "not a JSON object"),
        ("{", "cannot read api.json"),
        (None, "cannot read api.json"),
    ],
)
def test_load_refuses(tmp_path, text, message):
    (tmp_path / "manifest.json").write_text(json.dumps(manifest()), encoding="utf-8")
    if text is not None:
        (tmp_path / "api.json").write_text(text, encoding="utf-8")
    with pytest.raises(InvalidWorkflowError, match=message):
        load_workflow(tmp_path)


def test_load(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps(manifest()), encoding="utf-8")
    (tmp_path / "api.json").write_text(json.dumps(api()), encoding="utf-8")
    package = load_workflow(tmp_path)
    assert package.api_json == api() and package.manifest_json == manifest()
    assert package.manifest == parse_manifest(manifest())
    assert package.sha256 == workflow_sha256(api(), manifest())


def test_both_exports_are_here():
    assert {folder.name for folder in FOLDERS} == set(REQUIRED)


@pytest.mark.parametrize("folder", FOLDERS, ids=lambda folder: folder.name)
def test_exported_workflow(folder):
    package = load_workflow(folder)
    before = copy.deepcopy(package.api_json)
    required = {name for name, x in package.manifest.inputs.items() if x.required}
    assert required == set(REQUIRED[folder.name])
    assert set(package.manifest.outputs) == {"preview", "final", "upscaled"}
    filled = bind(package.api_json, package.manifest, REQUIRED[folder.name])
    assert package.api_json == before
    # Every photo is replaced, and the prompt reaches the switch that chooses it.
    loaded = [n["inputs"]["image"] for n in filled.values() if n["class_type"] == "LoadImage"]
    assert loaded and set(loaded) == {"a.png"}
    prompt = package.manifest.inputs["positive_prompt"]
    assert filled[prompt.targets[0].node]["inputs"]["value"] == "a cat"
    assert filled[prompt.toggle.node]["inputs"]["value"] is True
    switches = [
        n["inputs"]
        for n in filled.values()
        if n["class_type"] == "ComfySwitchNode" and n["inputs"]["switch"] == [prompt.toggle.node, 0]
    ]
    assert [prompt.targets[0].node, 0] in [inputs.get("on_true") for inputs in switches]


@pytest.mark.parametrize("folder", FOLDERS, ids=lambda folder: folder.name)
def test_exported_workflow_holds_nothing_private(folder):
    package = load_workflow(folder)
    for number, node in package.api_json.items():
        assert set(node) == {"class_type", "inputs"}, number
        for field, value in node["inputs"].items():
            if not isinstance(value, str):
                assert is_link(value) or isinstance(value, bool | int | float)
                continue
            if node["class_type"] == "LoadImage":
                assert value == "placeholder.png"
            elif node["class_type"] == "PrimitiveStringMultiline":
                assert value in ("", "placeholder")
            elif node["class_type"] == "CLIPTextEncode":
                assert value == "watermark, text"
            elif field.startswith("lora_"):
                assert value == "None"
            assert not re.search(r"[A-Za-z]:[\\/]|\\Users|/home/", value), (number, field)
            assert not re.search(
                r"\.(png|jpe?g|webp|mp4|webm|mov)$", value.removeprefix("placeholder.png")
            )


def test_the_video_ends_on_a_photo_only_when_asked():
    package = load_workflow(WORKFLOWS / "video-ltx2_5-v8")
    values = REQUIRED["video-ltx2_5-v8"]
    toggle = package.manifest.inputs["last_frame"].toggle
    image = package.manifest.inputs["last_frame"].targets[0]
    off = bind(package.api_json, package.manifest, values)
    on = bind(package.api_json, package.manifest, {**values, "last_frame": "b.png"})
    assert off[toggle.node]["inputs"]["value"] is False
    assert on[toggle.node]["inputs"]["value"] is True
    assert on[image.node]["inputs"]["image"] == "b.png"
