"""Pinned versions of what the app's ComfyUI runs: ComfyUI, its custom nodes and the model files.

deploy/pins.json holds them; ``python deploy/pins.py`` writes it again from this PC. The worker
calls check_installation() when it starts and refuses or warns on what it reports.

The pins hold names, sizes and hashes only, never a path on this PC. Hashing a 20 GB model takes
minutes, so hashes are cached in data/pins-cache.json by path, size and modification time.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from memcraft.workflows import WorkflowPackage, load_workflow

PROJECT = Path(__file__).resolve().parents[2]
PINS_FILE = PROJECT / "deploy" / "pins.json"
CACHE_FILE = PROJECT / "data" / "pins-cache.json"
# The two workflows the worker runs; their model files are pinned.
PINNED_WORKFLOWS = ("keyframe-qwen-image-2.1-v2", "clip-ltx-2.5-v1")

# Loader node and input -> the model folders ComfyUI searches for that file.
LOADERS: dict[tuple[str, str], tuple[str, ...]] = {
    ("UNETLoader", "unet_name"): ("diffusion_models", "unet"),
    ("CLIPLoader", "clip_name"): ("text_encoders", "clip"),
    ("VAELoader", "vae_name"): ("vae",),
    ("LatentUpscaleModelLoader", "model_name"): ("latent_upscale_models",),
    ("CheckpointLoaderSimple", "ckpt_name"): ("checkpoints",),
    ("LoraLoader", "lora_name"): ("loras",),
    ("LoraLoaderModelOnly", "lora_name"): ("loras",),
    ("UpscaleModelLoader", "model_name"): ("upscale_models",),
    ("LoadBackgroundRemovalModel", "bg_removal_name"): ("background_removal",),
}
_VERSION = re.compile(r'^\s*version\s*=\s*["\']([^"\']+)["\']', re.MULTILINE)


@dataclass(frozen=True)
class Install:
    """Where ComfyUI and its models are on this PC."""

    comfy_dir: Path  # the folder with main.py
    model_dirs: dict[str, tuple[Path, ...]]

    @classmethod
    def default(cls) -> Install:
        local = Path(os.environ.get("LOCALAPPDATA", ""))
        roaming = Path(os.environ.get("APPDATA", ""))
        comfy = local / "Comfy-Desktop" / "ComfyUI-Installs" / "ComfyUI" / "ComfyUI"
        yamls = sorted((roaming / "Comfy Desktop" / "instance-model-paths").glob("*.yaml"))
        dirs = model_dirs(yamls[0]) if len(yamls) == 1 else {}
        return cls(comfy, merge_dirs(dirs, {"": (comfy / "models",)}))


def model_dirs(yaml_file: Path) -> dict[str, tuple[Path, ...]]:
    """Folders per model kind from an extra_model_paths YAML file.

    Only the plain shape Comfy Desktop writes: sections with base_path and kind: 'sub/' or a
    block of several lines after '|-'.
    """
    found: dict[str, list[Path]] = {}
    base: Path | None = None
    block: str | None = None
    for raw in Path(yaml_file).read_text(encoding="utf-8").splitlines():
        line = raw.split(" #")[0].rstrip()
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip())
        if indent == 0:
            base, block = None, None
            continue
        if block is not None and indent > 2:
            if base is not None:
                found.setdefault(block, []).append(base / _unquote(line.strip()))
            continue
        block = None
        key, _, value = line.strip().partition(":")
        key, value = _unquote(key), _unquote(value.strip())
        if key == "base_path":
            base = Path(value)
        elif key == "is_default":
            continue
        elif value == "|-" or value == "|":
            block = key
        elif value and base is not None:
            found.setdefault(key, []).append(base / value)
    return {kind: tuple(paths) for kind, paths in found.items()}


def merge_dirs(*parts: Mapping[str, tuple[Path, ...]]) -> dict[str, tuple[Path, ...]]:
    merged: dict[str, tuple[Path, ...]] = {}
    for part in parts:
        for kind, paths in part.items():
            merged[kind] = merged.get(kind, ()) + tuple(paths)
    return merged


def _unquote(text: str) -> str:
    text = text.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "'\"":
        return text[1:-1]
    return text


def models_of(packages: Iterable[WorkflowPackage]) -> dict[str, tuple[str, ...]]:
    """Model file name -> the folders it may be in, for every loader in the graphs."""
    wanted: dict[str, tuple[str, ...]] = {}
    for package in packages:
        for node in package.api_json.values():
            for field, value in node["inputs"].items():
                kinds = LOADERS.get((node["class_type"], field))
                if kinds and isinstance(value, str) and value != "None":
                    wanted[value] = kinds
    return dict(sorted(wanted.items()))


def find_model(install: Install, name: str, kinds: tuple[str, ...]) -> Path | None:
    for kind in kinds:
        bases = [*install.model_dirs.get(kind, ())]
        bases += [root / kind for root in install.model_dirs.get("", ())]
        for base in bases:
            path = base / name
            if path.is_file():
                return path
    return None


class HashCache:
    """sha256 of large files, kept while path, size and modification time stay the same."""

    def __init__(self, path: Path = CACHE_FILE) -> None:
        self.path = Path(path)
        try:
            self.entries: dict[str, Any] = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self.entries = {}
        self.changed = False

    def sha256(self, file: Path) -> str:
        stat = file.stat()
        key = str(file.resolve())
        entry = self.entries.get(key)
        if entry and entry["size"] == stat.st_size and entry["mtime_ns"] == stat.st_mtime_ns:
            return entry["sha256"]
        digest = hashlib.sha256()
        with file.open("rb") as handle:
            while chunk := handle.read(16 * 1024 * 1024):
                digest.update(chunk)
        value = digest.hexdigest()
        self.entries[key] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": value}
        self.changed = True
        return value

    def save(self) -> None:
        if not self.changed:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        part = self.path.with_suffix(".part")
        part.write_text(json.dumps(self.entries, indent=1, sort_keys=True), encoding="utf-8")
        part.replace(self.path)
        self.changed = False


def comfy_version(comfy_dir: Path) -> str | None:
    try:
        text = (comfy_dir / "comfyui_version.py").read_text(encoding="utf-8")
    except OSError:
        return None
    match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', text)
    return match.group(1) if match else None


def git_commit(folder: Path) -> str | None:
    """The commit checked out in a git folder, read from .git without running git."""
    git = folder / ".git"
    if git.is_file():  # a worktree or submodule: "gitdir: <path>"
        target = git.read_text(encoding="utf-8").partition("gitdir:")[2].strip()
        git = (folder / target).resolve()
    try:
        head = (git / "HEAD").read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not head.startswith("ref:"):
        return head or None
    ref = head.removeprefix("ref:").strip()
    try:
        return (git / ref).read_text(encoding="utf-8").strip()
    except OSError:
        pass
    try:
        for line in (git / "packed-refs").read_text(encoding="utf-8").splitlines():
            if line.endswith(" " + ref):
                return line.split(" ", 1)[0]
    except OSError:
        pass
    return None


def files_sha256(folder: Path) -> str:
    """One hash over the Python files and pyproject.toml of a folder, names included.

    Settings files are left out: some custom nodes write them while they run.
    """
    digest = hashlib.sha256()
    files = sorted(
        path
        for path in folder.rglob("*")
        if path.is_file()
        and (path.suffix == ".py" or path.name == "pyproject.toml")
        and ".git" not in path.parts
        and "__pycache__" not in path.parts
        and "node_modules" not in path.parts
    )
    for path in files:
        digest.update(path.relative_to(folder).as_posix().encode() + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def custom_nodes(comfy_dir: Path) -> dict[str, dict[str, Any]]:
    nodes = {}
    root = comfy_dir / "custom_nodes"
    for folder in sorted(root.iterdir() if root.is_dir() else []):
        if not folder.is_dir() or folder.name.startswith((".", "__")):
            continue
        if folder.name.endswith(".disabled"):
            continue
        version = None
        try:
            match = _VERSION.search((folder / "pyproject.toml").read_text(encoding="utf-8"))
            version = match.group(1) if match else None
        except OSError:
            pass
        nodes[folder.name] = {
            "version": version,
            "git": git_commit(folder),
            "files_sha256": files_sha256(folder),
        }
    return nodes


def collect(
    install: Install,
    packages: Iterable[WorkflowPackage],
    cache: HashCache | None = None,
) -> dict[str, Any]:
    """What is installed now, in the shape of pins.json."""
    cache = cache or HashCache()
    models = {}
    for name, kinds in models_of(packages).items():
        path = find_model(install, name, kinds)
        if path is None:
            models[name] = {"folder": kinds[0], "size": None, "sha256": None}
            continue
        models[name] = {
            "folder": kinds[0],
            "size": path.stat().st_size,
            "sha256": cache.sha256(path),
        }
    cache.save()
    return {
        "format": 1,
        "comfyui": {
            "version": comfy_version(install.comfy_dir),
            "git": git_commit(install.comfy_dir),
        },
        "custom_nodes": custom_nodes(install.comfy_dir),
        "models": models,
    }


def compare(pins: Mapping[str, Any], actual: Mapping[str, Any]) -> list[str]:
    """Every difference, one line each. Empty means the installation matches the pins."""
    problems = []
    for field in ("version", "git"):
        want = pins.get("comfyui", {}).get(field)
        have = actual.get("comfyui", {}).get(field)
        if want != have:
            problems.append(f"ComfyUI {field} is {have}, pinned {want}")
    pinned_nodes = pins.get("custom_nodes", {})
    actual_nodes = actual.get("custom_nodes", {})
    for name in sorted(set(pinned_nodes) | set(actual_nodes)):
        if name not in actual_nodes:
            problems.append(f"custom node {name} is missing")
        elif name not in pinned_nodes:
            problems.append(f"custom node {name} is not pinned")
        else:
            for field in ("version", "git", "files_sha256"):
                want, have = pinned_nodes[name].get(field), actual_nodes[name].get(field)
                if want != have:
                    problems.append(f"custom node {name}: {field} is {have}, pinned {want}")
    pinned_models = pins.get("models", {})
    actual_models = actual.get("models", {})
    for name in sorted(set(pinned_models) | set(actual_models)):
        want, have = pinned_models.get(name), actual_models.get(name)
        if want is None:
            problems.append(f"model {name} is used but not pinned")
        elif have is None:
            problems.append(f"model {name} is pinned but no workflow uses it")
        elif have["sha256"] is None:
            problems.append(f"model {name} is missing from {want['folder']}")
        elif (have["size"], have["sha256"]) != (want["size"], want["sha256"]):
            problems.append(f"model {name} differs from the pinned file")
    return problems


def pinned_packages(workflows_dir: Path | None = None) -> list[WorkflowPackage]:
    root = workflows_dir or PROJECT / "workflows"
    return [load_workflow(root / name) for name in PINNED_WORKFLOWS]


def write_pins(
    path: Path = PINS_FILE,
    install: Install | None = None,
    packages: Iterable[WorkflowPackage] | None = None,
    cache: HashCache | None = None,
) -> dict[str, Any]:
    actual = collect(install or Install.default(), packages or pinned_packages(), cache)
    missing = [name for name, model in actual["models"].items() if model["sha256"] is None]
    if missing:
        raise FileNotFoundError(f"models not found: {', '.join(missing)}")
    Path(path).write_text(json.dumps(actual, indent=2) + "\n", encoding="utf-8")
    return actual


def check_installation(
    path: Path = PINS_FILE,
    install: Install | None = None,
    packages: Iterable[WorkflowPackage] | None = None,
    cache: HashCache | None = None,
) -> list[str]:
    """The worker's start check: one line per mismatch, empty when everything matches.

    The first run after a model changes hashes it again, which can take minutes.
    """
    try:
        pins = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        return [f"cannot read the pins: {error}"]
    actual = collect(install or Install.default(), packages or pinned_packages(), cache)
    return compare(pins, actual)
