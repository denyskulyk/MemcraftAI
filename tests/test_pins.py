import hashlib
import os

import pytest

from memcraft.pins import (
    HashCache,
    Install,
    check_installation,
    collect,
    compare,
    git_commit,
    model_dirs,
    models_of,
    write_pins,
)
from memcraft.workflows import WorkflowPackage, parse_manifest

GRAPH = {
    "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "big.safetensors"}},
    "2": {"class_type": "VAELoader", "inputs": {"vae_name": "vae.safetensors"}},
    "3": {"class_type": "SaveImage", "inputs": {"images": ["1", 0]}},
}
MANIFEST = {
    "format": 1,
    "name": "demo",
    "version": 1,
    "inputs": {},
    "outputs": {"image": {"node": "3", "class_type": "SaveImage"}},
}


@pytest.fixture
def packages():
    return [WorkflowPackage(GRAPH, MANIFEST, parse_manifest(MANIFEST))]


@pytest.fixture
def install(tmp_path):
    comfy = tmp_path / "ComfyUI"
    (comfy / "custom_nodes" / "node-a").mkdir(parents=True)
    (comfy / "custom_nodes" / "node-a" / "__init__.py").write_text("A = 1\n")
    (comfy / "custom_nodes" / "node-a" / "pyproject.toml").write_text('version = "1.2"\n')
    (comfy / "custom_nodes" / "__pycache__").mkdir()
    (comfy / "comfyui_version.py").write_text('__version__ = "0.46.2"\n')
    git = comfy / ".git"
    (git / "refs" / "heads").mkdir(parents=True)
    (git / "HEAD").write_text("ref: refs/heads/main\n")
    (git / "refs" / "heads" / "main").write_text("abc123\n")
    shared = tmp_path / "shared"
    (shared / "diffusion_models").mkdir(parents=True)
    (shared / "diffusion_models" / "big.safetensors").write_bytes(b"weights")
    (comfy / "models" / "vae").mkdir(parents=True)
    (comfy / "models" / "vae" / "vae.safetensors").write_bytes(b"vae")
    return Install(
        comfy, {"diffusion_models": (shared / "diffusion_models",), "": (comfy / "models",)}
    )


@pytest.fixture
def cache(tmp_path):
    return HashCache(tmp_path / "cache.json")


def test_model_dirs_reads_comfy_desktop_yaml(tmp_path):
    yaml = tmp_path / "paths.yaml"
    yaml.write_text(
        "# comment\n"
        "comfy.desktop_0:\n"
        "  base_path: 'D:\\models'\n"
        "  is_default: true\n"
        "  'vae': 'vae/'\n"
        "  'controlnet': |-\n"
        "    controlnet/\n"
        "    t2i_adapter/\n"
        "  'loras': 'loras/'\n",
        encoding="utf-8",
    )
    dirs = model_dirs(yaml)
    assert [p.name for p in dirs["controlnet"]] == ["controlnet", "t2i_adapter"]
    assert [p.name for p in dirs["vae"]] == ["vae"]
    assert "loras" in dirs and "is_default" not in dirs


def test_models_of_the_graphs(packages):
    assert models_of(packages) == {
        "big.safetensors": ("diffusion_models", "unet"),
        "vae.safetensors": ("vae",),
    }


def test_collect(install, packages, cache):
    found = collect(install, packages, cache)
    assert found["comfyui"] == {"version": "0.46.2", "git": "abc123"}
    assert set(found["custom_nodes"]) == {"node-a"}
    assert found["custom_nodes"]["node-a"]["version"] == "1.2"
    model = found["models"]["big.safetensors"]
    assert model == {
        "folder": "diffusion_models",
        "size": 7,
        "sha256": hashlib.sha256(b"weights").hexdigest(),
    }
    assert found["models"]["vae.safetensors"]["size"] == 3


def test_the_pins_hold_no_path(install, packages, cache, tmp_path):
    pins = write_pins(tmp_path / "pins.json", install, packages, cache)
    text = (tmp_path / "pins.json").read_text()
    assert str(tmp_path) not in text and str(tmp_path).replace("\\", "\\\\") not in text
    assert pins["models"]["big.safetensors"]["sha256"]


def test_check_reports_each_change(install, packages, cache, tmp_path):
    pins = tmp_path / "pins.json"
    write_pins(pins, install, packages, cache)
    assert check_installation(pins, install, packages, cache) == []
    big = install.model_dirs["diffusion_models"][0] / "big.safetensors"
    big.write_bytes(b"other weights")
    (install.comfy_dir / "custom_nodes" / "node-a" / "__init__.py").write_text("A = 2\n")
    (install.comfy_dir / "custom_nodes" / "node-b").mkdir()
    (install.comfy_dir / "models" / "vae" / "vae.safetensors").unlink()
    problems = check_installation(pins, install, packages, cache)
    assert "model big.safetensors differs from the pinned file" in problems
    assert "model vae.safetensors is missing from vae" in problems
    assert "custom node node-b is not pinned" in problems
    assert any(p.startswith("custom node node-a: files_sha256") for p in problems)


def test_write_refuses_missing_models(install, packages, cache, tmp_path):
    (install.comfy_dir / "models" / "vae" / "vae.safetensors").unlink()
    with pytest.raises(FileNotFoundError, match=r"vae.safetensors"):
        write_pins(tmp_path / "pins.json", install, packages, cache)


def test_unreadable_pins(install, packages, cache, tmp_path):
    assert check_installation(tmp_path / "none.json", install, packages, cache)[0].startswith(
        "cannot read the pins"
    )


def test_compare_versions():
    pins = {"comfyui": {"version": "1", "git": "a"}, "custom_nodes": {}, "models": {}}
    actual = {"comfyui": {"version": "2", "git": "a"}, "custom_nodes": {"x": {}}, "models": {}}
    assert compare(pins, actual) == [
        "ComfyUI version is 2, pinned 1",
        "custom node x is not pinned",
    ]


def test_hash_cache_skips_unchanged_files(tmp_path, monkeypatch):
    file = tmp_path / "m.bin"
    file.write_bytes(b"abc")
    cache = HashCache(tmp_path / "cache.json")
    first = cache.sha256(file)
    cache.save()
    reloaded = HashCache(tmp_path / "cache.json")
    monkeypatch.setattr(hashlib, "sha256", lambda: pytest.fail("hashed again"))
    assert reloaded.sha256(file) == first
    monkeypatch.undo()
    file.write_bytes(b"abcd")
    os.utime(file, ns=(1, 1))
    assert reloaded.sha256(file) != first


def test_git_commit_from_packed_refs(tmp_path):
    git = tmp_path / ".git"
    git.mkdir()
    (git / "HEAD").write_text("ref: refs/heads/main\n")
    (git / "packed-refs").write_text("# pack-refs\ndef456 refs/heads/main\n")
    assert git_commit(tmp_path) == "def456"
    (git / "HEAD").write_text("0123abcd\n")
    assert git_commit(tmp_path) == "0123abcd"
    assert git_commit(tmp_path / "nothing") is None
