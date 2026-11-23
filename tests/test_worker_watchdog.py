import subprocess

import pytest

from memcraft.worker import watchdog
from memcraft.worker.watchdog import GpuReading, Service, Watchdog, read_gpu


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


class Fake:
    def __init__(self, answers=True):
        self.answers = answers
        self.restarts = 0
        self.stops = 0

    def probe(self):
        return self.answers

    def restart(self):
        self.restarts += 1

    def stop(self):
        self.stops += 1


def dog(fake, clock, gpu=None, **kw):
    service = Service("comfyui", 8199, fake.probe, fake.restart)
    return Watchdog([service], gpu=lambda: gpu, clock=clock, stall_limit=100, answer_limit=30, **kw)


@pytest.mark.parametrize("port", [8188, 8080])
def test_never_watches_the_owners_servers(port):
    with pytest.raises(ValueError, match="not the app's own"):
        Service("x", port, lambda: True, lambda: None)


def test_nothing_happens_without_a_step():
    fake, clock = Fake(answers=False), Clock()
    watcher = dog(fake, clock)
    clock.now += 1000
    assert watcher.check() == []
    assert fake.restarts == 0


def test_a_service_that_stops_answering_is_restarted():
    fake, clock = Fake(), Clock()
    watcher = dog(fake, clock)
    watch = watcher.watch("comfyui", fake.stop)
    assert watcher.check() == []
    fake.answers = False
    clock.now += 30
    assert watcher.check() == []  # exactly at the limit is still fine
    clock.now += 1
    assert watcher.check() == ["comfyui"]
    assert (fake.restarts, fake.stops) == (1, 1)
    assert watch.hung.startswith("comfyui did not answer")
    clock.now += 1000
    assert watcher.check() == []  # once per step
    watch.close()


def test_no_progress_while_the_gpu_idles():
    fake, clock = Fake(), Clock()
    watcher = dog(fake, clock, gpu=GpuReading(0, 1000, 32000))
    watch = watcher.watch("comfyui", fake.stop)
    clock.now += 90
    watch.progress(0.1)
    clock.now += 90
    assert watcher.check() == []  # progress 90 s ago
    clock.now += 11
    assert watcher.check() == ["comfyui"]
    assert watch.hung.startswith("comfyui made no progress")


def test_the_same_progress_again_is_no_progress():
    fake, clock = Fake(), Clock()
    watcher = dog(fake, clock, gpu=None)  # no nvidia-smi: counts as idle
    watch = watcher.watch("comfyui", fake.stop)
    watch.progress(0.5)
    for _ in range(3):
        clock.now += 40
        watch.progress(0.5)
    assert watcher.check() == ["comfyui"]


def test_a_busy_gpu_gets_three_times_as_long():
    fake, clock = Fake(), Clock()
    watcher = dog(fake, clock, gpu=GpuReading(97, 30000, 32000))
    watch = watcher.watch("comfyui", fake.stop)
    clock.now += 300
    assert watcher.check() == []
    assert watcher.last_gpu.utilization == 97
    clock.now += 1
    assert watcher.check() == ["comfyui"]
    watch.close()


def test_a_finished_step_is_not_watched():
    fake, clock = Fake(answers=False), Clock()
    watcher = dog(fake, clock)
    watch = watcher.watch("comfyui", fake.stop)
    watch.close()
    clock.now += 1000
    assert watcher.check() == []
    assert watch.hung is None


def test_unknown_service_and_failing_callbacks():
    fake, clock = Fake(answers=False), Clock()
    watcher = dog(fake, clock)
    assert watcher.watch(None, fake.stop).hung is None
    assert watcher.watch("gemma", fake.stop).hung is None  # not watched here

    def boom():
        raise RuntimeError("boom")

    watch = watcher.watch("comfyui", boom)
    watcher._services["comfyui"].restart = boom
    watcher._services["comfyui"].probe = boom
    clock.now += 31
    assert watcher.check() == ["comfyui"]
    assert watch.hung


def test_read_gpu(monkeypatch):
    monkeypatch.setattr(watchdog.shutil, "which", lambda _name: "nvidia-smi")

    def run(args, **_kw):
        return subprocess.CompletedProcess(args, 0, stdout="42, 1234, 32607\n7, 1, 2\n")

    assert read_gpu(run) == GpuReading(42, 1234, 32607)

    def fails(args, **_kw):
        raise subprocess.CalledProcessError(9, args)

    assert read_gpu(fails) is None
    assert read_gpu(lambda args, **_kw: subprocess.CompletedProcess(args, 0, "junk")) is None
    monkeypatch.setattr(watchdog.shutil, "which", lambda _name: None)
    assert read_gpu(run) is None


def test_service_builders_refuse_other_hosts_and_ports():
    with pytest.raises(ValueError):
        watchdog.comfy_service(lambda: None, "http://127.0.0.1:8188")
    with pytest.raises(ValueError):
        watchdog.llama_server_service(lambda: None, "http://127.0.0.1:8080")
    with pytest.raises(ValueError):
        watchdog.llama_server_service(lambda: None, "http://example.com:8090")
    assert watchdog.comfy_service(lambda: None).port == 8199
