import json
import uuid

import httpx
import pytest

from memcraft.comfy import (
    CancelledRunError,
    ComfyClient,
    ComfyError,
    ComfyTimeoutError,
    ComfyUnavailableError,
    ExecutionFailedError,
    GraphRejectedError,
    OutputFile,
    check_url,
)

GRAPH = {
    "1": {"class_type": "LoadImage", "inputs": {"image": "a.png"}},
    "2": {"class_type": "SaveImage", "inputs": {"images": ["1", 0], "filename_prefix": "x"}},
    "3": {"class_type": "SaveImage", "inputs": {"images": ["1", 0], "filename_prefix": "y"}},
}


class FakeComfy:
    """Enough of ComfyUI's HTTP API: a prompt runs for `polls` looks at /queue, then ends."""

    def __init__(self, polls=2, outcome="success"):
        self.polls = polls
        self.outcome = outcome
        self.prompts = {}
        self.seen = {}
        self.requests = []
        self.uploads = {}
        self.freed = []
        self.interrupts = []
        self.deleted = []
        self.reject = None
        self.down = False
        self.pending_ahead = 0

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append((request.method, request.url.path))
        if self.down:
            raise httpx.ConnectError("refused", request=request)
        path, method = request.url.path, request.method
        if path == "/system_stats":
            return httpx.Response(200, json={"system": {"comfyui_version": "0.46.2"}})
        if path == "/prompt" and method == "POST":
            body = json.loads(request.content)
            if self.reject:
                return httpx.Response(400, json=self.reject)
            self.prompts[body["prompt_id"]] = body
            self.seen[body["prompt_id"]] = 0
            return httpx.Response(200, json={"prompt_id": body["prompt_id"], "number": 1})
        if path == "/queue" and method == "GET":
            running, pending = [], []
            ahead = [[0, str(uuid.uuid4()), {}, {}, []] for _ in range(self.pending_ahead)]
            for number, (pid, count) in enumerate(self.seen.items(), start=1):
                if pid in self.deleted or count > self.polls:
                    continue
                self.seen[pid] = count + 1
                item = [number, pid, {}, {}, []]
                (running if count >= 1 else pending).append(item)
            return httpx.Response(
                200, json={"queue_running": running, "queue_pending": ahead + pending}
            )
        if path == "/queue" and method == "POST":
            self.deleted += json.loads(request.content).get("delete", [])
            return httpx.Response(200)
        if path.startswith("/history/"):
            pid = path.removeprefix("/history/")
            if pid not in self.seen or self.seen[pid] <= self.polls or pid in self.deleted:
                return httpx.Response(200, json={})
            return httpx.Response(200, json={pid: self._entry()})
        if path == "/upload/image":
            name = request.content.split(b'filename="')[1].split(b'"')[0].decode()
            self.uploads[name] = request.content
            return httpx.Response(200, json={"name": name, "subfolder": "", "type": "input"})
        if path == "/view":
            return httpx.Response(
                200, content=b"video bytes " + request.url.params["filename"].encode()
            )
        if path == "/free":
            self.freed.append(json.loads(request.content))
            return httpx.Response(200)
        if path == "/interrupt":
            self.interrupts.append(json.loads(request.content))
            return httpx.Response(200)
        return httpx.Response(404)

    def _entry(self):
        if self.outcome == "error":
            messages = [
                ["execution_start", {}],
                [
                    "execution_error",
                    {
                        "node_id": "2",
                        "node_type": "SaveImage",
                        "exception_message": "out of memory",
                    },
                ],
            ]
            return {
                "outputs": {},
                "status": {"status_str": "error", "completed": False, "messages": messages},
            }
        if self.outcome == "interrupted":
            messages = [["execution_interrupted", {"node_id": "2"}]]
            return {
                "outputs": {},
                "status": {"status_str": "error", "completed": False, "messages": messages},
            }
        outputs = {
            "2": {"images": [{"filename": "x_00001_.png", "subfolder": "job", "type": "output"}]}
        }
        return {
            "outputs": outputs,
            "status": {"status_str": "success", "completed": True, "messages": []},
        }


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


@pytest.fixture
def fake():
    return FakeComfy()


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def client(fake, clock):
    with ComfyClient(
        transport=httpx.MockTransport(fake.handler),
        sleep=clock.sleep,
        clock=clock,
        unreachable_after=5,
    ) as client:
        yield client


@pytest.mark.parametrize(
    "url",
    ["http://127.0.0.1:8188", "http://localhost:8188/", "http://192.168.1.5:8199", "ftp://x"],
)
def test_refuses_other_instances(url):
    with pytest.raises(ValueError):
        check_url(url)
    with pytest.raises(ValueError):
        ComfyClient(url)


def test_is_up(client, fake):
    assert client.is_up()
    fake.down = True
    assert not client.is_up()


def test_run_reports_progress_and_files(client, fake):
    seen = []
    result = client.run(GRAPH, outputs=["2"], timeout=60, on_progress=seen.append)
    body = fake.prompts[result.prompt_id]
    assert body["partial_execution_targets"] == ["2"]
    assert seen[0].state == "pending"
    assert {p.state for p in seen} == {"pending", "running"}
    assert result.files("2") == [OutputFile("x_00001_.png", "job", "output")]
    assert result.files("3") == []
    assert result.files("2")[0].relpath == "job/x_00001_.png"
    assert result.seconds > 0


def test_position_counts_the_prompts_ahead(client, fake):
    fake.pending_ahead = 2
    seen = []
    client.run(GRAPH, timeout=60, on_progress=seen.append)
    assert seen[0].state == "pending" and seen[0].position == 2


def test_own_prompt_id_and_status(client, fake):
    pid = str(uuid.uuid4())
    assert client.status(pid) == "unknown"
    assert client.queue(GRAPH, prompt_id=pid) == pid
    assert client.status(pid) == "pending"
    assert client.result(pid) is None
    client.wait(pid, timeout=60)
    assert client.status(pid) == "done"
    assert client.result(pid).files("2")


def test_queue_refuses_unknown_outputs(client):
    with pytest.raises(ValueError):
        client.queue(GRAPH, outputs=["9"])
    with pytest.raises(ValueError):
        client.queue(GRAPH, outputs=[])


def test_rejected_graph(client, fake):
    fake.reject = {
        "error": {"type": "prompt_outputs_failed_validation", "message": "Prompt outputs failed"},
        "node_errors": {"1": {"errors": [{"message": "Value not in list"}]}},
    }
    with pytest.raises(GraphRejectedError) as caught:
        client.queue(GRAPH)
    assert "Prompt outputs failed" in str(caught.value)
    assert caught.value.node_errors["1"]["errors"][0]["message"] == "Value not in list"


def test_failed_run(client, fake):
    fake.outcome = "error"
    with pytest.raises(ExecutionFailedError) as caught:
        client.run(GRAPH, timeout=60)
    assert "out of memory" in str(caught.value)
    assert caught.value.details["node_id"] == "2"


def test_interrupted_run(client, fake):
    fake.outcome = "interrupted"
    with pytest.raises(CancelledRunError):
        client.run(GRAPH, timeout=60)


def test_timeout_cancels(client, fake):
    fake.polls = 1000
    with pytest.raises(ComfyTimeoutError):
        client.run(GRAPH, timeout=5)
    pid = next(iter(fake.prompts))
    assert pid in fake.deleted


def test_should_cancel_interrupts_a_running_prompt(client, fake):
    fake.polls = 1000
    calls = iter([False, False, False, True])
    with pytest.raises(CancelledRunError):
        client.run(GRAPH, timeout=60, should_cancel=lambda: next(calls))
    pid = next(iter(fake.prompts))
    assert pid in fake.deleted
    assert fake.interrupts == [] or fake.interrupts == [{"prompt_id": pid}]


def test_cancel_running_prompt_interrupts_only_that_prompt(client, fake):
    fake.polls = 1000
    pid = client.queue(GRAPH)
    fake.seen[pid] = 5  # running; the fake ignores the delete of a running prompt
    fake.deleted = []
    original = fake.handler

    def keep_running(request):
        if request.url.path == "/queue" and request.method == "POST":
            return httpx.Response(200)
        return original(request)

    client._http._transport = httpx.MockTransport(keep_running)
    client.cancel(pid)
    assert fake.interrupts == [{"prompt_id": pid}]


def test_server_gone_during_wait(client, fake):
    fake.polls = 1000
    pid = client.queue(GRAPH)
    fake.down = True
    with pytest.raises(ComfyUnavailableError):
        client.wait(pid, timeout=600)


def test_prompt_that_vanished(client, fake):
    pid = client.queue(GRAPH)
    fake.deleted.append(pid)
    with pytest.raises(ComfyError, match="neither queued"):
        client.wait(pid, timeout=600)


def test_upload(client, fake, tmp_path):
    photo = tmp_path / "p.png"
    photo.write_bytes(b"png")
    assert client.upload(photo, "k1.png") == "k1.png"
    assert b"png" in fake.uploads["k1.png"]
    assert b'name="type"' in fake.uploads["k1.png"]
    for bad in ("../x.png", "a/b.png", "", ".hidden"):
        with pytest.raises(ValueError):
            client.upload(b"x", bad)
    with pytest.raises(ValueError):
        client.upload(b"x", "ok.png", subfolder="../up")


def test_download(client, tmp_path):
    dest = client.download(OutputFile("x.mp4", "job"), tmp_path / "out" / "x.mp4")
    assert dest.read_bytes() == b"video bytes x.mp4"
    assert not (tmp_path / "out" / "x.mp4.part").exists()


def test_free_and_interrupt(client, fake):
    client.free()
    client.interrupt()
    assert fake.freed == [{"unload_models": True, "free_memory": True}]
    assert fake.interrupts == [{}]


def test_unreachable_raises(fake, clock):
    fake.down = True
    with ComfyClient(transport=httpx.MockTransport(fake.handler)) as client:
        with pytest.raises(ComfyUnavailableError):
            client.system_stats()
