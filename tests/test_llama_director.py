import contextlib

import httpx
import pytest

from memcraft.pipeline.services import DirectorConfig, LlamaDirector
from memcraft.roles.llama_text import LlamaTextError, LlamaTextRunner, request_body


def answer(text, status=200):
    return httpx.Response(
        status,
        json={"choices": [{"message": {"content": text}}], "timings": {"predicted_n": 3}},
    )


def test_the_body_has_the_pictures_first_and_no_thinking():
    body = request_body("sys", "the idea", [b"\x89PNG..", b"\xff\xd8.."], seed=7)
    system, user = body["messages"]
    assert system == {"role": "system", "content": "sys"}
    kinds = [part["type"] for part in user["content"]]
    assert kinds == ["image_url", "image_url", "text"]
    assert user["content"][0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert user["content"][1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert (body["seed"], body["max_tokens"], body["temperature"]) == (7, 1800, 0.7)


def test_the_runner_starts_the_server_before_each_call():
    events = []

    def post(url, json, timeout):
        events.append(("post", url))
        return answer("TITLE: x")

    runner = LlamaTextRunner(
        "http://127.0.0.1:8097/", before_call=lambda: events.append("up"), post=post
    )
    assert runner("s", "u", [b"x"], 1) == "TITLE: x"
    assert events == ["up", ("post", "http://127.0.0.1:8097/v1/chat/completions")]
    assert runner.last_timings == {"predicted_n": 3}


@pytest.mark.parametrize("response", [answer("", 200), answer("x", 500)])
def test_an_empty_or_failed_answer_is_an_error(response):
    runner = LlamaTextRunner("http://127.0.0.1:8097", post=lambda *a, **k: response)
    with pytest.raises(LlamaTextError):
        runner("s", "u", [b"x"], 1)


class FakeServer:
    url = "http://127.0.0.1:8097"

    def __init__(self):
        self.running = False
        self.events = []

    def ensure_up(self):
        self.events.append("up")
        self.running = True

    def stop(self):
        self.events.append("stop")
        self.running = False


class FakeComfy:
    def __init__(self, events):
        self.events = events

    def free(self):
        self.events.append("free")


def test_comfyui_frees_its_models_only_when_the_server_must_start():
    server = FakeServer()
    director = LlamaDirector(server, FakeComfy(server.events))
    director._up()
    director._up()
    director.release()
    director.release()  # nothing runs: no second stop
    director._up()
    assert server.events == ["free", "up", "up", "stop", "free", "up"]


def test_the_session_always_stops_the_server():
    server = FakeServer()
    director = LlamaDirector(server, FakeComfy(server.events))
    with contextlib.suppress(RuntimeError), director.session() as runner:
        assert isinstance(runner, LlamaTextRunner)
        raise RuntimeError("the step failed")
    assert server.events == ["stop"]


def test_the_director_server_fits_three_photos_and_uses_mtp():
    config = DirectorConfig()
    args = config.args()
    assert config.port == 8097 and config.context == 6144
    assert "--device" not in args and args[args.index("-ngl") + 1] == "99"
    assert args[args.index("--spec-type") + 1] == "draft-mtp"
    assert args[args.index("--image-max-tokens") + 1] == "320"
    assert str(config.model).endswith("IQ3_XXS-mtp.gguf")
