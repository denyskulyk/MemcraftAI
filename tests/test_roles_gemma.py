import json

import pytest

from memcraft.roles.gemma import GemmaClient, GemmaError, data_url, object_schema

JPEG = b"\xff\xd8\xff\xe0" + b"0" * 20
PNG = b"\x89PNG\r\n\x1a\n" + b"0" * 20


class FakeServer:
    def __init__(self, *answers):
        self.answers = list(answers)
        self.calls = []

    def __call__(self, url, body, timeout):
        self.calls.append((url, json.loads(json.dumps(body)), timeout))
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


def reply(content, finish="stop"):
    if not isinstance(content, str):
        content = json.dumps(content)
    return {
        "choices": [{"message": {"content": content}, "finish_reason": finish}],
        "timings": {"prompt_n": 123, "predicted_n": 45},
    }


def test_data_urls():
    assert data_url(JPEG).startswith("data:image/jpeg;base64,")
    assert data_url(PNG).startswith("data:image/png;base64,")
    assert data_url("data:image/png;base64,AA==") == "data:image/png;base64,AA=="
    with pytest.raises(ValueError):
        data_url(b"GIF89a")
    with pytest.raises(ValueError):
        data_url("C:/photo.png")


def test_a_request_is_cached_forced_json_with_images_first():
    server = FakeServer(reply({"a": 1}))
    client = GemmaClient("http://127.0.0.1:8098/v1/", timeout=30, post=server, model="gemma")
    answer = client.ask("sys", "hello", [JPEG, PNG], object_schema({"a": {"type": "integer"}}))
    assert answer.data == {"a": 1} and answer.prompt_tokens == 123 and answer.output_tokens == 45
    url, body, timeout = server.calls[0]
    assert url == "http://127.0.0.1:8098/v1/chat/completions" and timeout == 30
    assert body["cache_prompt"] is True
    assert body["model"] == "gemma"
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["messages"][0] == {"role": "system", "content": "sys"}
    content = body["messages"][1]["content"]
    assert [part["type"] for part in content] == ["image_url", "image_url", "text"]
    assert content[0]["image_url"]["url"].startswith("data:image/jpeg")
    form = body["response_format"]
    assert form["type"] == "json_schema" and form["json_schema"]["strict"] is True
    schema = form["json_schema"]["schema"]
    assert schema["required"] == ["a"] and schema["additionalProperties"] is False


@pytest.mark.parametrize(
    "answer",
    [
        reply("not json"),
        reply([1, 2]),
        reply({"a": 1}, finish="length"),
        {"choices": []},
        {"error": "boom"},
    ],
)
def test_unusable_answers_raise(answer):
    client = GemmaClient("http://x:1", post=FakeServer(answer))
    with pytest.raises(GemmaError):
        client.ask("s", "t", [], {"type": "object"})


def test_the_default_transport_refuses_other_schemes():
    with pytest.raises(GemmaError, match="http"):
        GemmaClient("file:///etc").ask("s", "t", [], {"type": "object"})


def test_an_unreachable_server_is_a_gemma_error():
    # Port 9 (discard) on loopback is closed on a normal PC.
    with pytest.raises(GemmaError):
        GemmaClient("http://127.0.0.1:9", timeout=2).ask("s", "t", [], {"type": "object"})


def test_an_unloaded_model_is_asked_once_more(monkeypatch):
    from memcraft.roles import gemma as module

    monkeypatch.setattr(module, "RELOAD_PAUSE", 0)
    calls = []

    def post(url, body, timeout):
        calls.append(url)
        if len(calls) == 1:
            raise module.GemmaError('Gemma answered 500: {"message":"group: model unloaded"}')
        return {"choices": [{"message": {"content": '{"ok": 1}'}, "finish_reason": "stop"}]}

    client = module.GemmaClient("http://127.0.0.1:1", post=post)
    assert client.ask("s", "t", [], {"type": "object"}).data == {"ok": 1}
    assert len(calls) == 2


def test_other_errors_are_not_repeated(monkeypatch):
    from memcraft.roles import gemma as module

    calls = []

    def post(url, body, timeout):
        calls.append(url)
        raise module.GemmaError("Gemma answered 500: something else")

    client = module.GemmaClient("http://127.0.0.1:1", post=post)
    import pytest

    with pytest.raises(module.GemmaError):
        client.ask("s", "t", [], {"type": "object"})
    assert len(calls) == 1
