"""A client for Gemma behind any OpenAI-compatible llama.cpp server (llama-server, llama-swap).

Answers are forced into a JSON schema by the server (response_format json_schema), so a valid
answer always parses. The system prompt comes first and cache_prompt is on, so the server
reuses its cached prefix when the next call has the same system prompt.
"""

from __future__ import annotations

import base64
import functools
import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

# (url, body, timeout in seconds) -> the decoded JSON answer
type Post = Callable[[str, Mapping[str, Any], float], dict[str, Any]]

DEFAULT_TIMEOUT = 600.0

_MIME = ((b"\xff\xd8\xff", "image/jpeg"), (b"\x89PNG\r\n\x1a\n", "image/png"))


class GemmaError(Exception):
    """The server failed or gave no usable answer."""


class GemmaTimeoutError(GemmaError):
    pass


@dataclass(frozen=True)
class Answer:
    data: dict[str, Any]
    seconds: float
    prompt_tokens: int | None = None
    output_tokens: int | None = None
    finish: str | None = None
    timings: dict[str, Any] = field(default_factory=dict)


def data_url(image: bytes | str) -> str:
    """A data URL for JPEG or PNG bytes; a string is taken to be a data URL already."""
    if isinstance(image, str):
        if not image.startswith("data:image/"):
            raise ValueError("an image string must be a data URL")
        return image
    for magic, mime in _MIME:
        if image.startswith(magic):
            return f"data:{mime};base64,{base64.b64encode(image).decode('ascii')}"
    raise ValueError("only JPEG and PNG images can go to Gemma")


def _urllib_post(
    url: str, body: Mapping[str, Any], timeout: float, api_key: str | None = None
) -> dict[str, Any]:
    if not url.startswith(("http://", "https://")):
        raise GemmaError("the Gemma URL must be http or https")
    request = urllib.request.Request(  # noqa: S310 - the scheme is checked above
        url,
        data=json.dumps(body).encode(),
        headers={
            "Content-Type": "application/json",
            **({"Authorization": f"Bearer {api_key}"} if api_key else {}),
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            return json.loads(response.read())
    except TimeoutError as error:
        raise GemmaTimeoutError(f"no answer from Gemma within {timeout:.0f} s") from error
    except urllib.error.HTTPError as error:
        detail = error.read()[:500].decode("utf-8", "replace")
        raise GemmaError(f"Gemma answered {error.code}: {detail}") from error
    except urllib.error.URLError as error:
        if isinstance(error.reason, TimeoutError):
            raise GemmaTimeoutError(f"no answer from Gemma within {timeout:.0f} s") from error
        raise GemmaError(f"cannot reach Gemma: {error.reason}") from error
    except ValueError as error:
        raise GemmaError("Gemma's answer is not JSON") from error


# Seconds to wait before asking again when llama-swap unloaded the model mid-request.
RELOAD_PAUSE = 2.0


class GemmaClient:
    """`base_url` is the server root, such as http://127.0.0.1:8098 (without /v1)."""

    def __init__(
        self,
        base_url: str,
        *,
        model: str | None = None,
        api_key: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        post: Post | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/").removesuffix("/v1")
        self.model = model
        self.timeout = timeout
        if post is None:
            post = functools.partial(_urllib_post, api_key=api_key)
        self._post = post

    def ask(
        self,
        system: str,
        text: str,
        images: Sequence[bytes | str],
        schema: Mapping[str, Any],
        *,
        max_tokens: int = 400,
        temperature: float = 0.7,
        seed: int = 1,
        timeout: float | None = None,
    ) -> Answer:
        """One forced-JSON answer. Images go before the text, as the E0 tests sent them."""
        content: list[dict[str, Any]] = [
            {"type": "image_url", "image_url": {"url": data_url(image)}} for image in images
        ]
        content.append({"type": "text", "text": text})
        body: dict[str, Any] = {
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 0.95,
            "top_k": 64,
            "seed": seed,
            "cache_prompt": True,
            "chat_template_kwargs": {"enable_thinking": False},
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "answer", "strict": True, "schema": dict(schema)},
            },
        }
        if self.model:
            body["model"] = self.model
        started = time.monotonic()
        url = self.base_url + "/v1/chat/completions"
        wait = self.timeout if timeout is None else timeout
        try:
            reply = self._post(url, body, wait)
        except GemmaError as error:
            # llama-swap drops a request whose model was unloaded meanwhile; ask once more.
            if "model unloaded" not in str(error):
                raise
            time.sleep(RELOAD_PAUSE)
            reply = self._post(url, body, wait)
        seconds = round(time.monotonic() - started, 1)
        try:
            choice = reply["choices"][0]
            content_text = choice["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise GemmaError("Gemma's answer has no message") from error
        if choice.get("finish_reason") == "length":
            raise GemmaError("Gemma's answer was cut off at max_tokens")
        try:
            data = json.loads(content_text)
        except (TypeError, ValueError) as error:
            raise GemmaError("Gemma's answer does not follow the schema") from error
        if not isinstance(data, dict):
            raise GemmaError("Gemma's answer is not a JSON object")
        timings = reply.get("timings") or {}
        return Answer(
            data=data,
            seconds=seconds,
            prompt_tokens=timings.get("prompt_n"),
            output_tokens=timings.get("predicted_n"),
            finish=choice.get("finish_reason"),
            timings=dict(timings),
        )


def object_schema(properties: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """A strict schema whose keys come in this order, all required: the order steers Gemma."""
    return {
        "type": "object",
        "properties": dict(properties),
        "required": list(properties),
        "additionalProperties": False,
    }
