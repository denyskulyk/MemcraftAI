"""Qwen 3.8 as a text model in the app's own llama-server on the GPU: the TextRunner the director
uses since 2026-11-24 (Denys's choice after the speed test).

The 27B file at IQ3_XXS with its MTP head fits the 12 GB card with the vision file, so it writes
about 61 tokens/s, where the 17 GB ComfyUI file streamed from RAM wrote 4.4. The pictures go as
JPEG data URLs, the system prompt first, thinking off, with the sampling of the ComfyUI graph.
"""

from __future__ import annotations

import base64
from collections.abc import Callable, Sequence
from typing import Any

import httpx

MAX_LENGTH = 1800
TIMEOUT = 600.0


class LlamaTextError(Exception):
    pass


def data_url(image: bytes) -> str:
    kind = "png" if image.startswith(b"\x89PNG") else "jpeg"
    return f"data:image/{kind};base64," + base64.b64encode(image).decode()


def request_body(
    system: str, prompt: str, images: Sequence[bytes], seed: int, max_length: int = MAX_LENGTH
) -> dict[str, Any]:
    content: list[dict[str, Any]] = [
        {"type": "image_url", "image_url": {"url": data_url(image)}} for image in images
    ]
    content.append({"type": "text", "text": prompt})
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ],
        "max_tokens": max_length,
        # The sampling of workflows/qwen3.8-director's TextGenerateV2 node.
        "temperature": 0.7,
        "top_k": 20,
        "top_p": 0.9,
        "min_p": 0.0,
        "repeat_penalty": 1.05,
        "seed": seed,
        "chat_template_kwargs": {"enable_thinking": False},
    }


class LlamaTextRunner:
    """A TextRunner on llama-server. `before_call` runs first each time, so the server can be
    started (and ComfyUI's models freed) only when a call needs it."""

    def __init__(
        self,
        url: str,
        *,
        before_call: Callable[[], None] | None = None,
        timeout: float = TIMEOUT,
        max_length: int = MAX_LENGTH,
        post: Callable[..., httpx.Response] | None = None,
    ) -> None:
        self.url = url.rstrip("/")
        self.before_call = before_call
        self.timeout = timeout
        self.max_length = max_length
        self._post = post or httpx.post
        self.last_timings: dict[str, Any] = {}

    def __call__(self, system: str, prompt: str, images: Sequence[bytes], seed: int) -> str:
        if self.before_call is not None:
            self.before_call()
        body = request_body(system, prompt, images, seed, self.max_length)
        try:
            response = self._post(
                f"{self.url}/v1/chat/completions", json=body, timeout=self.timeout
            )
        except httpx.HTTPError as error:
            raise LlamaTextError(f"cannot reach the director's llama-server: {error}") from error
        if response.status_code != 200:
            raise LlamaTextError(
                f"the director's llama-server answered {response.status_code}: "
                f"{response.text[:300]}"
            )
        try:
            answer = response.json()
            text = answer["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as error:
            raise LlamaTextError("the director's llama-server gave no text") from error
        if not isinstance(text, str) or not text.strip():
            raise LlamaTextError("the director's llama-server gave no text")
        self.last_timings = answer.get("timings") or {}
        return text
