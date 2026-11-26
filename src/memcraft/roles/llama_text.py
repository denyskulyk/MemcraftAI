"""Qwen 3.8 as a text model in the app's own llama-server on the GPU: the TextRunner the director
uses since 2026-11-24 (Denys's choice after the speed test).

The 27B file at IQ3_XXS with its MTP head fits the 12 GB card with the vision file, so it writes
about 61 tokens/s, where the 17 GB ComfyUI file streamed from RAM wrote 4.4. The pictures go as
JPEG data URLs, the system prompt first, thinking off, with the sampling of the ComfyUI graph.
"""

from __future__ import annotations

import base64
import json
import re
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


# Regime 3 (2026-11-26): Qwen3.8 Flash-Next IQ3_XXS on Strata, the "qwen-flash-next-strata"
# model of llama-swap (128K context, pictures, thinking on). Thinking costs thousands of tokens,
# so the answer streams and every piece moves the step's progress for the watchdog.
FLASH_MAX_LENGTH = 16384
FLASH_TIMEOUT = 1500.0
_THINK = re.compile(r"<think>.*?</think>", re.S)


def thinking_body(
    system: str,
    prompt: str,
    images: Sequence[bytes],
    seed: int,
    *,
    model: str,
    max_length: int = FLASH_MAX_LENGTH,
) -> dict[str, Any]:
    """The request with thinking on and Qwen's thinking sampling (0.6 / 0.95 / 20)."""
    body = request_body(system, prompt, images, seed, max_length)
    body.update(
        model=model,
        stream=True,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        chat_template_kwargs={"enable_thinking": True},
    )
    body.pop("repeat_penalty", None)
    return body


def answer_text(content: str) -> str:
    """The answer without any thinking the server left in it."""
    text = _THINK.sub("", content)
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    return text.strip()


class ThinkingTextRunner:
    """A TextRunner on an OpenAI-style server (llama-swap) that streams a thinking answer.

    `before_call` runs first each time (to make room for the model), `on_piece` after every
    streamed piece with the number of pieces so far. Only `content` is the answer: thinking
    (`reasoning_content`) and llama-swap's loading notes are counted, never returned."""

    def __init__(
        self,
        url: str,
        *,
        model: str,
        api_key: str | None = None,
        before_call: Callable[[], None] | None = None,
        on_piece: Callable[[int], None] | None = None,
        timeout: float = FLASH_TIMEOUT,
        max_length: int = FLASH_MAX_LENGTH,
        stream: Callable[..., Any] | None = None,
    ) -> None:
        self.url = url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.before_call = before_call
        self.on_piece = on_piece
        self.timeout = timeout
        self.max_length = max_length
        self._stream = stream or httpx.stream
        self.last_timings: dict[str, Any] = {}
        self.last_thinking = ""

    def __call__(self, system: str, prompt: str, images: Sequence[bytes], seed: int) -> str:
        if self.before_call is not None:
            self.before_call()
        body = thinking_body(
            system, prompt, images, seed, model=self.model, max_length=self.max_length
        )
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        content: list[str] = []
        thinking: list[str] = []
        pieces = 0
        finish = None
        try:
            with self._stream(
                "POST",
                f"{self.url}/v1/chat/completions",
                json=body,
                headers=headers,
                timeout=httpx.Timeout(self.timeout, connect=30.0),
            ) as response:
                if response.status_code != 200:
                    response.read()
                    raise LlamaTextError(
                        f"the director's server answered {response.status_code}: "
                        f"{response.text[:300]}"
                    )
                for line in response.iter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except ValueError:
                        continue
                    if chunk.get("timings"):
                        self.last_timings = chunk["timings"]
                    for choice in chunk.get("choices") or []:
                        delta = choice.get("delta") or {}
                        if delta.get("content"):
                            content.append(delta["content"])
                        if delta.get("reasoning_content"):
                            thinking.append(delta["reasoning_content"])
                        finish = choice.get("finish_reason") or finish
                    pieces += 1
                    if self.on_piece is not None:
                        self.on_piece(pieces)
        except httpx.HTTPError as error:
            raise LlamaTextError(f"cannot reach the director's server: {error}") from error
        self.last_thinking = "".join(thinking)
        text = answer_text("".join(content))
        if not text:
            why = " (it ran out of tokens while thinking)" if finish == "length" else ""
            raise LlamaTextError(f"the director's server gave no text{why}")
        return text
