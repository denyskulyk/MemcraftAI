import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import httpx

from memcraft.pipeline import services
from memcraft.pipeline.services import SwapGemma
from memcraft.roles.gemma import GemmaClient


class _Server(BaseHTTPRequestHandler):
    seen: list[dict] = []  # noqa: RUF012

    def do_POST(self):
        size = int(self.headers["Content-Length"])
        body = json.loads(self.rfile.read(size))
        _Server.seen.append({"auth": self.headers.get("Authorization"), "body": body})
        answer = {"choices": [{"message": {"content": '{"ok": 1}'}, "finish_reason": "stop"}]}
        data = json.dumps(answer).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


def test_the_api_key_and_model_reach_llama_swap(tmp_path):
    server = HTTPServer(("127.0.0.1", 0), _Server)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    key_file = tmp_path / "key.txt"
    key_file.write_text("secret-key\n", encoding="utf-8")
    try:
        gemma = SwapGemma(url=f"http://127.0.0.1:{server.server_port}", key_file=key_file)
        with gemma.client() as client:
            schema = {"type": "object", "properties": {"ok": {"type": "integer"}}}
            client.ask("s", "t", [], schema)
    finally:
        server.shutdown()
    seen = _Server.seen[-1]
    assert seen["auth"] == "Bearer secret-key"
    assert seen["body"]["model"] == "memcraft-gemma"


def test_no_key_sends_no_header():
    calls = []
    client = GemmaClient("http://127.0.0.1:1", post=lambda *a: calls.append(a) or {})
    assert client.model is None


def test_stop_unloads_only_this_model(tmp_path, monkeypatch):
    key_file = tmp_path / "key.txt"
    key_file.write_text("k", encoding="utf-8")
    calls = []
    monkeypatch.setattr(
        services.httpx, "post", lambda url, **kw: calls.append((url, kw["headers"]))
    )
    SwapGemma(key_file=key_file).stop()
    assert calls == [
        ("http://127.0.0.1:8080/api/models/unload/memcraft-gemma", {"Authorization": "Bearer k"})
    ]


def test_a_failed_unload_is_only_logged(tmp_path, monkeypatch):
    def fail(url, **kw):
        raise httpx.ConnectError("down")

    monkeypatch.setattr(services.httpx, "post", fail)
    SwapGemma(key_file=tmp_path / "missing.txt").stop()
