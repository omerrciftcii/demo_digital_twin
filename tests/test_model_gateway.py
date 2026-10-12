"""Model gateway: hashed keys, vLLM proxy, model switch lock, QA forward."""

import ast
import asyncio
import hashlib
import json
import logging
from pathlib import Path

import httpx
from fastapi.testclient import TestClient

from medical_ai_selective_context.model_gateway import (
    ModelSpec,
    create_gateway,
    key_matches,
    resolve_bind_host,
)

ROOT = Path(__file__).parents[1]
SECRET = "gateway-secret"
SECRET_HASH = hashlib.sha256(SECRET.encode()).hexdigest()
QUESTION = "gizli-soru-metni-xyz"


def _spec(root: Path) -> ModelSpec:
    start = root / "start.sh"
    stop = root / "stop.sh"
    start.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    stop.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    return ModelSpec(
        id="example-local",
        name="Example local weights",
        start="start.sh",
        stop="stop.sh",
        args=["--served-model-name", "example-local"],
    )


def _transport(seen: list[httpx.Request]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        path = request.url.path
        if path.endswith("/chat/completions"):
            return httpx.Response(200, json={
                "id": "chat",
                "choices": [{"message": {"role": "assistant", "content": "tamam"}}],
            })
        if path.endswith("/models"):
            return httpx.Response(200, json={"object": "list", "data": [{"id": "upstream"}]})
        if path.endswith("/api/v1/text/answer"):
            return httpx.Response(200, json={"status": "answered", "answer": "kayit"})
        return httpx.Response(404, json={"detail": "missing"})

    return httpx.MockTransport(handler)


def _gateway(tmp_path: Path, seen: list[httpx.Request] | None = None, **overrides):
    captured = seen if seen is not None else []
    spec = _spec(tmp_path)
    settings = {
        "key_hashes": [SECRET_HASH],
        "models": [spec],
        "vllm_base_url": "http://127.0.0.1:8000/v1",
        "qa_base_url": "http://127.0.0.1:8091",
        "vllm_api_key": "vllm-local",
        "root": tmp_path,
        "transport": _transport(captured),
        "log_content": False,
    }
    settings.update(overrides)
    return create_gateway(**settings), captured


def _auth() -> dict[str, str]:
    return {"Authorization": f"Bearer {SECRET}"}


def test_public_bind_requires_an_explicit_flag():
    assert resolve_bind_host("", False) == "127.0.0.1"
    assert resolve_bind_host("127.0.0.1", False) == "127.0.0.1"
    try:
        resolve_bind_host("0.0.0.0", False)
    except RuntimeError as exc:
        assert "refused" in str(exc)
    else:
        raise AssertionError("public bind was accepted")
    assert resolve_bind_host("0.0.0.0", True) == "0.0.0.0"


def test_key_is_compared_as_a_hash(tmp_path: Path):
    assert key_matches(SECRET, [SECRET_HASH])
    assert not key_matches(SECRET_HASH, [SECRET_HASH])
    assert not key_matches("nope", [SECRET_HASH])
    app, _seen = _gateway(tmp_path)
    client = TestClient(app)
    assert client.get("/health").status_code == 200
    assert "gateway-secret" not in client.get("/health").text
    assert SECRET_HASH not in client.get("/health").text
    closed, _ignored = _gateway(tmp_path, key_hashes=[])
    bare = TestClient(closed)
    assert bare.get("/models", headers=_auth()).status_code == 401
    assert client.get("/models").status_code == 401
    assert client.get("/models", headers={"Authorization": f"Bearer {SECRET_HASH}"}).status_code == 401
    assert client.get("/models", headers=_auth()).status_code == 200
    assert client.get("/models", headers={"x-api-key": SECRET}).status_code == 200


def test_chat_passthrough_and_configured_models(tmp_path: Path):
    app, seen = _gateway(tmp_path)
    client = TestClient(app)
    listed = client.get("/models", headers=_auth()).json()
    assert listed["models"][0]["id"] == "example-local"
    assert listed["models"][0]["args"] == ["--served-model-name", "example-local"]
    upstream = client.get("/v1/models", headers=_auth())
    assert upstream.status_code == 200
    assert upstream.json()["data"][0]["id"] == "upstream"
    reply = client.post("/v1/chat/completions", headers=_auth(), json={
        "model": "example-local",
        "temperature": 0.2,
        "max_tokens": 32,
        "enable_thinking": False,
        "system": "Kısa yaz.",
        "messages": [{"role": "user", "content": QUESTION}],
    })
    assert reply.status_code == 200
    sent = json.loads(seen[-1].content)
    assert sent["temperature"] == 0.2
    assert sent["max_tokens"] == 32
    assert sent["chat_template_kwargs"]["enable_thinking"] is False
    assert sent["messages"][0] == {"role": "system", "content": "Kısa yaz."}
    assert sent["messages"][1]["content"] == QUESTION
    assert seen[-1].headers["authorization"] == "Bearer vllm-local"


def test_qa_forwards_to_the_text_answer_api(tmp_path: Path):
    app, seen = _gateway(tmp_path)
    client = TestClient(app)
    reply = client.post("/qa", headers=_auth(), json={
        "recordId": "rec-1",
        "question": QUESTION,
        "history": [{"role": "user", "content": "onceki"}],
        "stream": False,
        "qaBaseUrl": "http://169.254.169.254/latest",
        "temperature": 0.9,
    })
    assert reply.status_code == 200
    assert reply.json()["answer"] == "kayit"
    sent = json.loads(seen[-1].content)
    assert sent == {
        "recordId": "rec-1",
        "question": QUESTION,
        "history": [{"role": "user", "content": "onceki"}],
        "stream": False,
    }
    assert seen[-1].url.host == "127.0.0.1"
    assert seen[-1].url.port == 8091
    assert seen[-1].url.path == "/api/v1/text/answer"
    assert all(request.url.host != "169.254.169.254" for request in seen)


def test_qa_forwards_the_validation_flag(tmp_path: Path):
    app, seen = _gateway(tmp_path)
    client = TestClient(app)
    reply = client.post("/qa", headers=_auth(), json={
        "recordId": "rec-1",
        "question": QUESTION,
        "validation": False,
        "qaBaseUrl": "http://169.254.169.254/latest",
    })
    assert reply.status_code == 200
    sent = json.loads(seen[-1].content)
    assert sent["validation"] is False
    assert "qaBaseUrl" not in sent
    assert seen[-1].url.host == "127.0.0.1"
    assert seen[-1].url.port == 8091
    assert all(request.url.host != "169.254.169.254" for request in seen)


def test_logs_timings_without_message_content(tmp_path: Path, caplog):
    caplog.set_level(logging.INFO)
    app, _seen = _gateway(tmp_path)
    client = TestClient(app)
    client.post("/qa", headers=_auth(), json={"recordId": "rec-1", "question": QUESTION})
    text = caplog.text
    assert QUESTION not in text
    assert SECRET not in text
    assert "elapsed_ms=" in text
    assert "path=/qa" in text


def test_switch_runs_scripts_and_rejects_a_second_switch(tmp_path: Path):
    calls: list[tuple[str, list[str]]] = []
    started = asyncio.Event()
    release = asyncio.Event()

    async def runner(path: Path, args: list[str], _timeout: float) -> int:
        calls.append((path.name, args))
        if not started.is_set():
            started.set()
            await release.wait()
        return 0

    app, _seen = _gateway(tmp_path, script_runner=runner)
    escaped = tmp_path / "escape.json"

    async def scenario():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://gateway") as client:
            headers = _auth()
            first = asyncio.create_task(client.post(
                "/models/switch", headers=headers, json={"id": "example-local"},
            ))
            await started.wait()
            second = await client.post(
                "/models/switch", headers=headers, json={"id": "example-local"},
            )
            assert second.status_code == 409
            status = await client.get("/models/status", headers=headers)
            assert status.json()["state"] == "stopping"
            missing = await client.post("/models/switch", headers=headers, json={"id": "yok"})
            assert missing.status_code == 404
            release.set()
            done = await first
            assert done.status_code == 200
            assert done.json()["state"] == "ready"
            assert done.json()["activeId"] == "example-local"

    asyncio.run(scenario())
    assert calls[0][0] == "stop.sh"
    assert calls[1] == ("start.sh", ["--served-model-name", "example-local"])
    outside = create_gateway(
        key_hashes=[SECRET_HASH],
        models=[ModelSpec("example-local", "x", "../outside.sh", "stop.sh", [])],
        root=tmp_path,
    )
    client = TestClient(outside)
    assert client.post("/models/switch", headers=_auth(), json={"id": "example-local"}).status_code == 400
    assert not escaped.exists()


def test_gateway_module_does_not_import_the_qa_app():
    source = (ROOT / "src" / "medical_ai_selective_context" / "model_gateway.py").read_text()
    tree = ast.parse(source)
    modules = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
    assert "medical_ai_selective_context.main" not in modules
    assert "neo4j" not in modules
