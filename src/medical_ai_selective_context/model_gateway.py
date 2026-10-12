"""Model gateway. Own process and port. It does not mount the QA app.

Other projects call an OpenAI-compatible proxy, list downloaded models, and
switch the local vLLM process. Keys are compared as hashes. Message bodies
stay out of the log unless that is turned on.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import os
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

log = logging.getLogger(__name__)

_PUBLIC_BIND = frozenset({"0.0.0.0", "::", "[::]"})
_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_HASH_CHARS = frozenset("0123456789abcdef")
_QA_FIELDS = ("recordId", "question", "history", "stream")
_SYSTEM_ROLE = "system"
_UPSTREAM_TIMEOUT = 120.0

ScriptRunner = Callable[[Path, list[str], float], Awaitable[int]]


@dataclass
class ModelSpec:
    id: str
    name: str
    start: str
    stop: str
    args: list[str]


@dataclass
class SwitchBoard:
    busy: bool = False
    state: str = "idle"
    progress: str = "idle"
    active_id: str | None = None
    error: str | None = None

    def try_begin(self, model_id: str) -> bool:
        """Claim the switch. No await, so two tasks cannot both pass."""
        if self.busy:
            return False
        self.busy = True
        self.state = "stopping"
        self.progress = "stopping"
        self.error = None
        self._target = model_id
        return True

    def mark(self, state: str) -> None:
        self.state = state
        self.progress = state

    def finish_ok(self, model_id: str) -> None:
        self.active_id = model_id
        self.state = "ready"
        self.progress = "ready"
        self.error = None
        self.busy = False

    def finish_err(self, message: str) -> None:
        self.state = "failed"
        self.progress = "failed"
        self.error = message
        self.busy = False

    def snapshot(self) -> dict:
        return {
            "state": self.state,
            "progress": self.progress,
            "activeId": self.active_id,
            "error": self.error,
        }


class SwitchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1, max_length=200)


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_bind_host(host: str, allow_public: bool) -> str:
    """Loopback unless a public bind was explicitly allowed."""
    cleaned = host.strip() or "127.0.0.1"
    if cleaned in _PUBLIC_BIND and not allow_public:
        raise RuntimeError("public bind refused")
    return cleaned


def _flag(value: str | None) -> bool:
    return (value or "").strip().lower() in _TRUE_VALUES


def _hash_token(value: str) -> str | None:
    token = value.strip().lower()
    if len(token) != 64 or any(char not in _HASH_CHARS for char in token):
        return None
    return token


def hashes_from_values(raw_keys: str, raw_hashes: str) -> list[str]:
    """SHA-256 hex from GATEWAY_API_KEY_SHA256, plus hashes of GATEWAY_API_KEYS."""
    found: list[str] = []
    for item in raw_hashes.split(","):
        token = _hash_token(item)
        if item.strip() and token is None:
            log.warning("ignored an API key hash with an unexpected shape")
            continue
        if token:
            found.append(token)
    for item in raw_keys.split(","):
        secret = item.strip()
        if secret:
            found.append(hashlib.sha256(secret.encode()).hexdigest())
    return found


def key_matches(presented: str, hashes: list[str]) -> bool:
    if not presented or not hashes:
        return False
    digest = hashlib.sha256(presented.encode()).hexdigest()
    return any(hmac.compare_digest(digest, stored) for stored in hashes)


def load_model_config(path: Path) -> list[ModelSpec]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("models", [])
    if not isinstance(rows, list):
        raise ValueError("models config must contain a list")
    specs: list[ModelSpec] = []
    seen: set[str] = set()
    for row in rows:
        model_id = str(row["id"]).strip()
        if not model_id or model_id in seen:
            raise ValueError("model id missing or duplicated")
        seen.add(model_id)
        specs.append(ModelSpec(
            id=model_id,
            name=str(row.get("name") or model_id),
            start=str(row["start"]),
            stop=str(row["stop"]),
            args=[str(arg) for arg in row.get("args", [])],
        ))
    return specs


def resolve_script(root: Path, raw: str) -> Path:
    if not raw or raw.strip() != raw or "\x00" in raw:
        raise ValueError("invalid script")
    candidate = Path(raw)
    resolved = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("script escapes root")
    if not resolved.is_file():
        raise ValueError("script is missing")
    return resolved


def _presented_key(request: Request) -> str:
    header = request.headers.get("authorization", "")
    prefix = "bearer "
    if header.lower().startswith(prefix):
        return header[len(prefix):].strip()
    return request.headers.get("x-api-key", "").strip()


def _require_key(request: Request) -> None:
    hashes = request.app.state.key_hashes
    if not key_matches(_presented_key(request), hashes):
        raise HTTPException(status_code=401, detail="unauthorized")


def _http_base(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _join(base: str, suffix: str) -> str:
    return base.rstrip("/") + "/" + suffix.lstrip("/")


def _is_system(message: dict) -> bool:
    return message.get("role") == _SYSTEM_ROLE


def prepare_chat(payload: dict) -> dict:
    """Pass temperature and max_tokens through. Map the two local extensions."""
    body = dict(payload)
    if "enable_thinking" in body:
        enabled = bool(body.pop("enable_thinking"))
        kwargs = dict(body.get("chat_template_kwargs") or {})
        kwargs["enable_thinking"] = enabled
        body["chat_template_kwargs"] = kwargs
    if "system" in body:
        prompt = body.pop("system")
        messages = list(body.get("messages") or [])
        if messages and _is_system(messages[0]):
            messages[0] = {**messages[0], "content": prompt}
        else:
            messages.insert(0, {"role": _SYSTEM_ROLE, "content": prompt})
        body["messages"] = messages
    return body


async def _default_runner(path: Path, args: list[str], timeout: float, root: Path) -> int:
    process = await asyncio.create_subprocess_exec(
        str(path),
        *args,
        cwd=str(root),
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    try:
        return await asyncio.wait_for(process.wait(), timeout)
    except TimeoutError:
        process.kill()
        await process.wait()
        return 124


def _client(app: FastAPI) -> httpx.AsyncClient:
    kwargs: dict = {"timeout": _UPSTREAM_TIMEOUT}
    transport = app.state.transport
    if transport is not None:
        kwargs["transport"] = transport
    return httpx.AsyncClient(**kwargs)


async def _proxy_json(app: FastAPI, method: str, url: str, body: dict | None, headers: dict) -> Response:
    try:
        async with _client(app) as client:
            upstream = await client.request(method, url, json=body, headers=headers)
    except httpx.HTTPError:
        log.warning("upstream request failed method=%s", method)
        return JSONResponse(status_code=502, content={"detail": "upstream unavailable"})
    media = upstream.headers.get("content-type")
    return Response(content=upstream.content, status_code=upstream.status_code, media_type=media)


def create_gateway(
    *,
    key_hashes: list[str] | None = None,
    models: list[ModelSpec] | None = None,
    models_path: Path | None = None,
    vllm_base_url: str = "http://127.0.0.1:8000/v1",
    qa_base_url: str = "http://127.0.0.1:8091",
    vllm_api_key: str = "",
    log_content: bool = False,
    root: Path | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
    script_runner: ScriptRunner | None = None,
    script_timeout: float = 120.0,
) -> FastAPI:
    loaded = list(models) if models is not None else []
    if models is None and models_path is not None:
        loaded = load_model_config(models_path)
    gateway = FastAPI(title="Model gateway", version="1")
    gateway.state.key_hashes = list(key_hashes or [])
    gateway.state.models = {spec.id: spec for spec in loaded}
    gateway.state.vllm_base_url = vllm_base_url.rstrip("/")
    gateway.state.qa_base_url = qa_base_url.rstrip("/")
    gateway.state.vllm_api_key = vllm_api_key
    gateway.state.log_content = log_content
    gateway.state.root = root or project_root()
    gateway.state.transport = transport
    gateway.state.script_timeout = script_timeout
    gateway.state.board = SwitchBoard()
    gateway.state.script_runner = script_runner

    @gateway.middleware("http")
    async def time_requests(request: Request, call_next):
        started = time.perf_counter()
        body_note = ""
        if gateway.state.log_content:
            raw = await request.body()
            body_note = raw.decode("utf-8", errors="replace")[:2000]
        response = await call_next(request)
        elapsed = int((time.perf_counter() - started) * 1000)
        if gateway.state.log_content:
            log.info(
                "method=%s path=%s status=%s elapsed_ms=%s body=%s",
                request.method, request.url.path, response.status_code, elapsed, body_note,
            )
        else:
            log.info(
                "method=%s path=%s status=%s elapsed_ms=%s",
                request.method, request.url.path, response.status_code, elapsed,
            )
        return response

    @gateway.get("/health")
    async def health() -> dict:
        board = gateway.state.board
        return {"status": "ok", "activeId": board.active_id}

    def _vllm_headers() -> dict[str, str]:
        key = gateway.state.vllm_api_key
        if not key:
            return {}
        return {"Authorization": f"Bearer {key}"}

    @gateway.get("/v1/models")
    async def vllm_models(request: Request) -> Response:
        _require_key(request)
        url = _join(gateway.state.vllm_base_url, "models")
        return await _proxy_json(gateway, "GET", url, None, _vllm_headers())

    @gateway.post("/v1/chat/completions")
    async def chat(request: Request) -> Response:
        _require_key(request)
        try:
            payload = await request.json()
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="invalid json") from exc
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="invalid json")
        body = prepare_chat(payload)
        url = _join(gateway.state.vllm_base_url, "chat/completions")
        if body.get("stream"):
            return await _proxy_stream(gateway, url, body, _vllm_headers())
        return await _proxy_json(gateway, "POST", url, body, _vllm_headers())

    @gateway.get("/models")
    async def configured_models(request: Request) -> dict:
        _require_key(request)
        rows = [
            {
                "id": spec.id,
                "name": spec.name,
                "start": spec.start,
                "stop": spec.stop,
                "args": spec.args,
            }
            for spec in gateway.state.models.values()
        ]
        return {"models": rows}

    @gateway.get("/models/status")
    async def model_status(request: Request) -> dict:
        _require_key(request)
        return gateway.state.board.snapshot()

    @gateway.post("/models/switch")
    async def switch_model(body: SwitchRequest, request: Request) -> JSONResponse:
        _require_key(request)
        spec = gateway.state.models.get(body.id)
        if spec is None:
            raise HTTPException(status_code=404, detail="unknown model")
        try:
            start = resolve_script(gateway.state.root, spec.start)
            stop = resolve_script(gateway.state.root, spec.stop)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="invalid model script") from exc
        board: SwitchBoard = gateway.state.board
        if not board.try_begin(spec.id):
            raise HTTPException(status_code=409, detail="switch in progress")
        code = await _run(gateway, stop, [])
        log.info("model script name=%s code=%s", stop.name, code)
        if code != 0:
            board.finish_err("stop failed")
            return JSONResponse(status_code=500, content=board.snapshot())
        board.mark("starting")
        code = await _run(gateway, start, spec.args)
        log.info("model script name=%s code=%s", start.name, code)
        if code != 0:
            board.finish_err("start failed")
            return JSONResponse(status_code=500, content=board.snapshot())
        board.finish_ok(spec.id)
        return JSONResponse(status_code=200, content=board.snapshot())

    @gateway.post("/qa")
    async def qa(request: Request) -> Response:
        _require_key(request)
        try:
            payload = await request.json()
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="invalid json") from exc
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="invalid json")
        base = str(payload.get("qaBaseUrl") or gateway.state.qa_base_url)
        if not _http_base(base):
            raise HTTPException(status_code=400, detail="invalid qa base")
        forwarded = {key: payload[key] for key in _QA_FIELDS if key in payload}
        url = _join(base, "/api/v1/text/answer")
        if forwarded.get("stream"):
            return await _proxy_stream(gateway, url, forwarded, {})
        return await _proxy_json(gateway, "POST", url, forwarded, {})

    return gateway


async def _run(app: FastAPI, path: Path, args: list[str]) -> int:
    runner: ScriptRunner | None = app.state.script_runner
    timeout = float(app.state.script_timeout)
    if runner is not None:
        return await runner(path, args, timeout)
    return await _default_runner(path, args, timeout, app.state.root)


async def _proxy_stream(app: FastAPI, url: str, body: dict, headers: dict) -> Response:
    client = _client(app)

    async def chunks():
        try:
            async with client.stream("POST", url, json=body, headers=headers) as upstream:
                async for piece in upstream.aiter_bytes():
                    yield piece
        except httpx.HTTPError:
            log.warning("upstream stream failed")
        finally:
            await client.aclose()

    return StreamingResponse(chunks(), media_type="text/event-stream")


def settings_from_environ() -> dict:
    config = os.environ.get("GATEWAY_MODELS_CONFIG", "").strip()
    return {
        "key_hashes": hashes_from_values(
            os.environ.get("GATEWAY_API_KEYS", ""),
            os.environ.get("GATEWAY_API_KEY_SHA256", ""),
        ),
        "models_path": Path(config) if config else None,
        "vllm_base_url": os.environ.get("GATEWAY_VLLM_BASE_URL", "http://127.0.0.1:8000/v1"),
        "qa_base_url": os.environ.get("GATEWAY_QA_BASE_URL", "http://127.0.0.1:8091"),
        "vllm_api_key": os.environ.get("GATEWAY_VLLM_API_KEY", ""),
        "log_content": _flag(os.environ.get("GATEWAY_LOG_CONTENT")),
        "script_timeout": float(os.environ.get("GATEWAY_SCRIPT_TIMEOUT_SECONDS", "120")),
    }


app = create_gateway(**settings_from_environ())
