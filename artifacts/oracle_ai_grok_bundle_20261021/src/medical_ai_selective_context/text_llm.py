"""One streaming chat completion on any OpenAI-compatible server.

The default body has no response_format. The evidence path may add a JSON
schema on a non-OpenRouter host; a 400 drops the schema and retries once.
OpenRouter zero-data-retention is sent only to
openrouter.ai, and a ZDR request aimed at another host fails closed.
Fallbacks stay inside that ZDR set unless the operator pins a provider.

Real records stay off the model unless the operator opts in. External
hosts need ALLOW_EXTERNAL_REAL_DATA. An on-prem vLLM needs the separate
LLM_LOCAL_INFERENCE flag, and even then only a loopback, private, or
explicitly listed host is accepted. openrouter.ai never counts as local.
"""

from __future__ import annotations

import asyncio
import ipaddress
import json
import math
import os
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass
from time import perf_counter
from typing import Any
from urllib.parse import urlparse

import httpx

DEFAULT_MODEL = "qwen/qwen3.5-9b"
_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}
_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"
_THINK_BLOCK = re.compile(r"<think>.*?(?:</think>|$)", re.DOTALL)
_RETRY_STATUS = {408, 429, 502, 503, 529}
_THINKING_NAMES = ("qwen3", "qwq", "deepseek-r1")


class _Retryable(Exception):
    """HTTP 429/5xx before any token. The caller retries; the message stays empty."""


class _SchemaRejected(Exception):
    """The host refused response_format. The same call can go out without it."""


class ModelError(RuntimeError):
    """Stable code only. Never the request text or the provider body."""


def env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    value = raw.strip().lower()
    if value in _TRUE:
        return True
    if value in _FALSE:
        return False
    raise ModelError("invalid_boolean_setting")


def env_optional_bool(name: str) -> bool | None:
    """Unset omits the parameter. Any other spelling refuses to start."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return None
    value = raw.strip().lower()
    if value in _TRUE:
        return True
    if value in _FALSE:
        return False
    raise ModelError("invalid_boolean_setting")


def env_number(name: str, default: float, low: float, high: float, *, integer: bool) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw.strip()) if integer else float(raw.strip())
    except ValueError as exc:
        raise ModelError("invalid_numeric_setting") from exc
    if isinstance(value, bool) or not math.isfinite(value) or not low <= value <= high:
        raise ModelError("invalid_numeric_setting")
    return float(value)


@dataclass(frozen=True)
class TextModelSettings:
    api_key: str = ""
    base_url: str = "https://openrouter.ai/api/v1"
    model: str = DEFAULT_MODEL
    timeout_seconds: float = 90
    provider: str = ""
    require_zdr: bool = True
    allow_external_real_data: bool = False
    # Separate from the external opt-in. True still refuses a public host.
    local_inference: bool = False
    local_hosts: tuple[str, ...] = ()
    answer_max_tokens: int = 2000
    # None omits an explicit override. Auto still disables thinking for known
    # reasoning models on OpenRouter; LLM_REASONING wins when it is set.
    reasoning: bool | None = None
    allow_fallbacks: bool = True
    retry_max: int = 2
    retry_backoff_seconds: float = 0.75
    # vLLM /tokenize. Empty derives it from base_url. Only a trusted local host is called.
    tokenize_url: str = ""
    # auto | true | false. Unset env is true: OpenRouter gets reasoning off,
    # and any other host gets chat_template_kwargs.enable_thinking false.
    # false on a non-OpenRouter host sends enable_thinking true.
    # auto limits the OpenRouter flag to known reasoning model names.
    disable_thinking: str = "auto"

    @classmethod
    def from_env(cls) -> TextModelSettings:
        model = os.getenv("LLM_MODEL", DEFAULT_MODEL).strip()
        if not model or len(model) > 200 or any(character.isspace() for character in model):
            raise ModelError("invalid_model_name")
        base_url = os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1").strip().rstrip("/")
        if not base_url:
            raise ModelError("invalid_model_endpoint")
        api_key = os.getenv("LLM_API_KEY", "").strip() or os.getenv("OPENROUTER_API_KEY", "").strip()
        return cls(
            api_key=api_key,
            base_url=base_url,
            model=model,
            timeout_seconds=env_number("LLM_TIMEOUT_SECONDS", 90, 1, 300, integer=False),
            provider=os.getenv("OPENROUTER_PROVIDER", "").strip(),
            require_zdr=env_bool("OPENROUTER_REQUIRE_ZDR", True),
            allow_external_real_data=env_bool("ALLOW_EXTERNAL_REAL_DATA", False),
            local_inference=env_bool("LLM_LOCAL_INFERENCE", False),
            local_hosts=_local_hosts_from_env(),
            answer_max_tokens=int(env_number("TEXT_ANSWER_MAX_TOKENS", 2000, 128, 16000, integer=True)),
            reasoning=env_optional_bool("LLM_REASONING"),
            allow_fallbacks=env_bool("OPENROUTER_ALLOW_FALLBACKS", True),
            retry_max=int(env_number("LLM_RETRY_MAX", 2, 0, 5, integer=True)),
            retry_backoff_seconds=env_number("LLM_RETRY_BACKOFF_SECONDS", 0.75, 0, 30, integer=False),
            tokenize_url=os.getenv("LLM_TOKENIZE_URL", "").strip().rstrip("/"),
            disable_thinking=_thinking_mode("LLM_DISABLE_THINKING"),
        )


def _hostname(base_url: str) -> str:
    return (urlparse(base_url).hostname or "").lower()


def _is_openrouter(base_url: str) -> bool:
    return _hostname(base_url) == "openrouter.ai"


def _is_openrouter_host(host: str) -> bool:
    return host == "openrouter.ai" or host.endswith(".openrouter.ai")


def _ip_literal(host: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    try:
        return ipaddress.ip_address(host)
    except ValueError:
        return None


def _is_local_ip(host: str) -> bool:
    """Loopback, RFC1918, link-local, or IPv6 ULA. No DNS lookup."""
    ip = _ip_literal(host)
    if ip is None:
        return False
    mapped = getattr(ip, "ipv4_mapped", None)
    if mapped is not None:
        ip = mapped
    return bool(ip.is_loopback or ip.is_private or ip.is_link_local)


def _host_is_trusted_local(host: str, extra_hosts: tuple[str, ...] = ()) -> bool:
    """On-prem only. A public name is trusted solely when the operator listed it."""
    if not host or _is_openrouter_host(host):
        return False
    if host == "localhost" or _is_local_ip(host):
        return True
    return host in extra_hosts


def _trusted_local_endpoint(settings: TextModelSettings) -> bool:
    return _host_is_trusted_local(_hostname(settings.base_url), settings.local_hosts)


def real_data_mode(settings: TextModelSettings) -> str:
    """Effective permission for a non-synthetic record: blocked, local, or external."""
    if settings.allow_external_real_data:
        return "external"
    if settings.local_inference and _trusted_local_endpoint(settings):
        return "local"
    return "blocked"


def _normalize_host_token(piece: str) -> str | None:
    token = piece.strip().lower().rstrip(".")
    if not token:
        return None
    if token.startswith("[") and "]" in token:
        host, _, rest = token[1:].partition("]")
        port = rest.removeprefix(":")
        if port and not port.isdigit():
            raise ModelError("invalid_local_hosts")
        token = host.rstrip(".")
    elif token.count(":") == 1:
        name, port = token.split(":")
        if port.isdigit() and name:
            token = name.rstrip(".")
    return token or None


def _host_token_ok(host: str) -> bool:
    if not host or len(host) > 253 or not re.fullmatch(r"[a-z0-9._:-]+", host):
        return False
    if _is_openrouter_host(host):
        return False
    ip = _ip_literal(host)
    return ip is None or _is_local_ip(host)


def _local_hosts_from_env() -> tuple[str, ...]:
    """Extra on-prem names. openrouter.ai and public IP literals are rejected."""
    hosts: list[str] = []
    for piece in os.getenv("LLM_LOCAL_HOSTS", "").split(","):
        host = _normalize_host_token(piece)
        if host is None:
            continue
        if not _host_token_ok(host):
            raise ModelError("invalid_local_hosts")
        if host not in hosts:
            hosts.append(host)
    return tuple(hosts)


def _thinking_mode(name: str) -> str:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return "true"
    value = raw.strip().lower()
    if value in {"auto"}:
        return "auto"
    if value in _TRUE:
        return "true"
    if value in _FALSE:
        return "false"
    raise ModelError("invalid_thinking_setting")


def _known_thinking_model(model: str) -> bool:
    name = model.lower()
    return any(token in name for token in _THINKING_NAMES)


def _disable_thinking(settings: TextModelSettings) -> bool:
    """True only when we will send reasoning.enabled=false. Explicit LLM_REASONING wins."""
    if settings.reasoning is not None or not _is_openrouter(settings.base_url):
        return False
    if settings.disable_thinking == "true":
        return True
    if settings.disable_thinking == "false":
        return False
    return _known_thinking_model(settings.model)


def chat_body(
    settings: TextModelSettings, messages: list[dict[str, str]],
    json_schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Portable chat-completions body. Patient text is already first in messages."""
    body: dict[str, Any] = {
        "model": settings.model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": settings.answer_max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if json_schema and not _is_openrouter(settings.base_url):
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "chart_answer", "strict": True, "schema": json_schema},
        }
    if settings.reasoning is not None:
        body["reasoning"] = {"enabled": settings.reasoning}
    elif _disable_thinking(settings):
        body["reasoning"] = {"enabled": False}
    elif not _is_openrouter(settings.base_url):
        # vLLM Qwen3 chat template. auto stays silent so an unknown field is not a 400.
        if settings.disable_thinking == "true":
            body["chat_template_kwargs"] = {"enable_thinking": False}
        elif settings.disable_thinking == "false":
            body["chat_template_kwargs"] = {"enable_thinking": True}
    if _is_openrouter(settings.base_url):
        provider: dict[str, Any] = {
            "allow_fallbacks": settings.allow_fallbacks,
            "data_collection": "deny",
            "zdr": settings.require_zdr,
        }
        if settings.provider:
            provider["only"] = [settings.provider]
        body["provider"] = provider
    return body


def strip_think_blocks(text: str, replacement: str = "") -> str:
    """Remove every think block, including one that never closes."""
    return _THINK_BLOCK.sub(replacement, text or "")


def _tag_prefix_len(buffer: str, tag: str) -> int:
    upper = min(len(tag) - 1, len(buffer))
    for size in range(upper, 0, -1):
        if tag.startswith(buffer[-size:]):
            return size
    return 0


class ThinkFilter:
    """Drop a leading chain-of-thought block so the status line can stream."""

    def __init__(self) -> None:
        self.buffer = ""
        self.inside = False

    def feed(self, chunk: str) -> str:
        self.buffer += chunk
        emitted: list[str] = []
        while self.buffer:
            if self.inside:
                close = self.buffer.find(_THINK_CLOSE)
                if close < 0:
                    self.buffer = self.buffer[-len(_THINK_CLOSE):]
                    break
                self.buffer = self.buffer[close + len(_THINK_CLOSE):]
                self.inside = False
                continue
            open_at = self.buffer.find(_THINK_OPEN)
            if open_at < 0:
                hold = _tag_prefix_len(self.buffer, _THINK_OPEN)
                emitted.append(self.buffer[:-hold] if hold else self.buffer)
                self.buffer = self.buffer[-hold:] if hold else ""
                break
            emitted.append(self.buffer[:open_at])
            self.buffer = self.buffer[open_at + len(_THINK_OPEN):]
            self.inside = True
        return "".join(emitted)

    def finish(self) -> str:
        if self.inside:
            self.buffer = ""
            return ""
        tail = self.buffer
        self.buffer = ""
        if _THINK_OPEN.startswith(tail):
            return ""
        return tail


def _usage(raw: Any) -> dict[str, float]:
    if not isinstance(raw, dict):
        return {}
    usage: dict[str, float] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = raw.get(key)
        if (
            isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value >= 0
        ):
            usage[key] = value
    return usage


def _has_reasoning(container: dict[str, Any]) -> bool:
    for key in ("reasoning", "reasoning_content"):
        value = container.get(key)
        if isinstance(value, str) and value.strip():
            return True
    return bool(container.get("reasoning_details"))


def _payload_overloaded(payload: Any) -> bool:
    if not isinstance(payload, dict) or not isinstance(payload.get("error"), dict):
        return False
    error = payload["error"]
    code = error.get("code")
    message = str(error.get("message") or "").lower()
    return code in {429, 502, 503, 529} or "overload" in message or "rate limit" in message


def _emit(shown: str, opened: bool) -> str:
    """Drop a leading run of whitespace. Reasoning streams are often nothing else."""
    if not shown:
        return ""
    if opened:
        return shown
    return shown.lstrip()


def _delta_text(
    payload: dict[str, Any],
) -> tuple[str, str | None, str | None, dict[str, float], bool]:
    choices = payload.get("choices")
    text = ""
    finish = None
    reasoning = False
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        choice = choices[0]
        delta = choice.get("delta") if isinstance(choice.get("delta"), dict) else {}
        message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
        reasoning = _has_reasoning(delta) or _has_reasoning(message)
        content = delta.get("content")
        if content is None:
            content = message.get("content")
        if isinstance(content, str):
            text = content
        finish_reason = choice.get("finish_reason")
        if isinstance(finish_reason, str):
            finish = finish_reason
    model = payload.get("model") if isinstance(payload.get("model"), str) else None
    return text, model, finish, _usage(payload.get("usage")), reasoning


class TextModelClient:
    def __init__(
        self, settings: TextModelSettings | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.settings = settings or TextModelSettings.from_env()
        self._transport = transport

    def status(self) -> dict[str, Any]:
        trusted = _trusted_local_endpoint(self.settings)
        return {
            "configured": bool(self.settings.api_key) or trusted,
            "model": self.settings.model,
            "host": _hostname(self.settings.base_url),
            "requireZdr": self.settings.require_zdr,
            "reasoning": self.settings.reasoning,
            "allowFallbacks": self.settings.allow_fallbacks,
            "retryMax": self.settings.retry_max,
            "disableThinking": self.settings.disable_thinking,
            "externalRealDataAllowed": self.settings.allow_external_real_data,
            "localInference": self.settings.local_inference,
            "trustedLocalHost": trusted,
            "realDataMode": real_data_mode(self.settings),
            "answerMaxTokens": self.settings.answer_max_tokens,
        }

    def guard(self, data_class: str) -> None:
        if data_class != "synthetic" and real_data_mode(self.settings) == "blocked":
            raise ModelError("external_real_data_blocked")
        if not self.settings.api_key and not _trusted_local_endpoint(self.settings):
            raise ModelError("api_key_missing")
        if self.settings.require_zdr and not _is_openrouter(self.settings.base_url):
            raise ModelError("zdr_requires_openrouter")

    def _endpoint(self) -> str:
        parsed = urlparse(self.settings.base_url)
        local_http = parsed.scheme == "http" and _trusted_local_endpoint(self.settings)
        if parsed.scheme != "https" and not local_http:
            raise ModelError("insecure_model_endpoint")
        if parsed.username or parsed.password or not parsed.hostname:
            raise ModelError("invalid_model_endpoint")
        return f"{self.settings.base_url}/chat/completions"

    def _headers(self) -> dict[str, str]:
        token = self.settings.api_key or "local"
        return {"Authorization": f"Bearer {token}"}

    def _tokenize_endpoint(self) -> str | None:
        """vLLM serves /tokenize at the server root, not under /v1."""
        url = self.settings.tokenize_url
        if not url:
            base = self.settings.base_url.rstrip("/")
            url = base.removesuffix("/v1") + "/tokenize"
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or parsed.username or parsed.password:
            return None
        if not _host_is_trusted_local((parsed.hostname or "").lower(), self.settings.local_hosts):
            return None
        return url

    async def count_tokens(
        self, messages: list[dict[str, str]] | None = None, prompt: str | None = None,
    ) -> dict[str, int | None] | None:
        """Served tokenizer and chat template. None means the count is unavailable.

        Messages are counted with the generation prompt, as a completion would be.
        Patient text goes only to a trusted local host; anything else returns None.
        """
        endpoint = self._tokenize_endpoint()
        if endpoint is None or (messages is None) == (prompt is None):
            return None
        body: dict[str, Any] = {"model": self.settings.model}
        if messages is not None:
            body["messages"] = messages
            body["add_generation_prompt"] = True
            kwargs = chat_body(self.settings, messages).get("chat_template_kwargs")
            if kwargs:
                body["chat_template_kwargs"] = kwargs
        else:
            body["prompt"] = prompt
            body["add_special_tokens"] = False
        try:
            async with httpx.AsyncClient(
                timeout=self.settings.timeout_seconds, transport=self._transport,
            ) as client:
                response = await client.post(endpoint, json=body, headers=self._headers())
            if response.status_code != 200:
                return None
            payload = response.json()
        except (httpx.HTTPError, ValueError):
            return None
        count = payload.get("count") if isinstance(payload, dict) else None
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            return None
        limit = payload.get("max_model_len")
        if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
            limit = None
        return {"count": count, "maxModelLen": limit}

    async def complete(
        self, messages: list[dict[str, str]], json_schema: dict[str, Any] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Yield token events, then one end event. Raises ModelError on a bad stream.

        A 429 or overloaded provider is retried with backoff only before the
        first token. A stream that already spoke is not replayed. A host that
        rejects a JSON schema is asked once more without the schema.
        """
        try:
            async for event in self._complete(messages, json_schema):
                yield event
        except _SchemaRejected:
            async for event in self._complete(messages, None):
                yield event

    async def _complete(
        self, messages: list[dict[str, str]], json_schema: dict[str, Any] | None,
    ) -> AsyncIterator[dict[str, Any]]:
        delay = self.settings.retry_backoff_seconds
        for attempt in range(self.settings.retry_max + 1):
            try:
                async for event in self._attempt(messages, json_schema):
                    yield event
                return
            except _Retryable:
                if attempt >= self.settings.retry_max:
                    break
                if delay > 0:
                    await asyncio.sleep(delay)
                delay = min(delay * 2, 8.0)
        raise ModelError("model_request_failed")

    async def _attempt(
        self, messages: list[dict[str, str]], json_schema: dict[str, Any] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        body = chat_body(self.settings, messages, json_schema)
        started = perf_counter()
        ttft_ms: float | None = None
        visible = ThinkFilter()
        parts: list[str] = []
        reported_model: str | None = None
        finish_reason: str | None = None
        usage: dict[str, float] = {}
        saw_reasoning = False
        opened = False
        try:
            async with httpx.AsyncClient(
                timeout=self.settings.timeout_seconds, transport=self._transport,
            ) as client, client.stream(
                "POST", self._endpoint(), json=body, headers=self._headers(),
            ) as response:
                if response.status_code in _RETRY_STATUS:
                    await response.aread()
                    raise _Retryable()
                if response.status_code >= 400:
                    await response.aread()
                    if json_schema and response.status_code == 400:
                        raise _SchemaRejected()
                    raise ModelError("model_request_failed")
                content_type = response.headers.get("content-type", "")
                if "application/json" in content_type:
                    payload = json.loads((await response.aread()).decode("utf-8"))
                    if _payload_overloaded(payload):
                        raise _Retryable()
                    text, reported_model, finish_reason, usage, saw_reasoning = _read_json_completion(
                        payload,
                    )
                    shown = _emit(visible.feed(text) + visible.finish(), opened)
                    opened = opened or bool(shown)
                    if shown:
                        ttft_ms = round((perf_counter() - started) * 1000, 2)
                        parts.append(shown)
                        yield {"type": "token", "text": shown}
                else:
                    async for line in response.aiter_lines():
                        if not line or line.startswith(":"):
                            continue
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if not data or data == "[DONE]":
                            continue
                        try:
                            payload = json.loads(data)
                        except json.JSONDecodeError as exc:
                            raise ModelError("invalid_model_json") from exc
                        if not isinstance(payload, dict):
                            raise ModelError("invalid_model_json")
                        if _payload_overloaded(payload) and not parts:
                            raise _Retryable()
                        if isinstance(payload.get("error"), dict):
                            raise ModelError("model_request_failed")
                        text, model, finish, chunk_usage, chunk_reasoning = _delta_text(payload)
                        saw_reasoning = saw_reasoning or chunk_reasoning
                        if model:
                            reported_model = model
                        if finish:
                            finish_reason = finish
                        if chunk_usage:
                            usage = chunk_usage
                        shown = _emit(visible.feed(text), opened)
                        if not shown:
                            continue
                        opened = True
                        parts.append(shown)
                        if ttft_ms is None:
                            ttft_ms = round((perf_counter() - started) * 1000, 2)
                        yield {"type": "token", "text": shown}
        except (ModelError, _Retryable, _SchemaRejected):
            raise
        except httpx.TimeoutException as exc:
            raise ModelError("model_timeout") from exc
        except (httpx.HTTPError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            raise ModelError("model_request_failed") from exc
        tail = _emit(visible.finish(), opened)
        if tail:
            parts.append(tail)
            if ttft_ms is None:
                ttft_ms = round((perf_counter() - started) * 1000, 2)
            yield {"type": "token", "text": tail}
        full = strip_think_blocks("".join(parts)).strip()
        if not full:
            raise ModelError("reasoning_only" if saw_reasoning else "invalid_model_json")
        # length means the cap was hit. Grounding still sees the text; a cut-off
        # number fails there. Other stop reasons are not a usable answer.
        if finish_reason not in {None, "stop", "length"}:
            raise ModelError("model_response_incomplete")
        if reported_model is not None:
            expected = self.settings.model.split(":", 1)[0].lower()
            got = reported_model.split(":", 1)[0].lower()
            if got != expected:
                raise ModelError("unexpected_model")
        yield {
            "type": "end",
            "text": full,
            "ttftMs": ttft_ms,
            "totalMs": round((perf_counter() - started) * 1000, 2),
            "usage": usage,
            "model": reported_model or self.settings.model,
            "finishReason": finish_reason or "stop",
        }


def _read_json_completion(
    payload: Any,
) -> tuple[str, str | None, str | None, dict[str, float], bool]:
    if not isinstance(payload, dict) or not isinstance(payload.get("choices"), list):
        raise ModelError("invalid_model_json")
    choice = payload["choices"][0]
    if not isinstance(choice, dict) or not isinstance(choice.get("message"), dict):
        raise ModelError("invalid_model_json")
    message = choice["message"]
    content = message.get("content")
    if not isinstance(content, str):
        raise ModelError("invalid_model_json")
    model = payload.get("model") if isinstance(payload.get("model"), str) else None
    finish = choice.get("finish_reason") if isinstance(choice.get("finish_reason"), str) else "stop"
    return content, model, finish, _usage(payload.get("usage")), _has_reasoning(message)
