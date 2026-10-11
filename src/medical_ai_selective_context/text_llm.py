"""One streaming chat completion on any OpenAI-compatible server.

No response_format, no provider routing, no reasoning field unless the
operator set LLM_REASONING. OpenRouter zero-data-retention is sent only to
openrouter.ai, and a ZDR request aimed at another host fails closed.
"""

from __future__ import annotations

import json
import math
import os
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
    answer_max_tokens: int = 800
    # None omits the field. Sending reasoning to a server that does not know it
    # is how a "portable" client stops being portable.
    reasoning: bool | None = None

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
            answer_max_tokens=int(env_number("TEXT_ANSWER_MAX_TOKENS", 800, 128, 4000, integer=True)),
            reasoning=env_optional_bool("LLM_REASONING"),
        )


def _hostname(base_url: str) -> str:
    return (urlparse(base_url).hostname or "").lower()


def _is_openrouter(base_url: str) -> bool:
    return _hostname(base_url) == "openrouter.ai"


def _is_localhost(base_url: str) -> bool:
    return _hostname(base_url) in {"127.0.0.1", "localhost", "::1"}


def chat_body(settings: TextModelSettings, messages: list[dict[str, str]]) -> dict[str, Any]:
    """Portable chat-completions body. Patient text is already first in messages."""
    body: dict[str, Any] = {
        "model": settings.model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": settings.answer_max_tokens,
        "stream": True,
    }
    if settings.reasoning is not None:
        body["reasoning"] = {"enabled": settings.reasoning}
    if _is_openrouter(settings.base_url):
        provider: dict[str, Any] = {
            "allow_fallbacks": False,
            "data_collection": "deny",
            "zdr": settings.require_zdr,
        }
        if settings.provider:
            provider["only"] = [settings.provider]
        body["provider"] = provider
    return body


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


def _delta_text(payload: dict[str, Any]) -> tuple[str, str | None, str | None, dict[str, float]]:
    choices = payload.get("choices")
    text = ""
    finish = None
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        choice = choices[0]
        delta = choice.get("delta") if isinstance(choice.get("delta"), dict) else {}
        message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
        content = delta.get("content")
        if content is None:
            content = message.get("content")
        if isinstance(content, str):
            text = content
        finish_reason = choice.get("finish_reason")
        if isinstance(finish_reason, str):
            finish = finish_reason
    model = payload.get("model") if isinstance(payload.get("model"), str) else None
    return text, model, finish, _usage(payload.get("usage"))


class TextModelClient:
    def __init__(
        self, settings: TextModelSettings | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.settings = settings or TextModelSettings.from_env()
        self._transport = transport

    def status(self) -> dict[str, Any]:
        return {
            "configured": bool(self.settings.api_key) or _is_localhost(self.settings.base_url),
            "model": self.settings.model,
            "host": _hostname(self.settings.base_url),
            "requireZdr": self.settings.require_zdr,
            "reasoning": self.settings.reasoning,
            "externalRealDataAllowed": self.settings.allow_external_real_data,
            "answerMaxTokens": self.settings.answer_max_tokens,
        }

    def guard(self, data_class: str) -> None:
        if data_class != "synthetic" and not self.settings.allow_external_real_data:
            raise ModelError("external_real_data_blocked")
        if not self.settings.api_key and not _is_localhost(self.settings.base_url):
            raise ModelError("api_key_missing")
        if self.settings.require_zdr and not _is_openrouter(self.settings.base_url):
            raise ModelError("zdr_requires_openrouter")

    def _endpoint(self) -> str:
        parsed = urlparse(self.settings.base_url)
        if parsed.scheme != "https" and not (parsed.scheme == "http" and _is_localhost(self.settings.base_url)):
            raise ModelError("insecure_model_endpoint")
        if parsed.username or parsed.password or not parsed.hostname:
            raise ModelError("invalid_model_endpoint")
        return f"{self.settings.base_url}/chat/completions"

    def _headers(self) -> dict[str, str]:
        token = self.settings.api_key or "local"
        return {"Authorization": f"Bearer {token}"}

    async def complete(self, messages: list[dict[str, str]]) -> AsyncIterator[dict[str, Any]]:
        """Yield token events, then one end event. Raises ModelError on a bad stream."""
        body = chat_body(self.settings, messages)
        started = perf_counter()
        ttft_ms: float | None = None
        visible = ThinkFilter()
        parts: list[str] = []
        reported_model: str | None = None
        finish_reason: str | None = None
        usage: dict[str, float] = {}
        try:
            async with httpx.AsyncClient(
                timeout=self.settings.timeout_seconds, transport=self._transport,
            ) as client, client.stream(
                "POST", self._endpoint(), json=body, headers=self._headers(),
            ) as response:
                if response.status_code >= 400:
                    await response.aread()
                    raise ModelError("model_request_failed")
                content_type = response.headers.get("content-type", "")
                if "application/json" in content_type:
                    payload = json.loads((await response.aread()).decode("utf-8"))
                    text, reported_model, finish_reason, usage = _read_json_completion(payload)
                    shown = visible.feed(text) + visible.finish()
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
                        text, model, finish, chunk_usage = _delta_text(payload)
                        if model:
                            reported_model = model
                        if finish:
                            finish_reason = finish
                        if chunk_usage:
                            usage = chunk_usage
                        shown = visible.feed(text)
                        if not shown:
                            continue
                        parts.append(shown)
                        if ttft_ms is None:
                            ttft_ms = round((perf_counter() - started) * 1000, 2)
                        yield {"type": "token", "text": shown}
        except ModelError:
            raise
        except httpx.TimeoutException as exc:
            raise ModelError("model_timeout") from exc
        except (httpx.HTTPError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            raise ModelError("model_request_failed") from exc
        tail = visible.finish()
        if tail:
            parts.append(tail)
            if ttft_ms is None:
                ttft_ms = round((perf_counter() - started) * 1000, 2)
            yield {"type": "token", "text": tail}
        full = "".join(parts).strip()
        if not full:
            raise ModelError("invalid_model_json")
        if finish_reason not in {None, "stop"}:
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


def _read_json_completion(payload: Any) -> tuple[str, str | None, str | None, dict[str, float]]:
    if not isinstance(payload, dict) or not isinstance(payload.get("choices"), list):
        raise ModelError("invalid_model_json")
    choice = payload["choices"][0]
    if not isinstance(choice, dict) or not isinstance(choice.get("message"), dict):
        raise ModelError("invalid_model_json")
    content = choice["message"].get("content")
    if not isinstance(content, str):
        raise ModelError("invalid_model_json")
    model = payload.get("model") if isinstance(payload.get("model"), str) else None
    finish = choice.get("finish_reason") if isinstance(choice.get("finish_reason"), str) else "stop"
    return content, model, finish, _usage(payload.get("usage"))
