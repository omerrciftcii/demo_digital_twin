from __future__ import annotations

import asyncio
import ipaddress
import json
import os
import re
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
from starlette.middleware.trustedhost import TrustedHostMiddleware

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STATIC_DIR = Path(__file__).parent / "static"
LEGACY_CCO_DIR = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "patient_centric_private"
    / "3_hasta_orijinal_ve_fact"
    / "orijinal_cco"
)
DEFAULT_CCO_DIR = PROJECT_ROOT / "data" / "cco"
LEGACY_PATIENT_FILES = (
    "response_1791739336553.json",
    "38c788d204d3-v2.json",
    "461429d00de8-v2.json",
)
PATIENT_FILES = {
    "P01": LEGACY_PATIENT_FILES[0],
    "P02": LEGACY_PATIENT_FILES[1],
    "P03": LEGACY_PATIENT_FILES[2],
}
PATIENT_LABELS = {patient_id: f"Hasta {index}" for index, patient_id in enumerate(PATIENT_FILES, 1)}

SYSTEM_PROMPT = """Türkçe, doğal ve hasta dostu yanıt ver; 250 kelimeyi geçme. Kişiye özel bilgiler yalnızca verilen hasta kaydından gelsin. Soruyla ilgisiz tanı, laboratuvar veya ölçümleri hiç anma. En yeni ilgili kaydı öncele; eski kaydın bugünkü durumu göstermediğini açıkça belirt ve onu halen süren tedavi sayma. Tek bir ölçüm veya geceden genel trend çıkarma; olumlu, olumsuz ve belirsiz bulguları birlikte değerlendir. Kayıtta bir veri türü varsa fakat soruyla ilişkili değilse "veri yok" deme; yalnızca soruyu açıklayan ilgili bir işaret olmadığını söyle. Wearable verisini yalnızca soruyla ilgili ve güncelse kullan, tanı gibi yorumlama. Genel tıbbi olasılıkları kişisel kayıtlardan açıkça ayır. Kesin tanı/neden, reçeteli ilaç veya doz önerme; eski tedaviyi sürdürmesini söyleme. Evidence ID, JSON ve teknik sistem dili kullanma."""

QUESTION_SUFFIX = """

Yanıtı şu sırayla kur:
1. Kısa ve doğrudan cevap; gerekiyorsa sık görülen genel olasılıklar.
2. "Son kayıtlarına göre" altında yalnızca doğrudan ilgili en güncel bilgiler.
3. "Şimdilik ne yapabilirsin?" altında güvenli genel adımlar.
4. Varsa acil uyarı işaretleri.
5. Yanıtı iyileştirecekse en sonda tek takip sorusu ve A), B), C) biçiminde 3-5 kısa seçenek. Seçenekleri ayrı soru cümleleri olarak değil, kullanıcının tıklayarak kendi yanıtı olarak gönderebileceği kısa ifadeler biçiminde yaz. Faydalı bir takip sorusu yoksa seçenek üretmek zorunda değilsin.

İlişkisiz kayıtları yanıta alma. Eski reçete veya tedaviyi bugün kullanmasını söyleme.
""".strip()

FOLLOW_UP_SELECTION_SUFFIX = """Bu mesaj, önceki yanıtındaki takip sorusuna kullanıcının seçtiği yanıttır. Önceki konuşma ile hasta kaydını birlikte değerlendir. Uzun ilk-yanıt şablonunu ve bölüm başlıklarını tekrarlama. Seçilen bilginin değerlendirmeyi nasıl değiştirdiğini veya neden değiştirmediğini 2-3 kısa, doğal cümlede açıkla. Yalnızca doğrudan ilgili klinik veya wearable bilgisini kullan; kişisel kayıtta bağlantı yoksa bunu zorlamadan söyle. Kesin tanı veya nedensellik kurma. Değerlendirme için gerçekten gerekli tek bir bilgi daha varsa en sonda kısa bir soru sorabilir ve 3-4 kısa A), B), C) yanıt seçeneği verebilirsin."""

MULTI_SELECT_QUESTION_SUFFIX = """

Yanıtı şu sırayla kur:
1. Kısa ve doğrudan cevap; gerekiyorsa sık görülen genel olasılıklar.
2. "Son kayıtlarına göre" altında yalnızca doğrudan ilgili en güncel bilgiler.
3. "Şimdilik ne yapabilirsin?" altında güvenli genel adımlar.
4. Varsa acil uyarı işaretleri.
5. En sonda, yanıtı kişiselleştirmek için kullanıcının aynı anda birden fazlasını seçebileceği tam 5 veya 6 kısa seçenek üret. Seçenekleri A), B), C), D), E) ve gerekiyorsa F) biçiminde ayrı satırlara yaz.

Seçenekler kullanıcının sorusuna ve verilen hasta kaydındaki ilgili klinik/wearable bağlama özel olsun. Kullanıcının kendisi hakkında işaretleyebileceği belirti, süre, şiddet, tetikleyici veya eşlik eden durum ifadeleri yaz; tanı, kesin neden ya da kayıtta zaten varmış gibi yeni bilgi yazma. Birlikte doğru olabilecek seçenekler üret. Uygunsa "Bunların hiçbiri" veya "Başka bir durum var" seçeneği ekle. İlişkisiz kayıtları yanıta alma. Eski reçete veya tedaviyi bugün kullanmasını söyleme.
""".strip()

MULTI_SELECTION_SUFFIX = """Bu mesaj, önceki yanıtındaki seçeneklerden kullanıcının aynı anda işaretlediği birden fazla bilgiyi içerir. Önceki konuşma, seçilen maddeler ve hasta kaydını birlikte değerlendir. İlk cümleyi "Seçimlerine göre, ... birlikte öne çıkıyor." yapısında kur; seçilen belirtileri doğal biçimde birlikte an ve bunları gerçekten değerlendirdiğini kullanıcıya açıkça hissettir. Bu ilk cümlede tanı koyma, neden-sonuç ilişkisi kurma veya verilerin bir sonucu kanıtladığını söyleme. Ardından seçimlerin değerlendirmeyi nasıl değiştirdiğini veya birbirini nasıl tamamladığını 2-4 kısa, doğal cümlede açıkla. Seçimleri yalnızca tekrarlama; kişisel kayıttaki ilgili bulgularla birlikte taşıdıkları anlamı anlat. Uzun ilk-yanıt şablonunu ve bölüm başlıklarını tekrarlama. Yanıt yalnızca birleşik değerlendirme metninden oluşsun; sonunda soru sorma, önceki seçenekleri tekrarlama ve A)-F) seçenek satırları üretme. Yalnızca doğrudan ilgili klinik veya wearable bilgisini kullan, kişisel kayıtta bağlantı yoksa bunu zorlamadan söyle. Seçimlerden biri acil uyarı işaretiyse bunu açık ve uygun öncelikle belirt. Kesin tanı veya nedensellik kurma."""


@dataclass(frozen=True)
class ModelSpec:
    id: str
    label: str
    short_label: str
    provider: Literal["local", "openai"]
    model: str
    endpoint_env: str
    model_env: str
    note: str
    featured: bool = False
    endpoint_default: str | None = None
    api_key_env: str | None = None

    @property
    def resolved_model(self) -> str:
        return os.getenv(self.model_env, self.model).strip() or self.model

    @property
    def endpoint(self) -> str | None:
        value = os.getenv(self.endpoint_env, self.endpoint_default or "").strip()
        return value.rstrip("/") or None


MODEL_SPECS = (
    ModelSpec(
        id="gpu-gemma4",
        label="Gemma 4 26B-A4B",
        short_label="GPU Gemma",
        provider="local",
        model="gemma4-26b-a4b",
        endpoint_env="DEMO_VLLM_BASE_URL",
        model_env="DEMO_GEMMA_MODEL",
        endpoint_default="http://127.0.0.1:18000",
        note="Birincil yerel model · Q4 · L40S 24 GB",
        featured=True,
    ),
    ModelSpec(
        id="openai-luna",
        label="OpenAI GPT-6 Luna",
        short_label="OpenAI Luna",
        provider="openai",
        model="gpt-6-luna",
        endpoint_env="OPENAI_BASE_URL",
        model_env="DEMO_OPENAI_MODEL",
        endpoint_default="https://api.openai.com/v1",
        api_key_env="OPENAI_API_KEY",
        note="Hızlı karşılaştırma · dış servis",
        featured=True,
    ),
    ModelSpec(
        id="gpu-qwen",
        label="Qwen 3.5 9B",
        short_label="Qwen 9B",
        provider="local",
        model="Qwen/Qwen3.5-9B",
        endpoint_env="DEMO_QWEN_BASE_URL",
        model_env="DEMO_QWEN_MODEL",
        note="Deney adayı · text-only · 24 GB uyumlu",
    ),
    ModelSpec(
        id="gpu-kimi",
        label="Kimi / Moonlight 16B-A3B",
        short_label="Kimi 16B",
        provider="local",
        model="moonshotai/Moonlight-16B-A3B-Instruct",
        endpoint_env="DEMO_KIMI_BASE_URL",
        model_env="DEMO_KIMI_MODEL",
        note="Deney adayı · Q4 quantization gerekli",
    ),
    ModelSpec(
        id="gpu-deepseek",
        label="DeepSeek R1 Distill Qwen 14B",
        short_label="DeepSeek 14B",
        provider="local",
        model="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        endpoint_env="DEMO_DEEPSEEK_BASE_URL",
        model_env="DEMO_DEEPSEEK_MODEL",
        note="Deney adayı · Q4 quantization gerekli",
    ),
    ModelSpec(
        id="gpu-nemotron",
        label="NVIDIA Nemotron Nano 9B v2",
        short_label="Nemotron 9B",
        provider="local",
        model="nvidia/NVIDIA-Nemotron-Nano-9B-v2",
        endpoint_env="DEMO_NEMOTRON_BASE_URL",
        model_env="DEMO_NEMOTRON_MODEL",
        note="Deney adayı · NVIDIA GPU için optimize",
    ),
)
MODEL_BY_ID = {spec.id: spec for spec in MODEL_SPECS}


@dataclass(frozen=True)
class PatientRecord:
    id: str
    label: str
    text: str
    payload: dict[str, Any]
    summary: tuple[str, ...]


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=12_000)


class ChatRequest(BaseModel):
    patient_id: str = Field(alias="patientId")
    model_id: str = Field(default="gpu-gemma4", alias="modelId")
    question: str = Field(min_length=1, max_length=2_000)
    history: list[ChatMessage] = Field(default_factory=list, max_length=8)
    quick_reply: bool = Field(default=False, alias="quickReply")
    interaction_mode: Literal["stepwise", "multi_select"] = Field(
        default="stepwise", alias="interactionMode"
    )
    selected_options: list[str] = Field(default_factory=list, alias="selectedOptions", max_length=6)

    @field_validator("patient_id", "model_id", "question")
    @classmethod
    def strip_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("blank_value")
        return value

    @field_validator("selected_options")
    @classmethod
    def clean_selected_options(cls, values: list[str]) -> list[str]:
        cleaned = [value.strip() for value in values]
        if any(not value or len(value) > 300 for value in cleaned):
            raise ValueError("invalid_selected_option")
        return cleaned


def build_question_prompt(body: ChatRequest, today: str) -> str:
    if body.interaction_mode == "multi_select" and body.selected_options:
        selected = "\n".join(f"- {option}" for option in body.selected_options)
        return (
            f"Bugünün tarihi {today}.\n"
            "Kullanıcının önceki seçeneklerden birlikte işaretlediği bilgiler:\n"
            f"{selected}\n\n{MULTI_SELECTION_SUFFIX}"
        )
    if body.quick_reply:
        return (
            f"Bugünün tarihi {today}.\n"
            "Kullanıcının önceki takip sorusuna seçtiği yanıt: "
            f"{body.question}\n\n{FOLLOW_UP_SELECTION_SUFFIX}"
        )
    suffix = (
        MULTI_SELECT_QUESTION_SUFFIX if body.interaction_mode == "multi_select" else QUESTION_SUFFIX
    )
    return f"Bugünün tarihi {today}.\nHasta sorusu: {body.question}\n\n{suffix}"


Completion = Callable[[list[dict[str, str]]], Awaitable[dict[str, Any]]]


def _extract_text(payload: Mapping[str, Any]) -> str | None:
    candidates: list[Mapping[str, Any]] = [payload]
    for key in ("document", "Document"):
        value = payload.get(key)
        if isinstance(value, Mapping):
            candidates.append(value)
    for candidate in candidates:
        for key in ("text", "Text"):
            value = candidate.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _clean_summary_line(value: str) -> str:
    value = re.sub(r"^#{1,6}\s+", "", value.strip())
    value = re.sub(r"^[-*]\s+", "", value)
    value = re.sub(r"\*\*(.+?)\*\*", r"\1", value)
    return re.sub(r"\s+", " ", value).strip()


def build_clinical_summary(text: str, limit: int = 4) -> tuple[str, ...]:
    keywords = (
        "tanı",
        "kronik",
        "akut",
        "laboratuvar",
        "ilaç",
        "muayene",
        "epikriz",
        "uyarı",
        "son ",
        "yüksek",
        "düşük",
        "risk",
    )
    candidates: list[tuple[int, int, str]] = []
    for index, raw_line in enumerate(text.splitlines()):
        line = _clean_summary_line(raw_line)
        if len(line) < 28 or len(line) > 280:
            continue
        lowered = line.casefold()
        score = sum(2 for keyword in keywords if keyword in lowered)
        score += min(3, len(re.findall(r"\d", line)) // 3)
        if raw_line.lstrip().startswith("##"):
            score += 2
        if raw_line.lstrip().startswith("-"):
            score += 1
        candidates.append((score, -index, line))
    selected: list[tuple[int, str]] = []
    seen: set[str] = set()
    for _, negative_index, line in sorted(candidates, reverse=True):
        fingerprint = re.sub(r"\W+", "", line.casefold())[:80]
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        selected.append((-negative_index, line))
        if len(selected) == limit:
            break
    return tuple(line for _, line in sorted(selected))


def _explicit_summary(payload: Mapping[str, Any]) -> tuple[str, ...]:
    for key in ("clinicalSummary", "clinical_summary", "summary"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return tuple(
                line
                for raw in value.splitlines()
                if (line := _clean_summary_line(raw))
            )[:4]
        if isinstance(value, list):
            return tuple(str(item).strip() for item in value if str(item).strip())[:4]
    return ()


def _source_directories(source_root: Path | None) -> list[Path]:
    if source_root is not None:
        return [source_root]
    configured = os.getenv("DEMO_CCO_DIR", "").strip()
    if configured:
        return [Path(configured).expanduser()]
    return [DEFAULT_CCO_DIR, LEGACY_CCO_DIR]


def load_patient_payloads(source_root: Path | None = None) -> dict[str, dict[str, Any]]:
    files: list[Path] = []
    for directory in _source_directories(source_root):
        if not directory.is_dir():
            continue
        discovered = {path.name: path for path in directory.glob("*.json") if path.is_file()}
        ordered = [discovered.pop(name) for name in LEGACY_PATIENT_FILES if name in discovered]
        files = ordered + sorted(discovered.values(), key=lambda path: path.name.casefold())
        if files:
            break

    records: dict[str, dict[str, Any]] = {}
    for index, path in enumerate(files[:99], start=1):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict) or not _extract_text(payload):
            continue
        records[f"P{index:02d}"] = payload
    return records


def load_patient_records(source_root: Path | None = None) -> dict[str, str]:
    """Compatibility loader for the existing mobile API."""
    return {
        patient_id: text
        for patient_id, payload in load_patient_payloads(source_root).items()
        if (text := _extract_text(payload)) is not None
    }


def prepare_patient_records(records: Mapping[str, str | Mapping[str, Any]]) -> dict[str, PatientRecord]:
    prepared: dict[str, PatientRecord] = {}
    for index, (patient_id, value) in enumerate(records.items(), start=1):
        payload = {"text": value} if isinstance(value, str) else dict(value)
        text = _extract_text(payload)
        if not text:
            continue
        summary = _explicit_summary(payload) or build_clinical_summary(text)
        prepared[patient_id] = PatientRecord(
            id=patient_id,
            label=f"Hasta {index}",
            text=text,
            payload=payload,
            summary=summary,
        )
    return prepared


REFERENCE_STOPWORDS = {
    "acaba",
    "ancak",
    "bana",
    "benim",
    "bile",
    "bir",
    "bunu",
    "çok",
    "daha",
    "gibi",
    "göre",
    "için",
    "ile",
    "ilgili",
    "kadar",
    "kayıt",
    "olan",
    "olarak",
    "son",
    "şimdi",
    "var",
    "veya",
    "yalnızca",
}


def _tokens(value: str) -> set[str]:
    return {
        token.casefold()
        for token in re.findall(r"[^\W_]{3,}|\d+(?:[.,]\d+)?", value, flags=re.UNICODE)
        if token.casefold() not in REFERENCE_STOPWORDS
    }


def build_source_references(answer: str, source_text: str, limit: int = 4) -> list[dict[str, Any]]:
    answer_parts = [
        _clean_summary_line(part)
        for part in re.split(r"\n\s*\n|(?<=[.!?])\s+(?=[A-ZÇĞİÖŞÜ])", answer)
    ]
    answer_parts = [part for part in answer_parts if len(part) >= 20]
    source_parts = [
        (match.start(), match.end(), match.group(0).strip())
        for match in re.finditer(r"\S(?:.*?\S)?(?=\n\s*\n|\Z)", source_text, flags=re.DOTALL)
        if len(match.group(0).strip()) >= 20
    ]
    matches: list[tuple[float, int, int, str, str]] = []
    for answer_part in answer_parts:
        answer_tokens = _tokens(answer_part)
        if not answer_tokens:
            continue
        answer_numbers = set(re.findall(r"\d+(?:[.,]\d+)?", answer_part))
        for start, end, source_part in source_parts:
            source_tokens = _tokens(source_part)
            common = answer_tokens & source_tokens
            number_hits = answer_numbers & set(re.findall(r"\d+(?:[.,]\d+)?", source_part))
            if len(common) < 2 and not (number_hits and common):
                continue
            score = len(common) / max(4, min(len(answer_tokens), 12))
            score += min(0.35, len(number_hits) * 0.15)
            if score >= 0.2:
                matches.append((score, start, end, answer_part, source_part))

    references: list[dict[str, Any]] = []
    used_offsets: set[int] = set()
    for score, start, end, answer_part, source_part in sorted(matches, reverse=True):
        if start in used_offsets:
            continue
        used_offsets.add(start)
        references.append(
            {
                "answerText": answer_part[:180],
                "sourceText": source_part[:260],
                "start": start,
                "end": end,
                "score": round(min(score, 1.0), 2),
            }
        )
        if len(references) == limit:
            break
    return references


def _truthy(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().casefold() in {"1", "true", "yes", "on"}


class ModelRouter:
    def __init__(self) -> None:
        self.allow_external = _truthy("DEMO_ALLOW_EXTERNAL_MODELS")

    async def complete(self, model_id: str, messages: list[dict[str, str]]) -> dict[str, Any]:
        spec = MODEL_BY_ID.get(model_id)
        if spec is None:
            raise HTTPException(422, "Bilinmeyen model seçimi.")
        endpoint = spec.endpoint
        if endpoint is None:
            raise HTTPException(503, f"{spec.short_label} için sunucu adresi yapılandırılmadı.")

        headers = {"Content-Type": "application/json"}
        payload: dict[str, Any] = {
            "model": spec.resolved_model,
            "messages": messages,
            "stream": False,
        }
        if spec.provider == "openai":
            if not self.allow_external:
                raise HTTPException(403, "Dış model aktarımı sunucu ayarlarında kapalı.")
            api_key = os.getenv(spec.api_key_env or "", "").strip()
            if not api_key:
                raise HTTPException(503, "OpenAI API anahtarı yapılandırılmadı.")
            headers["Authorization"] = f"Bearer {api_key}"
            payload.update({"max_completion_tokens": 900, "reasoning_effort": "low"})
        else:
            local_key = os.getenv("DEMO_LOCAL_API_KEY", "").strip()
            if local_key:
                headers["Authorization"] = f"Bearer {local_key}"
            payload.update(
                {
                    "temperature": 0.0,
                    "max_tokens": 900,
                    "chat_template_kwargs": {"enable_thinking": False},
                }
            )

        started = perf_counter()
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(180.0, connect=10.0)) as client:
                response = await client.post(
                    f"{endpoint}/v1/chat/completions",
                    headers=headers,
                    json=payload,
                )
            response.raise_for_status()
            response_payload = response.json()
            answer = response_payload["choices"][0]["message"]["content"]
            if not isinstance(answer, str) or not answer.strip():
                raise ValueError("empty_model_answer")
        except (httpx.HTTPError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError("model_request_failed") from exc
        return {
            "answer": answer,
            "model": response_payload.get("model", spec.resolved_model),
            "modelId": spec.id,
            "latencyMs": round((perf_counter() - started) * 1000),
        }

    async def status(self, spec: ModelSpec) -> str:
        if spec.provider == "openai":
            if not self.allow_external:
                return "locked"
            if not os.getenv(spec.api_key_env or "", "").strip():
                return "setup"
            return "ready"
        if spec.endpoint is None:
            return "setup"
        try:
            async with httpx.AsyncClient(timeout=1.5) as client:
                response = await client.get(f"{spec.endpoint}/v1/models")
            return "ready" if response.status_code == 200 else "offline"
        except httpx.HTTPError:
            return "offline"

    async def catalog(self, mocked: bool = False) -> list[dict[str, Any]]:
        statuses = (
            ["ready"] * len(MODEL_SPECS)
            if mocked
            else await asyncio.gather(*(self.status(spec) for spec in MODEL_SPECS))
        )
        return [
            {
                "id": spec.id,
                "label": spec.label,
                "shortLabel": spec.short_label,
                "provider": "OpenAI" if spec.provider == "openai" else "GPU",
                "privacy": "external" if spec.provider == "openai" else "on-prem",
                "note": spec.note,
                "featured": spec.featured,
                "status": status,
            }
            for spec, status in zip(MODEL_SPECS, statuses, strict=True)
        ]


class DirectVllmClient:
    """Compatibility client used by the existing mobile demo."""

    def __init__(self) -> None:
        self.router = ModelRouter()
        self.model = MODEL_BY_ID["gpu-gemma4"].resolved_model

    async def complete(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        return await self.router.complete("gpu-gemma4", messages)

    async def health(self) -> bool:
        return await self.router.status(MODEL_BY_ID["gpu-gemma4"]) == "ready"


def _allowed_networks() -> tuple[ipaddress.IPv4Network | ipaddress.IPv6Network, ...]:
    raw = os.getenv("DEMO_ALLOWED_NETWORKS", "").strip()
    if not raw:
        return ()
    try:
        return tuple(ipaddress.ip_network(item.strip()) for item in raw.split(",") if item.strip())
    except ValueError as exc:
        raise RuntimeError("DEMO_ALLOWED_NETWORKS contains an invalid CIDR") from exc


def create_app(
    *,
    records: Mapping[str, str | Mapping[str, Any]] | None = None,
    completion: Completion | None = None,
) -> FastAPI:
    patient_records = prepare_patient_records(records if records is not None else load_patient_payloads())
    model_router = ModelRouter()
    allowed_networks = _allowed_networks()
    allowed_hosts = [
        item.strip()
        for item in os.getenv(
            "DEMO_ALLOWED_HOSTS", "localhost,127.0.0.1,[::1],testserver"
        ).split(",")
        if item.strip()
    ]

    application = FastAPI(title="MedicalAI Digital Twin Demo", docs_url=None, redoc_url=None)
    application.add_middleware(TrustedHostMiddleware, allowed_hosts=allowed_hosts)

    @application.middleware("http")
    async def office_only(request: Request, call_next):
        if allowed_networks and request.client and request.client.host != "testclient":
            try:
                remote_ip = ipaddress.ip_address(request.client.host)
            except ValueError:
                return JSONResponse({"detail": "Erişim ağı doğrulanamadı."}, status_code=403)
            if not any(remote_ip in network for network in allowed_networks):
                return JSONResponse({"detail": "Bu ağdan erişime izin verilmiyor."}, status_code=403)

        origin = request.headers.get("origin")
        if (
            origin
            and request.method not in {"GET", "HEAD", "OPTIONS"}
            and origin.rstrip("/") != str(request.base_url).rstrip("/")
        ):
            return JSONResponse({"detail": "Cross-origin writes are disabled"}, status_code=403)
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Frame-Options"] = "DENY"
        if request.url.path == "/" or request.url.path.startswith("/static/"):
            response.headers["Content-Security-Policy"] = (
                "default-src 'self'; script-src 'self'; style-src 'self'; "
                "connect-src 'self'; img-src 'self' data:; frame-ancestors 'none'; "
                "object-src 'none'; base-uri 'none'; form-action 'self'"
            )
        return response

    @application.exception_handler(RequestValidationError)
    async def validation_error(_: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=422,
            content={
                "detail": [{"loc": error["loc"], "type": error["type"]} for error in exc.errors()]
            },
        )

    @application.get("/", include_in_schema=False)
    async def home():
        return FileResponse(STATIC_DIR / "index.html")

    application.mount(
        "/static", StaticFiles(directory=STATIC_DIR, check_dir=False), name="demo-static"
    )

    @application.get("/api/patients")
    async def patients():
        return [
            {
                "id": record.id,
                "label": record.label,
                "summary": list(record.summary),
            }
            for record in patient_records.values()
        ]

    @application.get("/api/patients/{patient_id}/record")
    async def patient_record(patient_id: str):
        record = patient_records.get(patient_id)
        if record is None:
            raise HTTPException(404, "Hasta kaydı bulunamadı.")
        return {
            "id": record.id,
            "label": record.label,
            "summary": list(record.summary),
            "readableText": record.text,
            "rawJson": record.payload,
        }

    @application.get("/api/models")
    async def models():
        return await model_router.catalog(mocked=completion is not None)

    @application.get("/api/health")
    async def health():
        catalog = await model_router.catalog(mocked=completion is not None)
        ready_models = sum(model["status"] == "ready" for model in catalog)
        return {
            "status": "ready" if ready_models and patient_records else "unavailable",
            "modelReady": ready_models > 0,
            "readyModelCount": ready_models,
            "patientCount": len(patient_records),
        }

    @application.post("/api/chat")
    async def chat(body: ChatRequest):
        record = patient_records.get(body.patient_id)
        if record is None:
            raise HTTPException(404, "Hasta kaydı bulunamadı.")
        if body.model_id not in MODEL_BY_ID:
            raise HTTPException(422, "Bilinmeyen model seçimi.")

        today = datetime.now().astimezone().strftime("%d.%m.%Y")
        messages: list[dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": record.text},
        ]
        messages.extend(item.model_dump() for item in body.history[-6:])
        current_prompt = build_question_prompt(body, today)
        messages.append({"role": "user", "content": current_prompt})
        try:
            result = (
                await completion(messages)
                if completion is not None
                else await model_router.complete(body.model_id, messages)
            )
        except HTTPException:
            raise
        except RuntimeError as exc:
            raise HTTPException(
                502,
                "Seçilen model bağlantısı kurulamadı. Model servisini ve ağ tünelini kontrol edin.",
            ) from exc
        result.setdefault("modelId", body.model_id)
        result["references"] = build_source_references(result["answer"], record.text)
        return result

    return application


app = create_app()
