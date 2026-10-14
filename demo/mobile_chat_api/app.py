from __future__ import annotations

import hmac
import ipaddress
import os
from collections.abc import Sequence
from datetime import datetime
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware

from demo.patient_chat.app import (
    PATIENT_FILES,
    PATIENT_LABELS,
    QUESTION_SUFFIX,
    SYSTEM_PROMPT,
    ChatRequest,
    Completion,
    DirectVllmClient,
    load_patient_records,
)

DEFAULT_OFFICE_CIDRS = ("10.34.14.0/23", "127.0.0.0/8")
DEFAULT_ALLOWED_HOSTS = ("10.34.14.76", "127.0.0.1", "localhost", "testserver")

FOLLOW_UP_SELECTION_SUFFIX = """Bu mesaj, önceki yanıtındaki takip sorusuna kullanıcının seçtiği yanıttır. Önceki konuşma ile hasta kaydını birlikte değerlendir. Uzun ilk-yanıt şablonunu ve bölüm başlıklarını tekrarlama. Seçilen bilginin değerlendirmeyi nasıl değiştirdiğini veya neden değiştirmediğini 2-3 kısa, doğal cümlede açıkla. Yalnızca doğrudan ilgili klinik veya wearable bilgisini kullan; kişisel kayıtta bağlantı yoksa bunu zorlamadan söyle. Kesin tanı veya nedensellik kurma. Değerlendirme için gerçekten gerekli tek bir bilgi daha varsa en sonda kısa bir soru sorabilir ve 3-4 kısa A), B), C) seçeneği verebilirsin."""


class MobileChatRequest(ChatRequest):
    quick_reply: bool = Field(default=False, alias="quickReply")


class PatientSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    label: str


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str
    modelReady: bool
    patientCount: int


class ChatResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str
    model: str
    latencyMs: int


def _csv_env(name: str, fallback: Sequence[str]) -> tuple[str, ...]:
    raw = os.getenv(name)
    if not raw:
        return tuple(fallback)
    values = tuple(value.strip() for value in raw.split(",") if value.strip())
    return values or tuple(fallback)


def _parse_networks(
    values: Sequence[str],
) -> tuple[ipaddress.IPv4Network | ipaddress.IPv6Network, ...]:
    networks: list[ipaddress.IPv4Network | ipaddress.IPv6Network] = []
    for value in values:
        networks.append(ipaddress.ip_network(value, strict=True))
    return tuple(networks)


def _client_is_allowed(
    request: Request,
    networks: Sequence[ipaddress.IPv4Network | ipaddress.IPv6Network],
) -> bool:
    if request.client is None:
        return False
    try:
        address = ipaddress.ip_address(request.client.host)
    except ValueError:
        return False
    return any(address.version == network.version and address in network for network in networks)


def create_mobile_app(
    *,
    records: dict[str, str] | None = None,
    completion: Completion | None = None,
    access_token: str | None = None,
    allowed_cidrs: Sequence[str] | None = None,
    allowed_hosts: Sequence[str] | None = None,
) -> FastAPI:
    patient_records = dict(records) if records is not None else load_patient_records()
    model_client = DirectVllmClient()
    complete = completion or model_client.complete
    demo_token = access_token if access_token is not None else os.getenv("MOBILE_DEMO_TOKEN", "")
    networks = _parse_networks(
        allowed_cidrs
        if allowed_cidrs is not None
        else _csv_env("MOBILE_ALLOWED_CIDRS", DEFAULT_OFFICE_CIDRS)
    )
    hosts = tuple(
        allowed_hosts
        if allowed_hosts is not None
        else _csv_env("MOBILE_ALLOWED_HOSTS", DEFAULT_ALLOWED_HOSTS)
    )

    application = FastAPI(
        title="MedicalAI Mobile Demo API",
        version="0.1.0",
        docs_url=None,
        redoc_url=None,
    )
    application.add_middleware(TrustedHostMiddleware, allowed_hosts=list(hosts))

    @application.middleware("http")
    async def office_network_and_token(request: Request, call_next):
        if request.url.path.startswith("/api/"):
            if not _client_is_allowed(request, networks):
                return JSONResponse(
                    status_code=403,
                    content={"detail": "Bu API yalnızca ofis ağı içinden kullanılabilir."},
                )
            if not demo_token:
                return JSONResponse(
                    status_code=503,
                    content={"detail": "Mobil demo erişim kodu yapılandırılmamış."},
                )
            supplied = request.headers.get("x-demo-token", "")
            if not hmac.compare_digest(supplied, demo_token):
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Geçersiz mobil demo erişim kodu."},
                )

        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Frame-Options"] = "DENY"
        return response

    @application.exception_handler(RequestValidationError)
    async def validation_error(_: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=422,
            content={
                "detail": [{"loc": error["loc"], "type": error["type"]} for error in exc.errors()]
            },
        )

    @application.get("/api/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        model_ready = await model_client.health() if completion is None else True
        return HealthResponse(
            status="ready" if model_ready and patient_records else "unavailable",
            modelReady=model_ready,
            patientCount=len(patient_records),
        )

    @application.get("/api/patients", response_model=list[PatientSummary])
    async def patients() -> list[PatientSummary]:
        return [
            PatientSummary(id=patient_id, label=PATIENT_LABELS[patient_id])
            for patient_id in PATIENT_FILES
            if patient_id in patient_records
        ]

    @application.post("/api/chat", response_model=ChatResponse)
    async def chat(body: MobileChatRequest) -> ChatResponse:
        patient_text = patient_records.get(body.patient_id)
        if patient_text is None:
            raise HTTPException(404, "Hasta kaydı bulunamadı.")

        today = datetime.now().astimezone().strftime("%d.%m.%Y")
        messages: list[dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": patient_text},
        ]
        messages.extend(item.model_dump() for item in body.history[-6:])
        if body.quick_reply:
            current_prompt = (
                f"Bugünün tarihi {today}.\n"
                "Kullanıcının önceki takip sorusuna seçtiği yanıt: "
                f"{body.question}\n\n{FOLLOW_UP_SELECTION_SUFFIX}"
            )
        else:
            current_prompt = (
                f"Bugünün tarihi {today}.\nHasta sorusu: {body.question}\n\n{QUESTION_SUFFIX}"
            )
        messages.append({"role": "user", "content": current_prompt})
        try:
            result: dict[str, Any] = await complete(messages)
        except RuntimeError as exc:
            raise HTTPException(502, "Gemma bağlantısı kurulamadı.") from exc
        return ChatResponse(
            answer=str(result["answer"]),
            model=str(result.get("model", model_client.model)),
            latencyMs=int(result.get("latencyMs", 0)),
        )

    return application


app = create_mobile_app()
