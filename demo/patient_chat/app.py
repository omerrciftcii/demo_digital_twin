from __future__ import annotations

import json
import os
from collections.abc import Awaitable, Callable
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
DEFAULT_CCO_DIR = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "patient_centric_private"
    / "3_hasta_orijinal_ve_fact"
    / "orijinal_cco"
)
PATIENT_FILES = {
    "P01": "response_1791739336553.json",
    "P02": "38c788d204d3-v2.json",
    "P03": "461429d00de8-v2.json",
}
PATIENT_LABELS = {
    "P01": "Hasta 1",
    "P02": "Hasta 2",
    "P03": "Hasta 3",
}

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


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=12_000)


class ChatRequest(BaseModel):
    patient_id: str = Field(alias="patientId")
    question: str = Field(min_length=1, max_length=2_000)
    history: list[ChatMessage] = Field(default_factory=list, max_length=8)
    quick_reply: bool = Field(default=False, alias="quickReply")
    interaction_mode: Literal["stepwise", "multi_select"] = Field(
        default="stepwise", alias="interactionMode"
    )
    selected_options: list[str] = Field(default_factory=list, alias="selectedOptions", max_length=6)

    @field_validator("patient_id", "question")
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


def load_patient_records(source_root: Path = DEFAULT_CCO_DIR) -> dict[str, str]:
    records: dict[str, str] = {}
    for patient_id, file_name in PATIENT_FILES.items():
        path = source_root / file_name
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        text = payload.get("text")
        if isinstance(text, str) and text.strip():
            records[patient_id] = text
    return records


class DirectVllmClient:
    def __init__(self) -> None:
        self.base_url = os.getenv("DEMO_VLLM_BASE_URL", "http://127.0.0.1:18000").rstrip("/")
        self.model = os.getenv("DEMO_MODEL", "gemma4-26b-a4b")

    async def complete(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        started = perf_counter()
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(180.0, connect=10.0)) as client:
                response = await client.post(
                    f"{self.base_url}/v1/chat/completions",
                    json={
                        "model": self.model,
                        "messages": messages,
                        "temperature": 0.0,
                        "max_tokens": 900,
                        "stream": False,
                        "chat_template_kwargs": {"enable_thinking": False},
                    },
                )
            response.raise_for_status()
            payload = response.json()
            answer = payload["choices"][0]["message"]["content"]
            if not isinstance(answer, str) or not answer.strip():
                raise ValueError("empty_model_answer")
        except (httpx.HTTPError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError("model_request_failed") from exc
        return {
            "answer": answer,
            "model": payload.get("model", self.model),
            "latencyMs": round((perf_counter() - started) * 1000),
        }

    async def health(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                response = await client.get(f"{self.base_url}/v1/models")
            return response.status_code == 200
        except httpx.HTTPError:
            return False


def create_app(
    *,
    records: dict[str, str] | None = None,
    completion: Completion | None = None,
) -> FastAPI:
    patient_records = dict(records) if records is not None else load_patient_records()
    model_client = DirectVllmClient()
    complete = completion or model_client.complete

    application = FastAPI(title="MedicalAI Hasta Sohbeti", docs_url=None, redoc_url=None)
    application.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["localhost", "127.0.0.1", "[::1]", "testserver"],
    )

    @application.middleware("http")
    async def local_only(request: Request, call_next):
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
            {"id": patient_id, "label": PATIENT_LABELS[patient_id]}
            for patient_id in PATIENT_FILES
            if patient_id in patient_records
        ]

    @application.get("/api/health")
    async def health():
        model_ready = await model_client.health() if completion is None else True
        return {
            "status": "ready" if model_ready and patient_records else "unavailable",
            "modelReady": model_ready,
            "patientCount": len(patient_records),
        }

    @application.post("/api/chat")
    async def chat(body: ChatRequest):
        patient_text = patient_records.get(body.patient_id)
        if patient_text is None:
            raise HTTPException(404, "Hasta kaydı bulunamadı.")

        today = datetime.now().astimezone().strftime("%d.%m.%Y")
        messages: list[dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": patient_text},
        ]
        messages.extend(item.model_dump() for item in body.history[-6:])
        current_prompt = build_question_prompt(body, today)
        messages.append({"role": "user", "content": current_prompt})
        try:
            result = await complete(messages)
        except RuntimeError as exc:
            raise HTTPException(
                502,
                "Gemma bağlantısı kurulamadı. GPU servisini ve SSH tünelini kontrol edin.",
            ) from exc
        return result

    return application


app = create_app()
