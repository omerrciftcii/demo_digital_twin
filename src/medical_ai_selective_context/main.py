import asyncio
import hashlib
import json
import os
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .evidence import build_evidence
from .graph import Neo4jTerminologyGraph, TerminologyGraphUnavailable
from .index import EvidenceIndex
from .llm import ModelClient, ModelError, local_plan
from .models import (
    CcoImportRequest,
    CcoSummary,
    ChatResponse,
    ContextPacket,
    DocumentSelectionRequest,
    SelectionRequest,
    TextAnswerRequest,
)
from .selector import select_context
from .store import CcoStore, content_hash, unwrap_and_validate_cco
from .terminology import catalog_from_glossaries
from .text_document import TextRecordError, TextRecordStore, loads_text_payload
from .text_llm import TextModelClient
from .text_qa import TextAnswerer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STATIC_DIR = Path(__file__).parent / "static"
SYNTHETIC_PATH = PROJECT_ROOT / "data" / "synthetic" / "cco-complete.json"
SYNTHETIC_SHA256 = "3db61f972c2c3aa7fffc26fa12df16375a84e5fc2d6706ddff758d849f1f35c2"

ERROR_MESSAGES = {
    "external_real_data_blocked": (
        "Gerçek hasta verisinin dış API'ye gönderimi kapalı. Yerel kanıt seçimi kullanılabilir."
    ),
    "api_key_missing": "OpenRouter anahtarı tanımlı değil. Yerel kanıt seçimi kullanılabilir.",
    "model_timeout": "Model isteği zaman aşımına uğradı. Kanıtlar yerel olarak korunuyor.",
    "model_request_failed": (
        "Model isteği tamamlanamadı. Anahtar, kredi ve sağlayıcının JSON/ZDR desteğini kontrol edin."
    ),
    "insufficient_evidence": "Eldeki CCO ile bu soruya yeterli ve bütçeye sığan kanıt bulunamadı.",
}


def _trusted_synthetic() -> dict[str, Any]:
    raw = SYNTHETIC_PATH.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SYNTHETIC_SHA256:
        raise ValueError("Bundled synthetic fixture integrity check failed")
    return unwrap_and_validate_cco(json.loads(raw))


class Services:
    def __init__(
        self, data_dir: Path, model: ModelClient, graph: Neo4jTerminologyGraph,
        embedding_model: str, load_sample: bool, text_model: TextModelClient,
    ) -> None:
        self.store = CcoStore(data_dir)
        self.index = EvidenceIndex(data_dir.parent / "runtime" / "evidence.sqlite3", embedding_model)
        self.model = model
        self.text_records = TextRecordStore(data_dir.parent / "text-records")
        self.text_model = text_model
        self.text_answers = TextAnswerer(text_model)
        self.graph = graph
        self.sample = _trusted_synthetic()
        # Only shipped, hash-verified public metadata enters the shared graph.
        # Arbitrary patient-provided narrative/coded display never enters Neo4j.
        self.catalog = catalog_from_glossaries([self.sample])
        self.load_sample = load_sample
        self.mutation_lock = asyncio.Lock()
        self.model_slots = asyncio.Semaphore(2)

    def reload(self) -> dict[str, Any]:
        result = self.store.reload()
        if self.load_sample:
            self.store.import_document(self.sample, data_class="synthetic")
        for item in self.store.list_summaries():
            snapshot = item["snapshotId"]
            document = self.store.get(snapshot)
            self.index.index_snapshot(snapshot, build_evidence(document, snapshot))
        result["loaded"] = self.store.count
        return result

    def document(self, snapshot_id: str) -> dict[str, Any]:
        document = self.store.get(snapshot_id)
        if document is None:
            raise HTTPException(404, "Snapshot bulunamadı; listeden snapshotId seçin.")
        return document

    def import_document(self, request: CcoImportRequest) -> dict[str, Any]:
        result = self.store.import_document(request.cco, request.file_name, data_class="real")
        snapshot = result["snapshotId"]
        self.index.index_snapshot(
            snapshot, build_evidence(self.store.get(snapshot), snapshot),
        )
        return result


def create_app(
    data_dir: Path | None = None, *, model_client: ModelClient | None = None,
    graph: Neo4jTerminologyGraph | None = None, embedding_model: str | None = None,
    load_sample: bool | None = None, text_model: TextModelClient | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(application: FastAPI):
        services = Services(
            data_dir or Path(os.getenv("CCO_DATA_DIR", str(PROJECT_ROOT / "data" / "cco"))),
            model_client or ModelClient(), graph or Neo4jTerminologyGraph(),
            embedding_model if embedding_model is not None else os.getenv(
                "EMBEDDING_MODEL", "intfloat/multilingual-e5-small",
            ),
            load_sample if load_sample is not None else (
                os.getenv("LOAD_SYNTHETIC_EXAMPLE", "true").lower() == "true"
            ),
            text_model or TextModelClient(),
        )
        application.state.services = services
        try:
            await run_in_threadpool(services.reload)
            await run_in_threadpool(services.text_records.load)
            try:
                await services.graph.initialize(services.catalog)
            except TerminologyGraphUnavailable:
                pass
            await services.model.check_capabilities()
            yield
        finally:
            await services.graph.close()
            services.index.close()
            services.store.close()

    application = FastAPI(
        title="MedicalAI Selective Context V2", version="0.3.0", lifespan=lifespan,
        description=(
            "Local, snapshot-bound CCO 0.7 evidence inspection. ContextPacket is not a CCO. "
            "Real-data external model calls are disabled by default. No production service calls."
        ),
    )
    application.add_middleware(
        TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1", "[::1]", "testserver"],
    )

    @application.middleware("http")
    async def local_security(request: Request, call_next):
        # No permissive CORS. Reject browser cross-origin writes before parsing PHI.
        origin = request.headers.get("origin")
        if (origin and request.method not in {"GET", "HEAD", "OPTIONS"}
                and origin.rstrip("/") != str(request.base_url).rstrip("/")):
            return JSONResponse({"detail": "Cross-origin writes are disabled"}, status_code=403)
        if request.method in {"POST", "PUT", "PATCH"}:
            length = request.headers.get("content-length")
            if length and (not length.isdigit() or int(length) > 2_000_000):
                return JSONResponse({"detail": "Request too large"}, status_code=413)
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
        # FastAPI's default error includes raw invalid input (possibly PHI).
        return JSONResponse(status_code=422, content={
            "detail": [{"loc": error["loc"], "type": error["type"]} for error in exc.errors()],
        })

    def services(request: Request) -> Services:
        return request.app.state.services

    @application.get("/", include_in_schema=False)
    async def home():
        return FileResponse(STATIC_DIR / "index.html")

    application.mount("/static", StaticFiles(directory=STATIC_DIR, check_dir=False), name="static")

    @application.get("/health")
    async def health(request: Request):
        svc = services(request)
        index_state = svc.index.status()
        return {
            "status": "ok" if svc.graph.available and index_state["status"] == "ready" else "degraded",
            "loadedSnapshots": svc.store.count,
            "textRecords": svc.text_records.count,
            "neo4j": svc.graph.status(), "index": index_state, "llm": svc.model.status(),
            "textLlm": svc.text_model.status(),
        }

    @application.get("/api/v1/ccos", response_model=list[CcoSummary])
    async def list_ccos(request: Request):
        return services(request).store.list_summaries()

    @application.post("/api/v1/ccos", response_model=CcoSummary, status_code=status.HTTP_201_CREATED)
    async def import_cco(body: CcoImportRequest, request: Request):
        svc = services(request)
        try:
            async with svc.mutation_lock:
                return await run_in_threadpool(svc.import_document, body)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @application.post("/api/v1/ccos/reload")
    async def reload_ccos(request: Request):
        svc = services(request)
        async with svc.mutation_lock:
            return await run_in_threadpool(svc.reload)

    @application.get("/api/v1/ccos/{snapshot_id}")
    async def get_cco(snapshot_id: str, request: Request):
        return services(request).document(snapshot_id)

    @application.get("/api/v1/terminology/status")
    async def terminology_status(request: Request):
        return services(request).graph.status()

    @application.post("/api/v1/terminology/reload")
    async def reload_terminology(request: Request):
        svc = services(request)
        try:
            return await svc.graph.initialize(svc.catalog)
        except TerminologyGraphUnavailable as exc:
            raise HTTPException(503, "Neo4j erişilemiyor; yerel arama kullanılabilir.") from exc

    async def packet_for(
        svc: Services, body: SelectionRequest, plan: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        document = svc.document(body.snapshot_id)
        started = perf_counter()
        search_plan = plan or local_plan(body.question)
        queries = list(search_plan["queries"])
        # Local-only inspection can still search the user's previous explicit terms.
        if plan is None and body.history:
            queries.extend(message.content for message in body.history[-2:] if message.role == "user")
        queries.extend(search_plan.get("concepts", []))
        graph_matches = []
        if svc.graph.available:
            try:
                graph_matches = await svc.graph.resolve(body.question)
            except TerminologyGraphUnavailable:
                pass
        for match in graph_matches:
            definition_id = match.get("id") or match.get("definitionId")
            if definition_id:
                queries.append(definition_id)
        ranked = await run_in_threadpool(svc.index.search, body.snapshot_id, queries)
        entries = await run_in_threadpool(svc.index.entries, body.snapshot_id)
        packet = await run_in_threadpool(
            select_context, document, body.question, entries, ranked,
            snapshot_id=body.snapshot_id, plan=search_plan, token_budget=body.token_budget,
            graph_status=svc.graph.status(),
        )
        packet["dataClass"] = svc.store.summary(body.snapshot_id)["dataClass"]
        packet["metrics"] = {"retrievalMs": round((perf_counter() - started) * 1000, 2)}
        packet["indexStatus"] = svc.index.status()
        packet["planner"] = "model" if plan is not None else "local_query"
        return packet

    @application.post("/api/v1/context/select", response_model=ContextPacket)
    async def select_loaded(body: SelectionRequest, request: Request):
        # This endpoint NEVER sends a question, history or patient record to a model.
        return await packet_for(services(request), body)

    @application.post("/api/v1/context/select-document", response_model=ContextPacket)
    async def select_document(body: DocumentSelectionRequest, request: Request):
        try:
            document = unwrap_and_validate_cco(body.cco)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        snapshot = "ephemeral_" + content_hash(document)
        entries = build_evidence(document, snapshot)
        # Isolated temporary in-memory index: raw-document endpoint does not persist PHI.
        index = EvidenceIndex(Path(":memory:"), embedding_model="")
        try:
            index.index_snapshot(snapshot, entries)
            ranked = index.search(snapshot, [body.question])
        finally:
            index.close()
        packet = select_context(
            document, body.question, entries, ranked, snapshot_id=snapshot,
            token_budget=body.token_budget, graph_status=services(request).graph.status(),
        )
        packet["dataClass"] = "real"
        return packet

    @application.post("/api/v1/chat", response_model=ChatResponse)
    async def chat(body: SelectionRequest, request: Request):
        svc = services(request)
        svc.document(body.snapshot_id)
        classification = svc.store.summary(body.snapshot_id)["dataClass"]
        history = [item.model_dump() for item in body.history]
        started = perf_counter()
        packet = None
        plan = None
        planner_metrics: dict[str, Any] = {}
        response: dict[str, Any] = {
            "answer": "", "citations": [], "metrics": {},
            "model": svc.model.settings.model, "provider": None, "validation": {},
        }
        try:
            svc.model.guard(classification)  # before *any* external planner or answer call
            async with svc.model_slots:
                try:
                    plan, planner_metrics = await svc.model.plan(
                        body.question, history, svc.catalog, classification,
                        datetime.now(UTC).isoformat(),
                    )
                except ModelError as exc:
                    # One bounded fallback to local search. No retry/model switch or raw body logging.
                    planner_metrics = {"error": str(exc), "fallback": "local_query"}
                packet = await packet_for(svc, body, plan)
                if plan and plan.get("needsClarification"):
                    response.update(
                        status="clarification_required",
                        answer=plan.get("clarificationQuestion") or "Sorunuzu biraz netleştirir misiniz?",
                    )
                elif not packet["canAnswer"]:
                    response.update(
                        status="insufficient_evidence", answer=ERROR_MESSAGES["insufficient_evidence"],
                    )
                else:
                    generated = await svc.model.answer(body.question, history, packet, classification)
                    response.update(
                        status="answered" if generated["validated"] else "answered_unvalidated",
                        answer=generated["answer"], citations=generated["citations"],
                        validation=generated["validation"],
                        provider=generated["metrics"].get("provider"),
                    )
                    response["metrics"]["answer"] = generated["metrics"]
        except ModelError as exc:
            code = str(exc)
            is_blocked = code == "external_real_data_blocked"
            invalid = code in {
                "citation_not_in_context", "value_citation_not_in_statement",
                "unsupported_value_field", "source_value_mismatch", "unsupported_numeric_claim",
                "invalid_answer_schema", "invalid_model_json", "unexpected_model",
                "model_response_incomplete",
            }
            response.update(
                status="blocked" if is_blocked else (
                    "validation_failed" if invalid else "model_unavailable"
                ),
                answer=ERROR_MESSAGES.get(
                    code, "Model yanıtı kaynak/biçim kontrollerinden geçmedi; yanıt gösterilmedi.",
                ),
                validation={"error": code, "clinicalMeaningVerified": False},
            )
        if packet is None:
            packet = await packet_for(svc, body)
        response["contextPacket"] = packet
        response["metrics"].update(
            planner=planner_metrics, retrievalMs=packet["metrics"]["retrievalMs"],
            totalMs=round((perf_counter() - started) * 1000, 2),
        )
        return response

    def _sse(event: str, data: dict[str, Any]) -> bytes:
        payload = json.dumps(data, ensure_ascii=False, allow_nan=False)
        return f"event: {event}\ndata: {payload}\n\n".encode()

    @application.get("/api/v1/text-records")
    async def list_text_records(request: Request):
        return services(request).text_records.list_summaries()

    @application.post("/api/v1/text-records")
    async def import_text_record(request: Request):
        svc = services(request)
        try:
            payload = loads_text_payload(await request.body())
            record, created = await run_in_threadpool(svc.text_records.import_payload, payload)
        except TextRecordError as exc:
            code = str(exc)
            raise HTTPException(409 if code == "record_id_conflict" else 422, code) from exc
        return JSONResponse(record.summary(), status_code=201 if created else 200)

    @application.get("/api/v1/text-records/{record_id}")
    async def get_text_record(record_id: str, request: Request):
        record = services(request).text_records.get(record_id.lower())
        if record is None:
            raise HTTPException(404, "Kayıt bulunamadı.")
        return record.public_document()

    @application.post("/api/v1/text/answer")
    async def answer_text(body: TextAnswerRequest, request: Request):
        svc = services(request)
        record = svc.text_records.get(body.record_id.lower())
        if record is None:
            raise HTTPException(404, "Kayıt bulunamadı.")
        if not body.stream:
            return await svc.text_answers.answer(record, body.question)

        async def events():
            async for event in svc.text_answers.events(record, body.question):
                name = "token" if event["type"] == "token" else "done"
                data = {key: value for key, value in event.items() if key != "type"}
                yield _sse(name, data)

        return StreamingResponse(events(), media_type="text/event-stream")

    return application


app = create_app()

