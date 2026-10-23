"""V3 real-CCO question pilot and full-context Oracle A/B/C contracts.

Question suggestions and predicted evidence are candidates only.  Human oracle
annotations remain a separate, reviewed artifact.  Source CCO files are read
but never rewritten or copied into the pilot output.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

import httpx
from pydantic import Field, model_validator

from .hybrid_synthetic import OpenAIDraftError
from .llm import strict_output_schema
from .synthetic_training import SyntheticModel
from .text_document import (
    EvidenceSpan,
    build_evidence_spans,
    extract_explicit_claims,
    parse_cco_text,
)
from .text_qa import FullContextPlan, build_full_context
from .text_relevance import question_terms, score_text

QUESTION_DRAFT_VERSION = "cco-oracle-question-drafts/0.2.0"
ORACLE_PILOT_VERSION = "cco-oracle-pilot/0.2.0"
ORACLE_REVIEW_VERSION = "cco-oracle-evidence-review/0.1.0"
ORACLE_ABC_VERSION = "cco-oracle-abc/0.2.0"

QuestionCategory = Literal[
    "headache", "fatigue", "gastrointestinal", "sleep", "palpitations",
    "medication", "daily_life", "acute_safety", "diagnosis_status", "general_symptom",
]
EvidenceAvailability = Literal["supported", "partial", "not_supported"]

_PII_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("email", re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)),
    ("url", re.compile(r"\b(?:https?://|www\.)\S+", re.IGNORECASE)),
    ("ipv4", re.compile(r"(?<!\d)(?:\d{1,3}\.){3}\d{1,3}(?!\d)")),
    ("turkish_phone", re.compile(r"(?<!\d)(?:\+?90\s*)?0?5\d{2}[\s().-]*\d{3}[\s.-]*\d{2}[\s.-]*\d{2}(?!\d)")),
    ("tckn", re.compile(r"(?<!\d)[1-9]\d{10}(?!\d)")),
    ("iban", re.compile(r"\bTR\d{2}(?:\s?\d{4}){5}\s?\d{2}\b", re.IGNORECASE)),
    ("explicit_name_field", re.compile(r"(?im)^\s*(?:hasta\s+ad[ıi]|ad[ıi]\s+soyad[ıi]|isim)\s*[:：]")),
    ("explicit_birth_date", re.compile(r"(?i)\b(?:doğum|dogum)\s+tarih[ıi]\s*[:：]")),
    ("explicit_address", re.compile(r"(?im)^\s*(?:adres|ikamet)\s*[:：]")),
    ("record_number", re.compile(r"(?i)\b(?:protokol|dosya|hasta)\s*(?:no|numarası|numarasi)\s*[:：]")),
    ("clinician_name", re.compile(r"\b(?:Dr\.?|Doktor)\s+[A-ZÇĞİÖŞÜ][a-zçğıöşü]+\s+[A-ZÇĞİÖŞÜ][a-zçğıöşü]+")),
    ("institution_name", re.compile(r"\b[A-ZÇĞİÖŞÜ][\wÇĞİÖŞÜçğıöşü.-]+(?:\s+[A-ZÇĞİÖŞÜ][\wÇĞİÖŞÜçğıöşü.-]+){0,4}\s+(?:Hastanesi|Tıp Merkezi|Kliniği)\b")),
)

_CATEGORY_SLOTS: tuple[tuple[QuestionCategory, EvidenceAvailability, str], ...] = (
    ("headache", "supported", r"baş ağr|migren"),
    ("headache", "partial", r"baş ağr|migren"),
    ("fatigue", "supported", r"halsiz|yorgun"),
    ("fatigue", "partial", r"halsiz|yorgun"),
    ("gastrointestinal", "supported", r"mide|bulant|reflü|karın"),
    ("gastrointestinal", "partial", r"mide|bulant|reflü|karın"),
    ("sleep", "supported", r"## Uyku"),
    ("sleep", "supported", r"## Uyku"),
    ("palpitations", "supported", r"çarpınt|nabız|## Kalp"),
    ("palpitations", "partial", r"çarpınt|nabız|## Kalp"),
    ("medication", "supported", r"## İlaçlar|## Kullandığı ilaçlar"),
    ("medication", "supported", r"## İlaçlar|## Kullandığı ilaçlar"),
    ("medication", "partial", r"## İlaçlar|## Kullandığı ilaçlar"),
    ("daily_life", "supported", r"## Hareket|günlük|egzersiz"),
    ("daily_life", "partial", r"## Hareket|günlük|egzersiz"),
    ("acute_safety", "supported", r"göğüs|nefes darl|bayıl|senkop"),
    ("acute_safety", "partial", r"göğüs|nefes darl|bayıl|senkop"),
    ("diagnosis_status", "supported", r"ön tanı"),
    ("general_symptom", "not_supported", r""),
    ("general_symptom", "not_supported", r""),
)


class OracleQuestionDraft(SyntheticModel):
    record_index: int = Field(alias="recordIndex", ge=1, le=20)
    question: str = Field(min_length=8, max_length=600)
    category: QuestionCategory
    evidence_availability: EvidenceAvailability = Field(alias="evidenceAvailability")
    evidence_line_numbers: list[int] = Field(
        alias="evidenceLineNumbers", min_length=0, max_length=8
    )
    irrelevant_line_numbers: list[int] = Field(
        alias="irrelevantLineNumbers", min_length=1, max_length=4
    )
    user_introduces_new_symptom: bool = Field(alias="userIntroducesNewSymptom")
    rationale: str = Field(min_length=5, max_length=500)
    forbidden_inferences: list[str] = Field(
        alias="forbiddenInferences", min_length=1, max_length=6
    )

    @model_validator(mode="after")
    def coherent_evidence(self) -> OracleQuestionDraft:
        self.question = self.question.strip()
        if self.evidence_availability == "not_supported" and self.evidence_line_numbers:
            raise ValueError("not_supported questions cannot cite CCO lines")
        if self.evidence_availability != "not_supported" and not self.evidence_line_numbers:
            raise ValueError("supported questions need candidate evidence lines")
        if len(self.evidence_line_numbers) != len(set(self.evidence_line_numbers)):
            raise ValueError("candidate evidence lines must be unique")
        if len(self.irrelevant_line_numbers) != len(set(self.irrelevant_line_numbers)):
            raise ValueError("candidate irrelevant lines must be unique")
        if set(self.evidence_line_numbers).intersection(self.irrelevant_line_numbers):
            raise ValueError("relevant and irrelevant candidate lines must not overlap")
        return self


class OracleQuestionBatch(SyntheticModel):
    schema_version: Literal[QUESTION_DRAFT_VERSION] = Field(alias="schemaVersion")
    questions: list[OracleQuestionDraft] = Field(min_length=20, max_length=20)

    @model_validator(mode="after")
    def one_per_record(self) -> OracleQuestionBatch:
        indexes = [item.record_index for item in self.questions]
        if sorted(indexes) != list(range(1, 21)):
            raise ValueError("question batch must contain each record index exactly once")
        return self


@dataclass(frozen=True)
class SelectedRecord:
    index: int
    source_path: Path
    record: Any
    requested_category: QuestionCategory
    requested_availability: EvidenceAvailability


def scan_identifiers(text: str) -> list[dict[str, Any]]:
    """Return locations and hashes only; never copy a possible identifier."""
    findings = []
    for kind, pattern in _PII_RULES:
        for match in pattern.finditer(text):
            findings.append({
                "kind": kind,
                "line": text.count("\n", 0, match.start()) + 1,
                "matchSha256": hashlib.sha256(match.group(0).encode("utf-8")).hexdigest(),
            })
    return findings


def audit_cco_directory(cco_dir: Path) -> tuple[list[dict[str, Any]], list[tuple[Path, Any]]]:
    rows: list[dict[str, Any]] = []
    eligible: list[tuple[Path, Any]] = []
    for path in sorted(Path(cco_dir).glob("*.json")):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            record = parse_cco_text(raw)
            findings = scan_identifiers(record.text)
            contract_valid = set(raw) <= {"id", "subjectId", "asOf", "text", "format", "language"}
            status = "eligible" if contract_valid and not findings else "blocked"
            if status == "eligible":
                eligible.append((path, record))
            rows.append({
                "sourceFile": path.name,
                "contractValid": contract_valid,
                "identifierFindings": findings,
                "status": status,
                "bytes": len(record.text.encode("utf-8")),
                "sectionCount": len(record.sections),
            })
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            rows.append({
                "sourceFile": path.name,
                "contractValid": False,
                "identifierFindings": [],
                "status": "invalid",
                "error": type(exc).__name__,
            })
    return rows, eligible


def select_oracle_pilot_records(
    eligible: list[tuple[Path, Any]], *, count: int = 20
) -> list[SelectedRecord]:
    if count != 20:
        raise ValueError("V3 oracle pilot requires exactly 20 records")
    ordered = sorted(eligible, key=lambda item: (len(item[1].text), item[0].name))
    selected: list[SelectedRecord] = []
    used: set[Path] = set()
    for category, availability, pattern in _CATEGORY_SLOTS:
        candidates = [
            item for item in ordered
            if item[0] not in used
            and (not pattern or re.search(pattern, item[1].text, re.IGNORECASE))
        ]
        if availability == "not_supported":
            candidates.sort(key=lambda item: (len(item[1].text), item[0].name))
        if not candidates:
            candidates = [item for item in ordered if item[0] not in used]
        if not candidates:
            raise ValueError("not_enough_distinct_eligible_ccos")
        path, record = candidates[len(selected) % len(candidates)]
        used.add(path)
        selected.append(SelectedRecord(
            len(selected) + 1, path, record, category, availability
        ))
    return selected


def _line_numbered(text: str) -> str:
    return "\n".join(f"L{index}: {line}" for index, line in enumerate(text.splitlines(), 1))


def build_openrouter_question_request(
    selected: list[SelectedRecord], *, model: str, seed: int, provider_slug: str
) -> dict[str, Any]:
    if provider_slug not in {"azure", "amazon-bedrock"}:
        raise ValueError("real CCO requests require an approved zero-retention provider")
    schema = strict_output_schema(OracleQuestionBatch.model_json_schema(by_alias=True))
    records = []
    for item in selected:
        spans = build_evidence_spans(item.record)
        selectable = sorted({
            line
            for span in spans
            for line in range(span.start_line, span.end_line + 1)
        } | {
            section.start_line
            for section in item.record.sections
            if section.heading != "Giriş"
        })
        records.append({
            "recordIndex": item.index,
            "asOf": item.record.as_of,
            "requestedCategory": item.requested_category,
            "requestedEvidenceAvailability": item.requested_availability,
            "selectableEvidenceLineNumbers": selectable,
            "lineNumberedText": _line_numbered(item.record.text),
        })
    system = (
        "Anonim Türkçe sağlık geçmişlerinden doğal kullanıcı soruları öner. Sorular gold değildir. "
        "Kullanıcı tüm geçmişini biliyormuş gibi yazma. Yeni bildirilen semptomu geçmiş kayda "
        "dönüştürme. Salt değer/tanı sorgusu yerine günlük hayatta sorulan yorum, güvenlik, ilaç "
        "ve yaşam soruları üret. evidenceLineNumbers yalnız verilen L numaralarıdır; "
        "not_supported için boş olmalıdır. irrelevantLineNumbers soruyu yanıtlamak için "
        "kullanılmaması gereken, görünüşte cazip olabilecek CCO satırlarıdır. İki liste "
        "çakışamaz. İki listeye de yalnız selectableEvidenceLineNumbers içindeki numaraları "
        "yaz. Bunlar gold değil, insan incelemesi için adaylardır."
    )
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps({
                "seed": seed, "records": records,
                "instruction": (
                    "Her recordIndex için tam bir soru üret. requestedCategory ve "
                    "requestedEvidenceAvailability deney tasarımıdır; bunları klinik yorumuna "
                    "göre değiştirme. Çıktıdaki category ve evidenceAvailability alanlarına "
                    "girdideki değerleri harfiyen kopyala. not_supported ise evidenceLineNumbers "
                    "kesinlikle boş kalır; diğer iki sınıfta en az bir seçilebilir satır olur."
                ),
            }, ensure_ascii=False)},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "oracle_question_candidates",
                "strict": True,
                "schema": schema,
            },
        },
        "reasoning": {"enabled": False},
        "max_tokens": 16000,
        "seed": seed,
        "stream": False,
        "provider": {
            "only": [provider_slug],
            "allow_fallbacks": False,
            "require_parameters": True,
            "data_collection": "deny",
            "zdr": True,
        },
    }


def request_openrouter_questions(
    *, api_key: str, selected: list[SelectedRecord], model: str,
    seed: int, provider_slug: str = "azure",
    base_url: str = "https://openrouter.ai/api/v1",
    transport: httpx.BaseTransport | None = None,
) -> tuple[OracleQuestionBatch, str, str | None]:
    if not api_key.strip():
        raise OpenAIDraftError("openrouter_api_key_missing")
    body = build_openrouter_question_request(
        selected, model=model, seed=seed, provider_slug=provider_slug
    )
    request_hash = hashlib.sha256(
        json.dumps(body, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    try:
        with httpx.Client(timeout=180, transport=transport) as client:
            response = client.post(
                f"{base_url.rstrip('/')}/chat/completions", json=body,
                headers={"Authorization": f"Bearer {api_key}"},
            )
            response.raise_for_status()
            payload = response.json()
        choice = payload["choices"][0]
        if choice.get("finish_reason") not in {"stop", None}:
            raise OpenAIDraftError("openrouter_question_response_not_completed")
        batch = OracleQuestionBatch.model_validate_json(choice["message"]["content"])
        provider = payload.get("provider")
    except OpenAIDraftError:
        raise
    except (httpx.HTTPError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise OpenAIDraftError("openrouter_question_request_failed") from exc
    return batch, request_hash, provider if isinstance(provider, str) else None


def _span_for_line(spans: list[EvidenceSpan], line: int) -> EvidenceSpan | None:
    return next((span for span in spans if span.start_line <= line <= span.end_line), None)


def _candidate_reference(span: EvidenceSpan) -> dict[str, Any]:
    return {
        "evidenceId": span.evidence_id,
        "sourceSpan": span.location(),
        "explicitClaims": sorted(extract_explicit_claims(span.section_heading, span.text)),
        "temporalStatus": "pending_human_review",
        "diagnosisStatus": "pending_human_review",
        "medicationActivity": "pending_human_review",
        "reviewStatus": "candidate_unreviewed",
    }


def _span_for_record_line(record: Any, spans: list[EvidenceSpan], line: int) -> EvidenceSpan | None:
    direct = _span_for_line(spans, line)
    if direct is not None:
        return direct
    source_lines = record.text.splitlines()
    if not 1 <= line <= len(source_lines) or not source_lines[line - 1].lstrip().startswith("## "):
        return None
    heading = source_lines[line - 1].lstrip()[3:].strip()
    return next(
        (
            span for span in spans
            if span.section_heading == heading and span.start_line > line
        ),
        None,
    )


def _references_for_lines(
    record: Any, spans: list[EvidenceSpan], lines: list[int]
) -> list[dict[str, Any]]:
    references: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line in lines:
        span = _span_for_record_line(record, spans, line)
        if span is None:
            raise ValueError("question_evidence_line_not_found")
        if span.evidence_id in seen:
            continue
        seen.add(span.evidence_id)
        reference = _candidate_reference(span)
        reference["modelSelectedLine"] = line
        reference["resolvedFromSectionHeading"] = not (
            span.start_line <= line <= span.end_line
        )
        references.append(reference)
    return references


def compile_question_candidates(
    selected: list[SelectedRecord], batch: OracleQuestionBatch
) -> list[dict[str, Any]]:
    by_index = {item.index: item for item in selected}
    rows = []
    for draft in sorted(batch.questions, key=lambda item: item.record_index):
        chosen = by_index[draft.record_index]
        if draft.category != chosen.requested_category:
            raise ValueError("question_category_mismatch")
        if draft.evidence_availability != chosen.requested_availability:
            raise ValueError("question_evidence_availability_mismatch")
        spans = build_evidence_spans(chosen.record)
        evidence = _references_for_lines(
            chosen.record, spans, draft.evidence_line_numbers
        )
        irrelevant = _references_for_lines(
            chosen.record, spans, draft.irrelevant_line_numbers
        )
        if {item["evidenceId"] for item in evidence}.intersection(
            item["evidenceId"] for item in irrelevant
        ):
            raise ValueError("candidate_evidence_overlap_after_span_resolution")
        rows.append({
            "schemaVersion": ORACLE_PILOT_VERSION,
            "caseId": f"oracle-pilot-{draft.record_index:03d}",
            "sourceFile": chosen.source_path.name,
            "snapshotId": chosen.record.id,
            "contentHash": chosen.record.content_hash,
            "question": draft.question,
            "category": draft.category,
            "evidenceAvailability": draft.evidence_availability,
            "userIntroducesNewSymptom": draft.user_introduces_new_symptom,
            "candidateEvidence": evidence,
            "candidateIrrelevantEvidence": irrelevant,
            "candidateRationale": draft.rationale,
            "candidateForbiddenInferences": draft.forbidden_inferences,
            "oracleEvidence": None,
            "goldAnswer": None,
            "reviewStatus": "pending",
            "trainable": False,
            "sourceCcoModified": False,
        })
    return rows


def build_evidence_review_queue(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Create review tasks without promoting model suggestions to verified evidence."""
    return [{
        "schemaVersion": ORACLE_REVIEW_VERSION,
        "caseId": case["caseId"],
        "snapshotId": case["snapshotId"],
        "contentHash": case["contentHash"],
        "question": case["question"],
        "questionReviewStatus": "pending",
        "candidateRelevantEvidence": case["candidateEvidence"],
        "candidateIrrelevantEvidence": case["candidateIrrelevantEvidence"],
        "verifiedRelevantEvidence": None,
        "verifiedIrrelevantEvidence": None,
        "verifiedForbiddenInferences": None,
        "temporalStatusReview": "pending",
        "diagnosisStatusReview": "pending",
        "medicationActivityReview": "pending",
        "reviewStatus": "pending",
        "reviewer": None,
        "reviewedAt": None,
        "reviewNotes": None,
        "gold": False,
        "trainable": False,
    } for case in cases]


def predicted_annotations(record: Any, question: str, *, limit: int = 8) -> list[dict[str, Any]]:
    terms = question_terms(question)
    ranked = sorted(
        (
            (score_text(span.text, terms), -span.start_line, span)
            for span in build_evidence_spans(record)
        ),
        key=lambda item: (item[0], item[1]),
        reverse=True,
    )
    return [
        {
            "evidenceId": span.evidence_id,
            "sourceSpan": span.location(),
            "importance": "helpful",
            "useRole": "direct_fact",
            "causalUseAllowed": False,
            "explicitClaims": sorted(extract_explicit_claims(span.section_heading, span.text)),
            "forbiddenInferences": ["Kayıtta açık olmayan aktiflik, kesinlik veya nedensellik çıkarma."],
            "annotationSource": "predicted_lexical_ranker",
        }
        for score, _line, span in ranked[:limit] if score > 0
    ]


def _validate_annotation_refs(record: Any, annotations: list[dict[str, Any]]) -> None:
    available = {span.evidence_id: span for span in build_evidence_spans(record)}
    for item in annotations:
        ref = item.get("evidenceId")
        span = available.get(ref)
        if span is None or item.get("sourceSpan") != span.location():
            raise ValueError("annotation_source_mismatch")


def _span_for_prompt_fact(spans: list[EvidenceSpan], fact: Any) -> EvidenceSpan | None:
    if fact.source_line:
        direct = _span_for_line(spans, fact.source_line)
        if direct is not None:
            return direct
    source = " ".join(str(fact.source_text or "").split()).casefold()
    if not source:
        return None
    candidates = [
        span for span in spans
        if span.section_heading == fact.source_section
    ]
    for span in candidates:
        span_text = " ".join(span.text.split()).casefold()
        if source in span_text or span_text.lstrip("- ") in source:
            return span
    return None


def build_oracle_abc_plan(
    record: Any, question: str, arm: Literal["A", "B", "B_AI", "C"], *,
    oracle_annotations: list[dict[str, Any]] | None = None,
    oracle_constraints: dict[str, list[str]] | None = None,
) -> FullContextPlan:
    """Keep the exact full CCO in all arms; add annotations only to the question turn."""
    base = build_full_context(record, question, [], structured_facts=None)
    if arm == "A":
        annotations: list[dict[str, Any]] = []
        source = "none"
    elif arm == "B":
        if oracle_annotations is None:
            raise ValueError("human_oracle_required")
        annotations = oracle_annotations
        source = "human_oracle"
    elif arm == "B_AI":
        if oracle_annotations is None:
            raise ValueError("ai_provisional_oracle_required")
        annotations = oracle_annotations
        source = "ai_provisional"
    elif arm == "C":
        annotations = predicted_annotations(record, question)
        source = "predicted_ranker"
    else:
        raise ValueError("unknown_oracle_arm")
    _validate_annotation_refs(record, annotations)
    spans = build_evidence_spans(record)
    prompt_to_stable: dict[str, str] = {}
    for fact in base.prompt_package:
        span = _span_for_prompt_fact(spans, fact)
        if span is not None:
            prompt_to_stable[fact.id] = span.evidence_id
    guide_annotations = []
    for annotation in annotations:
        stable_id = annotation["evidenceId"]
        prompt_ids = sorted(
            fact_id for fact_id, mapped_id in prompt_to_stable.items()
            if mapped_id == stable_id
        )
        if not prompt_ids:
            raise ValueError("annotation_prompt_evidence_mapping_missing")
        guide_annotations.append({**annotation, "promptEvidenceIds": prompt_ids})
    messages = [dict(message) for message in base.messages]
    constraints = oracle_constraints or {}
    if guide_annotations or constraints:
        guidance = json.dumps({
            "annotationSource": source,
            "rules": [
                "Tam CCO bağlamdır; anotasyonlar yalnız önem ve kullanım rehberidir.",
                "Anotasyonda olmayan kayıtlar görünmez veya yanlış kabul edilmez.",
                "Yasak çıkarımları yapma; kaynakta olmayan aktiflik, kesinlik ve nedenselliği ekleme.",
                "Yanıtın evidence_ids alanında yalnız promptEvidenceIds kimliklerini kullan.",
                "importance=irrelevant veya useRole=do_not_use olan kanıtı cevap desteği yapma.",
            ],
            "evidenceAnnotations": guide_annotations,
            "reviewConstraints": constraints,
        }, ensure_ascii=False, separators=(",", ":"))
        messages[-1]["content"] = f"KANIT REHBERİ:\n{guidance}\n\n{messages[-1]['content']}"
    report = {
        **base.report,
        "schemaVersion": ORACLE_ABC_VERSION,
        "arm": arm,
        "annotationSource": source,
        "annotationCount": len(annotations),
        "constraintCount": sum(len(items) for items in constraints.values()),
        "promptEvidenceIdMap": prompt_to_stable,
        "experimental": arm == "B_AI",
        "fullCcoUnchanged": base.report["ccoSha256"] == base.report["sentCcoSha256"],
    }
    return replace(base, messages=messages, report=report)


def pilot_summary(
    audit_rows: list[dict[str, Any]], cases: list[dict[str, Any]], *,
    request_sha256: str, requested_provider: str, actual_provider: str | None,
) -> dict[str, Any]:
    status_counts = Counter(row["status"] for row in audit_rows)
    return {
        "schemaVersion": ORACLE_PILOT_VERSION,
        "sourceAudit": {
            "examined": len(audit_rows),
            "statusCounts": dict(sorted(status_counts.items())),
            "identifierFindingFiles": sum(bool(row["identifierFindings"]) for row in audit_rows),
            "sourceFilesModified": 0,
        },
        "existingGoldQueue": {
            "tasks": 60,
            "humanReviewed": 0,
            "pending": 60,
            "usableAsOracleNow": False,
        },
        "pilot": {
            "cases": len(cases),
            "uniqueCcos": len({row["sourceFile"] for row in cases}),
            "categories": dict(sorted(Counter(row["category"] for row in cases).items())),
            "evidenceAvailability": dict(sorted(Counter(
                row["evidenceAvailability"] for row in cases
            ).items())),
            "reviewStatus": "pending",
            "trainable": False,
        },
        "externalProcessing": {
            "authorizedByUserRequest": True,
            "model": "openai/gpt-6-luna",
            "requestedProvider": requested_provider,
            "actualProvider": actual_provider,
            "zdr": True,
            "dataCollection": "deny",
            "allowFallbacks": False,
            "requestSha256": request_sha256,
            "identifiersSent": False,
            "recordIdsSent": False,
            "subjectIdsSent": False,
        },
        "abc": {
            "infrastructureReady": True,
            "realModelCallsExecuted": False,
            "blockedReason": "human_oracle_annotations_pending",
            "loraStarted": False,
        },
        "errorAttribution": {
            "observedModelErrors": 0,
            "status": "not_measured_without_gemma_abc_and_human_grades",
        },
        "syntheticScaleExpanded": False,
        "loraStarted": False,
    }
