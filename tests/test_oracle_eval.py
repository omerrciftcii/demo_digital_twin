import json
import uuid
from pathlib import Path

import httpx
import pytest

from medical_ai_selective_context.oracle_eval import (
    ORACLE_PILOT_VERSION,
    QUESTION_DRAFT_VERSION,
    OracleQuestionBatch,
    SelectedRecord,
    build_evidence_review_queue,
    build_openrouter_question_request,
    build_oracle_abc_plan,
    compile_question_candidates,
    request_openrouter_questions,
    scan_identifiers,
)
from medical_ai_selective_context.text_document import build_evidence_spans, parse_cco_text


def _record(index: int = 1):
    return parse_cco_text({
        "id": str(uuid.UUID(int=index)),
        "subjectId": str(uuid.UUID(int=1000 + index)),
        "asOf": "2026-10-21T12:00:00+03:00",
        "text": "42 yaşında kadın.\n\n## Laboratuvar (hastane)\n- TSH 2,1 mIU/L (20.10.2026).",
    })


def _selected() -> list[SelectedRecord]:
    categories = [
        "headache", "headache", "fatigue", "fatigue", "gastrointestinal",
        "gastrointestinal", "sleep", "sleep", "palpitations", "palpitations",
        "medication", "medication", "medication", "daily_life", "daily_life",
        "acute_safety", "acute_safety", "diagnosis_status", "general_symptom",
        "general_symptom",
    ]
    result = []
    for index, category in enumerate(categories, 1):
        availability = "not_supported" if index > 18 else "supported"
        result.append(SelectedRecord(
            index, Path(f"source-{index}.json"), _record(index), category, availability
        ))
    return result


def _batch(selected: list[SelectedRecord]) -> OracleQuestionBatch:
    return OracleQuestionBatch.model_validate({
        "schemaVersion": QUESTION_DRAFT_VERSION,
        "questions": [{
            "recordIndex": item.index,
            "question": f"Günlük hayatımda bu durumu nasıl yorumlamalıyım {item.index}?",
            "category": item.requested_category,
            "evidenceAvailability": item.requested_availability,
            "evidenceLineNumbers": [] if item.requested_availability == "not_supported" else [4],
            "irrelevantLineNumbers": [1],
            "userIntroducesNewSymptom": item.requested_category != "diagnosis_status",
            "rationale": "Doğal bir kullanıcı sorusu adayıdır.",
            "forbiddenInferences": ["Nedensellik kurma."],
        } for item in selected],
    })


def test_identifier_scanner_reports_hashes_without_echoing_values():
    secret = "omer@example.com"
    findings = scan_identifiers(f"İletişim: {secret}\nTelefon 0532 111 22 33")
    assert {item["kind"] for item in findings} == {"email", "turkish_phone"}
    assert secret not in json.dumps(findings)
    assert all(len(item["matchSha256"]) == 64 for item in findings)


def test_real_cco_request_is_provider_pinned_and_omits_record_identifiers():
    selected = _selected()
    body = build_openrouter_question_request(
        selected, model="openai/gpt-6-luna", seed=17, provider_slug="azure"
    )
    rendered = json.dumps(body, ensure_ascii=False)
    assert body["provider"] == {
        "only": ["azure"],
        "allow_fallbacks": False,
        "require_parameters": True,
        "data_collection": "deny",
        "zdr": True,
    }
    assert "subjectId" not in rendered
    assert str(selected[0].record.id) not in rendered
    assert body["messages"][1]["content"].count("selectableEvidenceLineNumbers") == 20
    with pytest.raises(ValueError, match="zero-retention"):
        build_openrouter_question_request(
            selected, model="x", seed=1, provider_slug="openai"
        )


def test_openrouter_question_response_is_schema_and_line_validated():
    selected = _selected()
    batch = _batch(selected)

    def handler(request: httpx.Request) -> httpx.Response:
        sent = json.loads(request.content)
        assert sent["provider"]["zdr"] is True
        return httpx.Response(200, json={
            "provider": "Azure",
            "choices": [{
                "finish_reason": "stop",
                "message": {"content": batch.model_dump_json(by_alias=True)},
            }],
        })

    received, request_hash, provider = request_openrouter_questions(
        api_key="test", selected=selected, model="test", seed=17,
        transport=httpx.MockTransport(handler),
    )
    cases = compile_question_candidates(selected, received)
    assert len(cases) == 20
    assert cases[0]["candidateEvidence"][0]["sourceSpan"]["sectionHeading"] == (
        "Laboratuvar (hastane)"
    )
    assert cases[0]["candidateIrrelevantEvidence"][0]["sourceSpan"]["sectionHeading"] == (
        "Giriş"
    )
    assert cases[-1]["candidateEvidence"] == []
    assert all(case["schemaVersion"] == ORACLE_PILOT_VERSION for case in cases)
    assert all(case["reviewStatus"] == "pending" for case in cases)
    assert all(case["trainable"] is False for case in cases)
    assert len(request_hash) == 64
    assert provider == "Azure"
    reviews = build_evidence_review_queue(cases)
    assert all(row["reviewStatus"] == "pending" for row in reviews)
    assert all(row["verifiedRelevantEvidence"] is None for row in reviews)
    assert all(row["gold"] is False and row["trainable"] is False for row in reviews)


def test_abc_uses_identical_full_cco_and_requires_human_oracle():
    record = _record()
    span = build_evidence_spans(record)[-1]
    oracle = [{
        "evidenceId": span.evidence_id,
        "sourceSpan": span.location(),
        "importance": "must_use",
        "useRole": "direct_fact",
        "causalUseAllowed": False,
        "explicitClaims": [],
        "forbiddenInferences": ["Nedensellik kurma."],
        "annotationSource": "human_oracle",
    }]
    a = build_oracle_abc_plan(record, "TSH sonucum ne ifade ediyor?", "A")
    b = build_oracle_abc_plan(
        record, "TSH sonucum ne ifade ediyor?", "B", oracle_annotations=oracle
    )
    c = build_oracle_abc_plan(record, "TSH sonucum ne ifade ediyor?", "C")

    assert {plan.report["sentCcoSha256"] for plan in (a, b, c)} == {
        a.report["ccoSha256"]
    }
    assert all(plan.report["fullCcoUnchanged"] for plan in (a, b, c))
    assert a.report["annotationCount"] == 0
    assert b.report["annotationSource"] == "human_oracle"
    assert c.report["annotationSource"] == "predicted_ranker"
    assert b.report["promptEvidenceIdMap"]
    assert span.evidence_id in b.report["promptEvidenceIdMap"].values()
    assert "promptEvidenceIds" in b.messages[-1]["content"]
    with pytest.raises(ValueError, match="human_oracle_required"):
        build_oracle_abc_plan(record, "TSH?", "B")
