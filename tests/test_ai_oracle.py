import json
import uuid

import httpx
import pytest

from medical_ai_selective_context.ai_oracle import (
    AI_ORACLE_DRAFT_VERSION,
    AIOracleDraft,
    ai_review_annotations,
    ai_review_constraints,
    build_ai_oracle_request,
    compile_ai_oracle_review,
    request_ai_oracle_review,
    validate_ai_review,
)
from medical_ai_selective_context.oracle_eval import build_oracle_abc_plan
from medical_ai_selective_context.text_document import build_evidence_spans, parse_cco_text


def _record():
    return parse_cco_text({
        "id": str(uuid.UUID(int=1)),
        "subjectId": str(uuid.UUID(int=2)),
        "asOf": "2026-10-21T12:00:00+03:00",
        "text": "42 yaşında kadın.\n\n## Laboratuvar\n- TSH 2,1 mIU/L (20.10.2026).",
    })


def _case(record):
    return {
        "caseId": "oracle-pilot-001",
        "sourceFile": "source.json",
        "snapshotId": str(record.id),
        "contentHash": record.content_hash,
        "question": "Son TSH sonucum neydi?",
        "evidenceAvailability": "supported",
    }


def _draft(record) -> AIOracleDraft:
    intro, lab = build_evidence_spans(record)
    return AIOracleDraft.model_validate({
        "schemaVersion": AI_ORACLE_DRAFT_VERSION,
        "caseId": "oracle-pilot-001",
        "answerability": "supported",
        "evidenceDecisions": [
            {
                "evidenceId": lab.evidence_id,
                "disposition": "relevant",
                "importance": "must_use",
                "useRole": "direct_fact",
                "causalUseAllowed": False,
                "temporalStatus": "recent",
                "diagnosisStatus": "not_applicable",
                "persistenceStatus": "not_applicable",
                "medicationActivity": "not_applicable",
                "forbiddenInferences": ["Sonucu tanıya dönüştürme."],
                "rationale": "Soru doğrudan bu laboratuvar sonucunu soruyor.",
            },
            {
                "evidenceId": intro.evidence_id,
                "disposition": "irrelevant",
                "importance": "irrelevant",
                "useRole": "do_not_use",
                "causalUseAllowed": False,
                "temporalStatus": "undated",
                "diagnosisStatus": "not_applicable",
                "persistenceStatus": "not_applicable",
                "medicationActivity": "not_applicable",
                "forbiddenInferences": ["Yaştan TSH yorumu çıkarma."],
                "rationale": "Demografik giriş doğrudan sonucu yanıtlamıyor.",
            },
        ],
        "forbiddenInferences": ["TSH değerinden tanı koyma."],
        "safetyWarnings": [],
        "reviewSummary": "Laboratuvar satırı zorunlu, demografik giriş ilgisizdir.",
    })


def test_ai_oracle_request_uses_astra_reasoning_and_closed_evidence_catalog():
    record = _record()
    body = build_ai_oracle_request(
        case=_case(record), record=record, model="openai/gpt-6-astra", seed=17
    )
    assert body["model"] == "openai/gpt-6-astra"
    assert body["reasoning"] == {"effort": "high"}
    assert body["response_format"]["json_schema"]["strict"] is True
    sent = json.loads(body["messages"][1]["content"])
    assert all(item["promptEvidenceIds"] for item in sent["evidenceCatalog"])
    assert {item["evidenceId"] for item in sent["evidenceCatalog"]} == {
        span.evidence_id for span in build_evidence_spans(record)
    }


def test_ai_review_is_compiled_from_source_and_stays_provisional():
    record = _record()
    case = _case(record)
    review = compile_ai_oracle_review(
        case=case, record=record, draft=_draft(record), model="openai/gpt-6-astra",
        seed=17, reasoning_effort="high", metadata={"requestSha256": "a" * 64},
    )
    validate_ai_review(case=case, record=record, review=review)
    assert review["reviewStatus"] == "ai_provisional"
    assert review["sourceIntegrityValidated"] is True
    assert review["humanVerified"] is False
    assert review["gold"] is False
    assert review["trainable"] is False
    assert review["experimentalOnly"] is True
    assert review["provisionalRelevantEvidence"][0]["sourceSpan"]["startLine"] == 4
    plan = build_oracle_abc_plan(
        record, case["question"], "B_AI",
        oracle_annotations=ai_review_annotations(review),
        oracle_constraints=ai_review_constraints(review),
    )
    assert plan.report["experimental"] is True
    assert plan.report["annotationSource"] == "ai_provisional"
    assert plan.report["fullCcoUnchanged"] is True
    assert plan.report["constraintCount"] == 1
    assert "TSH değerinden tanı koyma." in plan.messages[-1]["content"]


def test_not_supported_review_can_have_empty_evidence_but_keeps_constraints():
    record = parse_cco_text({
        "id": str(uuid.UUID(int=3)),
        "subjectId": str(uuid.UUID(int=4)),
        "asOf": "2026-10-21T12:00:00+03:00",
        "text": "42 yaşında kadın.",
    })
    case = {
        **_case(record),
        "question": "Boğazım ağrıyor, evde ne yapabilirim?",
        "evidenceAvailability": "not_supported",
    }
    draft = AIOracleDraft.model_validate({
        "schemaVersion": AI_ORACLE_DRAFT_VERSION,
        "caseId": case["caseId"],
        "answerability": "not_supported",
        "evidenceDecisions": [],
        "forbiddenInferences": ["Kayıtta boğaz ağrısı olduğunu varsayma."],
        "safetyWarnings": ["Güncel şiddet kaynakta yoktur."],
        "reviewSummary": "Kişisel klinik kanıt yoktur.",
    })
    review = compile_ai_oracle_review(
        case=case, record=record, draft=draft, model="openai/gpt-6.1-sol",
        seed=17, reasoning_effort="medium", metadata={"requestSha256": "a" * 64},
    )
    validate_ai_review(case=case, record=record, review=review)
    plan = build_oracle_abc_plan(
        record, case["question"], "B_AI", oracle_annotations=[],
        oracle_constraints=ai_review_constraints(review),
    )
    assert plan.report["annotationCount"] == 0
    assert plan.report["constraintCount"] == 2
    assert "Kayıtta boğaz ağrısı olduğunu varsayma." in plan.messages[-1]["content"]


def test_ai_review_rejects_unsupported_causal_permission():
    record = _record()
    raw = _draft(record).model_dump(mode="json", by_alias=True)
    raw["evidenceDecisions"][0]["causalUseAllowed"] = True
    draft = AIOracleDraft.model_validate(raw)
    with pytest.raises(ValueError, match="causal_use"):
        compile_ai_oracle_review(
            case=_case(record), record=record, draft=draft,
            model="openai/gpt-6-astra", seed=17, reasoning_effort="high",
            metadata={"requestSha256": "a" * 64},
        )


def test_ai_oracle_response_is_strictly_parsed():
    draft = _draft(_record())

    def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content)["reasoning"]["effort"] == "high"
        return httpx.Response(200, json={
            "id": "generation-1", "model": "openai/gpt-6-astra",
            "provider": "OpenAI", "usage": {"total_tokens": 100},
            "choices": [{
                "finish_reason": "stop",
                "message": {"content": draft.model_dump_json(by_alias=True)},
            }],
        })

    body = build_ai_oracle_request(
        case=_case(_record()), record=_record(), model="openai/gpt-6-astra", seed=17
    )
    received, metadata = request_ai_oracle_review(
        api_key="test", body=body, transport=httpx.MockTransport(handler)
    )
    assert received.case_id == "oracle-pilot-001"
    assert metadata["actualProvider"] == "OpenAI"
    assert len(metadata["requestSha256"]) == 64
