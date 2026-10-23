import json
import uuid
from pathlib import Path

import httpx
import pytest

from medical_ai_selective_context.oracle_benchmark import (
    analyze_results,
    build_completion_body,
    load_case_record,
    request_completion,
    response_evidence_ids,
    verified_oracle_annotations,
)
from medical_ai_selective_context.text_document import build_evidence_spans, parse_cco_text
from medical_ai_selective_context.text_verify import ANSWER_JSON_SCHEMA


def _raw_record() -> dict:
    return {
        "id": str(uuid.UUID(int=1)),
        "subjectId": str(uuid.UUID(int=2)),
        "asOf": "2026-10-21T12:00:00+03:00",
        "text": "42 yaşında kadın.\n\n## Laboratuvar\n- TSH 2,1 mIU/L (20.10.2026).",
    }


def _verified_review() -> dict:
    record = parse_cco_text(_raw_record())
    span = build_evidence_spans(record)[-1]
    evidence = [{
        "evidenceId": span.evidence_id,
        "sourceSpan": span.location(),
        "importance": "must_use",
        "useRole": "direct_fact",
        "causalUseAllowed": False,
        "explicitClaims": [],
        "forbiddenInferences": ["Nedensellik kurma."],
        "annotationSource": "human_oracle",
    }]
    return {
        "caseId": "c1",
        "questionReviewStatus": "approved",
        "verifiedRelevantEvidence": evidence,
        "verifiedIrrelevantEvidence": [],
        "verifiedForbiddenInferences": ["Nedensellik kurma."],
        "temporalStatusReview": "reviewed",
        "diagnosisStatusReview": "not_applicable",
        "medicationActivityReview": "not_applicable",
        "reviewStatus": "verified",
        "reviewer": "human-1",
        "reviewedAt": "2026-10-21T12:30:00+03:00",
    }


def test_source_snapshot_and_oracle_review_gates(tmp_path: Path):
    raw = _raw_record()
    source = tmp_path / "source.json"
    source.write_text(json.dumps(raw), encoding="utf-8")
    record = parse_cco_text(raw)
    case = {
        "sourceFile": source.name,
        "snapshotId": str(record.id),
        "contentHash": record.content_hash,
    }
    assert load_case_record(case, tmp_path).content_hash == record.content_hash
    review = _verified_review()
    assert verified_oracle_annotations(review)
    review["reviewStatus"] = "pending"
    with pytest.raises(ValueError, match="pending"):
        verified_oracle_annotations(review)
    case["contentHash"] = "bad"
    with pytest.raises(ValueError, match="content_hash"):
        load_case_record(case, tmp_path)


def test_completion_request_contract_and_response_parsing():
    body = build_completion_body(
        model="gemma", messages=[{"role": "user", "content": "x"}], seed=7,
        max_tokens=100, structured_output="response-format",
        disable_reasoning="chat-template",
    )
    assert body["temperature"] == 0
    assert body["response_format"]["json_schema"]["schema"] == ANSWER_JSON_SCHEMA
    assert body["chat_template_kwargs"] == {"enable_thinking": False}

    answer = {
        "conclusion": {
            "status": "supported", "text": "TSH 2,1.",
            "claim_type": "fact", "evidence_ids": ["E2"],
        },
        "sentences": [], "missing_information": [], "requires_clinician_review": False,
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content)["seed"] == 7
        return httpx.Response(200, json={
            "model": "gemma", "usage": {"total_tokens": 10},
            "choices": [{
                "finish_reason": "stop", "message": {"content": json.dumps(answer)},
            }],
        })

    content, meta = request_completion(
        base_url="http://127.0.0.1:8000/v1", api_key="", body=body, timeout=10,
        transport=httpx.MockTransport(handler),
    )
    assert response_evidence_ids(content) == ["E2"]
    assert meta["servedModel"] == "gemma"


def test_analysis_reports_paired_delta_and_evidence_metrics():
    review = _verified_review()
    evidence_id = review["verifiedRelevantEvidence"][0]["evidenceId"]
    runs = [
        {
            "runKey": "c1:1:A", "caseId": "c1", "repeat": 1, "arm": "A",
            "status": "completed", "latencyMs": 100, "responseEvidenceIds": [],
        },
        {
            "runKey": "c1:1:B", "caseId": "c1", "repeat": 1, "arm": "B",
            "status": "completed", "latencyMs": 120,
            "responseStableEvidenceIds": [evidence_id],
        },
        {
            "runKey": "c1:1:C", "caseId": "c1", "repeat": 1, "arm": "C",
            "status": "blocked", "modelCalled": False,
        },
    ]
    grades = [
        {
            "runKey": "c1:1:A", "verdict": "partial", "criticalSafetyFailure": False,
            "usedEvidenceIds": [], "errorSources": ["temporal_grounding"],
            "errorFlags": {"wrongDate": True},
        },
        {
            "runKey": "c1:1:B", "verdict": "pass", "criticalSafetyFailure": False,
            "usedEvidenceIds": [evidence_id], "errorSources": [], "errorFlags": {},
        },
    ]
    report = analyze_results(runs, grades, [review])
    assert report["arms"]["B"]["evidenceRecall"] == 1.0
    assert report["arms"]["B"]["evidencePrecision"] == 1.0
    assert report["pairedDeltas"]["B-A"] == {
        "pairs": 1, "wins": 1, "losses": 0, "ties": 0,
    }
    assert report["blockedRows"] == 1
