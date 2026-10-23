import json

import pytest
from pydantic import ValidationError

from medical_ai_selective_context.text_document import (
    build_evidence_spans,
    extract_explicit_claims,
    parse_cco_text,
)
from medical_ai_selective_context.training_dataset import EvidenceAnnotation


def _payload(text: str = "42 yaşında kadın.\n\n## Tanılar\n- Migren — ön tanı.") -> dict:
    return {
        "id": "11111111-1111-4111-8111-111111111111",
        "subjectId": "22222222-2222-4222-8222-222222222222",
        "asOf": "2026-10-21T12:00:00+03:00",
        "text": text,
    }


def test_four_field_text_cco_contract_builds_versioned_source_spans():
    record = parse_cco_text(_payload())

    spans = build_evidence_spans(record)

    assert spans
    assert all(span.evidence_id.startswith("evi_") for span in spans)
    assert all(span.snapshot_id == record.id for span in spans)
    assert all(span.content_hash == record.content_hash for span in spans)
    assert all(span.start_line <= span.end_line for span in spans)
    assert len({span.evidence_id for span in spans}) == len(spans)


def test_evidence_id_changes_with_snapshot_or_content():
    original = parse_cco_text(_payload())
    changed_content = parse_cco_text(_payload("42 yaşında kadın.\n\n## Tanılar\n- Migren."))
    changed_snapshot_payload = _payload()
    changed_snapshot_payload["id"] = "33333333-3333-4333-8333-333333333333"
    changed_snapshot = parse_cco_text(changed_snapshot_payload)

    original_ids = {span.evidence_id for span in build_evidence_spans(original)}

    assert original_ids.isdisjoint(
        {span.evidence_id for span in build_evidence_spans(changed_content)}
    )
    assert original_ids.isdisjoint(
        {span.evidence_id for span in build_evidence_spans(changed_snapshot)}
    )


def test_training_evidence_uses_text_location_and_does_not_require_fact_id():
    record = parse_cco_text(_payload())
    span = build_evidence_spans(record)[-1]
    annotation = EvidenceAnnotation.model_validate(
        {
            "evidenceId": span.evidence_id,
            "sourceSpan": span.location(),
            "factId": None,
            "requiredness": "must_use",
            "useRole": "direct_fact",
            "causalStrength": "none",
            "safetyCritical": False,
            "reasonCode": "explicit_preliminary_diagnosis",
            "explicitClaims": ["assertion_preliminary"],
            "independentClinicalValidation": False,
        }
    )

    assert annotation.fact_id is None
    assert annotation.evidence_id == span.evidence_id
    assert "Migren" not in json.dumps(annotation.model_dump(by_alias=True))


def test_training_evidence_rejects_fact_only_annotation():
    with pytest.raises(ValidationError):
        EvidenceAnnotation.model_validate(
            {
                "factId": "44444444-4444-4444-8444-444444444444",
                "requiredness": "must_use",
                "useRole": "direct_fact",
                "causalStrength": "none",
                "safetyCritical": False,
                "reasonCode": "legacy_fact_only",
                "explicitClaims": [],
                "independentClinicalValidation": False,
            }
        )


def test_explicit_claim_extraction_does_not_turn_prescription_into_active_use():
    claims = extract_explicit_claims(
        "İlaçlar (reçete kaydı)",
        "- ÖRNEK İLAÇ — son reçete 02.10.2026, 3 reçete.",
    )

    assert claims == {"medication_prescribed"}
    assert "medication_active" not in claims
