import json
from pathlib import Path

import pytest

from medical_ai_selective_context.training_sources import (
    SourceAuditError,
    audit_text_cco_and_facts,
)

SUBJECT_ID = "11111111-1111-4111-8111-111111111111"
CCO_ID = "22222222-2222-4222-8222-222222222222"
FACT_ID = "33333333-3333-4333-8333-333333333333"


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _source_corpus(tmp_path: Path) -> tuple[Path, Path]:
    cco_dir = tmp_path / "cco"
    facts_dir = tmp_path / "facts"
    cco_dir.mkdir()
    facts_dir.mkdir()
    _write_json(
        cco_dir / f"{CCO_ID}.json",
        {
            "id": CCO_ID,
            "subjectId": SUBJECT_ID,
            "asOf": "2026-10-20T10:00:00+03:00",
            "text": (
                "39 yaşında erkek.\n\n"
                "## Tanılar (hastane kaydı)\n"
                "Diğer:\n"
                "- Örnek tanı (A01) — kesin tanı; son kayıt 01.01.2024.\n\n"
                "## Alerjiler\nPolen (hasta beyanı)."
            ),
        },
    )
    _write_json(
        facts_dir / f"{SUBJECT_ID}.json",
        [
            {
                "factId": FACT_ID,
                "canonicalId": "diagnosis.recorded-item",
                "version": "1.1.0",
                "evaluationKey": "code:icd-10:A01",
                "evaluationStatus": "success",
                "validFrom": "2026-10-19T10:00:00+03:00",
                "validUntil": None,
                "occurredFrom": "2024-01-01T10:00:00+03:00",
                "occurredTo": "2024-01-01T10:00:00+03:00",
                "value": {
                    "coded": {
                        "display": "Örnek tanı",
                        "attributes": {"type": "preliminary"},
                    }
                },
            }
        ],
    )
    _write_json(
        facts_dir / "_fact-definitions.json",
        [{"canonicalId": "diagnosis.recorded-item", "version": "1.1.0"}],
    )
    _write_json(
        facts_dir / "_manifest.json",
        {
            "completedAt": "2026-10-21T15:00:00+03:00",
            "historicalCcoReconstruction": False,
            "includeEarlierDays": False,
        },
    )
    return cco_dir, facts_dir


def test_audit_is_aggregate_and_flags_derived_diagnosis_status(tmp_path: Path):
    cco_dir, facts_dir = _source_corpus(tmp_path)

    report = audit_text_cco_and_facts(cco_dir, facts_dir)

    assert report["valid"] is True
    assert report["disposition"] == "legacy_audit_only"
    assert report["facts"]["sourceRole"] == "legacy_audit_only"
    assert report["facts"]["goldLabelsAllowed"] is False
    assert report["facts"]["clinicalStateInferenceAllowed"] is False
    assert report["alignment"]["matchedSubjects"] == 1
    assert report["legacyComparisonDiagnostics"]["preliminaryRenderedDefinitive"] == 1
    assert report["textCco"]["allergyRecords"] == 1
    assert "derived_diagnosis_status_differs_from_source" not in report["warnings"]
    assert report["legacyObservations"]["preliminaryRenderedDefinitive"] == 1
    rendered = json.dumps(report, ensure_ascii=False)
    assert SUBJECT_ID not in rendered
    assert "Örnek tanı" not in rendered
    assert "Polen" not in rendered


def test_audit_counts_facts_created_after_text_snapshot(tmp_path: Path):
    cco_dir, facts_dir = _source_corpus(tmp_path)
    rows = json.loads((facts_dir / f"{SUBJECT_ID}.json").read_text(encoding="utf-8"))
    rows[0]["validFrom"] = "2026-10-21T10:00:00+03:00"
    _write_json(facts_dir / f"{SUBJECT_ID}.json", rows)

    report = audit_text_cco_and_facts(cco_dir, facts_dir)

    assert report["facts"]["validAfterCcoAsOf"] == 1
    assert report["legacyComparisonDiagnostics"]["eligibleSourceFacts"] == 0
    assert "facts_valid_after_cco_as_of" not in report["warnings"]
    assert report["legacyObservations"]["validAfterCcoAsOf"] == 1


def test_audit_rejects_duplicate_subjects_without_echoing_content(tmp_path: Path):
    cco_dir, facts_dir = _source_corpus(tmp_path)
    original = json.loads((cco_dir / f"{CCO_ID}.json").read_text(encoding="utf-8"))
    original["id"] = "44444444-4444-4444-8444-444444444444"
    original["text"] = "secret clinical phrase"
    _write_json(cco_dir / "duplicate.json", original)

    with pytest.raises(SourceAuditError) as caught:
        audit_text_cco_and_facts(cco_dir, facts_dir)

    assert str(caught.value) == "duplicate_text_cco_subject"
    assert "secret" not in str(caught.value)
