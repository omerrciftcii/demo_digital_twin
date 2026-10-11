from copy import deepcopy

import pytest
from evidence_support import synthetic_cco

from medical_ai_selective_context.evidence import build_evidence, resolve_pointer


def test_source_projection_is_lossless_and_refs_resolve() -> None:
    document = synthetic_cco()
    original = deepcopy(document)
    entries = build_evidence(document, "snapshot-a")
    assert len(entries) == 4
    assert len({entry["ref"] for entry in entries}) == 4
    for entry in entries:
        assert resolve_pointer(document, entry["sourcePointer"]) == entry["source"]
        assert entry["snapshotId"] == "snapshot-a"
    assert document == original
    assert entries == build_evidence(document, "snapshot-a")
    assert {item["ref"] for item in entries}.isdisjoint(
        {item["ref"] for item in build_evidence(document, "snapshot-b")})


def test_frozen_value_not_deduplicated_or_assigned_production_date() -> None:
    entries = build_evidence(synthetic_cco(), "a")
    fact = entries[0]
    insight = next(entry for entry in entries if entry["sourceKind"] == "insight")
    frozen = next(entry for entry in entries if entry["sourceKind"] == "insightSnapshot")
    assert fact["factId"] == frozen["factId"]
    assert fact["value"]["amount"] == 2.1
    assert frozen["value"]["amount"] == 2.2
    assert insight["dependencies"] == [frozen["ref"]]
    assert frozen["parentRef"] == insight["ref"]
    for name in ("occurredFrom", "occurredTo", "recordedAt", "producedAt"):
        assert frozen[name] is None
    assert insight["producedAt"] == "2026-02-01T00:00:00Z"
    assert fact["evidence"]


def test_boolean_flag_does_not_create_anomaly() -> None:
    document = synthetic_cco()
    document["sections"]["lab"]["facts"][0]["value"] = {
        "type": "boolean", "flag": True, "unit": "bool",
    }
    fact = build_evidence(document, "a")[0]
    assert fact["value"]["flag"] is True
    assert fact["severity"] is None
    assert "abnormal" not in fact


def test_rfc6901_escapes_and_invalid_array_indices() -> None:
    source = {"a/b": {"~": [1, 2]}}
    assert resolve_pointer(source, "/a~1b/~0/1") == 2
    for pointer in ("a/b", "/a~1b/~0/-1", "/a~1b/~0/01", "/a~2b"):
        with pytest.raises(ValueError):
            resolve_pointer(source, pointer)
