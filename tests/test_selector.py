import json
from copy import deepcopy
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import pytest

from medical_ai_selective_context.evidence import build_evidence
from medical_ai_selective_context.selector import select_context
from medical_ai_selective_context.store import unwrap_and_validate_cco

ROOT = Path(__file__).parents[1]
SNAPSHOT = "test-source-snapshot"


def _id(value: str) -> str:
    return str(uuid5(NAMESPACE_URL, value))


def _fact(name="lab.creatinine.series", year=2021, amount=2.4, key="old"):
    return {
        "factId": _id(key), "definition": name + "|1.0.0", "evaluationKey": key,
        "evaluationStatus": "success", "occurredFrom": f"{year}-01-01T10:00:00Z",
        "occurredTo": f"{year}-01-01T10:00:00Z", "recordedAt": "2026-01-01T10:00:00Z",
        "comparisons": [], "value": {"type": "quantity", "amount": amount, "unit": "mg/dL"},
    }


def _insight(fact, severity="info", name="lab.review", key="review"):
    return {
        "insightId": _id(key), "definition": name + "|1.0.0", "severity": severity,
        "producedAt": "2026-01-01T10:00:00Z",
        "facts": [{field: deepcopy(fact[field]) for field in (
            "factId", "definition", "evaluationKey", "evaluationStatus", "value"
        )}],
    }


def _document(facts=None, insights=None):
    document = json.loads((ROOT / "data/synthetic/cco-complete.json").read_text(encoding="utf-8"))
    document["sections"] = {"lab": {"facts": facts or [], "insights": insights or [], "missingData": []}}
    document["contentReferences"] = []
    document["glossary"] = {
        "families": {"lab": {"label": "Laboratuvar", "description": "Tarihli laboratuvar kayıtları."}},
        "facts": {}, "rules": {},
    }
    for fact in [*(facts or []), *(fact for insight in insights or [] for fact in insight["facts"])]:
        base = fact["definition"].split("|")[0]
        document["glossary"]["facts"][base] = {"label": base, "description": "Kaynak sonucu."}
    for insight in insights or []:
        base = insight["definition"].split("|")[0]
        document["glossary"]["rules"][base] = {"label": base, "description": "Kaynak kural yorumu."}
    # Every selector scenario starts with actual producer-schema validation.
    return unwrap_and_validate_cco(document)


def _select(document, indices=(0,), budget=8000, **kwargs):
    entries = build_evidence(document, SNAPSHOT)
    ranked = [{"ref": entries[index]["ref"], "score": 1 / (index + 1), "channels": ["exact"]}
              for index in indices]
    packet = select_context(document, "Kaynak sonuçlarımı açıkla", entries, ranked,
                            snapshot_id=SNAPSHOT, token_budget=budget, **kwargs)
    return packet, entries


def test_keeps_old_result_later_normal_and_latest_without_flag_inference():
    document = _document([
        _fact(), _fact(year=2022, amount=0.9, key="followup"),
        _fact("lab.creatinine.latest", year=2026, amount=1.0, key="latest"),
    ])
    before = deepcopy(document)
    packet, _ = _select(document, indices=(2,))
    assert packet["canAnswer"]
    assert packet["counts"]["selectedFacts"] == 3
    assert {item["value"]["amount"] for item in packet["evidence"]} == {2.4, 0.9, 1.0}
    assert all("abnormal" not in item for item in packet["evidence"])
    assert document == before
    assert "clinicalContext" not in packet


def test_no_fixed_fact_or_family_caps():
    document = _document([_fact(year=2001 + i, key=f"history-{i}") for i in range(20)])
    packet, _ = _select(document, budget=50000)
    assert packet["canAnswer"]
    assert packet["counts"]["selectedFacts"] == 20


def test_frozen_insight_values_keep_unknown_dates_glossary_and_content_closure():
    fact = _fact(amount=1.0)
    frozen = {**deepcopy(fact), "value": {"type": "quantity", "amount": 2.4, "unit": "mg/dL"}}
    insight = _insight(frozen, "warning")
    document = _document([fact], [insight])
    reference = {
        "insightId": insight["insightId"], "key": "review", "kind": "article",
        "provider": "catalog", "role": "education", "permanentId": "review",
        "definitionVersion": "1.0.0",
    }
    document["contentReferences"] = [reference, {**reference, "insightId": _id("unselected")}]
    unwrap_and_validate_cco(document)
    packet, _ = _select(document, indices=(1,))
    assert packet["canAnswer"]
    assert len(packet["evidence"]) == 3
    frozen_entry = next(item for item in packet["evidence"] if item["sourceKind"] == "insightSnapshot")
    assert frozen_entry["value"]["amount"] == 2.4
    assert frozen_entry["occurredFrom"] is None
    assert frozen_entry["occurredTo"] is None
    assert frozen_entry["producedAt"] is None
    assert packet["contentReferences"] == [reference]
    assert "lab.review" in packet["glossary"]["rules"]
    assert "lab.creatinine.series" in packet["glossary"]["facts"]
    assert any(item["code"] == "frozen_event_date_unknown" for item in packet["warnings"])


def test_independently_retrieved_frozen_value_includes_parent_and_siblings():
    insight = _insight(_fact())
    other = _insight(_fact("lab.other.latest", key="sibling"))["facts"][0]
    insight["facts"].append(other)
    packet, _ = _select(_document(insights=[insight]), indices=(1,))
    assert packet["canAnswer"]
    assert len(packet["evidence"]) == 3
    assert {item["sourceKind"] for item in packet["evidence"]} == {"insight", "insightSnapshot"}


def test_critical_bundle_preserved_even_when_it_exceeds_budget_and_answer_blocked():
    packet, _ = _select(_document(insights=[_insight(_fact(), "critical")]), budget=100)
    assert not packet["canAnswer"]
    assert "mandatory_context_exceeds_budget" in packet["blockers"]
    assert len(packet["evidence"]) == 2
    assert all(entry["selectedReason"] == "critical_safety" for entry in packet["evidence"])
    assert any(item["code"] == "human_review_required" for item in packet["warnings"])


def test_boolean_flag_is_not_an_abnormality_or_always_on_safety_signal():
    fact = _fact()
    fact["value"] = {"type": "boolean", "flag": True, "unit": "boolean"}
    packet, _ = _select(_document([fact]), indices=())
    assert packet["evidence"] == []
    assert not packet["canAnswer"]
    assert packet["warningIndex"] == []


def test_no_candidates_does_not_fallback_to_entire_cco():
    packet, _ = _select(_document([_fact()]), indices=())
    assert not packet["canAnswer"]
    assert packet["evidence"] == []
    assert {item["state"] for item in packet["coverage"]} >= {
        "not_found_in_snapshot", "history_unknown"
    }


def test_partial_budget_excludes_whole_related_group_and_never_truncates_value():
    large = _fact("lab.large.series")
    large["value"] = {
        "type": "narrative", "unit": "text", "sections": {"report": "ö" * 3900},
        "attributes": {}, "truncated": False,
    }
    document = _document([large, _fact("lab.small.latest", key="small")])
    packet, entries = _select(document, indices=(0, 1), budget=2500)
    assert packet["canAnswer"]
    assert [item["ref"] for item in packet["evidence"]] == [entries[1]["ref"]]
    assert packet["tokenEstimate"] <= packet["tokenBudget"]
    assert any(item["state"] == "budget_excluded" for item in packet["coverage"])


def test_source_withheld_truncation_preview_and_missing_data_are_distinct():
    document = _document([_fact()])
    document["consent"]["withheldFamilies"] = ["medication"]
    document["limits"]["truncated"] = True
    document["release"]["mode"] = "preview"
    document["release"]["unreviewedDefinitions"] = ["lab.review|1.0.0"]
    document["safety"]["requiresHumanReview"] = True
    document["sections"]["lab"]["missingData"] = [{
        "code": "lab.other.latest", "evaluationKey": "latest", "missing": ["source"],
    }]
    packet, _ = _select(document, graph_status={"status": "unavailable"})
    assert packet["consent"] == document["consent"]
    assert {item["state"] for item in packet["coverage"]} >= {
        "withheld", "source_truncated", "source_missing_data", "history_unknown"
    }
    assert {item["code"] for item in packet["warnings"]} >= {
        "preview_release", "unreviewed_definitions", "human_review_required", "terminology_unavailable"
    }


def test_withheld_family_fact_cannot_be_selected_even_if_present():
    document = _document([_fact()])
    document["consent"]["withheldFamilies"] = ["lab"]
    packet, _ = _select(document)
    assert packet["evidence"] == []
    assert packet["excluded"][0]["reason"] == "access_withheld"


def test_shared_prescription_definition_does_not_link_unrelated_drugs():
    first, second = _fact("medication.history-item"), _fact("medication.history-item", key="other")
    for fact, code in ((first, "A"), (second, "B")):
        fact["value"] = {"type": "coded", "unit": "code", "system": "source",
                         "code": code, "display": code, "attributes": {}}
    packet, _ = _select(_document([first, second]))
    assert packet["counts"]["selectedFacts"] == 1
    assert packet["evidence"][0]["value"]["code"] == "A"


def test_cross_snapshot_evidence_is_rejected():
    document = _document([_fact()])
    entries = build_evidence(document, "another-snapshot")
    with pytest.raises(ValueError, match="scope mismatch"):
        select_context(document, "question", entries, [], snapshot_id=SNAPSHOT)


def test_clarification_blocks_answer_without_discarding_safety_context():
    packet, _ = _select(_document(insights=[_insight(_fact(), "critical")]),
                        plan={"needsClarification": True, "clarificationQuestion": "Hangi sonuç?"})
    assert "query_needs_clarification" in packet["blockers"]
    assert len(packet["evidence"]) == 2


def test_missing_insight_dependency_is_not_silently_ignored():
    document = _document(insights=[_insight(_fact(), "critical")])
    entries = build_evidence(document, SNAPSHOT)
    packet = select_context(document, "question", entries[:1], [], snapshot_id=SNAPSHOT)
    assert "mandatory_dependency_missing" in packet["blockers"]
    assert not packet["canAnswer"]


def test_all_warning_insights_are_in_compact_index_without_invented_dates():
    packet, _ = _select(_document([_fact()], [_insight(_fact("lab.other.latest"), "warning")]))
    assert len(packet["warningIndex"]) == 1
    assert packet["warningIndex"][0]["detailStatus"] == "retrieve_sources_before_interpreting"
    assert all(item["sourceKind"] != "insight" for item in packet["evidence"])
    assert "lab.review" in packet["glossary"]["rules"]


@pytest.mark.parametrize("severity", ["warning", "critical"])
def test_cross_family_withheld_insight_cannot_leak_via_parent_or_warning_index(severity):
    insight = _insight(_fact(), severity, name="cardio.cross-family")
    document = _document(insights=[insight])
    document["sections"] = {"cardio": document["sections"]["lab"]}
    document["glossary"]["families"]["cardio"] = {"label": "Kalp", "description": "Kaynak."}
    document["consent"]["withheldFamilies"] = ["lab"]
    unwrap_and_validate_cco(document)
    packet, _ = _select(document)
    assert packet["evidence"] == []
    assert packet["warningIndex"] == []
    assert packet["modelContext"]["evidence"] == []
    assert all(item["reason"] == "access_withheld" for item in packet["excluded"])
    assert all(set(item) == {"ref", "reason"} for item in packet["excluded"])
    if severity == "critical":
        assert "critical_source_withheld" in packet["blockers"]


def test_cross_family_frozen_fact_keeps_its_own_family_glossary():
    document = _document(insights=[_insight(_fact(), name="cardio.cross-family")])
    document["sections"] = {"cardio": document["sections"]["lab"]}
    document["glossary"]["families"]["cardio"] = {"label": "Kalp", "description": "Kaynak."}
    packet, _ = _select(document)
    assert set(packet["glossary"]["families"]) == {"lab", "cardio"}


def test_semantic_only_candidates_are_uncertain_not_absent_or_auto_accepted():
    document = _document([_fact()])
    entries = build_evidence(document, SNAPSHOT)
    packet = select_context(document, "Alakasız bir soru", entries, [
        {"ref": entries[0]["ref"], "score": 0.016, "channels": ["semantic"], "semanticScore": 0.8}
    ], snapshot_id=SNAPSHOT)
    assert not packet["canAnswer"]
    assert packet["blockers"] == ["retrieval_uncertain"]
    assert packet["evidence"] == []
    assert "not_found_in_snapshot" not in {item["state"] for item in packet["coverage"]}
    assert packet["excluded"][0]["reason"] == "semantic_relevance_unverified"
    assert packet["excluded"][0]["sourcePointer"] == entries[0]["sourcePointer"]
    assert packet["excluded"][0]["definition"] == entries[0]["definition"]
    assert packet["excluded"][0]["sourceKind"] == "fact"


def test_permitted_excluded_metadata_has_navigation_but_no_patient_value():
    document = _document([_fact()])
    packet, entries = _select(document, indices=())
    assert packet["excluded"] == [{
        "ref": entries[0]["ref"], "reason": "not_retrieved",
        "sourcePointer": entries[0]["sourcePointer"],
        "definition": entries[0]["definition"], "sourceKind": "fact",
    }]


def test_semantic_only_unrelated_group_does_not_pollute_corroborated_query():
    document = _document([_fact(), _fact("lab.other.latest", key="unrelated")])
    entries = build_evidence(document, SNAPSHOT)
    packet = select_context(document, "Kreatinin", entries, [
        {"ref": entries[0]["ref"], "score": 0.03, "channels": ["lexical", "semantic"]},
        {"ref": entries[1]["ref"], "score": 0.016, "channels": ["semantic"]},
    ], snapshot_id=SNAPSHOT)
    assert packet["canAnswer"]
    assert [entry["ref"] for entry in packet["evidence"]] == [entries[0]["ref"]]


def test_date_only_query_prioritizes_event_not_recorded_date_without_removing_history():
    from medical_ai_selective_context.selector import _date_match

    fact = _fact(year=2021)
    entry = build_evidence(_document([fact]), SNAPSHOT)[0]
    assert _date_match(entry, {"dateFrom": "2021-01-01", "dateTo": "2021-01-01"})
    assert not _date_match(entry, {"dateFrom": "2026-01-01", "dateTo": "2026-01-01"})
    frozen = build_evidence(_document(insights=[_insight(fact)]), SNAPSHOT)[1]
    assert not _date_match(frozen, {"dateFrom": "2026-01-01", "dateTo": "2026-01-01"})
