import json
from copy import deepcopy
from pathlib import Path

import pytest
from evidence_support import synthetic_cco
from pydantic import ValidationError

from medical_ai_selective_context.evaluation import (
    EvaluationCase,
    aggregate_results,
    evaluate_dataset,
    read_cases,
    resolve_cco_file,
    score_selection,
)
from medical_ai_selective_context.evidence import build_evidence


def annotation(**changes):
    return EvaluationCase.model_validate({
        "caseId": "case-001", "ccoFile": "cco.json", "question": "Kreatinin ölçümleri",
        "requiredSourcePointers": ["/sections/lab/facts/0", "/sections/lab/facts/1"],
        "critical": True, "reviewStatus": "unreviewed", "failureOrigin": "unknown", **changes,
    })


def sources():
    document = synthetic_cco()
    entries = build_evidence(document, "snapshot-test")
    return document, entries


def test_empty_and_unreviewed_evaluations_cannot_pass():
    empty = aggregate_results([], 0)
    assert not empty["retrievalAcceptance"]["passed"]
    assert empty["allCases"]["requiredEvidenceRecall"] is None
    document, entries = sources()
    result = score_selection(document, annotation(), entries, {"evidence": entries})
    unreviewed = aggregate_results([result] * 100, 20)
    assert unreviewed["allCases"]["requiredEvidenceRecall"] == 1
    assert unreviewed["reviewedCases"]["cases"] == 0
    assert not unreviewed["retrievalAcceptance"]["passed"]


def test_existing_required_source_recall_and_absent_source_attribution():
    document, entries = sources()
    case = annotation(requiredSourcePointers=[
        "/sections/lab/facts/0", "/sections/lab/facts/1", "/sections/lab/facts/999",
    ])
    result = score_selection(document, case, entries, {"evidence": [entries[0]]})
    assert result["requiredPresent"] == 2
    assert result["matchedRequired"] == 1
    assert result["requiredEvidenceRecall"] == .5
    assert result["retrievalMisses"] == ["/sections/lab/facts/1"]
    assert result["missingInCco"] == ["/sections/lab/facts/999"]
    assert result["observedFailureOrigins"] == ["retrieval", "missing_in_cco"]
    assert result["criticalRetrievalMisses"] == 1
    assert result["criticalUpstreamGaps"] == 1


def test_absent_gold_is_not_counted_as_a_retrieval_failure_or_perfect_score():
    document, entries = sources()
    case = annotation(requiredSourcePointers=["/sections/lab/facts/999"],
                      failureOrigin="missing_in_cco")
    result = score_selection(document, case, entries, {"evidence": entries})
    assert result["requiredEvidenceRecall"] is None
    assert result["retrievalMisses"] == []
    assert result["observedFailureOrigins"] == ["missing_in_cco"]
    assert not result["failureAnnotationMismatch"]


def test_empty_gold_never_scores_perfect():
    document, entries = sources()
    result = score_selection(document, annotation(requiredSourcePointers=[]), entries,
                             {"evidence": entries})
    assert result["requiredEvidenceRecall"] is None
    assert result["emptyGoldSet"] is True


def test_wrong_selected_value_or_cross_snapshot_reference_is_invalid_citation():
    document, entries = sources()
    changed = deepcopy(entries[0])
    changed["source"]["value"]["amount"] = 100
    other_snapshot = deepcopy(entries[1])
    other_snapshot["snapshotId"] = "other-patient"
    result = score_selection(document, annotation(), entries,
                             {"evidence": [changed, other_snapshot]})
    assert result["invalidSelectedCitations"] == 2
    assert result["matchedRequired"] == 0


def test_changed_projected_value_is_invalid_even_if_raw_source_is_untouched():
    document, entries = sources()
    changed = deepcopy(entries[0])
    changed["value"]["amount"] = 100
    result = score_selection(document, annotation(), entries, {"evidence": [changed]})
    assert result["invalidSelectedCitations"] == 1
    assert result["matchedRequired"] == 0


def test_real_but_non_evidence_pointer_is_invalid_annotation():
    document, entries = sources()
    document["sections"]["lab"]["facts"][0]["facts"] = [{"sentinel": True}]
    # Bypass the case validator to exercise the scorer's independent validation.
    case = annotation().model_copy(update={
        "requiredSourcePointers": ["/sections/lab/facts/0/facts/0"],
    })
    result = score_selection(document, case, entries, {"evidence": []})
    assert result["invalidAnnotations"] == ["/sections/lab/facts/0/facts/0"]
    assert result["missingInCco"] == []


@pytest.mark.parametrize("pointer", [
    "/sections/lab/facts/-1", "/sections/lab/facts/00", "/sections/lab/facts/0/value",
    "/sections/lab/facts/0/facts/0", "/sections/a~3b/facts/0", "",
])
def test_bad_source_pointer_syntax_rejected(pointer):
    with pytest.raises(ValidationError):
        annotation(requiredSourcePointers=[pointer])


def test_duplicate_json_keys_and_duplicate_case_ids_are_rejected(tmp_path):
    path = tmp_path / "cases.jsonl"
    path.write_text('{"question":"a","question":"b"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="line 1"):
        read_cases(path)
    payload = annotation().model_dump_json()
    path.write_text(payload + "\n" + payload + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 2"):
        read_cases(path)


def test_cco_path_is_local_relative_scoped_and_unambiguous(tmp_path):
    dataset = tmp_path / "data" / "evaluation" / "cases.jsonl"
    dataset.parent.mkdir(parents=True)
    local_cco = dataset.parent / "cco.json"
    local_cco.write_text("{}", encoding="utf-8")
    assert resolve_cco_file("cco.json", dataset, tmp_path) == local_cco.resolve()
    for rejected in ("../secret.json", "https://example.org/a.json", str(local_cco.resolve())):
        with pytest.raises(ValueError):
            resolve_cco_file(rejected, dataset, tmp_path)


def test_valid_reviewed_retrieval_gate_and_each_gap_blocks_acceptance():
    document, entries = sources()
    result = score_selection(document, annotation(reviewStatus="reviewed"), entries,
                             {"evidence": entries, "canAnswer": True})
    assert aggregate_results([result] * 100, 15)["retrievalAcceptance"]["passed"]
    assert not aggregate_results([result] * 99, 15)["retrievalAcceptance"]["passed"]
    assert not aggregate_results([result] * 100, 14)["retrievalAcceptance"]["passed"]
    incomplete = deepcopy(result)
    incomplete["missingInCco"] = ["/sections/lab/facts/999"]
    assert not aggregate_results([result] * 99 + [incomplete], 15)["retrievalAcceptance"]["passed"]


def test_offline_runner_validates_and_scores_without_network_or_question_in_report(tmp_path):
    dataset = tmp_path / "cases.jsonl"
    (tmp_path / "cco.json").write_text(json.dumps(synthetic_cco()), encoding="utf-8")
    case = annotation()
    dataset.write_text(case.model_dump_json() + "\n", encoding="utf-8")
    report = evaluate_dataset(dataset, tmp_path, tmp_path / "runtime")
    assert report["modelApiCalls"] == 0
    assert report["index"]["semantic"] == "disabled"
    assert report["allCases"]["requiredEvidenceRecall"] == 1
    assert report["allCases"]["invalidSelectedCitations"] == 0
    assert not report["retrievalAcceptance"]["passed"]
    rendered = json.dumps(report)
    assert case.question not in rendered
    assert synthetic_cco()["subjectId"] not in rendered
    assert "cco.json" not in rendered


def test_seed_annotations_are_unreviewed_and_every_pointer_is_source_grounded():
    project_root = Path(__file__).resolve().parents[1]
    dataset = project_root / "docs" / "evaluation-cases.example.jsonl"
    cases = read_cases(dataset)
    assert len(cases) >= 15
    for case in cases:
        assert case.reviewStatus == "unreviewed"
        path = resolve_cco_file(case.ccoFile, dataset, project_root)
        document = json.loads(path.read_text(encoding="utf-8"))
        entries = build_evidence(document, "synthetic")
        result = score_selection(document, case, entries, {"evidence": entries})
        assert not result["missingInCco"]
        assert not result["invalidAnnotations"]
        assert result["requiredEvidenceRecall"] == 1
