import json
from copy import deepcopy
from pathlib import Path

import pytest
from pydantic import ValidationError

from medical_ai_selective_context.text_document import evidence_span_index, parse_cco_text
from medical_ai_selective_context.training_dataset import (
    DatasetValidationError,
    RankerPrediction,
    TrainingExample,
    read_training_examples,
    score_ranker_prediction,
    validate_training_dataset,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEED_ROOT = PROJECT_ROOT / "data" / "training_v2" / "seed"


def _seed_payload() -> dict:
    first = read_training_examples(SEED_ROOT / "annotations.jsonl")[0]
    return first.model_dump(by_alias=True)


def _seed_example(example_id: str) -> TrainingExample:
    return next(
        example
        for example in read_training_examples(SEED_ROOT / "annotations.jsonl")
        if example.example_id == example_id
    )


def _available_refs() -> set[str]:
    refs = set()
    for path in (SEED_ROOT / "sources").glob("*.json"):
        record = parse_cco_text(json.loads(path.read_text(encoding="utf-8")))
        refs.update(evidence_span_index(record))
    return refs


def _manifest(*, max_not_supported: float = 1.0) -> dict:
    return {
        "schemaVersion": "cco-text-training-manifest/0.3.0",
        "datasetId": "test-dataset-001",
        "purpose": "schema_smoke_test",
        "policy": {
            "personalizationPriority": True,
            "primarySource": "text_cco",
            "legacyFactRole": "legacy_audit_only",
            "maxNotSupportedFraction": max_not_supported,
            "minReviewedFraction": 0.0,
        },
    }


def _write_dataset(tmp_path: Path, rows: list[dict], manifest: dict | None = None):
    dataset = tmp_path / "annotations.jsonl"
    dataset.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest or _manifest(), ensure_ascii=False), encoding="utf-8"
    )
    return dataset, manifest_path


def test_committed_seed_is_source_grounded_and_keeps_none_as_regression_only():
    report = validate_training_dataset(
        SEED_ROOT / "annotations.jsonl", SEED_ROOT / "manifest.json", PROJECT_ROOT
    )
    assert report["valid"] is True
    assert report["examples"] == 10
    assert report["personalizationDecisions"] == {
        "required": 10,
    }
    assert report["notSupportedFraction"] == 0
    assert report["legacyFactRole"] == "legacy_audit_only"
    assert report["factIdsUsedAsPrimaryEvidence"] is False
    assert report["clinicalCorrectnessEvaluated"] is False
    rendered = json.dumps(report)
    assert "Son ana uykum" not in rendered
    assert "ee8e0db6-905b-50c2-8828-947cd48fc3cb" not in rendered


def test_not_supported_cannot_hide_positive_personal_context():
    payload = _seed_payload()
    payload["personalizationDecision"] = "not_supported"
    payload["answerPolicy"]["personalizedAnswerExpected"] = False
    with pytest.raises(ValidationError, match="cannot contain positive evidence"):
        TrainingExample.model_validate(payload)


def test_helpful_cannot_be_used_as_a_softer_name_for_required():
    payload = _seed_payload()
    payload["personalizationDecision"] = "helpful"
    with pytest.raises(ValidationError, match="helpful but no must_use"):
        TrainingExample.model_validate(payload)


def test_status_check_cannot_establish_causality():
    payload = _seed_payload()
    evidence = payload["evidenceAnnotations"][0]
    evidence["useRole"] = "status_check"
    evidence["causalStrength"] = "possible"
    with pytest.raises(ValidationError, match="Status checks cannot establish causality"):
        TrainingExample.model_validate(payload)


def test_missing_cco_reference_fails_without_echoing_question(tmp_path):
    payload = _seed_payload()
    payload["evidenceAnnotations"][0]["evidenceId"] = "evi_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    dataset, manifest = _write_dataset(tmp_path, [payload])
    with pytest.raises(DatasetValidationError, match="example number 1") as caught:
        validate_training_dataset(dataset, manifest, PROJECT_ROOT)
    assert payload["question"]["text"] not in str(caught.value)


def test_root_level_split_leakage_is_rejected(tmp_path):
    train = _seed_payload()
    test = deepcopy(train)
    test["exampleId"] = "seed-sleep-direct-test-001"
    test["split"] = "test"
    dataset, manifest = _write_dataset(tmp_path, [train, test])
    with pytest.raises(DatasetValidationError, match="roots cross"):
        validate_training_dataset(dataset, manifest, PROJECT_ROOT)


def test_manifest_caps_not_supported_instead_of_optimizing_for_it(tmp_path):
    payload = _seed_payload()
    payload["exampleId"] = "seed-no-context-test-001"
    payload["rootId"] = "seed-no-context-test-001"
    payload["personalizationDecision"] = "not_supported"
    payload["evidenceAnnotations"] = []
    payload["answerPolicy"]["personalizedAnswerExpected"] = False
    dataset, manifest = _write_dataset(
        tmp_path, [payload], _manifest(max_not_supported=0.05)
    )
    with pytest.raises(DatasetValidationError, match="exceed"):
        validate_training_dataset(dataset, manifest, PROJECT_ROOT)


def test_duplicate_json_keys_are_rejected_by_line_number(tmp_path):
    dataset = tmp_path / "annotations.jsonl"
    dataset.write_text('{"exampleId":"first","exampleId":"second"}\n', encoding="utf-8")
    with pytest.raises(DatasetValidationError, match="line 1"):
        read_training_examples(dataset)


def test_ranker_target_excludes_question_cco_path_and_gold_answer():
    example = TrainingExample.model_validate(_seed_payload())
    target = example.ranker_target()
    rendered = json.dumps(target, ensure_ascii=False)
    assert "question" not in target
    assert "ccoFile" not in target
    assert "goldAnswer" not in target
    assert example.question.text not in rendered
    assert target["personalizationDecision"] == "required"


def test_unreviewed_example_cannot_be_trainable():
    payload = _seed_payload()
    payload["trainable"] = True

    with pytest.raises(ValueError):
        TrainingExample.model_validate(payload)


def test_perfect_ranker_target_scores_perfectly():
    gold = _seed_example("synthetic-000003")
    score = score_ranker_prediction(gold, gold.ranker_target(), _available_refs())
    assert score["personalizationDecisionMatch"] is True
    assert score["requiredEvidenceRecall"] == 1
    assert score["adjudicatedSelectionPrecision"] == 1
    assert score["criticalEvidenceRecall"] == 1
    assert score["causalOverreachCount"] == 0
    assert score["statusRoleMisses"] == 0


def test_false_personal_link_is_not_hidden_inside_precision():
    gold = _seed_example("synthetic-000001")
    prediction = gold.ranker_target()
    irrelevant = prediction["evidence"][-1]
    irrelevant.update({
        "requiredness": "helpful",
        "useRole": "background",
        "causalStrength": "possible",
    })
    score = score_ranker_prediction(gold, prediction, _available_refs())
    assert score["adjudicatedSelectionPrecision"] == 0.5
    assert score["falsePositiveRefs"] == [irrelevant["evidenceId"]]
    assert score["causalOverreachCount"] == 1


def test_one_complete_alternative_evidence_set_satisfies_requirement():
    payload = _seed_example("synthetic-000005").model_dump(by_alias=True)
    first_ref = payload["evidenceAnnotations"][0]["evidenceId"]
    second = payload["evidenceAnnotations"][1]
    second["requiredness"] = "must_use"
    second["useRole"] = "direct_fact"
    second["reasonCode"] = "alternative_procedure_confirmation"
    second_ref = second["evidenceId"]
    payload["alternativeEvidenceSets"] = [[first_ref], [second_ref]]
    gold = TrainingExample.model_validate(payload)
    prediction = gold.ranker_target()
    prediction["evidence"] = prediction["evidence"][:1]
    prediction["alternativeEvidenceSets"] = []
    score = score_ranker_prediction(gold, prediction, _available_refs())
    assert score["alternativeEvidenceSatisfied"] is True
    assert score["requiredEvidenceUnits"] == 1
    assert score["requiredEvidenceRecall"] == 1


def test_source_grounded_but_unreviewed_selection_is_reported_not_assumed_wrong():
    gold = _seed_example("synthetic-000001")
    prediction = gold.ranker_target()
    other = _seed_example("synthetic-000002").evidence_annotations[0]
    unreviewed = other.model_dump(by_alias=True)
    unreviewed_ref = unreviewed["evidenceId"]
    unreviewed.update({
        "requiredness": "helpful", "useRole": "background",
        "causalStrength": "none", "safetyCritical": False,
        "reasonCode": "unreviewed_possible_context",
    })
    prediction["evidence"].append(unreviewed)
    score = score_ranker_prediction(gold, RankerPrediction.model_validate(prediction), _available_refs())
    assert score["adjudicatedSelectionPrecision"] == 1
    assert score["unadjudicatedRefs"] == [unreviewed_ref]
    assert score["invalidRefs"] == []

