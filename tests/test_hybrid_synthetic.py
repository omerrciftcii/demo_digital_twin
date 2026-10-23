import json
from pathlib import Path

import httpx
import pytest

from medical_ai_selective_context.hybrid_synthetic import (
    DRAFT_SCHEMA_VERSION,
    DraftTaxonomyError,
    build_fixture_drafts,
    build_openai_draft_request,
    build_openrouter_draft_request,
    draft_taxonomy_errors,
    generate_hybrid_pilot,
    request_openai_drafts,
    request_openrouter_drafts,
)
from medical_ai_selective_context.training_dataset import validate_training_dataset


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_openai_request_contains_only_fictional_brief_and_strict_schema():
    body = build_openai_draft_request(model="test-model", root_count=20, seed=17)
    rendered = json.dumps(body, ensure_ascii=False)

    assert body["text"]["format"]["type"] == "json_schema"
    assert body["text"]["format"]["strict"] is True
    assert body["text"]["format"]["schema"]["additionalProperties"] is False
    assert body["store"] is False
    draft_schema = body["text"]["format"]["schema"]["$defs"]["FictionalScenarioDraft"]
    assert {"eventType", "recordSection", "subjectScope"} <= set(
        draft_schema["required"]
    )
    assert "subjectId" not in rendered
    assert "C:\\DigitalTwinData" not in rendered


def test_openai_response_is_schema_validated_before_use():
    batch = build_fixture_drafts(2)

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == "https://api.openai.com/v1/responses"
        assert request.headers["Authorization"] == "Bearer test-key"
        return httpx.Response(200, json={
            "status": "completed",
            "output": [{
                "type": "message",
                "content": [{
                    "type": "output_text",
                    "text": batch.model_dump_json(by_alias=True),
                }],
            }],
        })

    received, request_hash = request_openai_drafts(
        api_key="test-key",
        model="test-model",
        root_count=2,
        seed=17,
        transport=httpx.MockTransport(handler),
    )

    assert received.schema_version == DRAFT_SCHEMA_VERSION
    assert len(received.drafts) == 2
    assert len(request_hash) == 64


def test_openrouter_uses_strict_schema_and_validates_response():
    batch = build_fixture_drafts(2)
    body = build_openrouter_draft_request(
        model="test/model", root_count=2, seed=17
    )
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["strict"] is True
    assert body["provider"] == {"require_parameters": True}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == "https://openrouter.ai/api/v1/chat/completions"
        return httpx.Response(200, json={
            "choices": [{
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": batch.model_dump_json(by_alias=True),
                },
            }],
        })

    received, request_hash = request_openrouter_drafts(
        api_key="test-key",
        model="test/model",
        root_count=2,
        seed=17,
        transport=httpx.MockTransport(handler),
    )
    assert len(received.drafts) == 2
    assert len(request_hash) == 64


def test_hybrid_pilot_assigns_split_before_four_variants(tmp_path: Path):
    package = generate_hybrid_pilot(
        tmp_path,
        build_fixture_drafts(20),
        seed=17,
        draft_provider="fixture",
    )
    rows = _rows(package.annotations)
    report = json.loads(package.report.read_text(encoding="utf-8"))
    validation = validate_training_dataset(
        package.annotations, package.manifest, package.project_root
    )

    assert len(rows) == 80
    assert len(_rows(package.event_graphs)) == 80
    assert len(list((package.package_root / "sources").glob("*.json"))) == 80
    by_root: dict[str, list[dict]] = {}
    for row in rows:
        by_root.setdefault(row["rootId"], []).append(row)
    assert len(by_root) == 20
    for variants in by_root.values():
        assert len(variants) == 4
        assert len({item["split"] for item in variants}) == 1
        assert {item["variant"] for item in variants} == {
            "base", "relevant_counterfactual", "irrelevant_counterfactual",
            "order_paraphrase",
        }
        assert all(item["trainable"] is False for item in variants)
        assert all(item["goldAnswer"] is None for item in variants)
        assert all(item["candidateAnswer"] for item in variants)
        assert all(item["provenance"]["reviewStatus"] == "unreviewed" for item in variants)

        indexed = {item["variant"]: item for item in variants}
        assert indexed["base"]["candidateAnswer"] != (
            indexed["relevant_counterfactual"]["candidateAnswer"]
        )
        assert indexed["base"]["candidateAnswer"] == (
            indexed["irrelevant_counterfactual"]["candidateAnswer"]
        )
        assert indexed["base"]["question"]["text"] != (
            indexed["order_paraphrase"]["question"]["text"]
        )

    assert validation["valid"] is True
    assert validation["trainable"] == {"false": 80}
    assert validation["readyForTraining"] is False
    assert report["sourceIntegrity"]["rootSplitLeakage"] == 0
    assert report["pilot"]["rootSplitCounts"] == {"dev": 2, "test": 2, "train": 16}
    assert report["pilot"]["exampleSplitCounts"] == {
        "dev": 8, "test": 8, "train": 64,
    }
    assert report["correctness"]["modelClinicalRelationshipsAcceptedAsTrue"] is False
    assert report["correctness"]["taxonomyErrors"] == 0
    assert report["correctness"]["taxonomyRenderMismatches"] == 0
    assert report["dataLeakage"]["realCcoSentToExternalApi"] is False
    assert report["review"]["trainableExamples"] == 0


@pytest.mark.parametrize(
    ("field", "value", "error_prefix"),
    [
        ("eventType", "family_history", "eventType:"),
        ("recordSection", "Aile öyküsü (hasta beyanı)", "recordSection:"),
        ("subjectScope", "family_member", "subjectScope:"),
    ],
)
def test_taxonomy_contract_rejects_wrong_field_matches(
    tmp_path: Path, field: str, value: str, error_prefix: str
):
    payload = build_fixture_drafts(1).model_dump(by_alias=True, mode="json")
    payload["drafts"][0][field] = value
    batch_type = type(build_fixture_drafts(1))
    batch = batch_type.model_validate(payload)

    errors = draft_taxonomy_errors(batch.drafts[0])
    assert any(error.startswith(error_prefix) for error in errors)
    with pytest.raises(DraftTaxonomyError) as captured:
        generate_hybrid_pilot(tmp_path, batch)
    assert captured.value.rejections[0]["draftId"] == "fixture-001"
    assert not (tmp_path / "data" / "training_v2" / "hybrid-pilot").exists()


@pytest.mark.parametrize(
    ("template_index", "bad_label", "semantic_error"),
    [
        (1, "günlük adım ortalaması", "semantic:non_sleep_wearable_concept"),
        (5, "anne kalp hastalığı aile öyküsü", "semantic:persistent_history_requires_patient_diagnosis"),
    ],
)
def test_taxonomy_contract_rejects_semantically_incompatible_concepts(
    template_index: int, bad_label: str, semantic_error: str
):
    payload = build_fixture_drafts(6).model_dump(by_alias=True, mode="json")
    payload["drafts"][template_index]["primaryLabel"] = bad_label
    batch_type = type(build_fixture_drafts(1))
    batch = batch_type.model_validate(payload)

    assert semantic_error in draft_taxonomy_errors(batch.drafts[template_index])


def _event_semantics(event: dict) -> dict:
    return {
        key: value
        for key, value in event.items()
        if key not in {"eventId"}
    }


def test_all_templates_preserve_the_four_variant_contract(tmp_path: Path):
    package = generate_hybrid_pilot(
        tmp_path, build_fixture_drafts(6), seed=91, package_name="variant-contract"
    )
    graph_rows = _rows(package.event_graphs)
    by_root: dict[str, dict[str, dict]] = {}
    for row in graph_rows:
        by_root.setdefault(row["rootId"], {})[row["variant"]] = row

    expected_kinds = {
        "laboratory_result": "laboratory",
        "sleep_wearable_measurement": "wearable",
        "prescription_record": "medication_prescription",
        "diagnosis_record": "diagnosis",
        "procedure_record": "procedure",
    }
    for variants in by_root.values():
        base = variants["base"]
        relevant = variants["relevant_counterfactual"]
        irrelevant = variants["irrelevant_counterfactual"]
        order = variants["order_paraphrase"]

        def primary(row: dict) -> dict:
            return next(
                event for event in row["scenario"]["events"]
                if event["eventId"].endswith("-primary")
            )

        def distractor(row: dict) -> dict:
            return next(
                event for event in row["scenario"]["events"]
                if event["eventId"].endswith("-distractor")
            )

        assert _event_semantics(primary(base)) != _event_semantics(primary(relevant))
        assert base["scenario"]["answer"] != relevant["scenario"]["answer"]

        assert _event_semantics(primary(base)) == _event_semantics(primary(irrelevant))
        assert base["scenario"]["answer"] == irrelevant["scenario"]["answer"]
        assert _event_semantics(distractor(base)) != _event_semantics(distractor(irrelevant))

        assert _event_semantics(primary(base)) == _event_semantics(primary(order))
        assert _event_semantics(distractor(base)) == _event_semantics(distractor(order))
        assert base["scenario"]["answer"] == order["scenario"]["answer"]
        assert base["scenario"]["question"] != order["scenario"]["question"]
        assert base["scenario"]["events"][0]["eventId"].endswith("-primary")
        assert order["scenario"]["events"][0]["eventId"].endswith("-distractor")

        taxonomy = base["taxonomy"]
        assert all(row["taxonomy"] == taxonomy for row in variants.values())
        assert primary(base)["kind"] == expected_kinds[taxonomy["eventType"]]
        source_path = package.package_root / "sources" / f'{base["scenario"]["snapshotId"]}.json'
        source = json.loads(source_path.read_text(encoding="utf-8"))
        assert f'## {taxonomy["recordSection"]}' in source["text"]
        assert taxonomy["subjectScope"] == "patient"

