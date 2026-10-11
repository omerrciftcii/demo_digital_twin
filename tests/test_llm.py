import asyncio
import json

import httpx
import pytest

from medical_ai_selective_context.llm import (
    ModelClient,
    ModelError,
    ModelSettings,
    outbound_context,
    strict_output_schema,
    validate_answer,
)
from medical_ai_selective_context.models import GroundedAnswer, SearchPlan


def test_strict_schema_requires_nullable_fields_and_nested_arrays():
    original = SearchPlan.model_json_schema(by_alias=True)
    schema = strict_output_schema(original)
    assert set(schema["required"]) == set(schema["properties"])
    assert "dateFrom" in schema["required"]
    assert "default" not in schema["properties"]["dateFrom"]
    assert "default" in original["properties"]["dateFrom"]
    answer_schema = strict_output_schema(GroundedAnswer.model_json_schema(by_alias=True))
    statement = answer_schema["$defs"]["AnswerStatement"]
    assert "values" in statement["required"]
    assert statement["additionalProperties"] is False


def packet():
    evidence = [{
        "ref": "local-ref", "snapshotId": "private-snapshot", "subjectId": "private-subject",
        "factId": "private-fact", "sourcePointer": "/sections/lab/facts/0",
        "value": {"type": "quantity", "amount": 0.9, "unit": "mg/dL"},
        "occurredFrom": "2026-01-01T00:00:00Z", "occurredTo": "2026-01-01T00:00:00Z",
        "producedAt": None,
    }]
    return {
        "canAnswer": True, "evidence": evidence,
        "modelContext": {"snapshotId": "private-snapshot", "evidence": evidence, "coverage": []},
    }


def answer(**updates):
    statement = {
        "text": "Kaynakta kreatinin 0,9 mg/dL olarak kayıtlı.",
        "evidenceRefs": ["E1"],
        "values": [
            {"ref": "E1", "field": "value.amount", "value": 0.9},
            {"ref": "E1", "field": "value.unit", "value": "mg/dL"},
        ],
    }
    statement.update(updates)
    return GroundedAnswer.model_validate({"statements": [statement]})


def test_outbound_context_remaps_citations_and_omits_internal_identifiers():
    context, reverse = outbound_context(packet())
    assert reverse == {"E1": "local-ref"}
    assert context["evidence"][0]["ref"] == "E1"
    assert "private-" not in json.dumps(context)
    assert "sourcePointer" not in json.dumps(context)


def test_valid_source_values_and_citations():
    text, citations, checks = validate_answer(answer(), packet(), {"E1": "local-ref"})
    assert "0,9" in text
    assert citations == [{"ref": "local-ref", "label": "E1"}]
    assert checks["clinicalMeaningVerified"] is False
    assert checks["exactValueSpansChecked"] is True


@pytest.mark.parametrize("text", ["Kreatinin 0,9 mg/dL.", "Kreatinin 0.9mg/dL."])
def test_exact_amount_and_unit_spans_allow_decimal_separator_and_spacing(text):
    result, _, _ = validate_answer(answer(text=text), packet(), {"E1": "local-ref"})
    assert text in result


@pytest.mark.parametrize("text", [
    "Kreatinin 0,9 mg/dL.", "Kreatinin 2026 mg/dL.", "Kreatinin 0,9 kg.",
])
def test_empty_values_never_support_a_numeric_statement(text):
    with pytest.raises(ModelError, match="unsupported_numeric_claim"):
        validate_answer(answer(text=text, values=[]), packet(), {"E1": "local-ref"})


@pytest.mark.parametrize("text", [
    "Kreatinin 2026 mg/dL.", "Kreatinin 0,9 kg.", "Kreatinin 0,9 mg/dLkg.",
    "Kreatinin 0,90 mg/dL.", "Kreatinin 0,9 mg/dL ve 2026 mg/dL.",
])
def test_correct_assertions_cannot_validate_wrong_or_extra_numeric_unit_spans(text):
    with pytest.raises(ModelError, match="unsupported_numeric_claim"):
        validate_answer(answer(text=text), packet(), {"E1": "local-ref"})


def test_amount_assertion_requires_unit_assertion_from_the_same_record():
    source = packet()
    source["evidence"].append({**source["evidence"][0], "ref": "other-ref"})
    parsed = answer(evidenceRefs=["E1", "E2"], values=[
        {"ref": "E1", "field": "value.amount", "value": 0.9},
        {"ref": "E2", "field": "value.unit", "value": "mg/dL"},
    ])
    with pytest.raises(ModelError, match="unsupported_numeric_claim"):
        validate_answer(parsed, source, {"E1": "local-ref", "E2": "other-ref"})


def test_full_asserted_date_literal_is_valid_but_date_parts_cannot_sponsor_measurements():
    values = [{"ref": "E1", "field": "occurredFrom", "value": "2026-01-01T00:00:00Z"}]
    validate_answer(answer(text="Kaynak olay tarihi 2026-01-01T00:00:00Z.", values=values),
                    packet(), {"E1": "local-ref"})
    for text in ("Kaynak tarihi 2026.", "Kreatinin 2026 mg/dL.",
                 "2026-01-01T00:00:00Z tarihinde kreatinin 2026 mg/dL."):
        with pytest.raises(ModelError, match="unsupported_numeric_claim"):
            validate_answer(answer(text=text, values=values), packet(), {"E1": "local-ref"})


def test_numeric_coded_display_must_be_exact_source_quotation():
    source = packet()
    source["evidence"][0]["value"] = {
        "type": "coded", "display": "Kaynak derece 2", "code": "TEST", "unit": "code",
    }
    values = [{"ref": "E1", "field": "value.display", "value": "Kaynak derece 2"}]
    validate_answer(answer(text='Kaynak tanımı "Kaynak derece 2".', values=values), source,
                    {"E1": "local-ref"})
    with pytest.raises(ModelError, match="unsupported_numeric_claim"):
        validate_answer(answer(text="Kaynak derece 2.", values=values), source, {"E1": "local-ref"})


def test_nonnumeric_explanation_can_have_no_structured_value_assertions():
    _, _, checks = validate_answer(answer(text="Kaynakta bir laboratuvar kaydı var.", values=[]),
                                   packet(), {"E1": "local-ref"})
    assert checks["checkedValueAssertions"] == 0
    assert checks["clinicalMeaningVerified"] is False


@pytest.mark.parametrize(("updates", "error"), [
    ({"evidenceRefs": ["E404"]}, "citation_not_in_context"),
    ({"values": [{"ref": "E1", "field": "value.amount", "value": 8.1}]}, "source_value_mismatch"),
    ({"text": "Kreatinin 8,1 mg/dL."}, "unsupported_numeric_claim"),
    ({"values": [{"ref": "E1", "field": "value.amount", "value": True}]}, "source_value_mismatch"),
    ({"values": [{"ref": "E2", "field": "value.amount", "value": 0.9}]},
     "value_citation_not_in_statement"),
])
def test_untrusted_model_output_is_rejected(updates, error):
    with pytest.raises(ModelError, match=error):
        validate_answer(answer(**updates), packet(), {"E1": "local-ref"})


def test_real_data_gate_is_before_transport():
    def forbidden(_):
        pytest.fail("Network called with blocked real patient data")
    client = ModelClient(ModelSettings(api_key="test-key"), httpx.MockTransport(forbidden))
    with pytest.raises(ModelError, match="external_real_data_blocked"):
        asyncio.run(client.plan("Soru", [], {}, "real", "2026-10-11T00:00:00Z"))
    with pytest.raises(ModelError, match="external_real_data_blocked"):
        asyncio.run(client.answer("Soru", [], packet(), "real"))


def test_synthetic_call_uses_fixed_model_strict_json_and_privacy_routing():
    requests = []
    def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        assert body["model"] == "qwen/qwen3.5-9b"
        assert body["provider"]["allow_fallbacks"] is False
        assert body["provider"]["zdr"] is True
        assert body["provider"]["data_collection"] == "deny"
        assert body["response_format"]["json_schema"]["strict"] is True
        assert body["reasoning"]["enabled"] is False
        return httpx.Response(200, json={
            "model": body["model"], "provider": "test-provider",
            "choices": [{"finish_reason": "stop", "message": {
                "content": answer().model_dump_json(by_alias=True),
            }}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 30},
        })
    client = ModelClient(ModelSettings(api_key="test-key"), httpx.MockTransport(handler))
    result = asyncio.run(client.answer("Kreatinin kaç?", [], packet(), "synthetic"))
    assert result["metrics"]["usage"]["prompt_tokens"] == 100
    assert len(requests) == 1


def test_provider_error_does_not_echo_sensitive_body():
    def handler(_):
        return httpx.Response(400, json={"error": "PHI AND SECRET THAT MUST NOT ESCAPE"})
    client = ModelClient(ModelSettings(api_key="test-key"), httpx.MockTransport(handler))
    with pytest.raises(ModelError) as error:
        asyncio.run(client.answer("Soru", [], packet(), "synthetic"))
    assert str(error.value) == "model_request_failed"


def test_partial_generation_is_not_a_valid_answer():
    def handler(_):
        return httpx.Response(200, json={
            "choices": [{"finish_reason": "length", "message": {"content": "{}"}}],
        })
    client = ModelClient(ModelSettings(api_key="test-key"), httpx.MockTransport(handler))
    with pytest.raises(ModelError, match="model_response_incomplete"):
        asyncio.run(client.answer("Soru", [], packet(), "synthetic"))


def test_model_substitution_is_not_silent():
    def handler(_):
        return httpx.Response(200, json={
            "model": "different-model", "choices": [{"finish_reason": "stop",
                                                    "message": {"content": "{}"}}],
        })
    client = ModelClient(ModelSettings(api_key="test-key"), httpx.MockTransport(handler))
    with pytest.raises(ModelError, match="unexpected_model"):
        asyncio.run(client.answer("Soru", [], packet(), "synthetic"))


@pytest.mark.parametrize("payload", [[], {"data": []}, {"data": {"endpoints": [None]}}])
def test_malformed_capability_metadata_degrades_without_crashing_startup(payload):
    client = ModelClient(ModelSettings(api_key="test-key"), httpx.MockTransport(
        lambda _: httpx.Response(200, json=payload),
    ))
    assert asyncio.run(client.check_capabilities())["status"] == "unavailable"


@pytest.mark.parametrize("payload", [
    [], {"choices": [None]}, {"choices": [{"message": []}]},
    {"model": 42, "choices": [{"finish_reason": "stop", "message": {"content": "{}"}}]},
    {"provider": [], "choices": [{"finish_reason": "stop", "message": {"content": "{}"}}]},
    {"usage": ["secret"], "choices": [{"finish_reason": "stop", "message": {"content": "{}"}}]},
])
def test_malformed_model_envelopes_return_stable_error_without_type_exceptions(payload):
    client = ModelClient(ModelSettings(api_key="test-key"), httpx.MockTransport(
        lambda _: httpx.Response(200, json=payload),
    ))
    with pytest.raises(ModelError, match="invalid_model_json"):
        asyncio.run(client.answer("Soru", [], packet(), "synthetic"))


def test_provider_usage_diagnostics_exclude_unknown_content_and_non_numeric_fields():
    payload = {
        "choices": [{"finish_reason": "stop", "message": {
            "content": answer().model_dump_json(by_alias=True),
        }}],
        "usage": {"prompt_tokens": 100, "cost": 0.001, "completion_tokens": "private",
                  "unexpected": {"private": "not a metric"}},
    }
    client = ModelClient(ModelSettings(api_key="test-key"), httpx.MockTransport(
        lambda _: httpx.Response(200, json=payload),
    ))
    result = asyncio.run(client.answer("Soru", [], packet(), "synthetic"))
    assert result["metrics"]["usage"] == {"prompt_tokens": 100, "cost": 0.001}


def unsupported_answer_transport():
    rejected = answer(text="Kreatinin 8,1 mg/dL.", evidenceRefs=["E1", "E404"])
    return httpx.MockTransport(lambda request: httpx.Response(200, json={
        "model": json.loads(request.content)["model"],
        "choices": [{"finish_reason": "stop", "message": {
            "content": rejected.model_dump_json(by_alias=True),
        }}],
    }))


def test_enforced_validation_hides_unsupported_answer():
    client = ModelClient(ModelSettings(api_key="test-key"), unsupported_answer_transport())
    with pytest.raises(ModelError, match="citation_not_in_context"):
        asyncio.run(client.answer("Soru", [], packet(), "synthetic"))


def test_disabled_validation_returns_answer_marked_unvalidated():
    client = ModelClient(
        ModelSettings(api_key="test-key", enforce_answer_validation=False),
        unsupported_answer_transport(),
    )
    result = asyncio.run(client.answer("Soru", [], packet(), "synthetic"))
    assert result["validated"] is False
    assert result["answer"] == "Kreatinin 8,1 mg/dL. [E1, E404]"
    assert result["citations"] == [{"ref": "local-ref", "label": "E1"}]
    assert result["validation"]["error"] == "citation_not_in_context"
    assert result["validation"]["enforced"] is False
    assert result["validation"]["clinicalMeaningVerified"] is False


def test_disabled_validation_still_rejects_real_data_before_transport():
    def forbidden(_):
        pytest.fail("Network called with blocked real patient data")
    client = ModelClient(
        ModelSettings(api_key="test-key", enforce_answer_validation=False),
        httpx.MockTransport(forbidden),
    )
    with pytest.raises(ModelError, match="external_real_data_blocked"):
        asyncio.run(client.answer("Soru", [], packet(), "real"))
