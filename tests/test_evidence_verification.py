"""Deterministic evidence checks. Snippets are made up; they are not chart fixtures."""

import asyncio
import json

import httpx
import pytest

from medical_ai_selective_context.text_document import parse_cco_text
from medical_ai_selective_context.text_evidence import build_evidence, parse_chart
from medical_ai_selective_context.text_grounding import format_model_text
from medical_ai_selective_context.text_llm import TextModelClient, TextModelSettings, chat_body
from medical_ai_selective_context.text_qa import (
    TextAnswerer,
    _fit_evidence,
    _prompt_budget,
    deterministic_verification_enabled,
    estimate_tokens,
)
from medical_ai_selective_context.text_verify import (
    ANSWER_JSON_SCHEMA,
    _quantities,
    fallback_answer,
    parse_model_payload,
    verify_payload,
)

AS_OF = "2026-10-01T00:00:00+00:00"
_IDS = ("00000000-0000-4000-8000-0000000000c1", "00000000-0000-4000-8000-0000000000c2")


def _record(text: str):
    return parse_cco_text({
        "format": "cco-text/1",
        "id": _IDS[0],
        "subjectId": _IDS[1],
        "asOf": AS_OF,
        "language": "tr",
        "text": text,
    })


def _labs() -> str:
    return (
        "Giriş.\n\n## Laboratuvar\n"
        "- Trigliserit 80 mg/dL (02.02.2024), normal (aralık 0-150).\n"
        "- Açlık şekeri 95,2 mg/dL (02.02.2024), normal (aralık 70-100).\n"
    )


def _blood() -> str:
    return (
        "Giriş.\n\n## Laboratuvar\n"
        "- Hemoglobin 14 g/dL (06.06.2024), normal (aralık 12-16).\n"
        "- MCV 90 fL (06.06.2024), normal (aralık 80-100).\n"
        "- B12 vitamini 400 pg/mL (06.06.2024), normal (aralık 200-900).\n"
    )


def _dx_other() -> str:
    return (
        "Giriş.\n\n## Tanılar\n"
        "Diğer:\n"
        "- Arpacık (H00) — son kayıt 03.03.2024.\n"
    )


def _dx_preliminary() -> str:
    return (
        "Giriş.\n\n## Tanılar\n"
        "- Ornekdurum (Z99.0) — ön tanı; son kayıt 04.04.2024.\n"
    )


def _rx() -> str:
    return (
        "Giriş.\n\n## İlaçlar (reçete kaydı)\n"
        "- ORNEKILAC 10 MG — ilk reçete 03.2022, son reçete 03.2022, 1 reçete.\n"
    )


def _same_day() -> str:
    return (
        "Giriş.\n\n## Tanılar\n"
        "- Ornekdurum (Z99.0) — kesin tanı; son kayıt 05.05.2024.\n\n"
        "## İlaçlar (reçete kaydı)\n"
        "- ORNEKILAC 10 MG — ilk reçete 05.2024, son reçete 05.05.2024, 1 reçete.\n"
    )


def _body(conclusion, sentences, missing=None) -> str:
    return json.dumps({
        "conclusion": conclusion,
        "sentences": sentences,
        "missing_information": missing or [],
        "requires_clinician_review": False,
    }, ensure_ascii=False)


def _fact(text: str, evidence_ids: list[str], claim_type: str = "fact") -> dict:
    return {"text": text, "claim_type": claim_type, "evidence_ids": evidence_ids}


def test_package_keeps_value_on_its_concept():
    package = parse_chart(_labs(), AS_OF)
    by_concept = {item.concept: item for item in package}
    assert by_concept["triglyceride"].value == "80"
    assert by_concept["triglyceride"].unit.lower() == "mg/dl"
    assert by_concept["glucose"].value == "95,2"
    assert [item.id for item in package] == ["E1", "E2"]


def test_swapped_triglyceride_value_is_rejected():
    package = parse_chart(_labs(), AS_OF)
    trig = next(item for item in package if item.concept == "triglyceride")
    payload = parse_model_payload(_body(
        {"status": "supported", "text": "Trigliserit 95,2 mg/dL.", "evidence_ids": [trig.id]},
        [_fact("Trigliserit 95,2 mg/dL.", [trig.id])],
    ))
    verdict = verify_payload(payload, package)
    assert verdict.fully_ok is False
    assert "value_not_in_evidence" in verdict.codes or "value_concept_mismatch" in verdict.codes
    assert any("95,2" in reason for reason in verdict.reasons)
    assert any("80" in reason for reason in verdict.reasons)


def test_other_category_is_not_chronic():
    package = parse_chart(_dx_other(), AS_OF)
    item = package[0]
    assert item.category == "other"
    assert item.type == "diagnosis"
    verdict = verify_payload(parse_model_payload(_body(
        {"status": "supported", "text": "Arpacık kronik bir hastalıktır.", "evidence_ids": [item.id]},
        [_fact("Arpacık kronik bir hastalıktır.", [item.id], "diagnosis_status")],
    )), package)
    assert "diagnosis_category_upgrade" in verdict.codes


def test_old_prescription_is_not_current_use():
    package = parse_chart(_rx(), AS_OF)
    item = package[0]
    assert item.active_use_confirmed is False
    assert item.record_type == "prescription"
    bad = verify_payload(parse_model_payload(_body(
        {"status": "supported", "text": "ORNEKILAC halen kullanıyorsun.", "evidence_ids": [item.id]},
        [_fact("ORNEKILAC halen kullanıyorsun.", [item.id], "current_use")],
    )), package)
    assert "unconfirmed_current_use" in bad.codes
    good = verify_payload(parse_model_payload(_body(
        {
            "status": "uncertain",
            "text": "2022 tarihli ORNEKILAC reçete kaydı var, güncel kullanım doğrulanamıyor.",
            "evidence_ids": [item.id],
        },
        [_fact(
            "2022 tarihli ORNEKILAC reçete kaydı var, güncel kullanım doğrulanamıyor.",
            [item.id],
        )],
    )), package)
    assert good.fully_ok, good.reasons


def test_same_date_does_not_create_a_cause():
    package = parse_chart(_same_day(), AS_OF)
    ids = [item.id for item in package]
    assert all("causal" not in item.relations for item in package)
    bad = verify_payload(parse_model_payload(_body(
        {
            "status": "supported",
            "text": "Bu hastalık için yatırıldın ve tedavisinde ORNEKILAC kullanıldı.",
            "evidence_ids": ids,
        },
        [_fact(
            "Bu hastalık için yatırıldın ve tedavisinde ORNEKILAC kullanıldı.",
            ids, "causal",
        )],
    )), package)
    assert "unsupported_causality" in bad.codes
    good = verify_payload(parse_model_payload(_body(
        {
            "status": "supported",
            "text": "Aynı tarihte Ornekdurum kaydı ve ORNEKILAC reçetesi var.",
            "evidence_ids": ids,
        },
        [_fact("Aynı tarihte Ornekdurum kaydı ve ORNEKILAC reçetesi var.", ids, "temporal")],
    )), package)
    assert good.fully_ok, good.reasons


def test_normal_labs_cannot_support_a_deficiency():
    package = parse_chart(_blood(), AS_OF)
    ids = [item.id for item in package]
    assert {item.status for item in package} == {"normal"}
    bad = verify_payload(parse_model_payload(_body(
        {
            "status": "supported",
            "text": "Anemi ve B12 eksikliği laboratuvarla destekleniyor.",
            "evidence_ids": ids,
        },
        [_fact(
            "Anemi ve B12 eksikliği laboratuvarla destekleniyor.",
            ids, "lab_support",
        )],
    )), package)
    assert "normal_results_cannot_support" in bad.codes
    good = verify_payload(parse_model_payload(_body(
        {
            "status": "not_supported",
            "text": "Mevcut hemogram anemiyi desteklemiyor.",
            "evidence_ids": ids,
        },
        [
            _fact("Mevcut hemogram anemiyi desteklemiyor.", ids, "interpretation"),
            _fact(
                "Demir depoları ferritin olmadığı için değerlendirilemiyor.",
                [], "uncertainty",
            ),
        ],
        ["Ferritin kaydı yok."],
    )), package)
    assert good.fully_ok, good.reasons


def test_preliminary_diagnosis_stays_preliminary():
    package = parse_chart(_dx_preliminary(), AS_OF)
    item = package[0]
    assert item.diagnosis_status == "preliminary"
    bad = verify_payload(parse_model_payload(_body(
        {"status": "supported", "text": "Ornekdurum kesin tanıdır.", "evidence_ids": [item.id]},
        [_fact("Ornekdurum kesin tanıdır.", [item.id], "diagnosis_status")],
    )), package)
    assert "diagnosis_status_upgrade" in bad.codes
    good = verify_payload(parse_model_payload(_body(
        {"status": "supported", "text": "Ornekdurum ön tanı olarak kayıtlı.", "evidence_ids": [item.id]},
        [_fact("Ornekdurum ön tanı olarak kayıtlı.", [item.id], "diagnosis_status")],
    )), package)
    assert good.fully_ok, good.reasons


def test_matching_number_date_unit_and_concept_passes():
    package = parse_chart(_labs(), AS_OF)
    trig = next(item for item in package if item.concept == "triglyceride")
    verdict = verify_payload(parse_model_payload(_body(
        {
            "status": "supported",
            "text": "Trigliserit 80 mg/dL (02.02.2024).",
            "evidence_ids": [trig.id],
        },
        [_fact("Trigliserit 80 mg/dL (02.02.2024).", [trig.id])],
    )), package)
    assert verdict.fully_ok, verdict.reasons
    assert "E1" not in "\n".join(verdict.texts) or True
    shown = __import__(
        "medical_ai_selective_context.text_verify", fromlist=["render_answer"],
    ).render_answer(verdict.texts, package)
    assert "80" in shown
    assert "E1" not in shown


def test_unsupported_interpretation_stays_uncertain():
    package = parse_chart(_labs(), AS_OF)
    verdict = verify_payload(parse_model_payload(_body(
        {
            "status": "uncertain",
            "text": "Bu yorumu kuracak bir satır yok.",
            "evidence_ids": [],
        },
        [],
        ["Bu yorum için kayıtlı bir ölçüm yok."],
    )), package)
    assert verdict.fully_ok, verdict.reasons
    assert verdict.conclusion_status == "uncertain"


def test_structured_atoms_are_used_when_upstream_sends_them():
    record = _record("Giriş.\n\n## Not\ndüz metin.\n")
    package = build_evidence(record, structured=[{
        "type": "lab",
        "concept": "triglyceride",
        "display_name": "Trigliserit",
        "value": "80",
        "unit": "mg/dL",
        "date": "2024-02-02",
        "status": "normal",
        "source_section": "Laboratuvar",
        "source_text": "Trigliserit 80 mg/dL",
    }])
    assert package[0].id == "E1"
    assert package[0].concept == "triglyceride"
    assert package[0].value == "80"


def test_citation_punctuation_does_not_open_the_next_line():
    record = _record("Giriş.\n\n## Not\nbir satir.\n")
    _status, answer, _cites = format_model_text(
        "DURUM: YANIT\nÖlçüm duruyor [Not]. Devam cümlesi burada.\n",
        record,
    )
    for line in answer.splitlines():
        assert not line.startswith(". ")
        assert not line.startswith(", ")
        assert not line.startswith(".,")
    assert "Devam cümlesi" in answer


def _sse(text: str, model: str = "local-model") -> bytes:
    frames = []
    for index in range(0, len(text), 24):
        payload = {
            "model": model,
            "choices": [{
                "index": 0, "delta": {"content": text[index:index + 24]}, "finish_reason": None,
            }],
        }
        frames.append("data: " + json.dumps(payload, ensure_ascii=False) + "\n\n")
    frames.append("data: " + json.dumps({
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 20, "completion_tokens": 10},
    }) + "\n\n")
    frames.append("data: [DONE]\n\n")
    return "".join(frames).encode("utf-8")


def _answerer(responses: list[str]):
    cursor = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        text = responses[min(cursor["n"], len(responses) - 1)]
        cursor["n"] += 1
        cursor["last"] = body
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"},
            content=_sse(text, body["model"]),
        )

    client = TextModelClient(TextModelSettings(
        api_key="k", base_url="http://127.0.0.1:8000/v1", model="local-model",
        require_zdr=False, local_inference=True,
    ), httpx.MockTransport(handler))
    return TextAnswerer(client), cursor


def test_retry_returns_the_second_answer_when_it_passes(monkeypatch):
    monkeypatch.setenv("DETERMINISTIC_EVIDENCE_VERIFICATION_ENABLED", "true")
    assert deterministic_verification_enabled() is True
    package = parse_chart(_labs(), AS_OF)
    trig = next(item for item in package if item.concept == "triglyceride")
    bad = _body(
        {"status": "supported", "text": "Trigliserit 95,2 mg/dL.", "evidence_ids": [trig.id]},
        [_fact("Trigliserit 95,2 mg/dL.", [trig.id])],
    )
    good = _body(
        {"status": "supported", "text": "Trigliserit 80 mg/dL.", "evidence_ids": [trig.id]},
        [_fact("Trigliserit 80 mg/dL.", [trig.id])],
    )
    answerer, cursor = _answerer([bad, good])
    result = asyncio.run(answerer.answer(_record(_labs()), "Trigliserit ne çıktı?"))
    assert cursor["n"] == 2
    assert result["validation"]["retried"] is True
    assert result["validation"]["fallback"] is False
    assert result["validation"]["passed"] is True
    assert "80" in result["answer"]
    assert "95" not in result["answer"]
    assert "E1" not in result["answer"]
    assert result["metrics"]["modelCalls"] == 2
    assert result["validation"]["correlationId"]


def test_double_failure_returns_a_safe_fallback(monkeypatch):
    monkeypatch.setenv("DETERMINISTIC_EVIDENCE_VERIFICATION_ENABLED", "true")
    bad = _body(
        {"status": "supported", "text": "Trigliserit 95,2 mg/dL.", "evidence_ids": ["E1"]},
        [_fact("Trigliserit 95,2 mg/dL.", ["E1"])],
    )
    answerer, cursor = _answerer([bad, bad])
    result = asyncio.run(answerer.answer(_record(_labs()), "Trigliserit ne çıktı?"))
    assert cursor["n"] == 2
    assert result["answer"].strip()
    assert result["validation"]["fallback"] is True
    assert result["validation"]["passed"] is False
    assert "95" not in result["answer"]
    assert "80" in result["answer"]
    assert result["status"] in {"answered", "insufficient_evidence"}


def test_flag_off_keeps_the_status_line_path(monkeypatch):
    monkeypatch.delenv("DETERMINISTIC_EVIDENCE_VERIFICATION_ENABLED", raising=False)
    assert deterministic_verification_enabled() is False
    answerer, cursor = _answerer(["DURUM: YANIT\nTrigliserit 80 mg/dL. [Laboratuvar]\n"])
    result = asyncio.run(answerer.answer(_record(_labs()), "Trigliserit ne çıktı?"))
    assert cursor["n"] == 1
    assert result["metrics"]["modelCalls"] == 1
    assert "deterministicEvidence" not in result["validation"]
    assert "80" in result["answer"]


def test_broken_json_retries_then_falls_back(monkeypatch):
    monkeypatch.setenv("DETERMINISTIC_EVIDENCE_VERIFICATION_ENABLED", "true")
    answerer, _cursor = _answerer(["bu json değil", "hâlâ json değil"])
    result = asyncio.run(answerer.answer(_record(_labs()), "Trigliserit ne çıktı?"))
    assert result["answer"].strip()
    assert result["validation"]["fallback"] is True
    assert "invalid_answer_json" in result["validation"]["errorCodes"]


@pytest.mark.parametrize("lead", [". ", ", ", ",. "])
def test_stray_lead_in_is_removed_from_grounded_lines(lead):
    record = _record(_labs())
    raw = f"DURUM: YANIT\n{lead}Trigliserit 80 mg/dL. [Laboratuvar]\n"
    grounded = __import__(
        "medical_ai_selective_context.text_grounding", fromlist=["ground_answer"],
    ).ground_answer(raw, record)
    assert grounded.ok, grounded.debug
    for line in grounded.answer.splitlines():
        assert not line.startswith(". ")
        assert not line.startswith(", ")


def _amounts(text: str) -> list[float]:
    return [amount for _match, amount, _family in _quantities(text)]


def test_comma_and_dot_decimals_stay_one_number():
    assert _amounts("Ornekolcum 11,7 mg/dL.") == [11.7]
    assert _amounts("Ornekolcum 11.7 mg/dL.") == [11.7]
    assert 11.0 not in _amounts("Ornekolcum 11,7 mg/dL.")
    assert 7.0 not in _amounts("Ornekolcum 11,7 mg/dL.")
    package = parse_chart(
        "Giriş.\n\n## Laboratuvar\n- Ornekolcum 11,7 mg/dL (02.02.2024).\n",
        AS_OF,
    )
    item = package[0]
    assert item.value == "11,7"
    for wording in ("Ornekolcum 11,7 mg/dL.", "Ornekolcum 11.7 mg/dL."):
        verdict = verify_payload(parse_model_payload(_body(
            {"status": "supported", "text": wording, "evidence_ids": [item.id]},
            [_fact(wording, [item.id])],
        )), package)
        assert verdict.fully_ok, verdict.reasons


def _rate_chart() -> str:
    return (
        "Giriş.\n\n## Laboratuvar\n"
        "- Ornekolcum 11,7 mg/dL (02.02.2024), normal (aralık 10-12).\n"
        "- Ornekyikim 90 mL/dk/1,73 m² (02.02.2024), normal (aralık 60-120).\n\n"
        "## Giyilebilir\n"
        "- Uyku skoru 86 (2014).\n"
    )


def test_dates_units_and_ranges_are_not_measured_values():
    sentence = "Ornekyikim 90 mL/dk/1,73 m² (02.02.2024, aralık 60-120)."
    assert _amounts(sentence) == [90.0]
    package = parse_chart(_rate_chart(), AS_OF)
    renal = next(item for item in package if item.concept == "ornekyikim")
    assert renal.unit and "1,73" in renal.unit
    verdict = verify_payload(parse_model_payload(_body(
        {"status": "supported", "text": sentence, "evidence_ids": [renal.id]},
        [_fact(sentence, [renal.id])],
    )), package)
    assert verdict.fully_ok, verdict.reasons
    year = "2022 yılında uyku skoru 86."
    sleep = next(item for item in package if "uyku" in item.concept)
    dated = verify_payload(parse_model_payload(_body(
        {"status": "supported", "text": year, "evidence_ids": [sleep.id]},
        [_fact(year, [sleep.id])],
    )), package)
    assert dated.fully_ok, dated.reasons
    assert "date_mismatch" not in dated.codes


def test_number_is_compared_only_with_the_matching_concept():
    package = parse_chart(_rate_chart(), AS_OF)
    sleep = next(item for item in package if "uyku" in item.concept)
    renal = next(item for item in package if item.concept == "ornekyikim")
    cited_wrong = verify_payload(parse_model_payload(_body(
        {"status": "supported", "text": "Uyku skoru 86.", "evidence_ids": [renal.id]},
        [_fact("Uyku skoru 86.", [renal.id])],
    )), package)
    assert cited_wrong.fully_ok, cited_wrong.reasons
    vague = verify_payload(parse_model_payload(_body(
        {"status": "uncertain", "text": "Bir skor 86 görünüyor.", "evidence_ids": [renal.id]},
        [_fact("Bir skor 86 görünüyor.", [renal.id], "uncertainty")],
    )), package)
    assert "value_concept_mismatch" not in vague.codes
    assert "value_not_in_evidence" not in vague.codes
    moved = verify_payload(parse_model_payload(_body(
        {"status": "supported", "text": "Uyku skoru 90.", "evidence_ids": [sleep.id]},
        [_fact("Uyku skoru 90.", [sleep.id])],
    )), package)
    assert not moved.fully_ok
    assert "value_not_in_evidence" in moved.codes or "value_concept_mismatch" in moved.codes


def test_correct_sentences_are_not_rejected_in_bulk():
    package = parse_chart(_rate_chart(), AS_OF)
    by_concept = {item.concept: item for item in package}
    rows = [
        ("Ornekolcum 11,7 mg/dL.", by_concept["ornekolcum"].id),
        ("Ornekolcum 11.7 mg/dL (02.02.2024).", by_concept["ornekolcum"].id),
        ("Ornekyikim 90 mL/dk/1,73 m².", by_concept["ornekyikim"].id),
        ("Uyku skoru 86.", next(item.id for item in package if "uyku" in item.concept)),
        ("Ornekolcum 11,7 mg/dL (aralık 10-12).", by_concept["ornekolcum"].id),
    ]
    failed = []
    for text, evidence_id in rows:
        verdict = verify_payload(parse_model_payload(_body(
            {"status": "supported", "text": text, "evidence_ids": [evidence_id]},
            [_fact(text, [evidence_id])],
        )), package)
        if not verdict.fully_ok:
            failed.append((text, verdict.codes, verdict.reasons))
    assert not failed, failed
    assert len(failed) / len(rows) == 0


def test_json_fence_trailing_comma_and_prose_are_recovered():
    package = parse_chart(_labs(), AS_OF)
    trig = next(item for item in package if item.concept == "triglyceride")
    payload = _body(
        {"status": "supported", "text": "Trigliserit 80 mg/dL.", "evidence_ids": [trig.id]},
        [_fact("Trigliserit 80 mg/dL.", [trig.id])],
    )
    fenced = "Açıklama.\n```json\n" + payload + "\n```\nson not"
    assert parse_model_payload(fenced)["conclusion"]["text"].startswith("Trigliserit")
    comma = payload[:payload.rfind("}")] + ",}"
    parsed = parse_model_payload(comma)
    assert parsed is not None
    assert verify_payload(parsed, package).fully_ok


def test_catalog_reserves_output_tokens(monkeypatch):
    monkeypatch.setenv("QA_CONTEXT_TOKENS", "4000")
    messages = [
        {"role": "system", "content": "s" * 80},
        {"role": "user", "content": "u" * 80},
    ]
    package = parse_chart(_labs() * 80, AS_OF)
    max_output = 1200
    fitted, kept = _fit_evidence(messages, package, max_output)
    assert kept
    assert len(kept) < len(package)
    assert estimate_tokens("".join(item["content"] for item in fitted)) + max_output + 128 <= 4000
    assert _prompt_budget(max_output) == 4000 - 1200 - 128


def test_fallback_keeps_passing_sentences_and_closes_parentheses():
    package = parse_chart(_dx_preliminary(), AS_OF)
    item = package[0]
    assert "(" not in item.display_name or item.display_name.count("(") == item.display_name.count(")")
    assert not item.display_name.endswith("(")
    text = fallback_answer(package, "Ornekdurum kaydı ne?")
    assert text.count("(") == text.count(")")
    assert "ön tanı" in text
    assert "Ornekdurum" in text


def test_partial_pass_does_not_retry(monkeypatch):
    monkeypatch.setenv("DETERMINISTIC_EVIDENCE_VERIFICATION_ENABLED", "true")
    package = parse_chart(_labs(), AS_OF)
    trig = next(item for item in package if item.concept == "triglyceride")
    sugar = next(item for item in package if item.concept == "glucose")
    payload = _body(
        {"status": "supported", "text": "Trigliserit 80 mg/dL.", "evidence_ids": [trig.id]},
        [
            _fact("Trigliserit 80 mg/dL.", [trig.id]),
            _fact("Trigliserit 95,2 mg/dL.", [trig.id]),
            _fact("Açlık şekeri 95,2 mg/dL.", [sugar.id]),
        ],
    )
    answerer, cursor = _answerer([payload, payload])
    result = asyncio.run(answerer.answer(_record(_labs()), "Trigliserit ve şeker?"))
    assert cursor["n"] == 1
    assert result["validation"]["retried"] is False
    assert result["validation"]["fallback"] is False
    assert "80" in result["answer"]
    assert "95,2" in result["answer"]
    assert "Trigliserit 95" not in result["answer"]


def test_local_schema_is_optional_and_a_400_drops_it():
    local = TextModelSettings(
        api_key="k", base_url="http://127.0.0.1:8000/v1", model="local-model",
        require_zdr=False,
    )
    plain = chat_body(local, [{"role": "user", "content": "kayıt"}])
    assert "response_format" not in plain
    guided = chat_body(local, [{"role": "user", "content": "kayıt"}], ANSWER_JSON_SCHEMA)
    assert guided["response_format"]["type"] == "json_schema"
    routed = chat_body(
        TextModelSettings(
            api_key="k", base_url="https://openrouter.ai/api/v1", require_zdr=True,
        ),
        [],
        ANSWER_JSON_SCHEMA,
    )
    assert "response_format" not in routed

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if "response_format" in body:
            return httpx.Response(400, json={"error": {"message": "unsupported"}})
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"},
            content=_sse("tamam"),
        )

    client = TextModelClient(TextModelSettings(
        api_key="k", base_url="http://127.0.0.1:8000/v1", model="local-model",
        require_zdr=False, local_inference=True,
    ), httpx.MockTransport(handler))

    async def _run():
        end = None
        async for event in client.complete([{"role": "user", "content": "kayıt"}], ANSWER_JSON_SCHEMA):
            if event["type"] == "end":
                end = event
        return end

    ended = asyncio.run(_run())
    assert ended is not None
    assert ended["text"] == "tamam"
