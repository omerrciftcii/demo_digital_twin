"""Drug index, hybrid section selection, and answer shape. No chart-specific rules."""

import json
from pathlib import Path
from time import perf_counter

from medical_ai_selective_context.index import normalize_text
from medical_ai_selective_context.text_document import parse_cco_text
from medical_ai_selective_context.text_drug_index import drug_index, drug_names
from medical_ai_selective_context.text_embed import embedding_model_name, semantic_section_scores
from medical_ai_selective_context.text_grounding import ground_answer
from medical_ai_selective_context.text_qa import SYSTEM_PROMPT
from medical_ai_selective_context.text_relevance import relevant_context, relevant_sections

ROOT = Path(__file__).parents[1]
PATIENTS = ROOT / "tests" / "fixtures" / "text" / "patients"
DICTIONARY = ROOT / "data" / "drug_classes.json"


def _record(name: str):
    payload = json.loads((PATIENTS / f"{name}.json").read_text(encoding="utf-8"))
    return parse_cco_text(payload)


def _tiny(text: str):
    return parse_cco_text({
        "format": "cco-text/1",
        "id": "00000000-0000-0000-0000-0000000000b1",
        "subjectId": "00000000-0000-0000-0000-0000000000b2",
        "asOf": "2026-10-11T12:00:00+03:00",
        "language": "tr",
        "text": text,
    })


def test_relevant_context_stays_under_twenty_milliseconds():
    drug_index()
    record = _record("patient-01")
    questions = [
        "Son laboratuvar değerim ne?",
        "Hâlâ Coumadin kullanıyor muyum?",
        "Tansiyonum nasıl?",
    ]
    for question in questions:
        relevant_context(record, question)
    started = perf_counter()
    repeats = 20
    for _ in range(repeats):
        for question in questions:
            relevant_context(record, question)
    elapsed = (perf_counter() - started) / (repeats * len(questions))
    assert elapsed < 0.020, elapsed


def test_one_character_typo_still_finds_the_dictionary_entry():
    payload = json.loads(DICTIONARY.read_text(encoding="utf-8"))
    entry = next(
        row for row in payload["entries"]
        if len(row["names"]) >= 2 and any(len(name) >= 8 for name in row["names"])
    )
    name = max((row for row in entry["names"] if len(row) >= 8), key=len)
    middle = len(name) // 2
    replacement = "q" if name[middle] != "q" else "x"
    typo = name[:middle] + replacement + name[middle + 1:]
    found = drug_names(typo)
    siblings = {other for other in entry["names"] if other != name}
    assert found & (siblings | {name}), (name, typo, found)


def test_embedding_miss_leaves_lexical_sections(monkeypatch):
    monkeypatch.setenv("TEXT_EMBEDDING_DISABLE", "true")
    import medical_ai_selective_context.text_embed as embed

    embed._ATTEMPTED = False
    record = _tiny(
        "Giriş.\n\n## Alfa ölçüm\nalfa sonucu burada.\n\n## Beta notu\nbeta baska satir.\n"
    )
    scores = semantic_section_scores(record, "alfa sonucu")
    assert scores == {}
    assert embedding_model_name() == ""
    sections = relevant_sections(record, "alfa sonucu")
    assert any("alfa sonucu" in section.text for section in sections)
    assert all("beta baska" not in section.text for section in sections)


def test_semantic_outlier_can_join_lexical_sections(monkeypatch):
    record = _tiny(
        "Giriş.\n\n## Alfa ölçüm\nalfa sonucu burada.\n\n"
        "## Beta notu\nbeta baska satir.\n\n## Gama notu\ngama ayri konu.\n"
    )

    def fake(_record, _question):
        return {0: 0.20, 1: 0.20, 2: 0.21, 3: 0.90}

    monkeypatch.setattr(
        "medical_ai_selective_context.text_relevance.semantic_section_scores", fake,
    )
    sections = relevant_sections(record, "alfa sonucu")
    headings = [section.heading for section in sections]
    assert "Alfa ölçüm" in headings
    assert "Gama notu" in headings
    assert "Beta notu" not in headings


def test_flat_semantic_scores_do_not_add_sections(monkeypatch):
    record = _tiny(
        "Giriş.\n\n## Alfa ölçüm\nalfa sonucu burada.\n\n## Beta notu\nbeta baska satir.\n"
    )

    def fake(_record, _question):
        return {index: 0.40 + index * 0.01 for index in range(len(record.sections))}

    monkeypatch.setattr(
        "medical_ai_selective_context.text_relevance.semantic_section_scores", fake,
    )
    sections = relevant_sections(record, "alfa sonucu")
    assert any("alfa sonucu" in section.text for section in sections)
    assert all("beta baska" not in section.text for section in sections)


def test_prompt_states_general_answer_shape_without_digits():
    assert "Önce sorunun kendisini cevapla" in SYSTEM_PROMPT
    assert "sen diye" in SYSTEM_PROMPT
    assert "atım sıklığı" in SYSTEM_PROMPT
    assert "Aynı cümleyi tekrarlama" in SYSTEM_PROMPT
    assert "uzmanlık dalı" in SYSTEM_PROMPT
    assert not __import__("re").search(r"\d", SYSTEM_PROMPT)


def test_repeated_sentence_is_dropped():
    record = _tiny("Giriş.\n\n## Laboratuvar (hastane)\n- eGFR 41 mL/dk (04.08.2026).\n")
    raw = (
        "DURUM: YANIT\n"
        "eGFR 41 mL/dk (04.08.2026). [Laboratuvar (hastane)]\n"
        "eGFR 41 mL/dk (04.08.2026). [Laboratuvar (hastane)]\n"
    )
    grounded = ground_answer(raw, record)
    assert grounded.ok
    assert normalize_text(grounded.answer).count(normalize_text("eGFR 41")) == 1


def test_think_block_cannot_add_a_number():
    record = _tiny("Giriş.\n\n## Ölçüm\n- Nabız 80/dk.\n")
    raw = "<think>999 mg</think>\nDURUM: YANIT\nNabız 80/dk. [Ölçüm]\n<think>12 °C"
    grounded = ground_answer(raw, record)
    assert grounded.ok
    assert "80/dk" in grounded.answer
    assert "999" not in grounded.answer
    assert "12" not in grounded.answer


def test_near_duplicate_sentence_is_dropped():
    record = _tiny("Giriş.\n\n## Ölçüm\n- Nabız 80/dk.\n- Kilo 70 kg.\n")
    raw = (
        "DURUM: YANIT\n"
        "Nabız 80/dk olarak duruyor. [Ölçüm]\n"
        "Nabız 80/dk olarak duruyor! [Ölçüm]\n"
        "Kilo 70 kg. [Ölçüm]\n"
    )
    grounded = ground_answer(raw, record)
    assert grounded.ok
    assert grounded.answer.count("80/dk") == 1
    assert "70 kg" in grounded.answer


def test_incompatible_unit_drops_only_that_sentence():
    record = _tiny(
        "Giriş.\n\n## Ölçüm\n- Nabız 80/dk.\n- Kilo 70 kg.\n"
        "- Klirens 44 mL/dk/1,73 m².\n"
    )
    rate = ground_answer("DURUM: YANIT\nNabız 80 bpm. [Ölçüm]\n", record)
    assert rate.ok and "bpm" in rate.answer
    beats = ground_answer("DURUM: YANIT\nNabız 80 atım/dk. [Ölçüm]\n", record)
    assert beats.ok and "atım/dk" in beats.answer
    duration = ground_answer("DURUM: YANIT\nNabız 80 dakika sürdü. [Ölçüm]\n", record)
    assert not duration.ok
    temperature = ground_answer("DURUM: YANIT\nNabız 80 °C. [Ölçüm]\n", record)
    assert not temperature.ok
    short = ground_answer("DURUM: YANIT\nKlirens 44 mL/dk. [Ölçüm]\n", record)
    assert short.ok and "44 mL/dk" in short.answer
    mixed = ground_answer(
        "DURUM: YANIT\nNabız 80 ms. [Ölçüm]\nKilo 70 kg. [Ölçüm]\n",
        record,
    )
    assert mixed.ok
    assert "70 kg" in mixed.answer
    assert "ms" not in mixed.answer


def test_startup_warmup_is_configurable(monkeypatch):
    import medical_ai_selective_context.text_drug_index as drugs
    import medical_ai_selective_context.text_embed as embed

    drugs.drug_index.cache_clear()
    monkeypatch.delenv("TEXT_STARTUP_WARMUP", raising=False)
    monkeypatch.setattr(embed, "warmup_encoder", lambda: False)
    embed.warmup_text_indexes()
    assert drugs.drug_index.cache_info().currsize == 1
    drugs.drug_index.cache_clear()
    monkeypatch.setenv("TEXT_STARTUP_WARMUP", "false")
    embed.warmup_text_indexes()
    assert drugs.drug_index.cache_info().currsize == 0
    drugs.drug_index()


def test_encoder_warmup_runs_one_batch_when_a_model_is_loaded(monkeypatch):
    import medical_ai_selective_context.text_embed as embed

    calls: list[list[str]] = []

    class _Fake:
        def encode(self, texts, **_kwargs):
            calls.append(list(texts))
            return texts

    monkeypatch.setattr(embed, "_load_model", lambda: _Fake())
    assert embed.warmup_encoder() is True
    assert calls and calls[0][0].startswith("passage:")
    monkeypatch.setattr(embed, "_load_model", lambda: None)
    assert embed.warmup_encoder() is False
