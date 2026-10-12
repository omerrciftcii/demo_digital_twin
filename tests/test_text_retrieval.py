"""Drug index, hybrid section selection, and answer shape. No chart-specific rules."""

import json
from pathlib import Path
from time import perf_counter

from medical_ai_selective_context.index import normalize_text
from medical_ai_selective_context.text_document import parse_cco_text
from medical_ai_selective_context.text_drug_index import drug_index, drug_names
from medical_ai_selective_context.text_embed import embedding_model_name, semantic_section_scores
from medical_ai_selective_context.text_grounding import _sentences, ground_answer
from medical_ai_selective_context.text_qa import ACK, SYSTEM_PROMPT, build_messages
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


def test_derived_sum_or_difference_is_grounded():
    record = _tiny("Giriş.\n\n## Laboratuvar\n- LDL 168 mg/dL. Önceki 71 mg/dL.\n")
    drop = ground_answer("DURUM: YANIT\nLDL 97 mg/dL düştü. [Laboratuvar]\n", record)
    assert drop.ok and "97" in drop.answer
    invented = ground_answer("DURUM: YANIT\nLDL 50 mg/dL düştü. [Laboratuvar]\n", record)
    assert not invented.ok
    other_unit = ground_answer("DURUM: YANIT\nLDL 97 kg düştü. [Laboratuvar]\n", record)
    assert not other_unit.ok
    total = _tiny("Giriş.\n\n## Ölçüm\n- Parça 20 mg. Diğer parça 22 mg.\n")
    summed = ground_answer("DURUM: YANIT\nToplam 42 mg. [Ölçüm]\n", total)
    assert summed.ok and "42" in summed.answer


def test_ordinal_period_does_not_split_a_sentence():
    assert _sentences("1. saat 80/dk sürdü. 2. doz 70 kg kaldı.") == [
        "1. saat 80/dk sürdü.",
        "2. doz 70 kg kaldı.",
    ]
    assert _sentences("Bitti. Yeni cümle başladı.") == ["Bitti.", "Yeni cümle başladı."]
    record = _tiny("Giriş.\n\n## Ölçüm\n- 1. saat 80/dk.\n- 2. doz 70 kg.\n")
    grounded = ground_answer(
        "DURUM: YANIT\n1. saat 80/dk sürdü. Uydurma 999 mg eklendi. [Ölçüm]\n",
        record,
    )
    assert grounded.ok
    assert "1. saat 80/dk" in grounded.answer
    assert "999" not in grounded.answer


def test_unsupported_cause_clause_is_removed():
    record = _tiny("Giriş.\n\n## Laboratuvar\n- LDL 168 mg/dL.\n")
    stripped = ground_answer(
        "DURUM: YANIT\nLDL 168 mg/dL çünkü yeni bir ilaç. [Laboratuvar]\n",
        record,
    )
    assert stripped.ok
    assert "168" in stripped.answer
    assert "çünkü" not in stripped.answer.casefold()
    assert "ilaç" not in stripped.answer.casefold()
    supported = _tiny("Giriş.\n\n## Laboratuvar\n- LDL 168 mg/dL, ilaç kesildi.\n")
    kept = ground_answer(
        "DURUM: YANIT\nLDL 168 mg/dL çünkü ilaç kesildi. [Laboratuvar]\n",
        supported,
    )
    assert kept.ok
    assert "çünkü" in kept.answer.casefold()
    assert "ilaç" in kept.answer.casefold()


def test_trend_direction_follows_the_record_order():
    wearable = _tiny("Giriş.\n\n## Giyilebilir\n- HRV olağanı 78 ms, güncel 41 ms.\n")
    missing = ground_answer(
        "DURUM: YANIT\nHRV 41 ms'den 79 ms'ye arttı. [Giyilebilir]\n",
        wearable,
    )
    assert not missing.ok
    swapped = ground_answer(
        "DURUM: YANIT\nHRV 41 ms'den 78 ms'ye arttı. [Giyilebilir]\n",
        wearable,
    )
    assert swapped.ok
    assert "79" not in swapped.answer
    folded = normalize_text(swapped.answer)
    assert "azal" in folded or "dus" in folded
    assert folded.find("78") < folded.find("41")
    steady = ground_answer(
        "DURUM: YANIT\nHRV 78 ms'den 41 ms'ye düştü. [Giyilebilir]\n",
        wearable,
    )
    assert steady.ok and "düştü" in steady.answer
    dated = _tiny(
        "Giriş.\n\n## Laboratuvar\n"
        "- LDL 80 mg/dL (01.01.2020).\n- LDL 100 mg/dL (01.01.2024).\n"
    )
    backward = ground_answer(
        "DURUM: YANIT\nLDL 100 mg/dL'den 80 mg/dL'ye düştü. [Laboratuvar]\n",
        dated,
    )
    assert backward.ok
    assert normalize_text(backward.answer).find("80") < normalize_text(backward.answer).find("100")
    assert "yükseldi" in backward.answer


def test_history_follows_the_cached_record_and_feeds_retrieval():
    record = _tiny(
        "Giriş.\n\n## Alfa ölçüm\nalfa sonucu burada.\n\n## Beta notu\nbeta baska satir.\n"
    )
    alone = build_messages(record, "peki o nasıl")
    assert [item["role"] for item in alone] == ["system", "user", "assistant", "user"]
    assert alone[2]["content"] == ACK
    assert "alfa sonucu" not in alone[-1]["content"]
    messages = build_messages(record, "peki o nasıl", [
        {"role": "user", "content": "alfa sonucu ne?"},
        {"role": "assistant", "content": "Buna bakıyorum."},
    ])
    assert messages[1]["content"].startswith("HASTA KAYDI")
    assert "peki o nasıl" not in messages[1]["content"]
    assert messages[3] == {"role": "user", "content": "alfa sonucu ne?"}
    assert messages[4]["role"] == "assistant"
    assert messages[-1]["content"].endswith("peki o nasıl")
    assert "alfa sonucu" in messages[-1]["content"]
    long_history = [{"role": "user", "content": f"tur {index} alfa"} for index in range(9)]
    capped = build_messages(record, "peki", long_history)
    prior = [item for item in capped if item["content"].startswith("tur ")]
    assert len(prior) == 6


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
