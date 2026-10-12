"""Drug index, hybrid section selection, and answer shape. No chart-specific rules."""

import json
from pathlib import Path
from time import perf_counter

from medical_ai_selective_context.index import normalize_text
from medical_ai_selective_context.text_document import parse_cco_text
from medical_ai_selective_context.text_drug_index import drug_index, drug_names
from medical_ai_selective_context.text_embed import embedding_model_name, semantic_section_scores
from medical_ai_selective_context.text_grounding import (
    _sentences,
    format_model_text,
    ground_answer,
)
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
    assert "Evet" in SYSTEM_PROMPT and "Hayır" in SYSTEM_PROMPT
    assert "Kısmen" in SYSTEM_PROMPT
    assert "Kayıtta yeterli bilgi yok" in SYSTEM_PROMPT
    assert "hüküm değildir" in SYSTEM_PROMPT
    assert "Evet, düzelmiş" in SYSTEM_PROMPT
    assert "Klinik rapor gibi yazma" in SYSTEM_PROMPT
    assert "Geçmiş olsun" in SYSTEM_PROMPT
    assert "Dikkat:" in SYSTEM_PROMPT
    assert "merak ediyorsun" in SYSTEM_PROMPT
    assert "Hekimden en fazla bir kez" in SYSTEM_PROMPT
    assert "tire ile" in SYSTEM_PROMPT
    assert "- kısa bir mola" in SYSTEM_PROMPT
    assert "on ile yirmi" in SYSTEM_PROMPT
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
    assert missing.error == "wrong_value"
    assert missing.answer == ""
    flagged = next(item for item in missing.debug if item["number"] == "79")
    assert flagged["nearest"] == "78"
    assert flagged["check"] == "wrong_value"
    assert flagged["detail"] == "reversed_direction"
    assert flagged["citation"] == "Giyilebilir"
    off_by_one = ground_answer(
        "DURUM: YANIT\nHRV 78 ms'den 40 ms'ye düştü. [Giyilebilir]\n",
        wearable,
    )
    assert not off_by_one.ok
    assert any(item["number"] == "40" and item["nearest"] == "41" for item in off_by_one.debug)
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


def test_named_trend_ignores_a_later_sentence_and_reads_case_endings():
    wearable = _tiny(
        "Giriş.\n\n## Giyilebilir\n"
        "Dinlenik nabız 71/dk (11.10); son 7 gün: 71, 74, 76, 69, 73, 70, 68. "
        "HRV 41 ms (11.10), kişinin olağanı 78 ms; son 7 gün: 34, 31, 29, 36, 38, 44. "
        "Uyku skoru değeri 79.\n"
    )
    leaked = ground_answer(
        "DURUM: YANIT\nHRV değeri 41 ms'den 79 ms'ye arttı. [Giyilebilir]\n",
        wearable,
    )
    assert not leaked.ok
    assert leaked.error == "wrong_value"
    flagged = next(item for item in leaked.debug if item["number"] == "79")
    assert flagged["nearest"] == "78"
    possessive = ground_answer(
        "DURUM: YANIT\nHRV değeri 41 ms'sinden 79 ms'sine arttı. [Giyilebilir]\n",
        wearable,
    )
    assert not possessive.ok
    assert possessive.error == "wrong_value"
    assert any(item["number"] == "79" and item["nearest"] == "78" for item in possessive.debug)


def test_absence_without_a_new_fact_passes_grounding():
    record = _tiny("Giriş.\n\n## Giyilebilir\n- HRV olağanı 78 ms, güncel 41 ms.\n")
    missing = ground_answer(
        "DURUM: YANIT\nBunu gösteren bir kayıt bulunmuyor. [Giyilebilir]\n",
        record,
    )
    assert missing.ok, missing.debug
    assert "bulunmuyor" in missing.answer.casefold()
    gap = ground_answer(
        "DURUM: YANIT\nKayıtta bu yakınma yok. [Giyilebilir]\n",
        record,
    )
    assert gap.ok, gap.debug
    assert "yok" in gap.answer.casefold()
    invented = ground_answer(
        "DURUM: YANIT\nKayıtta 999 yok. [Giyilebilir]\n",
        record,
    )
    assert not invented.ok


def test_line_unit_covers_a_later_bare_amount_and_its_difference():
    record = _tiny(
        "Giriş.\n\n## Laboratuvar\n"
        "- LDL 168 mg/dL (01.06.2024); önceki 71 (12.03.2026).\n"
    )
    drop = ground_answer(
        "DURUM: YANIT\nLDL 168 mg/dL'den 71 mg/dL'ye düşerek 97 mg/dL azaldı. [Laboratuvar]\n",
        record,
    )
    assert drop.ok, drop.debug
    assert "97" in drop.answer
    assert "168" in drop.answer
    stated = _tiny(
        "Giriş.\n\n## Laboratuvar\n"
        "- LDL 168 mg/dL (01.06.2024); önceki 71 (12.03.2026); bir öncekine göre −97.\n"
    )
    repeated = ground_answer(
        "DURUM: YANIT\nDüşüş 97 mg/dL. [Laboratuvar]\n",
        stated,
    )
    assert repeated.ok, repeated.debug
    assert "97" in repeated.answer
    other = ground_answer("DURUM: YANIT\nDüşüş 97 kg. [Laboratuvar]\n", stated)
    assert not other.ok
    assert any(item["check"] == "unsupported_number" and item["number"] == "97" for item in other.debug)


def test_cause_on_the_item_line_is_kept_and_a_blank_sentence_is_ignored():
    symptoms = _tiny("Giriş.\n\n## Not\n- Alfa kabus ve uyku sorunu yazıyor. Alfa 10 mg.\n")
    kept = ground_answer(
        "DURUM: YANIT\nAlfa 10 mg. Alfa'yı kabus ve uyku sorunu nedeniyle kestin. [Not]\n",
        symptoms,
    )
    assert kept.ok, kept.debug
    assert "10 mg" in kept.answer
    assert "kabus" in kept.answer.casefold()
    assert "nedeniyle" in kept.answer.casefold()
    inflected = _tiny("Giriş.\n\n## İlaçlar\n- Betafil 10 mg, kabuslar ve uyku sorunları.\n")
    stemmed = ground_answer(
        "DURUM: YANIT\nBetafil'i kabus ve uyku sorunu nedeniyle bırakmıştık. [İlaçlar]\n",
        inflected,
    )
    assert stemmed.ok, stemmed.debug
    assert "kabus" in stemmed.answer.casefold()
    assert "nedeniyle" in stemmed.answer.casefold()
    split = _tiny("Giriş.\n\n## İlaçlar\n- Alfa 10 mg.\n- Kabus ve uyku sorunu başka satırda.\n")
    other_line = ground_answer(
        "DURUM: YANIT\nAlfa'yı kabus ve uyku sorunu nedeniyle kestin. [İlaçlar]\n",
        split,
    )
    assert not other_line.ok
    assert "kabus" not in other_line.answer.casefold()
    blank = ground_answer("DURUM: YANIT\nAlfa 10 mg. . [Not]\n", symptoms)
    assert blank.ok, blank.debug
    assert "10" in blank.answer
    assert blank.error != "unsupported_claim"
    only_dot = ground_answer("DURUM: YANIT\n. [Not]\n", symptoms)
    assert not only_dot.ok


def test_cause_fragment_and_record_event_are_not_kept():
    dated = _tiny(
        "Giriş.\n\n## İlaçlar\n- Alfa 02.2025'te kesildi. Son reçete 01.02.2025.\n"
    )
    cause = ground_answer(
        "DURUM: YANIT\n02.2025'te kesildi, çünkü 01.02.2025'te son reçete verildi. [İlaçlar]\n",
        dated,
    )
    assert cause.ok, cause.debug
    assert "çünkü" not in cause.answer.casefold()
    assert "kesildi" in cause.answer.casefold()
    assert "reçete" not in cause.answer.casefold()


def test_unsupported_cause_fragment_drops_the_sentence_and_a_supported_cause_stays():
    """A dose left behind after the verb is not a sentence. A reason on the item line stays."""
    bare = _tiny("Giriş.\n\n## İlaçlar\n- Alfa 10 MG 28 TABLET.\n")
    broken = ground_answer(
        "DURUM: YANIT\n"
        "Alfa 10 MG 28 TABLET ilacını kabus ve uyku sorunu nedeniyle bıraktık. [İlaçlar]\n",
        bare,
    )
    assert not broken.ok
    assert "kabus" not in broken.answer.casefold()
    assert "ilacını" not in broken.answer.casefold()
    dated = ground_answer(
        "DURUM: YANIT\n"
        "Ayrıca Alfa 10 MG ilacın, 03.2025 tarihinde kabus ve uyku sorunu nedeniyle kesildi. "
        "[İlaçlar]\n",
        _tiny("Giriş.\n\n## İlaçlar\n- Alfa 10 MG. İlk satır 03.2025.\n"),
    )
    assert not dated.ok
    assert "kabus" not in dated.answer.casefold()
    assert "sorunu" not in dated.answer.casefold()
    supported = _tiny(
        "Giriş.\n\n## İlaçlar\n- Alfa 10 MG 28 TABLET, kabus ve uyku sorunu.\n"
    )
    kept = ground_answer(
        "DURUM: YANIT\n"
        "Alfa 10 MG 28 TABLET ilacını kabus ve uyku sorunu nedeniyle bıraktık. [İlaçlar]\n",
        supported,
    )
    assert kept.ok, kept.debug
    assert "kabus" in kept.answer.casefold()
    assert "nedeniyle" in kept.answer.casefold()
    assert "bıraktık" in kept.answer.casefold()


def test_decimal_spellings_match_and_one_bad_sentence_does_not_block_the_rest():
    record = _tiny("Giriş.\n\n## Ölçüm\n- Ateş 38,5 °C.\n- Nabız 80/dk.\n")
    dotted = ground_answer("DURUM: YANIT\nAteş 38.5 °C ölçüldü. [Ölçüm]\n", record)
    assert dotted.ok, dotted.debug
    assert "38" in dotted.answer
    bare = ground_answer("DURUM: YANIT\nAteş 38.5 C ölçüldü. [Ölçüm]\n", record)
    assert bare.ok, bare.debug
    mixed = ground_answer(
        "DURUM: YANIT\nAteş 38.5 °C ölçüldü. Nabız 999/dk oldu. [Ölçüm]\n",
        record,
    )
    assert mixed.ok, mixed.debug
    assert "38" in mixed.answer
    assert "999" not in mixed.answer
    assert mixed.debug
    blocked = ground_answer("DURUM: YANIT\nAteş 99 °C ölçüldü. [Ölçüm]\n", record)
    assert not blocked.ok
    assert blocked.answer == ""


def test_trimmed_answer_capitalizes_the_first_letter():
    record = _tiny(
        "Giriş.\n\n## Not\n"
        "- İstanbul notu duruyor.\n"
        "- Israrla not düzeldi.\n"
        "- Diyetisyene yönlendirildin.\n"
        "- eGFR 41 mL/dk duruyor.\n"
    )
    istanbul = ground_answer(
        "DURUM: YANIT\nUydurma 999 mg eklendi. istanbul notu duruyor. [Not]\n",
        record,
    )
    assert istanbul.ok, istanbul.debug
    assert istanbul.answer.startswith("İstanbul")
    assert "999" not in istanbul.answer
    dotless = ground_answer(
        "DURUM: YANIT\nUydurma 999 mg eklendi. ısrarla not düzeldi. [Not]\n",
        record,
    )
    assert dotless.ok, dotless.debug
    assert dotless.answer.startswith("Israrla")
    diet = ground_answer(
        "DURUM: YANIT\nUydurma 999 mg eklendi. diyetisyene yönlendirildin. [Not]\n",
        record,
    )
    assert diet.ok, diet.debug
    assert diet.answer.startswith("Diyetisyene")
    kidney = ground_answer(
        "DURUM: YANIT\nUydurma 999 mg eklendi. eGFR 41 mL/dk duruyor. [Not]\n",
        record,
    )
    assert kidney.ok, kidney.debug
    assert kidney.answer.startswith("eGFR")


def test_flag_word_follows_the_record_range_or_label():
    low = _tiny("Giriş.\n\n## Laboratuvar\n- Alfa 58 mL/dk, düşük (aralık 90-120).\n")
    flipped = ground_answer("DURUM: YANIT\nAlfa 58 mL/dk yüksek. [Laboratuvar]\n", low)
    assert flipped.ok, flipped.debug
    assert "düşük" in flipped.answer.casefold()
    assert "yüksek" not in flipped.answer.casefold()
    ranged = _tiny("Giriş.\n\n## Laboratuvar\n- Alfa 58 mL/dk (aralık 90-120).\n")
    from_range = ground_answer("DURUM: YANIT\nAlfa 58 mL/dk yüksek. [Laboratuvar]\n", ranged)
    assert from_range.ok, from_range.debug
    assert "düşük" in from_range.answer.casefold()
    assert "yüksek" not in from_range.answer.casefold()
    bare = _tiny("Giriş.\n\n## Laboratuvar\n- Alfa 58 mL/dk.\n")
    untouched = ground_answer("DURUM: YANIT\nAlfa 58 mL/dk yüksek. [Laboratuvar]\n", bare)
    assert untouched.ok, untouched.debug
    assert "58" in untouched.answer
    assert "yüksek" in untouched.answer.casefold()
    agreed = ground_answer("DURUM: YANIT\nAlfa 58 mL/dk düşük. [Laboratuvar]\n", low)
    assert agreed.ok, agreed.debug
    assert "düşük" in agreed.answer.casefold()
    inside = _tiny("Giriş.\n\n## Laboratuvar\n- Beta 15 mg/dL, normal (aralık 10-20).\n")
    within = ground_answer("DURUM: YANIT\nBeta 15 mg/dL yüksek. [Laboratuvar]\n", inside)
    assert within.ok, within.debug
    assert "normal" in within.answer.casefold()
    assert "yüksek" not in within.answer.casefold()
    peak = _tiny("Giriş.\n\n## Ölçüm\n- En yüksek nabız 80/dk.\n")
    extreme = ground_answer("DURUM: YANIT\nEn yüksek nabız 80/dk. [Ölçüm]\n", peak)
    assert extreme.ok, extreme.debug
    assert "en yüksek" in extreme.answer.casefold()
    prose = _tiny("Giriş.\n\n## Ölçüm\n- Nabız 47/dk.\n")
    kept = ground_answer(
        "DURUM: YANIT\nNabız 47/dk yüksek değerine göre kalmış. [Ölçüm]\n",
        prose,
    )
    assert kept.ok, kept.debug
    assert "yüksek" in kept.answer.casefold()
    assert "değerine göre kalmış" in kept.answer.casefold()
    labeled = _tiny("Giriş.\n\n## Ölçüm\n- Nabız 47/dk, düşük (aralık 60-100).\n")
    rewritten = ground_answer(
        "DURUM: YANIT\nNabız 47/dk yüksek değerine göre kalmış. [Ölçüm]\n",
        labeled,
    )
    assert rewritten.ok, rewritten.debug
    assert "düşük" in rewritten.answer.casefold()
    assert "yüksek" not in rewritten.answer.casefold()
    assert "değerine göre kalmış" in rewritten.answer.casefold()


def test_numberless_paraphrase_inherits_a_supported_neighbor():
    record = _tiny(
        "Giriş.\n\n## Laboratuvar\n- Alfa 58 mL/dk.\n\n## Not\n- baska bir satir yazildi.\n"
    )
    before = ground_answer(
        "DURUM: OZET\n"
        "Bu ölçüm son seferinde biraz zayıf duruyor. [Laboratuvar]\n"
        "Alfa 58 mL/dk. [Laboratuvar]\n",
        record,
    )
    assert before.ok, before.debug
    assert before.status == "answered"
    assert "DURUM" not in before.answer
    assert "zayıf" in before.answer.casefold()
    assert "58" in before.answer
    chained = ground_answer(
        "DURUM: YANIT\n"
        "Bu ölçüm son seferinde biraz zayıf duruyor. [Laboratuvar]\n"
        "Günlük dilde bu güç biraz zayıf kalmış. [Laboratuvar]\n"
        "Alfa 58 mL/dk. [Laboratuvar]\n",
        record,
    )
    assert chained.ok, chained.debug
    assert chained.answer.casefold().count("zayıf") == 2
    other = ground_answer(
        "DURUM: YANIT\n"
        "Bu ölçüm son seferinde biraz zayıf duruyor. [Not]\n"
        "Alfa 58 mL/dk. [Laboratuvar]\n",
        record,
    )
    assert other.ok, other.debug
    assert "58" in other.answer
    assert "zayıf" not in other.answer.casefold()
    alone = ground_answer(
        "DURUM: YANIT\nBu ölçüm son seferinde biraz zayıf duruyor. [Laboratuvar]\n",
        record,
    )
    assert alone.ok is False
    assert alone.error == "unsupported_claim"
    invented = ground_answer(
        "DURUM: YANIT\n"
        "Bu ölçüm son seferinde biraz zayıf duruyor. [Laboratuvar]\n"
        "Alfa 999 mL/dk. [Laboratuvar]\n",
        record,
    )
    assert invented.ok is False


def test_any_durum_line_is_a_status_line():
    record = _tiny("Giriş.\n\n## Laboratuvar\n- Alfa 58 mL/dk.\n")
    status, answer, citations = format_model_text(
        "DURUM: OZET\nBu cümle duruyor. [Laboratuvar]\n",
        record,
    )
    assert status == "answered"
    assert "DURUM" not in answer
    assert "OZET" not in answer
    assert answer.endswith(".")
    assert citations and citations[0]["section"] == "Laboratuvar"
    named = ground_answer(
        "DURUM: BELIRSIZ\nLaboratuvar satırı duruyor. [Laboratuvar]\n",
        record,
    )
    assert named.ok, named.debug
    assert named.status == "answered"
    assert "DURUM" not in named.answer
    assert "BELIRSIZ" not in named.answer


def test_wrapped_decimal_stays_one_sentence_and_bullets_stay():
    record = _tiny("Giriş.\n\n## Ölçüm\n- Önceki %7,6. Daha önceki %7,2.\n")
    wrapped = (
        "DURUM: YANIT\n"
        "Bir önceki ölçüm %7,6\n"
        ", ondan önceki ise %7,2\n"
        "olarak kaydedilmiş. [Ölçüm]\n"
    )
    _status, shown, _citations = format_model_text(wrapped, record)
    assert "%7,6.\n," not in shown
    assert "%7,6, ondan önceki ise %7,2" in shown
    grounded = ground_answer(wrapped, record)
    assert grounded.ok, grounded.debug
    assert "7,6" in grounded.answer
    assert "7,2" in grounded.answer
    assert "%7,6.\n," not in grounded.answer
    dotted = _sentences("Değer 7. 6 kaldı. Sonra durdu.")
    assert dotted[0].startswith("Değer 7. 6")
    assert "Sonra durdu." in dotted
    listed = (
        "DURUM: YANIT\n"
        "Geçmiş olsun. Değer duruyor.\n\n"
        "Şunlar iyi gelir:\n"
        "- dinlen\n"
        "- su iç\n\n"
        "Dikkat: ağırlaşırsa yardım iste. [Ölçüm]\n"
    )
    _listed_status, layout, cites = format_model_text(listed, record)
    assert "\n\n" in layout
    assert "- dinlen" in layout
    assert "- su iç" in layout
    assert "Dikkat:" in layout
    assert "gelir:." not in layout
    assert "DURUM" not in layout
    assert "[" not in layout
    assert cites and cites[0]["section"] == "Ölçüm"


def test_dose_schedule_and_visit_date_in_the_section_are_kept():
    record = _tiny(
        "Giriş.\n\n## İlaçlar\n- Alfa 2x2 puf.\n\n## Başvuru\n- Kontrol 12.04.2024.\n"
    )
    dose = ground_answer(
        "DURUM: YANIT\nGeçmiş olsun. Günde 2x2 puf kullanıyorsun. [İlaçlar]\n",
        record,
    )
    assert dose.ok, dose.debug
    assert dose.answer.strip()
    assert "2x2" in dose.answer
    assert "Geçmiş olsun" in dose.answer
    visit = ground_answer(
        "DURUM: YANIT\n12.04.2024 tarihinde kontrole gittin. [Başvuru]\n",
        record,
    )
    assert visit.ok, visit.debug
    assert "12.04.2024" in visit.answer
    missing = ground_answer(
        "DURUM: YANIT\nGünde 9x9 puf kullanıyorsun. [İlaçlar]\n",
        record,
    )
    assert missing.ok is False
    assert missing.answer == ""


def test_chart_abbreviation_is_recalled_without_a_nearby_digit():
    record = _tiny(
        "Giriş.\n\n## Giyilebilir\nHRV ölçümü güvenilir değil.\n\n"
        "## Laboratuvar\nTSH sonucu burada.\n"
    )
    context = relevant_context(record, "HRV değerim ne durumda?")
    assert "güvenilir" in context
    thyroid = relevant_context(record, "Tiroid hormon tahlilimde TSH ne çıktı?")
    assert "TSH" in thyroid
    assert "HRV" not in thyroid


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
