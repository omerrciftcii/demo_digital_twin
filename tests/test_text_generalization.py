"""Generalization checks. Held-out questions are scored once and are not a tuning set."""

import json
import re
from pathlib import Path

from medical_ai_selective_context.index import normalize_text
from medical_ai_selective_context.text_document import parse_cco_text
from medical_ai_selective_context.text_eval import tokens_cover
from medical_ai_selective_context.text_relevance import relevant_context, relevant_sections
from medical_ai_selective_context.text_safety import (
    _MESSAGES,
    _TREATMENT_WITH_RX,
    _TREATMENT_WITHOUT_RX,
    DISCLAIMER,
    classify_question,
    render_refusal,
)

ROOT = Path(__file__).parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "text"
DEV_QUESTIONS = FIXTURES / "eval_questions.jsonl"
DEV_PATIENTS = FIXTURES / "patients"
HELD = FIXTURES / "heldout"
SRC = ROOT / "src"

_AUDIT = (
    "proksimaline",
    "hipoaktif",
    "pretibial",
    "rizatriptan",
    "128/dk",
    "5 sa 18",
    "9 ng",
    "phq-9",
    "ekstrasistol",
    "e11.9",
    "i82.4",
    "l71.9",
    "+1,3",
    "4 cm morluk",
    "ferritin",
    "en fazla 2",
)


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _records(folder: Path) -> dict:
    return {
        path.stem: parse_cco_text(json.loads(path.read_text(encoding="utf-8")))
        for path in folder.glob("*.json")
    }


def test_month_names_and_suffixes_match_without_a_chart_value():
    assert tokens_cover("04.05.2021", "4 Mayıs 2021 tarihinde")
    assert tokens_cover("4 Mayıs 2021", "04.05.2021")
    assert not tokens_cover("09.09.2015", "2015 yılında")
    assert tokens_cover("metforminin kullanımı", "metformin kullanımı sürüyor")


def _source_blobs() -> list[tuple[Path, str]]:
    blobs = []
    for path in SRC.rglob("*"):
        if path.suffix.lower() not in {".py", ".json", ".md"} or not path.is_file():
            continue
        blobs.append((path, normalize_text(path.read_text(encoding="utf-8"))))
    return blobs


def _refusal_copy() -> str:
    """General safety sentences. A short fact may quote one of these."""
    chunks = [DISCLAIMER, _TREATMENT_WITH_RX, _TREATMENT_WITHOUT_RX, *_MESSAGES.values()]
    return normalize_text("\n".join(chunks))


def _eval_literals() -> tuple[set[str], set[str]]:
    """Full questions, plus facts that are a chart value or a multi-word phrase."""
    questions: set[str] = set()
    phrases: set[str] = set()
    for path in (DEV_QUESTIONS, HELD / "questions.jsonl"):
        for row in _rows(path):
            question = " ".join(row["question"].split())
            if len(question) >= 12:
                questions.add(normalize_text(question))
            for fact in row.get("expected_facts", []):
                text = " ".join(str(fact).split())
                folded = normalize_text(text)
                words = folded.split()
                numeric = re.search(r"\d", folded) and len(folded) >= 4
                verbal = len(words) >= 3 and len(folded) >= 12
                if numeric or verbal:
                    phrases.add(folded)
    return questions, phrases


def test_source_has_no_literal_record_values():
    """Source must not embed chart values, eval questions, or eval phrases."""
    blobs = _source_blobs()
    py_blobs = [(path, text) for path, text in blobs if path.suffix == ".py"]
    for needle in _AUDIT:
        folded = normalize_text(needle)
        for path, text in py_blobs:
            assert folded not in text, f"{needle} in {path}"
    values = set()
    for path in DEV_PATIENTS.glob("*.json"):
        chart = json.loads(path.read_text(encoding="utf-8"))["text"]
        values.update(re.findall(r"\d+,\d+", chart))
    for value in sorted(values):
        if len(value) < 4:
            continue
        for path, text in py_blobs:
            cleaned = re.sub(r"\{\d+,\d+\}", "", text)
            assert value not in cleaned, f"{value} in {path}"
    questions, phrases = _eval_literals()
    allowed = _refusal_copy()
    safety_name = "text_safety.py"
    for phrase in sorted(questions | phrases):
        in_refusal = phrase not in questions and phrase in allowed and not re.search(r"\d", phrase)
        for path, text in blobs:
            if in_refusal and path.name == safety_name:
                continue
            assert phrase not in text, f"{phrase} in {path}"


def test_perturbed_chart_values_follow_the_row():
    """The model sees the number that is in the chart now."""
    patient = json.loads((DEV_PATIENTS / "patient-08.json").read_text(encoding="utf-8"))
    patient["text"] = patient["text"].replace("Hemoglobin 10,2 g/dL", "Hemoglobin 9,7 g/dL", 1)
    record = parse_cco_text(patient)
    context = relevant_context(record, "Hemoglobin sonucum ne?")
    assert "9,7" in context
    assert "10,2" not in context

    other = json.loads((DEV_PATIENTS / "patient-02.json").read_text(encoding="utf-8"))
    other["text"] = other["text"].replace("FEV1 3,12 L (%71)", "FEV1 2,87 L (%71)", 1)
    replaced = relevant_context(parse_cco_text(other), "Solunum testimde FEV1 kaç çıktı?")
    assert "2,87" in replaced
    assert "3,12" not in replaced


def test_heldout_is_separate_and_safety_paraphrases_refuse():
    dev_text = DEV_QUESTIONS.read_text(encoding="utf-8")
    held_rows = _rows(HELD / "questions.jsonl")
    assert len(held_rows) >= 30
    families: dict[str, int] = {}
    for row in held_rows:
        assert row["question"] not in dev_text
        family = row["id"].rsplit("-", 1)[0]
        families[family] = families.get(family, 0) + 1
    assert all(count >= 3 for count in families.values()), families
    records = _records(HELD / "patients")
    for row in held_rows:
        if row["expected_status"] != "safety_refusal":
            continue
        decision = classify_question(row["question"])
        assert decision.action == "refuse", row["id"]
        answer, _citations = render_refusal(records[row["patient"]], row["question"], decision)
        for fact in row["expected_facts"]:
            assert fact in answer or tokens_cover(fact, answer), (row["id"], fact, answer)


def test_heldout_context_is_the_full_section():
    """A held-out question receives the whole overlapping section, wearable line included only on overlap."""
    record = _records(HELD / "patients")["heldout-01"]
    question = "Tiroid hormon tahlilimde TSH ne çıktı?"
    sections = relevant_sections(record, question)
    context = relevant_context(record, question)
    assert any("TSH 7,8" in section.text for section in sections)
    for section in sections:
        assert section.text.strip() in context
    assert "HRV" not in context
    assert "71/dk" not in context
