"""Generalization checks. Held-out questions are scored once and are not a tuning set."""

import json
import re
import sys
from pathlib import Path

from medical_ai_selective_context.text_document import parse_cco_text
from medical_ai_selective_context.text_eval import score_case, summarize, tokens_cover
from medical_ai_selective_context.text_focus import align_answer
from medical_ai_selective_context.text_grounding import GroundingResult
from medical_ai_selective_context.text_safety import classify_question, render_refusal

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


def _answer(record, row: dict) -> dict:
    decision = classify_question(row["question"])
    if decision.action == "refuse":
        answer, citations = render_refusal(record, row["question"], decision)
        return {"status": "safety_refusal", "answer": answer, "citations": citations, "metrics": {}}
    aligned = align_answer(
        record,
        row["question"],
        GroundingResult(True, "insufficient_evidence", "Bu bilgi kayıtta yok.", [], ""),
    )
    return {
        "status": aligned.status,
        "answer": aligned.answer,
        "citations": aligned.citations,
        "metrics": {},
    }


def _coverage(questions: Path, patients: Path) -> dict:
    records = _records(patients)
    scored = [score_case(row, _answer(records[row["patient"]], row)) for row in _rows(questions)]
    summary = summarize(scored)
    summary["missing"] = [
        {"id": row["id"], "missingFacts": row["missingFacts"], "status": row["status"]}
        for row in scored
        if row["missingFacts"] or not row["statusOk"]
    ]
    return summary


def test_month_names_and_suffixes_match_without_a_chart_value():
    assert tokens_cover("04.05.2021", "4 Mayıs 2021 tarihinde")
    assert tokens_cover("4 Mayıs 2021", "04.05.2021")
    assert not tokens_cover("09.09.2015", "2015 yılında")
    assert tokens_cover("metforminin kullanımı", "metformin kullanımı sürüyor")


def test_source_has_no_literal_record_values():
    """Python source must not embed chart values or the retired audit needles."""
    blobs = [(path, path.read_text(encoding="utf-8").casefold()) for path in SRC.rglob("*.py")]
    for needle in _AUDIT:
        for path, text in blobs:
            assert needle not in text, f"{needle} in {path}"
    values = set()
    for path in DEV_PATIENTS.glob("*.json"):
        chart = json.loads(path.read_text(encoding="utf-8"))["text"]
        values.update(re.findall(r"\d+,\d+", chart))
    for value in sorted(values):
        if len(value) < 4:
            continue
        for path, text in blobs:
            cleaned = re.sub(r"\{\d+,\d+\}", "", text)
            assert value not in cleaned, f"{value} in {path}"


def test_perturbed_chart_values_follow_the_row():
    """Changing a retrieved number changes the answer. The old number is not kept."""
    patient = json.loads((DEV_PATIENTS / "patient-08.json").read_text(encoding="utf-8"))
    patient["text"] = patient["text"].replace("Hemoglobin 10,2 g/dL", "Hemoglobin 9,7 g/dL", 1)
    record = parse_cco_text(patient)
    aligned = align_answer(
        record,
        "Hemoglobin sonucum ne?",
        GroundingResult(True, "insufficient_evidence", "Bu bilgi kayıtta yok.", [], ""),
    )
    assert "9,7" in aligned.answer
    assert "10,2" not in aligned.answer

    other = json.loads((DEV_PATIENTS / "patient-02.json").read_text(encoding="utf-8"))
    other["text"] = other["text"].replace("FEV1 3,12 L (%71)", "FEV1 2,87 L (%71)", 1)
    replaced = align_answer(
        parse_cco_text(other),
        "Solunum testimde FEV1 kaç çıktı?",
        GroundingResult(True, "insufficient_evidence", "Bu bilgi kayıtta yok.", [], ""),
    )
    assert "2,87" in replaced.answer
    assert "3,12" not in replaced.answer


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


def test_heldout_labs_do_not_quote_the_wearable():
    record = _records(HELD / "patients")["heldout-01"]
    for question in (
        "Tahlillerim nasıl?",
        "Laboratuvar sonuçlarım iyi mi?",
        "Kalp için tahlillerim ne diyor?",
    ):
        aligned = align_answer(
            record,
            question,
            GroundingResult(True, "answered", "Nabız 71/dk ve HRV 19 ms.", [], ""),
        )
        assert "71/dk" not in aligned.answer, question
        assert "HRV" not in aligned.answer, question


def test_dev_and_heldout_coverage_is_reported(capsys):
    """Print both scores. No coverage floor: held-out is not used to add rules."""
    dev = _coverage(DEV_QUESTIONS, DEV_PATIENTS)
    held = _coverage(HELD / "questions.jsonl", HELD / "patients")
    report = {
        "dev": {key: dev[key] for key in ("questions", "statusAccuracy", "factCoverage", "factsFound", "expectedFacts")},
        "heldout": {key: held[key] for key in ("questions", "statusAccuracy", "factCoverage", "factsFound", "expectedFacts")},
    }
    print(json.dumps(report, ensure_ascii=False))
    captured = capsys.readouterr()
    sys.stderr.write(captured.out + "\n")
    assert "dev" in captured.out and "heldout" in captured.out
    assert dev["questions"] == 100
    assert held["questions"] == len(_rows(HELD / "questions.jsonl"))
