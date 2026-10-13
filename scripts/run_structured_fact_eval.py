"""Run text answers with a cco-text/1 JSON and cco-derived-facts/1 JSON pair.

The script does not print or persist the raw CCO text or fact rows. It writes:
- metrics.json: aggregate rates and timing/token summaries.
- questions.md: one row per question with status, answer, validation codes, and metrics.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from medical_ai_selective_context.text_document import parse_cco_text
from medical_ai_selective_context.text_llm import TextModelClient
from medical_ai_selective_context.text_qa import TextAnswerer


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_questions(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _safe_id(row: dict, index: int) -> str:
    raw = str(row.get("id") or row.get("question_id") or f"q{index:03d}")
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in raw)[:80]


def _numbers(rows: list[dict], dotted_key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        current: Any = row
        for part in dotted_key.split("."):
            if not isinstance(current, dict):
                current = None
                break
            current = current.get(part)
        if isinstance(current, (int, float)) and not isinstance(current, bool):
            values.append(float(current))
    return values


def _rate(rows: list[dict], predicate) -> float:
    if not rows:
        return 0.0
    return round(sum(1 for row in rows if predicate(row)) / len(rows), 4)


def _p95(values: list[float]) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return round(values[0], 2)
    return round(statistics.quantiles(values, n=20, method="inclusive")[18], 2)


def _avg(values: list[float]) -> float | None:
    return None if not values else round(sum(values) / len(values), 2)


def _codes(row: dict) -> list[str]:
    validation = row.get("validation") or {}
    codes: list[str] = []
    for key in ("errorCodes", "droppedCodes", "warningCodes"):
        value = validation.get(key)
        if isinstance(value, list):
            codes.extend(str(item) for item in value)
    return codes


def _status_letter(row: dict) -> str:
    """Manual D/K/Y/C labels are accepted when the questions file already has them."""
    for key in ("grade", "expected_grade", "manual_grade"):
        value = str(row.get(key) or "").strip().upper()
        if value in {"D", "K", "Y", "C"}:
            return value
    return ""


def _summarize(rows: list[dict]) -> dict:
    total = len(rows)
    grade_counts = {letter: 0 for letter in ("D", "K", "Y", "C", "ungraded")}
    code_counts: dict[str, int] = {}
    for row in rows:
        grade = _status_letter(row) or "ungraded"
        grade_counts[grade] += 1
        for code in _codes(row):
            code_counts[code] = code_counts.get(code, 0) + 1
    total_ms = _numbers(rows, "metrics.totalMs")
    input_tokens = _numbers(rows, "metrics.usage.prompt_tokens")
    output_tokens = _numbers(rows, "metrics.usage.completion_tokens")
    retried = [row for row in rows if bool((row.get("metrics") or {}).get("retry"))]
    return {
        "questionCount": total,
        "gradeCounts": grade_counts,
        "validJsonRate": _rate(rows, lambda row: "invalid_answer_json" not in _codes(row)),
        "firstPassRate": _rate(rows, lambda row: not (row.get("metrics") or {}).get("retry")),
        "retryRate": _rate(rows, lambda row: bool((row.get("metrics") or {}).get("retry"))),
        "retrySuccessRate": _rate(
            retried,
            lambda row: bool((row.get("validation") or {}).get("passed")),
        ),
        "retryCount": len(retried),
        "retrySuccessCount": sum(
            1 for row in retried if bool((row.get("validation") or {}).get("passed"))
        ),
        "fallbackRate": _rate(rows, lambda row: bool((row.get("metrics") or {}).get("fallback"))),
        "emptyAnswerRate": _rate(rows, lambda row: not str(row.get("answer") or "").strip()),
        "latencyMs": {"avg": _avg(total_ms), "p95": _p95(total_ms)},
        "tokens": {
            "inputAvg": _avg(input_tokens),
            "inputP95": _p95(input_tokens),
            "outputAvg": _avg(output_tokens),
            "outputP95": _p95(output_tokens),
        },
        "errorCodeDistribution": dict(sorted(code_counts.items())),
    }


def _md(rows: list[dict], summary: dict) -> str:
    lines = [
        "# Structured fact eval",
        "",
        "Raw CCO text and fact rows are intentionally not included.",
        "",
        "```json",
        json.dumps(summary, ensure_ascii=False, indent=2),
        "```",
        "",
    ]
    for row in rows:
        lines.extend([
            f"## {row['id']}",
            "",
            f"- status: `{row.get('status')}`",
            f"- grade: `{_status_letter(row) or 'ungraded'}`",
            f"- retry: `{bool((row.get('metrics') or {}).get('retry'))}`",
            f"- fallback: `{bool((row.get('metrics') or {}).get('fallback'))}`",
            f"- validation_passed: `{bool((row.get('validation') or {}).get('passed'))}`",
            f"- error_codes: `{', '.join(_codes(row)) or '-'}`",
            "",
            "Answer:",
            "",
            str(row.get("answer") or "").strip() or "(empty)",
            "",
        ])
    return "\n".join(lines).strip() + "\n"


async def _run(args: argparse.Namespace) -> dict:
    os.environ.setdefault("DETERMINISTIC_EVIDENCE_VERIFICATION_ENABLED", "true")
    cco = _load_json(args.cco_json)
    facts = _load_json(args.facts_json)
    record = parse_cco_text(cco)
    questions = _load_questions(args.questions_jsonl)
    answerer = TextAnswerer(TextModelClient())
    rows: list[dict] = []
    for index, question_row in enumerate(questions, start=1):
        qid = _safe_id(question_row, index)
        started = perf_counter()
        try:
            result = await answerer.answer(
                record,
                str(question_row["question"]),
                structured_facts=facts,
            )
            row = {
                "id": qid,
                "status": result.get("status"),
                "answer": result.get("answer"),
                "metrics": result.get("metrics") or {},
                "validation": result.get("validation") or {},
                "elapsedMs": round((perf_counter() - started) * 1000, 2),
            }
            for key in ("grade", "expected_grade", "manual_grade"):
                if key in question_row:
                    row[key] = question_row[key]
        except Exception as exc:  # noqa: BLE001 - CLI report row, not server code.
            row = {
                "id": qid,
                "status": "error",
                "answer": "",
                "metrics": {"totalMs": round((perf_counter() - started) * 1000, 2)},
                "validation": {"errorCodes": [type(exc).__name__]},
                "elapsedMs": round((perf_counter() - started) * 1000, 2),
            }
        rows.append(row)
    summary = _summarize(rows)
    payload = {
        "createdAt": datetime.now(UTC).isoformat(),
        "ccoFile": args.cco_json.name,
        "factsFile": args.facts_json.name,
        "questionsFile": args.questions_jsonl.name,
        "summary": summary,
        "rows": rows,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    (args.out_dir / "questions.md").write_text(_md(rows, summary), encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cco-json", required=True, type=Path)
    parser.add_argument("--facts-json", required=True, type=Path)
    parser.add_argument("--questions-jsonl", required=True, type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("data/evaluation/structured-facts"))
    args = parser.parse_args()
    payload = asyncio.run(_run(args))
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
