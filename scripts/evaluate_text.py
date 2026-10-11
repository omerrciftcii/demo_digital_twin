"""Score cco-text/1 answers through the running API. No API key is read here.

The server reports provider time-to-first-token. This script also times the
first SSE token the client receives. Reports stay under data/evaluation/,
which is gitignored.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from medical_ai_selective_context.text_eval import _latency, case_error, score_case, summarize

DEFAULT_QUESTIONS = PROJECT_ROOT / "tests" / "fixtures" / "text" / "eval_questions.jsonl"
DEFAULT_PATIENTS = PROJECT_ROOT / "tests" / "fixtures" / "text" / "patients"


def load_questions(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def parse_events(lines: list[str]) -> list[tuple[str, str]]:
    events: list[tuple[str, str]] = []
    name = "message"
    data: list[str] = []
    for line in lines:
        if line == "":
            if data:
                events.append((name, "\n".join(data)))
            name = "message"
            data = []
            continue
        if line.startswith("event:"):
            name = line[6:].strip()
        elif line.startswith("data:"):
            data.append(line[5:].lstrip())
    if data:
        events.append((name, "\n".join(data)))
    return events


def _numbers(rows: list[dict], key: str) -> list[float]:
    return [row[key] for row in rows if isinstance(row.get(key), (int, float))]


def _public_error(exc: Exception) -> str:
    if isinstance(exc, httpx.TimeoutException):
        return "timeout"
    if isinstance(exc, RuntimeError):
        text = str(exc)
        if re.fullmatch(r"[A-Za-z0-9_.:-]{1,80}", text):
            return text
    return type(exc).__name__


def ask_case(client: httpx.Client, base_url: str, case: dict, record_id: str) -> dict:
    start = time.perf_counter()
    client_ttft = None
    collected: list[str] = []
    try:
        with client.stream(
            "POST", f"{base_url}/api/v1/text/answer",
            json={"recordId": record_id, "question": case["question"], "stream": True},
        ) as response:
            if response.status_code != 200:
                raise RuntimeError(f"answer_failed:{case['id']}:{response.status_code}")
            for line in response.iter_lines():
                if client_ttft is None and line.startswith("event: token"):
                    client_ttft = round((time.perf_counter() - start) * 1000, 2)
                collected.append(line)
        done = None
        for name, data in parse_events(collected):
            if name == "done":
                done = json.loads(data)
        if not isinstance(done, dict):
            raise RuntimeError(f"missing_done:{case['id']}")
        row = score_case(case, done)
    except (httpx.HTTPError, json.JSONDecodeError, OSError, RuntimeError, KeyError, ValueError) as exc:
        row = case_error(case, _public_error(exc))
    row["clientTimeToFirstTokenMs"] = client_ttft
    row["clientTotalMs"] = round((time.perf_counter() - start) * 1000, 2)
    return row


def run(
    base_url: str, questions_path: Path, patients_dir: Path, timeout: float,
    transport: httpx.BaseTransport | None = None,
) -> dict:
    questions = load_questions(questions_path)
    rows: list[dict] = []
    with httpx.Client(timeout=timeout, transport=transport) as client:
        record_ids: dict[str, str] = {}
        for path in sorted(patients_dir.glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            response = client.post(f"{base_url}/api/v1/text-records", json=payload)
            if response.status_code not in {200, 201}:
                raise RuntimeError(f"import_failed:{path.stem}:{response.status_code}")
            record_ids[path.stem] = response.json()["recordId"]
        for case in questions:
            record_id = record_ids.get(case["patient"])
            if record_id is None:
                row = case_error(case, f"missing_patient:{case['patient']}")
            else:
                row = ask_case(client, base_url, case, record_id)
            rows.append(row)
    summary = summarize(rows)
    summary["clientLatencyMs"] = {
        "timeToFirstToken": _latency(_numbers(rows, "clientTimeToFirstTokenMs")),
        "total": _latency(_numbers(rows, "clientTotalMs")),
    }
    return {"baseUrl": base_url, "summary": summary, "cases": rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8091")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--patients", type=Path, default=DEFAULT_PATIENTS)
    parser.add_argument("--timeout", type=float, default=180)
    args = parser.parse_args()
    try:
        report = run(args.base_url.rstrip("/"), args.questions, args.patients, args.timeout)
    except RuntimeError as exc:
        print(f"Evaluation failed ({exc}).", file=sys.stderr)
        return 1
    except (OSError, httpx.HTTPError, json.JSONDecodeError, KeyError) as exc:
        print(f"Evaluation failed ({type(exc).__name__}).", file=sys.stderr)
        return 1
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    work_dir = PROJECT_ROOT / "data" / "evaluation" / f"text-run-{run_id}"
    work_dir.mkdir(parents=True, exist_ok=True)
    output = work_dir / "report.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = report["summary"]
    failures = [
        {
            "id": row["id"], "status": row["status"], "expected": row["expectedStatus"],
            "missingFacts": row["missingFacts"],
            "answer": (row.get("answer") or "")[:240],
            "error": row.get("error"),
            "validationError": row.get("validationError"),
        }
        for row in report["cases"]
        if not row["statusOk"] or not row["factsOk"]
    ]
    print(json.dumps({
        "report": str(output.relative_to(PROJECT_ROOT)),
        "questions": summary["questions"],
        "statusMatches": summary["statusMatches"],
        "factCoverage": summary["factCoverage"],
        "criticalFailures": summary["criticalFailures"],
        "modelCalls": summary["modelCalls"],
        "conversational": summary["conversational"],
        "latencyMs": summary["latencyMs"],
        "clientLatencyMs": summary["clientLatencyMs"],
        "failures": failures,
    }, ensure_ascii=False, indent=2))
    return 2 if summary["criticalFailureCount"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
