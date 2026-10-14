"""Full Context and Oracle Evidence answers over the private patient-centric benchmark.

The full arm sends the whole cco-text/1 chart and every fact. The oracle arm
sends only the gold facts and their own source lines. Both use the same system
text, JSON schema, and deterministic verifier, so a gap between them points at
context size rather than interpretation.

--dry-run builds and checks every prompt without a chat completion.
--tokenize adds served-tokenizer counts from vLLM /tokenize on a trusted local
host. Outputs stay under data/evaluation: they hold model answers.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any
from urllib.parse import urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from medical_ai_selective_context.text_document import parse_cco_text
from medical_ai_selective_context.text_llm import ModelError, TextModelClient
from medical_ai_selective_context.text_qa import TextAnswerer, build_full_context, fit_report
from scripts.run_patient_centric_retrieval_eval import (
    _canonical_hash,
    _case_groups,
    _group_fact_ids,
    _load_json,
    _mean,
    _p95,
    _recall,
    _satisfied,
    expand_cases,
)

ARMS = ("full", "oracle")
EVAL_ROOT = PROJECT_ROOT / "data" / "evaluation"
NUMERIC_CODES = frozenset({
    "value_not_in_evidence", "value_concept_mismatch", "evidence_concept_mismatch",
})
TEMPORAL_CODES = frozenset({"date_mismatch", "date_unconfirmed", "stale_as_current"})
GRADE_SCALE = {
    "clinical": ("dogru", "kismen", "yanlis"),
    "numericTemporal": ("dogru", "hatali", "yok"),
    "criticalLoss": ("yok", "var"),
    "naturalness": ("iyi", "orta", "kotu"),
}
GRADE_FIELDS = tuple(GRADE_SCALE)
# Dry runs have no answers. A prior answer at the history cap gives the worst-case prompt.
_PLACEHOLDER_ANSWER = ("Önceki yanıt yer tutucusu. " * 80)[:2000].strip()


def _inside_eval_root(path: Path) -> bool:
    try:
        path.resolve().relative_to(EVAL_ROOT.resolve())
    except ValueError:
        return False
    return True


def _gold(groups: list[dict[str, Any]], cited: set[str]) -> dict[str, Any]:
    rows = [
        {
            "groupId": group["group_id"], "role": group["role"],
            "critical": bool(group.get("critical")), "satisfied": _satisfied(group, cited),
        }
        for group in groups
    ]
    critical = [row for row in rows if row["critical"]]
    return {
        "recall": _recall(groups, cited),
        "hasCritical": bool(critical),
        "criticalMiss": any(not row["satisfied"] for row in critical),
        "groups": rows,
    }


def _expectations(case: dict[str, Any], profiles: dict[str, Any]) -> list[str]:
    seen: list[str] = []
    for ref in case["profile_refs"]:
        for label in profiles[ref].get("clinical_expectations", []):
            if label not in seen:
                seen.append(label)
    return seen


async def _served_parts(
    client: TextModelClient, messages: list[dict[str, str]], total: int,
) -> dict[str, Any]:
    """Per-part served counts. Template overhead is what the parts do not explain."""
    parts = {
        "system": messages[0]["content"],
        "record": messages[1]["content"],
        "ack": messages[2]["content"],
        "history": "\n".join(item["content"] for item in messages[3:-1]),
        "question": messages[-1]["content"],
    }
    counts: dict[str, Any] = {}
    for name, text in parts.items():
        counted = await client.count_tokens(prompt=text) if text else {"count": 0}
        counts[name] = None if counted is None else counted["count"]
    known = [value for value in counts.values() if value is not None]
    counts["templateOverhead"] = total - sum(known) if len(known) == len(parts) else None
    counts["total"] = total
    return counts


def _row_base(case: dict[str, Any], arm: str, profiles: dict[str, Any]) -> dict[str, Any]:
    return {
        "key": f"{arm}:{case['case_id']}",
        "arm": arm,
        "caseId": case["case_id"],
        "patientId": case["patient_id"],
        "symptom": case["symptom"],
        "conversationId": case.get("conversation_id"),
        "turn": case.get("turn"),
        "question": case["question"],
        "clinicalExpectations": _expectations(case, profiles),
    }


def _from_result(result: dict[str, Any]) -> dict[str, Any]:
    validation = result.get("validation") or {}
    metrics = result.get("metrics") or {}
    trace = validation.get("trace") or {}
    usage = metrics.get("usage") or {}
    context = validation.get("context") or {}
    check = None
    if context.get("tokenCountSource") == "vllm_tokenize" and metrics.get("modelCalls") == 1:
        reported = usage.get("prompt_tokens")
        check = {
            "counted": context.get("promptTokens"), "reported": reported,
            "match": reported == context.get("promptTokens"),
        }
    return {
        "status": result.get("status"),
        "answer": result.get("answer"),
        "passed": validation.get("passed"),
        "errorCodes": validation.get("errorCodes") or [],
        "droppedCodes": validation.get("droppedCodes") or [],
        "warningCodes": validation.get("warningCodes") or [],
        "attemptCodes": trace.get("validation_error_codes") or [],
        "claims": validation.get("claims"),
        "attemptClaims": trace.get("attempt_claims") or [],
        "citedFactIds": trace.get("cited_fact_ids") or [],
        "retry": bool(metrics.get("retry")),
        "retrySkipped": trace.get("retry_skipped"),
        "fallback": bool(metrics.get("fallback")),
        "modelCalls": metrics.get("modelCalls", 0),
        "ttftMs": metrics.get("timeToFirstTokenMs"),
        "totalMs": metrics.get("totalMs"),
        "stagesMs": metrics.get("stagesMs"),
        "usage": usage,
        "promptTokenCheck": check,
        "context": context,
        "refused": validation.get("refused"),
    }


async def _one(
    args: argparse.Namespace, arm: str, case: dict[str, Any], history: list[dict[str, str]],
    patient: dict[str, Any], groups: list[dict[str, Any]],
    client: TextModelClient, answerer: TextAnswerer,
) -> dict[str, Any]:
    oracle = None
    if arm == "oracle":
        oracle = sorted({fact for group in groups for fact in _group_fact_ids(group)})
    record, facts = patient["record"], patient["facts"]
    if args.tokenize:
        client.guard(record.data_class)
    plan = build_full_context(record, case["question"], history, facts, oracle_fact_ids=oracle)
    counted = await client.count_tokens(plan.messages) if args.tokenize else None
    served = await _served_parts(client, plan.messages, counted["count"]) if counted else None
    if args.dry_run:
        context = {**plan.report, **fit_report(
            plan.messages, counted, client.settings.answer_max_tokens,
        )}
        return {
            "status": "dry_run" if context["fits"] else "context_overflow",
            "answer": None, "modelCalls": 0, "context": context, "servedTokens": served,
            "citedFactIds": [],
        }
    result = await answerer.answer(
        record, case["question"], history, structured_facts=facts, oracle_fact_ids=oracle,
    )
    return {**_from_result(result), "servedTokens": served}


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    os.environ["DETERMINISTIC_EVIDENCE_VERIFICATION_ENABLED"] = "true"
    os.environ["QA_FULL_CONTEXT_ENABLED"] = "true"
    benchmark = _load_json(args.benchmark)
    profiles = benchmark["evidence_profiles"]
    cases = expand_cases(benchmark)
    patients: dict[str, dict[str, Any]] = {}
    for patient_id, entry in benchmark["patients"].items():
        raw = _load_json(args.data_root / entry["cco_file"])
        record = parse_cco_text(raw)
        patients[patient_id] = {
            "record": record,
            "facts": _load_json(args.data_root / entry["facts_file"]),
            "rawTextMatchesRecord": raw.get("text") == record.text,
        }
    client = TextModelClient()
    answerer = TextAnswerer(client)
    rows: list[dict[str, Any]] = []
    args.out_dir.mkdir(parents=True, exist_ok=True)
    progress = (args.out_dir / "rows.jsonl").open("w", encoding="utf-8")
    total = len(args.arms) * len(cases)
    for arm in args.arms:
        histories: dict[str, list[dict[str, str]]] = defaultdict(list)
        for case in cases:
            history = list(histories[case["conversation_id"]]) if case.get("conversation_id") else []
            groups = _case_groups(case, profiles)
            row = _row_base(case, arm, profiles)
            started = perf_counter()
            try:
                outcome = await _one(
                    args, arm, case, history, patients[case["patient_id"]], groups,
                    client, answerer,
                )
            except ModelError as exc:
                outcome = {"status": "error", "answer": None, "errorCodes": [str(exc)],
                           "modelCalls": 0, "citedFactIds": []}
            row.update(outcome)
            row["elapsedMs"] = round((perf_counter() - started) * 1000, 2)
            row["gold"] = _gold(groups, set(row.get("citedFactIds") or []))
            row["historyIsPlaceholder"] = bool(args.dry_run and history)
            rows.append(row)
            progress.write(json.dumps(row, ensure_ascii=False) + "\n")
            progress.flush()
            print(
                f"[{len(rows)}/{total}] {row['key']} {row['status']} "
                f"{row['elapsedMs'] / 1000:.1f}s",
                file=sys.stderr, flush=True,
            )
            if case.get("conversation_id"):
                answer = row.get("answer") or (_PLACEHOLDER_ANSWER if args.dry_run else "")
                histories[case["conversation_id"]].extend([
                    {"role": "user", "content": case["question"]},
                    {"role": "assistant", "content": answer},
                ])
    progress.close()
    grades = _load_json(args.grades) if args.grades else {}
    summary = {arm: summarize([row for row in rows if row["arm"] == arm], grades) for arm in args.arms}
    settings = client.settings
    return {
        "schemaVersion": "full-context-eval/1",
        "createdAt": datetime.now(UTC).isoformat(),
        "benchmarkHash": _canonical_hash(benchmark),
        "dryRun": args.dry_run,
        "tokenize": args.tokenize,
        "arms": list(args.arms),
        "model": settings.model,
        "host": urlparse(settings.base_url).hostname,
        "answerMaxTokens": settings.answer_max_tokens,
        "modelCalled": any((row.get("modelCalls") or 0) > 0 for row in rows),
        "rawTextMatchesRecord": {
            patient_id: patient["rawTextMatchesRecord"] for patient_id, patient in patients.items()
        },
        "summary": summary,
        "rows": rows,
    }


def _rate(numerator: int, denominator: int) -> float | None:
    return None if not denominator else round(numerator / denominator, 4)


def _prefix_stability(rows: list[dict[str, Any]]) -> dict[str, int]:
    prefixes: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        sha = (row.get("context") or {}).get("prefixSha256")
        if row.get("conversationId") and sha:
            prefixes[row["conversationId"]].add(sha)
    return {
        "conversations": len(prefixes),
        "stable": sum(1 for values in prefixes.values() if len(values) == 1),
    }


def summarize(rows: list[dict[str, Any]], grades: dict[str, Any] | None = None) -> dict[str, Any]:
    """Aggregate one arm. Clinical grades come only from a human-filled file."""
    grades = grades or {}
    contexts = [row.get("context") or {} for row in rows]
    called = [row for row in rows if (row.get("modelCalls") or 0) > 0]
    first = [row["attemptClaims"][0] for row in called if row.get("attemptClaims")]
    final = [row["claims"] for row in called if row.get("claims")]
    codes = Counter(code for row in called for code in row.get("attemptCodes") or [])
    checks = [row["promptTokenCheck"] for row in rows if row.get("promptTokenCheck")]
    # Citations exist only where the model answered; a dry run or refusal is not a miss.
    critical = [row for row in called if row["gold"]["hasCritical"]]
    recalls = [row["gold"]["recall"] for row in called if row["gold"]["recall"] is not None]
    retried = [row for row in called if row.get("retry")]
    prompt_tokens = [float(c["promptTokens"]) for c in contexts if c.get("promptTokens") is not None]
    ttft = [float(row["ttftMs"]) for row in called if row.get("ttftMs") is not None]
    total = [float(row["totalMs"]) for row in called if row.get("totalMs") is not None]
    grade_counts: dict[str, Counter] = {field: Counter() for field in GRADE_FIELDS}
    for row in rows:
        given = grades.get(row["key"]) or {}
        for field in GRADE_FIELDS:
            grade_counts[field][str(given.get(field) or "ungraded")] += 1
    return {
        "caseCount": len(rows),
        "statusCounts": dict(sorted(Counter(str(row.get("status")) for row in rows).items())),
        "context": {
            "overflow": sum(1 for row in rows if row.get("status") == "context_overflow"),
            "ccoComplete": sum(1 for c in contexts if c.get("ccoComplete")),
            "tokenCountSource": dict(Counter(str(c.get("tokenCountSource")) for c in contexts)),
            "promptTokens": {"mean": _mean(prompt_tokens), "max": max(prompt_tokens, default=None)},
            "promptTokenMatch": {
                "checked": len(checks), "matched": sum(1 for item in checks if item["match"]),
            },
            "prefix": _prefix_stability(rows),
            "historyTruncatedRows": sum(
                1 for c in contexts if (c.get("history") or {}).get("truncated")
                or (c.get("history") or {}).get("dropped")
            ),
        },
        "verifier": {
            "modelCalledCount": len(called),
            "passRate": _rate(sum(1 for row in called if row.get("passed")), len(called)),
            "unsupportedClaimRateFirst": _rate(
                sum(item["rejected"] for item in first), sum(item["total"] for item in first),
            ),
            "unsupportedClaimRateFinal": _rate(
                sum(item["rejected"] for item in final), sum(item["total"] for item in final),
            ),
            "numericErrorRows": sum(
                1 for row in called if NUMERIC_CODES & set(row.get("attemptCodes") or [])
            ),
            "temporalErrorRows": sum(
                1 for row in called if TEMPORAL_CODES & set(row.get("attemptCodes") or [])
            ),
            "errorCodeDistribution": dict(sorted(codes.items())),
        },
        "retry": {
            "count": len(retried),
            "successCount": sum(1 for row in retried if row.get("passed")),
            "skippedForContext": sum(1 for row in called if row.get("retrySkipped")),
        },
        "fallbackCount": sum(1 for row in called if row.get("fallback")),
        "latencyMs": {
            "ttft": {"mean": _mean(ttft), "p95": _p95(ttft)},
            "total": {"mean": _mean(total), "p95": _p95(total)},
        },
        "gold": {
            "citationMacroRecall": _mean(recalls),
            "criticalQuestionCount": len(critical),
            "criticalMissRate": _rate(
                sum(1 for row in critical if row["gold"]["criticalMiss"]), len(critical),
            ),
        },
        "manualGrades": {field: dict(counts) for field, counts in grade_counts.items()},
    }


def grading_template(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {row["key"]: {field: None for field in GRADE_FIELDS} for row in rows}


def grade_errors(grades: dict[str, Any], rows: list[dict[str, Any]]) -> list[str]:
    keys = {row["key"] for row in rows}
    errors = [f"{key}: not in this run" for key in grades if key not in keys]
    for key, given in grades.items():
        for field, value in (given or {}).items():
            if field not in GRADE_SCALE:
                errors.append(f"{key}.{field}: unknown field")
            elif value is not None and value not in GRADE_SCALE[field]:
                errors.append(f"{key}.{field}: {value!r} not in {'/'.join(GRADE_SCALE[field])}")
    return errors


def regrade(payload: dict[str, Any], grades: dict[str, Any]) -> dict[str, Any]:
    """Summaries for a finished run. The model is not called again."""
    rows = payload["rows"]
    return {
        **payload,
        "summary": {
            arm: summarize([row for row in rows if row["arm"] == arm], grades)
            for arm in payload["arms"]
        },
    }


def _md(payload: dict[str, Any]) -> str:
    lines = [
        "# Full Context değerlendirmesi",
        "",
        "Klinik doğruluk otomatik ölçülmez. `grades.template.json` doldurulup",
        "`--regrade --grades` ile, model yeniden çağrılmadan özetlenir.",
        "Ham CCO metni bu dosyada yoktur; yanıtlar vardır.",
        "",
        *(f"- `{field}`: {' / '.join(values)}" for field, values in GRADE_SCALE.items()),
        "",
    ]
    for row in payload["rows"]:
        gold = row["gold"]
        missed = [item["groupId"] for item in gold["groups"] if item["critical"] and not item["satisfied"]]
        lines.extend([
            f"## {row['key']}",
            "",
            f"- soru: {row['question']}",
            f"- durum: `{row.get('status')}`  doğrulama: `{row.get('passed')}`",
            f"- kodlar: `{', '.join(row.get('attemptCodes') or []) or '-'}`",
            f"- kaçan kritik gruplar (atıfa göre): `{', '.join(missed) or '-'}`",
            f"- beklentiler: {', '.join(row['clinicalExpectations']) or '-'}",
            "",
            str(row.get("answer") or "(yanıt yok)").strip(),
            "",
        ])
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--tokenize", action="store_true")
    parser.add_argument("--grades", type=Path)
    parser.add_argument(
        "--regrade", action="store_true",
        help="re-summarize out-dir/metrics.json with --grades; no model call",
    )
    args = parser.parse_args()
    args.arms = tuple(item.strip() for item in args.arms.split(",") if item.strip())
    if not args.arms or set(args.arms) - set(ARMS):
        parser.error(f"--arms must be a subset of {','.join(ARMS)}")
    if not _inside_eval_root(args.out_dir):
        parser.error("--out-dir must stay under data/evaluation")
    if args.regrade:
        if not args.grades:
            parser.error("--regrade needs --grades")
        payload = _load_json(args.out_dir / "metrics.json")
        grades = _load_json(args.grades)
        errors = grade_errors(grades, payload["rows"])
        if errors:
            parser.error("invalid grades:\n" + "\n".join(errors))
        payload = regrade(payload, grades)
        (args.out_dir / "metrics.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        print(json.dumps(
            {arm: summary["manualGrades"] for arm, summary in payload["summary"].items()},
            ensure_ascii=False, indent=2,
        ))
        return 0
    if not args.benchmark or not args.data_root:
        parser.error("--benchmark and --data-root are required unless --regrade")
    payload = asyncio.run(_run(args))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    (args.out_dir / "answers.md").write_text(_md(payload), encoding="utf-8")
    template = args.out_dir / "grades.template.json"
    if not template.exists():
        template.write_text(
            json.dumps(grading_template(payload["rows"]), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    print(json.dumps({
        "benchmarkHash": payload["benchmarkHash"],
        "dryRun": payload["dryRun"],
        "tokenize": payload["tokenize"],
        "modelCalled": payload["modelCalled"],
        "summary": payload["summary"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
