"""Aggregate the isolated experiment without equating a verifier to a clinician."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

ARMS = ("oracle", "full", "selective", "hybrid")
CLINICAL_FIELDS = (
    "unsupported_clinical_claims",
    "numeric_accuracy",
    "temporal_accuracy",
    "diagnosis_overcertainty",
    "incorrect_causality",
    "risk_omission",
    "missing_data_management",
    "history_retention",
    "clinical_quality",
)


def mean(values):
    values = [v for v in values if v is not None]
    return round(statistics.mean(values), 4) if values else None


def percentile95(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return (
        round(statistics.quantiles(values, n=20, method="inclusive")[18], 2)
        if len(values) > 1
        else values[0]
    )


def summarize(rows):
    called = [r for r in rows if r["modelCalls"] > 0]
    first = [r["attempts"][0]["verdict"] for r in called if r["attempts"]]
    # Per-attempt outcomes are not the post-filter user-visible pass rate.
    last = [r["attempts"][-1]["verdict"] for r in called if r["attempts"]]
    validations = [r["result"].get("validation", {}) for r in called]
    metrics = [r["result"].get("metrics", {}) for r in called]
    codes = Counter(c for r in called for a in r["attempts"] for c in a["verdict"]["codes"])
    result = {
        "rows": len(rows),
        "calledRows": len(called),
        "statusCounts": dict(Counter(r["result"].get("status") for r in rows)),
        "modelCalls": sum(r["modelCalls"] for r in called),
        "verifierPassCount": sum(bool(v.get("passed")) for v in validations),
        "verifierPassRate": mean([bool(v.get("passed")) for v in validations]),
        "fullyCleanFirstCount": sum(bool(v["fully_ok"]) for v in first),
        "fullyCleanLastCount": sum(bool(v["fully_ok"]) for v in last),
        "firstClaimCount": sum(v["claim_count"] for v in first),
        "firstRejectedClaimCount": sum(len(v["rejected_claims"]) for v in first),
        "retryCount": sum(bool(m.get("retry")) for m in metrics),
        "fallbackCount": sum(bool(m.get("fallback")) for m in metrics),
        "retryPassCount": sum(
            bool(m.get("retry")) and bool(v.get("passed")) for m, v in zip(metrics, validations)
        ),
        "verifierErrorCodeCounts": dict(codes),
        "latencyMs": {
            "engineMean": mean([m.get("totalMs") for m in metrics]),
            "engineP95": percentile95([m.get("totalMs") for m in metrics]),
            "ttftMean": mean([m.get("timeToFirstTokenMs") for m in metrics]),
            "ttftP95": percentile95([m.get("timeToFirstTokenMs") for m in metrics]),
            "harnessMean": mean([r["compositionAndAnswerMs"] for r in called]),
        },
        "promptTokens": {
            "min": min((r["context"]["promptTokens"] for r in rows), default=None),
            "max": max((r["context"]["promptTokens"] for r in rows), default=None),
        },
        "rawCcoCompleteCount": sum(bool(r["context"].get("ccoComplete")) for r in rows),
        "historyDroppedMessages": sum(r["context"]["history"]["dropped"] for r in rows),
        "historyTruncatedMessages": sum(r["context"]["history"]["truncated"] for r in rows),
        "defaultSafetyDecisionCounts": dict(
            Counter(r["defaultSafetyDecision"]["reason"] for r in rows)
        ),
        "clinicalMetrics": {
            field: {"value": None, "graded": 0, "status": "requires_blinded_clinical_review"}
            for field in CLINICAL_FIELDS
        },
    }
    for stage, name in (("catalogCoverage", "catalog"), ("rawTextCoverage", "rawText")):
        # Coverage of attempted composition includes overflows, so expose sent-only separately.
        for subset_name, subset in (("Composed", rows), ("Sent", called)):
            result[name + subset_name] = {
                "groupRecall": mean([r["context"][stage]["groupRecall"] for r in subset]),
                "criticalMissRate": mean([r["context"][stage]["criticalMiss"] for r in subset]),
                "criticalDenominator": sum(
                    r["context"][stage]["criticalMiss"] is not None for r in subset
                ),
            }
    result["citation"] = {
        "groupRecall": mean([r["citationCoverage"]["groupRecall"] for r in called]),
        "criticalMissRate": mean([r["citationCoverage"]["criticalMiss"] for r in called]),
    }
    return result


def aggregate(rows):
    # Matched questions only, excluding any arm's overflow/failure, avoid full-arm patient bias.
    by_case = {}
    for r in rows:
        by_case.setdefault(r["case"]["case_id"], {})[r["arm"]] = r
    matched = {
        cid
        for cid, arms in by_case.items()
        if set(arms) == set(ARMS) and all(r["modelCalls"] > 0 for r in arms.values())
    }
    return {
        "totalRows": len(rows),
        "complete": len(rows) == 132,
        "matchedCaseCount": len(matched),
        "all": {a: summarize([r for r in rows if r["arm"] == a]) for a in ARMS},
        "matched": {
            a: summarize([r for r in rows if r["arm"] == a and r["case"]["case_id"] in matched])
            for a in ARMS
        },
        "singleTurn": {
            a: summarize(
                [r for r in rows if r["arm"] == a and not r["case"].get("conversation_id")]
            )
            for a in ARMS
        },
        "conversationTurns": {
            a: summarize([r for r in rows if r["arm"] == a and r["case"].get("conversation_id")])
            for a in ARMS
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = [
        json.loads(s)
        for s in (args.run_dir / "rows.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    summary = aggregate(rows)
    if not summary["complete"]:
        raise ValueError("incomplete_run_not_final")
    (args.run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    template = args.run_dir / "clinical_review.template.json"
    if not template.exists():
        template.write_text(
            json.dumps(
                {
                    r["key"]: {
                        "reviewer": None,
                        "adjudicated": False,
                        "claim_annotations": [],
                        **{field: None for field in CLINICAL_FIELDS},
                    }
                    for r in rows
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    print(
        json.dumps(
            {
                a: {k: v for k, v in s.items() if k not in ("clinicalMetrics",)}
                for a, s in summary["all"].items()
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
