"""Score text-record answers. The harness speaks HTTP; this module only compares."""

from __future__ import annotations

import math
from typing import Any


def score_case(case: dict[str, Any], response: dict[str, Any]) -> dict[str, Any]:
    expected = case["expected_status"]
    status = response.get("status")
    status_ok = status == expected
    haystack = response.get("answer") or ""
    for citation in response.get("citations") or []:
        if isinstance(citation, dict):
            haystack += "\n" + str(citation.get("quote") or "")
    facts = list(case.get("expected_facts") or [])
    missing = [fact for fact in facts if fact not in haystack]
    facts_ok = not missing
    critical = bool(case.get("critical"))
    metrics = response.get("metrics") or {}
    return {
        "id": case["id"],
        "patient": case["patient"],
        "critical": critical,
        "expectedStatus": expected,
        "status": status,
        "statusOk": status_ok,
        "factsOk": facts_ok,
        "expectedFactCount": len(facts),
        "factsFound": len(facts) - len(missing),
        "missingFacts": missing,
        "criticalFailure": critical and not (status_ok and facts_ok),
        "modelCalls": metrics.get("modelCalls"),
        "timeToFirstTokenMs": metrics.get("timeToFirstTokenMs"),
        "totalMs": metrics.get("totalMs"),
    }


def _percentile(values: list[float], percent: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(percent / 100 * len(ordered)) - 1))
    return ordered[index]


def _latency(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"p50": None, "p95": None, "mean": None}
    return {
        "p50": _percentile(values, 50),
        "p95": _percentile(values, 95),
        "mean": round(sum(values) / len(values), 2),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    facts_expected = sum(row["expectedFactCount"] for row in rows)
    facts_found = sum(row["factsFound"] for row in rows)
    status_matches = sum(1 for row in rows if row["statusOk"])
    ttft = [row["timeToFirstTokenMs"] for row in rows if isinstance(row["timeToFirstTokenMs"], (int, float))]
    total = [row["totalMs"] for row in rows if isinstance(row["totalMs"], (int, float))]
    calls = [row["modelCalls"] for row in rows if isinstance(row["modelCalls"], int)]
    return {
        "questions": len(rows),
        "statusMatches": status_matches,
        "statusAccuracy": round(status_matches / len(rows), 4) if rows else None,
        "expectedFacts": facts_expected,
        "factsFound": facts_found,
        "factCoverage": round(facts_found / facts_expected, 4) if facts_expected else None,
        "criticalFailureCount": sum(1 for row in rows if row["criticalFailure"]),
        "criticalFailures": [row["id"] for row in rows if row["criticalFailure"]],
        "modelCalls": sum(calls),
        "latencyMs": {"timeToFirstToken": _latency(ttft), "total": _latency(total)},
    }
