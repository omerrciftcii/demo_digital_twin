"""Score text-record answers. The harness speaks HTTP; this module only compares.

Exact substring match is still recorded. A fact also counts when its drug,
dose, date, and the load-bearing words sit together, so "56 FİLM TABLET" is
not a clinical miss. Pack size and glue words are not keys.
"""

from __future__ import annotations

import math
import re
from typing import Any

from .index import normalize_text

_DATE = re.compile(r"\d{1,2}\.\d{1,2}\.\d{4}|\d{1,2}\.\d{4}")
_YEAR = re.compile(r"\d{4}")
_NUM_UNIT = re.compile(
    r"\d+(?:[.,]\d+)?\s*(?:mg/dL|g/dL|pg/mL|ng/mL|mIU/L|mmHg|mL/dk|µg|mcg|kg|mg|mL|mm|cm|g)\b",
    re.IGNORECASE,
)
_LIMIT = re.compile(r"en fazla\s+\d+", re.IGNORECASE)
_CLOCK = re.compile(r"\d+\s*saat(?:te|ta)?", re.IGNORECASE)
_PACK_PHRASE = re.compile(
    r"\b\d+\s+(?:film\s+)?(?:tablet|kapsül|kapsul|draje)\b",
    re.IGNORECASE,
)
_DROP_WORDS = re.compile(r"\b(?:film|tablet|kapsül|kapsul|draje|itibarıyla|itibariyla)\b", re.IGNORECASE)
_WORD = re.compile(r"[^\W\d_]{5,}", re.UNICODE)
_PHQ = re.compile(r"\b(PHQ-\d{1,2})(?:\s+(\d{1,2}))?", re.IGNORECASE)
_PLUS = re.compile(r"\+\d+(?:,\d+)?")
_DAYS = re.compile(r"\d+\s*gün(?:de|lük|luk)?", re.IGNORECASE)
_CHANGE = frozenset({
    "cikarildi", "yukseltildi", "artirildi", "arttirildi",
    "dusuruldu", "azaltildi", "indirildi",
})


def _spans(pattern: re.Pattern[str], text: str) -> list[tuple[int, int, str]]:
    return [(match.start(), match.end(), match.group(0)) for match in pattern.finditer(text)]


def fact_tokens(fact: str) -> list[tuple[str, str]]:
    """Drug, measurement, date, and long words. Packaging and glue are ignored."""
    text = _PACK_PHRASE.sub(" ", fact)
    text = _DROP_WORDS.sub(" ", text).replace("(", " ").replace(")", " ")
    occupied: list[tuple[int, int]] = []
    tokens: list[tuple[str, str]] = []
    for match in _PHQ.finditer(text):
        occupied.append((match.start(1), match.end(1)))
        tokens.append(("scale", normalize_text(match.group(1))))
        if match.group(2):
            occupied.append((match.start(2), match.end(2)))
            tokens.append(("num", match.group(2)))
    for match in _PLUS.finditer(text):
        if any(match.start() < right and match.end() > left for left, right in occupied):
            continue
        occupied.append((match.start(), match.end()))
        tokens.append(("plus", normalize_text(match.group(0))))
    for kind, pattern in (
        ("date", _DATE), ("days", _DAYS), ("limit", _LIMIT), ("unit", _NUM_UNIT),
        ("clock", _CLOCK), ("year", _YEAR),
    ):
        for start, end, raw in _spans(pattern, text):
            if any(start < right and end > left for left, right in occupied):
                continue
            occupied.append((start, end))
            tokens.append((kind, normalize_text(re.sub(r"\s+", " ", raw.strip()))))
    for match in _WORD.finditer(text):
        if any(match.start() < right and match.end() > left for left, right in occupied):
            continue
        tokens.append(("word", normalize_text(match.group(0))))
    return tokens


def _needle_in(needle: str, region: str) -> bool:
    if needle in region:
        return True
    if needle.startswith("+"):
        bare = re.escape(needle[1:])
        if re.search(rf"(?<!\d)\+?{bare}(?!\d)", region):
            return True
    dated = re.fullmatch(r"0?(\d{1,2})\.0?(\d{1,2})\.(\d{4})", needle)
    if dated:
        day, month, year = (int(part) for part in dated.groups())
        if re.search(rf"(?<!\d)0?{day}\.0?{month}\.{year}(?!\d)", region):
            return True
    if needle in _CHANGE and any(verb in region for verb in _CHANGE):
        return True
    clock = re.sub(r"saat(?:te|ta)$", "saat", needle)
    if clock != needle and clock in region:
        return True
    compact = needle.replace(" ", "")
    return bool(compact) and compact in region.replace(" ", "")


def tokens_cover(fact: str, haystack: str) -> bool:
    tokens = fact_tokens(fact)
    if len(tokens) < 2:
        return False
    folded = normalize_text(haystack)
    words = [token for kind, token in tokens if kind == "word"]
    anchor = max(words or [token for _kind, token in tokens], key=len)
    # A dose has to sit on a long drug name. A short anchor may sit further
    # from its number ("son 2 günde" / "1,3 kg"). "Eliquis" stays tight so a
    # Coumadin 5 mg does not satisfy an Eliquis 5 mg fact.
    tight = {"unit", "limit", "clock"}
    long_anchor = bool(words) and len(anchor) >= 7
    doses = [token for kind, token in tokens if kind == "unit"]
    flexible_change = len(doses) >= 2
    start = 0
    while True:
        at = folded.find(anchor, start)
        if at < 0:
            return False
        matched = True
        for kind, token in tokens:
            before, after = (8, 48) if kind in tight and long_anchor else (40, 72)
            region = folded[max(0, at - before): at + len(anchor) + after]
            if token in _CHANGE and not flexible_change:
                if token not in region:
                    matched = False
                    break
                continue
            if not _needle_in(token, region):
                matched = False
                break
        if matched:
            return True
        start = at + 1


def score_case(case: dict[str, Any], response: dict[str, Any]) -> dict[str, Any]:
    expected = case["expected_status"]
    status = response.get("status")
    status_ok = status == expected
    haystack = response.get("answer") or ""
    for citation in response.get("citations") or []:
        if isinstance(citation, dict):
            haystack += "\n" + str(citation.get("quote") or "")
    facts = list(case.get("expected_facts") or [])
    missing: list[str] = []
    exact_found = 0
    for fact in facts:
        if fact in haystack:
            exact_found += 1
        elif not tokens_cover(fact, haystack):
            missing.append(fact)
    facts_ok = not missing
    critical = bool(case.get("critical"))
    metrics = response.get("metrics") or {}
    row = {
        "id": case["id"],
        "patient": case["patient"],
        "critical": critical,
        "expectedStatus": expected,
        "status": status,
        "statusOk": status_ok,
        "factsOk": facts_ok,
        "expectedFactCount": len(facts),
        "factsFound": len(facts) - len(missing),
        "exactFactsFound": exact_found,
        "missingFacts": missing,
        "criticalFailure": critical and not (status_ok and facts_ok),
        "modelCalls": metrics.get("modelCalls"),
        "timeToFirstTokenMs": metrics.get("timeToFirstTokenMs"),
        "totalMs": metrics.get("totalMs"),
        "answer": response.get("answer") if isinstance(response.get("answer"), str) else None,
    }
    raw = response.get("rawOutput")
    if isinstance(raw, str) and raw.strip():
        row["rawOutput"] = raw
    validation = response.get("validation")
    if isinstance(validation, dict) and validation.get("error"):
        row["validationError"] = validation["error"]
    return row


def case_error(case: dict[str, Any], code: str) -> dict[str, Any]:
    """One question failed in transport. The rest of the run still gets a report."""
    facts = list(case.get("expected_facts") or [])
    critical = bool(case.get("critical"))
    return {
        "id": case.get("id"),
        "patient": case.get("patient"),
        "critical": critical,
        "expectedStatus": case.get("expected_status"),
        "status": None,
        "statusOk": False,
        "factsOk": False,
        "expectedFactCount": len(facts),
        "factsFound": 0,
        "exactFactsFound": 0,
        "missingFacts": facts,
        "criticalFailure": critical,
        "modelCalls": None,
        "timeToFirstTokenMs": None,
        "totalMs": None,
        "answer": None,
        "error": code,
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
    exact_found = sum(row.get("exactFactsFound", 0) for row in rows)
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
        "exactFactCoverage": round(exact_found / facts_expected, 4) if facts_expected else None,
        "caseErrors": sum(1 for row in rows if row.get("error")),
        "criticalFailureCount": sum(1 for row in rows if row["criticalFailure"]),
        "criticalFailures": [row["id"] for row in rows if row["criticalFailure"]],
        "modelCalls": sum(calls),
        "latencyMs": {"timeToFirstToken": _latency(ttft), "total": _latency(total)},
    }
