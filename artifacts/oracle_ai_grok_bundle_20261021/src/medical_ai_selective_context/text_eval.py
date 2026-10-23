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

_MONTHS = (
    "ocak", "subat", "mart", "nisan", "mayis", "haziran",
    "temmuz", "agustos", "eylul", "ekim", "kasim", "aralik",
)
_MONTH_OF = {name: index for index, name in enumerate(_MONTHS, start=1)}
_NAMED_DATE = re.compile(
    r"\b(\d{1,2})\s+("
    r"ocak|şubat|subat|mart|nisan|mayıs|mayis|haziran|temmuz|"
    r"ağustos|agustos|eylül|eylul|ekim|kasım|kasim|aralık|aralik"
    r")\s+(\d{4})\b",
    re.IGNORECASE,
)
_DATE = re.compile(r"\d{1,2}\.\d{1,2}\.\d{4}|\d{1,2}\.\d{4}")
_YEAR = re.compile(r"\d{4}")
_LETTER_SUFFIXES = (
    "lerinden", "larindan", "imizden", "imizdan",
    "nin", "nın", "nun", "nün", "den", "dan", "ten", "tan",
    "yi", "yı", "yu", "yü", "ler", "lar",
    "in", "ın", "un", "ün", "de", "da", "te", "ta",
    "i", "ı", "u", "ü", "e", "a",
)
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


def _canonical_date(day: str, month: int, year: str) -> str:
    return f"{int(day):02d}.{month:02d}.{year}"


def _month_forms(token: str) -> list[str]:
    dated = re.fullmatch(r"0?(\d{1,2})\.0?(\d{1,2})\.(\d{4})", token)
    if not dated:
        return []
    day, month, year = (int(part) for part in dated.groups())
    if not 1 <= month <= 12:
        return []
    name = _MONTHS[month - 1]
    return [f"{day} {name} {year}", f"{day:02d} {name} {year}"]


def _letter_stems(token: str) -> list[str]:
    """Stems of an all-letter token. Every suffix length is tried."""
    folded = normalize_text(token)
    if not folded.isalpha() or len(folded) < 5:
        return []
    stems: list[str] = []
    for suffix in _LETTER_SUFFIXES:
        ending = normalize_text(suffix)
        if folded.endswith(ending) and len(folded) - len(ending) >= 4:
            stem = folded[: len(folded) - len(ending)]
            if stem not in stems:
                stems.append(stem)
    return stems


def fact_tokens(fact: str) -> list[tuple[str, str]]:
    """Drug, measurement, date, and long words. Packaging and glue are ignored."""
    text = _PACK_PHRASE.sub(" ", fact)
    text = _DROP_WORDS.sub(" ", text).replace("(", " ").replace(")", " ")
    occupied: list[tuple[int, int]] = []
    tokens: list[tuple[str, str]] = []
    for match in _NAMED_DATE.finditer(text):
        month = _MONTH_OF.get(normalize_text(match.group(2)))
        if not month:
            continue
        occupied.append((match.start(), match.end()))
        tokens.append(("date", _canonical_date(match.group(1), month, match.group(3))))
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
        for form in _month_forms(needle):
            if form in region:
                return True
    if needle.isalpha():
        for stem in _letter_stems(needle):
            if re.search(rf"(?<![a-z]){re.escape(stem)}(?![a-z])", region):
                return True
    if needle in _CHANGE and any(verb in region for verb in _CHANGE):
        return True
    clock = re.sub(r"saat(?:te|ta)$", "saat", needle)
    if clock != needle and clock in region:
        return True
    compact = needle.replace(" ", "")
    return bool(compact) and compact in region.replace(" ", "")


def _anchor_hits(folded: str, anchor: str, start: int) -> list[tuple[int, int]]:
    """Places the anchor can sit: the token, a month-name date, or a letter stem."""
    found: list[tuple[int, int]] = []
    at = folded.find(anchor, start)
    if at >= 0:
        found.append((at, len(anchor)))
    for form in _month_forms(anchor):
        alt = folded.find(form, start)
        if alt >= 0:
            found.append((alt, len(form)))
    if anchor.isalpha():
        for stem in _letter_stems(anchor):
            match = re.search(rf"(?<![a-z]){re.escape(stem)}(?![a-z])", folded[start:])
            if match:
                found.append((start + match.start(), match.end() - match.start()))
    return found


def tokens_cover(fact: str, haystack: str) -> bool:
    tokens = fact_tokens(fact)
    if not tokens:
        return False
    standalone = {"date", "unit", "plus", "scale", "limit", "days", "clock"}
    if len(tokens) < 2 and tokens[0][0] not in standalone:
        return False
    folded = normalize_text(haystack)
    words = [token for kind, token in tokens if kind == "word"]
    anchor = max(words or [token for _kind, token in tokens], key=len)
    # A dose has to sit on a long drug name. A short anchor may sit further
    # from its number. A dose stays tied to its own drug name.
    tight = {"unit", "limit", "clock"}
    long_anchor = bool(words) and len(anchor) >= 7
    doses = [token for kind, token in tokens if kind == "unit"]
    flexible_change = len(doses) >= 2
    start = 0
    while True:
        hits = _anchor_hits(folded, anchor, start)
        if not hits:
            return False
        at, span = min(hits, key=lambda item: item[0])
        matched = True
        for kind, token in tokens:
            before, after = (8, 48) if kind in tight and long_anchor else (40, 72)
            region = folded[max(0, at - before): at + span + after]
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


_DISCLAIMER_MARK = "Bu yanıt tıbbi tanı, tedavi veya acil müdahale önerisi değildir."
_SIZ = re.compile(
    r"\b(yapmalisiniz|kullanmalisiniz|degeriniz|tahliliniz|hekiminize)\b"
)


def clinical_body(answer: str) -> str:
    """The sentences the patient reads, without the fixed siz-form disclaimer."""
    body = answer or ""
    if _DISCLAIMER_MARK in body:
        body = body.split(_DISCLAIMER_MARK)[0]
    return re.sub(r"\s*\[[^\[\]\n]{2,160}\]", "", body).strip()


def conversation_issues(answer: str, status: str) -> list[str] | None:
    """None when the row is a refusal or a blocked validation. Otherwise a problem list.

    A raw chart dump, a broken sentence, or a body outside the length window fails.
    """
    if status not in {"answered", "insufficient_evidence"}:
        return None
    body = clinical_body(answer)
    issues: list[str] = []
    if len(body) < 30 or len(body) > 4000:
        issues.append("length")
    sentences = [
        part.strip() for part in re.split(r"\n+|(?<=[.!?])\s+", body) if part.strip()
    ]
    sentences = [part for part in sentences if not re.fullmatch(r"\[[^\[\]]+\]", part)]
    if not sentences:
        issues.append("unfinished")
        return issues
    for sentence in sentences:
        folded = normalize_text(sentence)
        dumped = (
            sentence.startswith("-")
            or "Şikâyet:" in sentence
            or "Şikayet:" in sentence
            or "ŞİKAYET:" in sentence
            or "Hikaye:" in sentence
            or "Hikâye:" in sentence
            or "HİKAYE:" in sentence
            or "Epikriz:" in sentence
            or "Bulgular:" in sentence
            or "BULGULAR:" in sentence
            or "Özgeçmiş:" in sentence
            or len(sentence) > 450
        )
        if dumped:
            issues.append("dump")
        if any(token in folded for token in ("seyrindi", "bilinmeyen cumle", "kayittaki cumleyi")):
            issues.append("broken")
        if sentence[-1] not in ".!?":
            issues.append("unfinished")
        if _SIZ.search(folded):
            issues.append("siz_form")
    return list(dict.fromkeys(issues))


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
    folded_haystack = haystack.casefold()
    for fact in facts:
        if fact in haystack or fact.casefold() in folded_haystack:
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
    issues = conversation_issues(row["answer"] or "", status or "")
    row["conversationOk"] = None if issues is None else not issues
    row["conversationIssues"] = issues
    raw = response.get("rawOutput")
    if isinstance(raw, str) and raw.strip():
        row["rawOutput"] = raw
    validation = response.get("validation")
    if isinstance(validation, dict) and validation.get("error"):
        row["validationError"] = validation["error"]
        if validation.get("debug"):
            row["validationDebug"] = validation["debug"]
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
        "conversationOk": None,
        "conversationIssues": None,
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
        return {"average": None, "p50": None, "p95": None, "mean": None}
    average = round(sum(values) / len(values), 2)
    return {
        "average": average,
        "p50": _percentile(values, 50),
        "p95": _percentile(values, 95),
        "mean": average,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    facts_expected = sum(row["expectedFactCount"] for row in rows)
    facts_found = sum(row["factsFound"] for row in rows)
    exact_found = sum(row.get("exactFactsFound", 0) for row in rows)
    status_matches = sum(1 for row in rows if row["statusOk"])
    ttft = [row["timeToFirstTokenMs"] for row in rows if isinstance(row["timeToFirstTokenMs"], (int, float))]
    total = [row["totalMs"] for row in rows if isinstance(row["totalMs"], (int, float))]
    calls = [row["modelCalls"] for row in rows if isinstance(row["modelCalls"], int)]
    checked = [row for row in rows if row.get("conversationOk") is not None]
    passed = sum(1 for row in checked if row["conversationOk"])
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
        "conversational": {
            "checked": len(checked),
            "passed": passed,
            "average": round(passed / len(checked), 4) if checked else None,
        },
        "latencyMs": {"timeToFirstToken": _latency(ttft), "total": _latency(total)},
    }
