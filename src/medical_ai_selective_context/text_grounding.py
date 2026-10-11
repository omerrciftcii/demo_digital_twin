"""Parse a plain-text answer and keep only numbers the record actually contains.

The model does not emit JSON. The first line is a status token, and citations
are lightweight markers such as [Laboratuvar (hastane) 18.09.2026]. Line
numbers are resolved here. A number, date, unit pair, relative time, or causal
phrase that the cited section does not contain drops the answer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .index import normalize_text
from .text_document import TextRecord
from .text_safety import answer_has_unsafe_advice

_FENCE = re.compile(r"^```[a-zA-Z0-9_-]*\n?|\n?```$")
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)
_MARKER = re.compile(r"\[([^\[\]\n]{2,160})\]")
_NUMBER = re.compile(
    r"\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2})?(?:Z|[+-]\d{2}:\d{2})?)?"
    r"|\d{1,2}\.\d{1,2}\.\d{4}(?:[ T]\d{1,2}:\d{2})?"
    r"|\d{1,2}\.\d{1,2}"
    r"|\d{1,2}\.\d{4}"
    r"|\d{1,2}:\d{2}"
    r"|\d{1,3}(?:\.\d{3})+(?:,\d+)?"
    r"|\d+,\d+"
    r"|\d+"
)
_UNIT = re.compile(
    r"^(?:mg/dL|g/dL|pg/mL|ng/mL|mIU/L|µIU/mL|U/L|mmHg|mL/dk|µg|mcg|kg|mg|mL|mm|cm|°C)",
    re.IGNORECASE,
)
_RELATIVE = re.compile(
    r"\d+(?:[.,]\d+)?\s*(?:gün|gun|ay|yıl|yil|hafta)\s+önce",
    re.IGNORECASE,
)
_CAUSAL = (
    "nedeniyle", "yüzünden", "sebep oldu", "neden oldu", "bağlı olarak", "sonucu olarak",
)
_PAREN = re.compile(r"\([^)]*\)")


@dataclass
class GroundingResult:
    ok: bool
    status: str
    answer: str
    citations: list[dict]
    error: str = ""


def strip_model_wrappers(text: str) -> str:
    cleaned = _THINK.sub("", text or "")
    cleaned = cleaned.strip()
    if cleaned.startswith("```"):
        cleaned = _FENCE.sub("", cleaned).strip()
    return cleaned


def _status_and_body(text: str) -> tuple[str, str] | None:
    lines = text.splitlines()
    index = 0
    while index < len(lines) and not lines[index].strip():
        index += 1
    if index >= len(lines):
        return None
    folded = normalize_text(lines[index])
    status = None
    for token, name in (("durum: yanit", "answered"), ("durum: bilgi_yok", "insufficient_evidence")):
        if folded == token or folded.startswith(token + " "):
            status = name
            break
    if status is None:
        return None
    tail = ""
    if folded not in {"durum: yanit", "durum: bilgi_yok"}:
        colon = re.search(r"[:：]", lines[index])
        if colon:
            tail = re.sub(r"^\s*\S+\s*", "", lines[index][colon.end():]).strip()
    body = "\n".join(part for part in (tail, *lines[index + 1 :]) if part).strip()
    return status, body


def _heading_forms(heading: str) -> list[str]:
    folded = normalize_text(heading)
    short = " ".join(_PAREN.sub(" ", folded).split())
    forms = [folded]
    if short and short != folded:
        forms.append(short)
    return [form for form in forms if len(form) >= 4]


def _heading_matches(heading: str, marker: str) -> bool:
    folded_marker = normalize_text(marker)
    return any(form in folded_marker for form in _heading_forms(heading))


def _numbers(text: str) -> list[re.Match[str]]:
    return [
        match for match in _NUMBER.finditer(text)
        if not re.fullmatch(r"\d", match.group(0))
    ]


def _token_in_text(token: str, blob: str) -> bool:
    if re.fullmatch(r"\d{1,2}\.\d{1,2}", token):
        start = 0
        while True:
            found = blob.find(token, start)
            if found < 0:
                return False
            after = blob[found + len(token):found + len(token) + 5]
            if not re.match(r"\.\d{4}", after):
                return True
            start = found + 1
    return token in blob


def _unit_after(text: str, end: int) -> str | None:
    match = _UNIT.match(text[end:].lstrip(" \t"))
    return match.group(0) if match else None


def _near(blob: str, token: str, unit: str) -> bool:
    folded = blob.casefold()
    needle = token.casefold()
    unit_folded = unit.casefold()
    start = 0
    while True:
        found = folded.find(needle, start)
        if found < 0:
            return False
        window = folded[found:found + len(needle) + 48]
        if unit_folded in window:
            return True
        start = found + 1


def _numbers_supported(claim: str, blob: str) -> bool:
    for match in _numbers(claim):
        token = match.group(0)
        if not _token_in_text(token, blob):
            return False
        unit = _unit_after(claim, match.end())
        if unit and not _near(blob, token, unit):
            return False
        prefixed = match.start() > 0 and claim[match.start() - 1] == "%"
        if prefixed and f"%{token}" not in blob and f"% {token}" not in blob:
            return False
        suffixed = match.end() < len(claim) and claim[match.end()] == "%"
        if suffixed and f"{token}%" not in blob and f"%{token}" not in blob:
            return False
    def squash(value: str) -> str:
        return re.sub(r"\s+", " ", value.casefold())

    squashed = squash(blob)
    for match in _RELATIVE.finditer(claim):
        if squash(match.group(0)) not in squashed:
            return False
    return True


def _causal_supported(body: str, blob: str) -> bool:
    folded_body = normalize_text(body)
    folded_blob = normalize_text(blob)
    return all(
        normalize_text(phrase) not in folded_body or normalize_text(phrase) in folded_blob
        for phrase in _CAUSAL
    )


def _resolve_section(marker: str, claim: str, record: TextRecord) -> int | None:
    candidates = [
        index for index, section in enumerate(record.sections)
        if _heading_matches(section.heading, marker)
    ]
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    def score(index: int) -> int:
        section = record.sections[index]
        return sum(1 for match in _numbers(claim + " " + marker) if match.group(0) in section.text)

    ranked = sorted(candidates, key=score, reverse=True)
    if score(ranked[0]) == 0:
        return None
    return ranked[0]


def _best_line(section_index: int, claim: str, record: TextRecord) -> tuple[int, str]:
    section = record.sections[section_index]
    best_number = section.start_line
    best_text = section.lines[0] if section.lines else section.heading
    best_score = -1
    claim_numbers = [match.group(0) for match in _numbers(claim)]
    for line in record.lines:
        if line.section_index != section_index or line.text.startswith("##"):
            continue
        score = sum(1 for token in claim_numbers if token in line.text)
        if score > best_score:
            best_score = score
            best_number = line.number
            best_text = line.text
    return best_number, best_text


def _pairs(body: str) -> list[tuple[str, str | None]]:
    pieces = _MARKER.split(body)
    # split with a capturing group yields text, marker, text, marker, text
    pairs: list[tuple[str, str | None]] = []
    pending = ""
    for index, piece in enumerate(pieces):
        if index % 2 == 1:
            pairs.append((pending, piece.strip()))
            pending = ""
        else:
            pending += piece
    if pending.strip() or not pairs:
        pairs.append((pending, None))
    return pairs


def ground_answer(raw: str, record: TextRecord) -> GroundingResult:
    parsed = _status_and_body(strip_model_wrappers(raw))
    if parsed is None:
        return GroundingResult(False, "validation_failed", "", [], "missing_status")
    status, body = parsed
    if not body:
        return GroundingResult(False, "validation_failed", "", [], "empty_answer")
    if answer_has_unsafe_advice(body):
        return GroundingResult(False, "safety_refusal", "", [], "unsafe_advice")
    citations: list[dict] = []
    seen: set[tuple] = set()
    resolved: list[tuple[str, int | None]] = []
    last_index: int | None = None
    for claim, marker in _pairs(body):
        section_index: int | None = None
        if marker:
            section_index = _resolve_section(marker, claim, record)
            if section_index is not None:
                last_index = section_index
                section = record.sections[section_index]
                if not _numbers_supported(marker, section.text):
                    return GroundingResult(False, "validation_failed", "", [], "unsupported_number")
                number, quote = _best_line(section_index, claim + " " + marker, record)
                key = (section.heading, number, quote)
                if key not in seen:
                    seen.add(key)
                    citations.append({
                        "section": section.heading,
                        "startLine": number,
                        "endLine": number,
                        "quote": quote,
                        "marker": marker,
                    })
        resolved.append((claim, section_index if section_index is not None else last_index))

    if status == "answered" and not citations:
        return GroundingResult(False, "validation_failed", "", [], "missing_citation")

    for claim, section_index in resolved:
        if not claim.strip():
            continue
        blob = record.sections[section_index].text if section_index is not None else ""
        if (_numbers(claim) or _RELATIVE.search(claim)) and (
            section_index is None or not _numbers_supported(claim, blob)
        ):
            return GroundingResult(False, "validation_failed", "", [], "unsupported_number")
        if not _causal_supported(claim, blob):
            return GroundingResult(False, "validation_failed", "", [], "unsupported_causal_link")
    return GroundingResult(True, status, body, citations)
