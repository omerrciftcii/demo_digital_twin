"""Parse a plain-text answer and keep only numbers the record actually contains.

The model does not emit JSON. The first line is a status token, and citations
are lightweight markers such as [Laboratuvar (hastane)]. Line
numbers are resolved here. A number, date, unit pair, relative time, or causal
phrase that the cited section does not contain drops the answer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from rapidfuzz import fuzz

from .index import normalize_text
from .text_document import TextRecord
from .text_llm import strip_think_blocks
from .text_safety import answer_has_unsafe_advice

_FENCE = re.compile(r"^```[a-zA-Z0-9_-]*\n?|\n?```$")
# rapidfuzz ratio is 0–100. About 0.9 catches a repeated sentence with light edits.
_NEAR_DUPLICATE = 90
_MARKER = re.compile(
    r"\[([^\[\]\n]{2,160})\]|【([^】\n]{2,160})】|［([^］\n]{2,160})］"
)
_SENTENCE = re.compile(r"\n+|(?<=[.!?])\s+")
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
# A diagnosis code is one token. Its inner digits are not a lab value.
_ICD = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]\d{2}(?:\.\d{1,4})?(?!\d)")
# "(referans 0-41)" and "(0–41)" are ranges the model often adds. They are not the result.
_REF_PAREN = re.compile(
    r"\([^)\n]{0,96}(?:referans|aral[ıi]k|\d+(?:[.,]\d+)?\s*[-–—]\s*\d+(?:[.,]\d+)?)[^)\n]{0,48}\)",
    re.IGNORECASE,
)


@dataclass
class GroundingResult:
    ok: bool
    status: str
    answer: str
    citations: list[dict]
    error: str = ""


def strip_model_wrappers(text: str) -> str:
    cleaned = strip_think_blocks(text).strip()
    if cleaned.startswith("```"):
        cleaned = _FENCE.sub("", cleaned).strip()
    return cleaned


_STATUS_PREFIXES = (
    ("durum: bilgi_yok", "insufficient_evidence"),
    ("durum: bilgi yok", "insufficient_evidence"),
    ("durum: yanit", "answered"),
)


def _status_of_line(line: str) -> str | None:
    folded = normalize_text(line).strip(" *_`#>")
    for token, status in _STATUS_PREFIXES:
        if folded == token or folded.startswith((token + " ", token + ".", token + ":")):
            return status
    return None


def _line_is_only_status(line: str) -> bool:
    folded = normalize_text(line).strip(" *_`#>:-")
    return folded in {token for token, _status in _STATUS_PREFIXES}


def _infer_status(text: str) -> str | None:
    compact = text.strip()
    if not compact:
        return None
    folded = normalize_text(compact)
    has_number = bool(_numbers(compact))
    absence = (
        "bilgi yok", "bilgi_yok", "kayitta yok", "kayitta bulunmuyor",
        "bilinmiyor", "bulunamadi",
    )
    if any(phrase in folded for phrase in absence) and not has_number:
        return "insufficient_evidence"
    if has_number or _MARKER.search(compact):
        return "answered"
    if len(compact) < 80:
        return None
    return "answered"


def _status_and_body(text: str) -> tuple[str, str] | None:
    lines = text.splitlines()
    for index, line in enumerate(lines[:8]):
        if not line.strip() or _status_of_line(line) is None:
            continue
        status = _status_of_line(line)
        tail = "" if _line_is_only_status(line) else _tail_after_status(line)
        body = "\n".join(part for part in (tail, *lines[index + 1 :]) if part).strip()
        return status or "answered", body
    inferred = _infer_status(text)
    if inferred is None:
        return None
    return inferred, text.strip()


def _tail_after_status(line: str) -> str:
    """Keep the answer when a small model puts it on the DURUM line."""
    cleaned = line.strip().strip("*_`")
    folded = normalize_text(cleaned)
    for token, _status in _STATUS_PREFIXES:
        if folded.startswith(token):
            # The folded prefix can be shorter than the original (İ → i).
            # Drop leading words until the folded remainder matches.
            words = cleaned.split()
            for count in range(1, len(words) + 1):
                rest = " ".join(words[count:])
                if normalize_text(rest) == folded[len(token):].strip(" :.-") or (
                    not folded[len(token):].strip(" :.-") and not rest
                ):
                    return rest.strip(" :.-")
            return ""
    return ""


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


def _fold_unit(text: str) -> str:
    folded = normalize_text(text).replace("µ", "u").replace("μ", "u")
    return folded.replace("²", "2").replace("º", "°")


# Same family means the same quantity. Aliases are spelling, not a conversion.
# A body-surface suffix on a per-minute volume is stripped before lookup.
_UNIT_FAMILIES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("per_min", ("atım/dakika", "nabız/dk", "vuru/dk", "atım/dk", "/dakika", "bpm", "/dk")),
    ("l_per_min", ("l/min", "l/dk")),
    ("ml_per_kg_min", ("ml/kg/min", "ml/kg/dk")),
    ("ml_per_min", ("ml/min", "ml/dk")),
    ("mg_per_dl", ("mg/dl",)),
    ("g_per_dl", ("g/dl",)),
    ("ng_per_ml", ("ng/ml",)),
    ("pg_per_ml", ("pg/ml",)),
    ("ng_per_dl", ("ng/dl",)),
    ("ug_per_dl", ("µg/dl", "mcg/dl")),
    ("mmol_per_l", ("mmol/l",)),
    ("miu_per_l", ("miu/l",)),
    ("uiu_per_ml", ("µiu/ml",)),
    ("u_per_l", ("iu/l", "u/l")),
    ("mmhg", ("mm hg", "mmhg")),
    ("celsius", ("celsius", "°c")),
    ("percent", ("%",)),
    ("ms", ("milisaniye", "millisaniye", "msec", "ms")),
    ("minute", ("dakika", "dk")),
    ("second", ("saniye", "sn")),
    ("hour", ("saat", "sa")),
    ("day", ("gün",)),
    ("week", ("hafta",)),
    ("year", ("yıl",)),
    ("kg", ("kilogram", "kg")),
    ("ug", ("µg", "mcg")),
    ("mg", ("mg",)),
    ("ml", ("ml",)),
    ("liter", ("l",)),
    ("mm", ("mm",)),
    ("cm", ("cm",)),
    ("fl", ("fl",)),
    ("gram", ("g",)),
)


def _unit_surfaces() -> tuple[tuple[str, str], ...]:
    rows: list[tuple[str, str]] = []
    seen: set[str] = set()
    for family, surfaces in _UNIT_FAMILIES:
        for surface in surfaces:
            folded = re.sub(r"\s*/\s*", "/", _fold_unit(surface))
            if not folded or folded in seen:
                continue
            seen.add(folded)
            rows.append((folded, family))
    rows.sort(key=lambda item: len(item[0]), reverse=True)
    return tuple(rows)


_UNIT_SURFACES = _unit_surfaces()
_BSA_SUFFIX = re.compile(r"/1[.,]73\s*m2")
_RATIO_TAIL = re.compile(r"^\s*/\s*\d+(?:[.,]\d+)?")
_UNIT_CUT = re.compile(r"[\(\[;\n]|\.(?!\d)")
_SYMBOLIC_UNIT = re.compile(r"^(?:10\^\d+/[a-z0-9]+)")


def _unit_window(text: str, end: int) -> str:
    rest = _RATIO_TAIL.sub("", text[end:end + 48], count=1).lstrip(" \t")
    rest = _UNIT_CUT.split(rest, maxsplit=1)[0]
    folded = _fold_unit(rest)
    folded = _BSA_SUFFIX.sub("", folded, count=1)
    return re.sub(r"\s*/\s*", "/", folded)


def _boundary(text: str, size: int) -> bool:
    return size >= len(text) or not text[size].isalnum()


def _match_family(window: str) -> str | None:
    compact = re.sub(r"°\s+", "°", window)
    for surface, family in _UNIT_SURFACES:
        if compact.startswith(surface) and _boundary(compact, len(surface)):
            return family
    symbolic = _SYMBOLIC_UNIT.match(compact)
    if symbolic and _boundary(compact, symbolic.end()):
        return symbolic.group(0)
    return None


def _unit_family(text: str, match: re.Match[str]) -> str | None:
    if match.start() > 0 and text[match.start() - 1] == "%":
        return "percent"
    return _match_family(_unit_window(text, match.end()))


def _families_for_token(blob: str, token: str) -> set[str]:
    found: set[str] = set()
    for match in _numbers(blob):
        if match.group(0) != token:
            continue
        family = _unit_family(blob, match)
        if family:
            found.add(family)
    return found


def _unit_conflict(claim: str, blob: str) -> bool:
    """Drop a sentence that hangs a different kind of unit on a record number."""
    if not blob:
        return False
    for match in _numbers(claim):
        if _skipped_number(claim, match):
            continue
        family = _unit_family(claim, match)
        if family is None:
            continue
        recorded = _families_for_token(blob, match.group(0))
        if recorded and family not in recorded:
            return True
    return False


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


def _skipped_number(claim: str, match: re.Match[str]) -> bool:
    """ICD digits and a parenthetical reference range are not separate measurements."""
    for code in _ICD.finditer(claim):
        if code.start() <= match.start() and match.end() <= code.end():
            return True
    return any(span.start() <= match.start() and match.end() <= span.end() for span in _REF_PAREN.finditer(claim))


def _codes_supported(claim: str, blob: str) -> bool:
    folded = blob.casefold()
    return all(code.group(0).casefold() in folded for code in _ICD.finditer(claim))


def _numbers_supported(claim: str, blob: str) -> bool:
    if not _codes_supported(claim, blob):
        return False
    for match in _numbers(claim):
        if _skipped_number(claim, match):
            continue
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


_GENERIC_WORDS = frozenset({
    "atif", "bilgi", "bolum", "cumle", "cumleyi", "durum", "gibi", "hasta",
    "kaydi", "kayit", "kayitta", "kayittaki", "olan", "olarak", "oldugu",
    "yaz", "yazili", "yok", "yanit",
})


def content_words(text: str) -> list[str]:
    folded = normalize_text(text)
    return [
        token for token in re.findall(r"[a-z]{5,}", folded)
        if token not in _GENERIC_WORDS
    ]


def _best_line(section_index: int, claim: str, record: TextRecord) -> tuple[int, str]:
    section = record.sections[section_index]
    best_number = section.start_line
    best_text = section.lines[0] if section.lines else section.heading
    best_score = -1
    claim_numbers = [match.group(0) for match in _numbers(claim)]
    words = content_words(claim)
    for line in record.lines:
        if line.section_index != section_index or line.text.startswith("##"):
            continue
        folded = normalize_text(line.text)
        score = sum(3 for token in claim_numbers if token in line.text)
        score += sum(1 for word in words if word in folded)
        if score > best_score:
            best_score = score
            best_number = line.number
            best_text = line.text
    return best_number, best_text


def _section_supporting_numbers(claim: str, record: TextRecord) -> int | None:
    """The section that actually contains the claim's numbers, not the bracket that named one."""
    numbers = _numbers(claim)
    if not numbers and not _RELATIVE.search(claim):
        return None
    words = content_words(claim)
    best: tuple[int, int] | None = None
    for index, section in enumerate(record.sections):
        if not _numbers_supported(claim, section.text):
            continue
        folded = normalize_text(section.text)
        score = sum(1 for match in numbers if match.group(0) in section.text)
        score += sum(2 for word in words if word in folded)
        if best is None or score > best[0]:
            best = (score, index)
    return None if best is None else best[1]


def _overlaps_section(sentence: str, blob: str) -> bool:
    words = content_words(sentence)
    if not words:
        return False
    folded = normalize_text(blob)
    hits = [word for word in words if word in folded]
    return any(len(word) >= 6 for word in hits) or len(hits) >= 2


def _marker_supported(marker: str, record: TextRecord) -> bool:
    if not _numbers(marker) and not _RELATIVE.search(marker):
        return True
    return any(_numbers_supported(marker, section.text) for section in record.sections)


def _marker_text(match: re.Match[str]) -> str:
    return next(group.strip() for group in match.groups() if group)


def _pairs(body: str) -> list[tuple[str, str | None]]:
    pairs: list[tuple[str, str | None]] = []
    cursor = 0
    for match in _MARKER.finditer(body):
        pairs.append((body[cursor:match.start()], _marker_text(match)))
        cursor = match.end()
    tail = body[cursor:]
    if tail.strip() or not pairs:
        pairs.append((tail, None))
    return pairs


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in _SENTENCE.split(text) if part.strip()]


def _section_for_claim(claim: str, record: TextRecord) -> int | None:
    """Attach a section when the model forgot the brackets but the numbers are real."""
    folded = normalize_text(claim)
    by_heading = [
        index for index, section in enumerate(record.sections)
        if section.heading != "Giriş" and any(form in folded for form in _heading_forms(section.heading))
    ]
    if len(by_heading) == 1:
        return by_heading[0]
    numbers = _numbers(claim)
    if not numbers and not _RELATIVE.search(claim):
        return by_heading[0] if by_heading else None
    scored: list[tuple[int, int]] = []
    pool = by_heading or range(len(record.sections))
    for index in pool:
        blob = record.sections[index].text
        if not _numbers_supported(claim, blob):
            continue
        folded = normalize_text(blob)
        score = sum(1 for match in numbers if match.group(0) in blob)
        score += sum(2 for word in content_words(claim) if word in folded)
        if score or not numbers:
            scored.append((score, index))
    if not scored:
        return None
    scored.sort(key=lambda item: (-item[0], item[1]))
    return scored[0][1]


def _section_overlapping(sentence: str, record: TextRecord) -> int | None:
    """A numberless chart clause still belongs to the section that contains its words."""
    words = content_words(sentence)
    if not words:
        return None
    best: tuple[int, int] | None = None
    for index, section in enumerate(record.sections):
        if not _overlaps_section(sentence, section.text):
            continue
        folded = normalize_text(section.text)
        score = sum(2 if len(word) >= 6 else 1 for word in words if word in folded)
        if best is None or score > best[0]:
            best = (score, index)
    return None if best is None else best[1]


def _remember(
    citations: list[dict], seen: set[tuple], record: TextRecord,
    section_index: int, claim: str, marker: str | None,
) -> None:
    section = record.sections[section_index]
    number, quote = _best_line(section_index, claim + " " + (marker or ""), record)
    key = (section.heading, number, quote)
    if key in seen:
        return
    seen.add(key)
    citations.append({
        "section": section.heading,
        "startLine": number,
        "endLine": number,
        "quote": quote,
        "marker": marker or section.heading,
    })


def ground_answer(raw: str, record: TextRecord) -> GroundingResult:
    parsed = _status_and_body(strip_model_wrappers(raw))
    if parsed is None:
        return GroundingResult(False, "validation_failed", "", [], "missing_status")
    status, body = parsed
    if not body:
        return GroundingResult(False, "validation_failed", "", [], "empty_answer")
    citations: list[dict] = []
    seen: set[tuple] = set()
    seen_sentences: list[str] = []
    kept: list[str] = []
    errors: list[str] = []
    saw_unsafe = False
    last_index: int | None = None
    sentence_count = 0
    hide_markers = False
    moved = False
    for claim, marker in _pairs(body):
        section_index: int | None = None
        if marker:
            if not _marker_supported(marker, record):
                hide_markers = True
            else:
                section_index = _resolve_section(marker, claim, record)
                if section_index is not None:
                    last_index = section_index
        pieces = _sentences(claim) or ([claim.strip()] if claim.strip() else [])
        for sentence in pieces:
            sentence_count += 1
            if answer_has_unsafe_advice(sentence):
                saw_unsafe = True
                continue
            index = section_index if section_index is not None else last_index
            if index is None:
                index = _section_for_claim(sentence, record)
            blob = record.sections[index].text if index is not None else ""
            needs_number = bool(_numbers(sentence) or _RELATIVE.search(sentence))
            if needs_number and (index is None or not _numbers_supported(sentence, blob)):
                relocated = _section_supporting_numbers(sentence, record)
                if relocated is None:
                    errors.append("unsupported_number")
                    continue
                if index != relocated:
                    moved = True
                index = relocated
                blob = record.sections[index].text
            elif needs_number and content_words(sentence) and not _overlaps_section(sentence, blob):
                relocated = _section_supporting_numbers(sentence, record)
                if (
                    relocated is not None
                    and _overlaps_section(sentence, record.sections[relocated].text)
                    and relocated != index
                ):
                    moved = True
                    index = relocated
                    blob = record.sections[index].text
            if not _causal_supported(sentence, blob):
                errors.append("unsupported_causal_link")
                continue
            if _unit_conflict(sentence, record.text):
                errors.append("incompatible_unit")
                continue
            if not needs_number and (index is None or not _overlaps_section(sentence, blob)):
                relocated = _section_overlapping(sentence, record)
                if relocated is None:
                    errors.append("unsupported_claim" if index is not None else "missing_citation")
                    continue
                if index != relocated:
                    moved = True
                index = relocated
                blob = record.sections[index].text
            if status == "answered" and index is None:
                errors.append("missing_citation")
                continue
            signature = " ".join(normalize_text(_MARKER.sub(" ", sentence)).split())
            if signature and any(
                fuzz.ratio(signature, prior) >= _NEAR_DUPLICATE for prior in seen_sentences
            ):
                continue
            seen_sentences.append(signature)
            kept.append(sentence)
            if index is not None:
                last_index = index
                _remember(citations, seen, record, index, sentence, marker)
    if not kept:
        if saw_unsafe:
            return GroundingResult(False, "safety_refusal", "", [], "unsafe_advice")
        return GroundingResult(
            False, "validation_failed", "", [], errors[-1] if errors else "missing_citation",
        )
    if status == "answered" and not citations:
        return GroundingResult(False, "validation_failed", "", [], "missing_citation")
    answer = body if len(kept) == sentence_count and not hide_markers and not moved else "\n".join(kept)
    answer = answer.strip()
    if status == "answered" and _absence_only(answer):
        status = "insufficient_evidence"
    return GroundingResult(True, status, answer, citations)


_ABSENCE = (
    "bilinmiyor", "kayitta yok", "kayitta bulunmuyor", "planlandi", "bulunamadi", "bilgi yok",
)


def _absence_only(answer: str) -> bool:
    """A supported answer that only says the fact is missing or still planned."""
    if _numbers(answer) or _ICD.search(answer):
        return False
    folded = normalize_text(answer)
    return any(phrase in folded for phrase in _ABSENCE)
