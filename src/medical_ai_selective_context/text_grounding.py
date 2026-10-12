"""Parse a plain-text answer and keep only numbers the record actually contains.

The model does not emit JSON. The first line is a status token, and citations
are lightweight markers such as [Laboratuvar (hastane)]. Line
numbers are resolved here. A number, date, unit pair, relative time, or causal
phrase that the cited section does not contain drops the answer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

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
_SENTENCE_BREAK = re.compile(r"\n+|(?<=[.!?])\s+")
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
# Causal connectives. Longer phrases win. These are grammar, not chart values.
_CAUSE_MARKERS = (
    "sonucu olarak",
    "bagli olarak",
    "dolayisiyla",
    "bu nedenle",
    "bu yuzden",
    "o yuzden",
    "sebebiyle",
    "nedeniyle",
    "yuzunden",
    "sebep oldu",
    "neden oldu",
    "cunku",
    "dolayi",
    "zira",
)
_CAUSE_STOP = frozenset({
    "olan", "olarak", "sonra", "once", "daha", "cok", "gibi", "diye", "yeni",
    "icin", "ayrica", "fakat", "ama", "veya", "hem", "degil",
})
# Rounding slack for a sum or difference, not a clinical limit.
_DERIVED_ABS = 0.51
_DERIVED_REL = 0.01
# Nearest recorded amount for a from-to claim. Off by one still counts as that amount.
_SNAP_ABS = 1.0
_SNAP_REL = 0.01
# A dated chart event is not a cause. These are record-keeping words, not clinical facts.
_EVENT_WORDS = frozenset({
    "belge", "duzenlendi", "duzenlenmis", "epikriz", "kaydedildi", "kaydedilmis",
    "kaydi", "kayit", "kayitta", "rapor", "raporu", "recete", "recetesi", "receteye",
    "tarih", "tarihi", "tarihinde", "tarihte", "verildi", "verilmis", "verilmistir",
    "yazildi", "yazilmis",
})
# Finite endings, longest first. A past ending counts only after a consonant,
# so a noun plus an accusative vowel is not a verb.
_FINITE_SUFFIXES = (
    "diniz", "dunuz", "tiniz", "tunuz",
    "yordu", "yormus", "mekte", "makta",
    "mistir", "mustur", "ecektir", "acaktir",
    "ecek", "acak", "yor", "mis", "mus",
    "dim", "din", "dik", "dum", "dun", "duk",
    "tim", "tin", "tik", "tum", "tun", "tuk",
    "di", "du", "ti", "tu",
)
_PAST_SUFFIXES = frozenset({
    "di", "du", "ti", "tu", "dim", "din", "dik", "dum", "dun", "duk",
    "tim", "tin", "tik", "tum", "tun", "tuk", "diniz", "dunuz", "tiniz", "tunuz",
})
_BASELINE_WORDS = ("onceki", "olagan", "bazal", "baslangic", "eski")
_CURRENT_WORDS = ("guncel", "simdiki", "bugunku", "mevcut", "suanki")
_UP_DOWN = (
    ("arttı", "azaldı"),
    ("artmış", "azalmış"),
    ("artıyor", "azalıyor"),
    ("artış", "azalış"),
    ("artarak", "azalarak"),
    ("yükseldi", "düştü"),
    ("yükselmiş", "düşmüş"),
    ("yükseliyor", "düşüyor"),
    ("yükselme", "düşme"),
    ("yükselerek", "düşerek"),
    ("çıktı", "indi"),
    ("çıkmış", "inmiş"),
    ("çıkıyor", "iniyor"),
)
_DATE_TOKEN = re.compile(r"\d{1,2}\.\d{1,2}\.\d{4}|\d{4}-\d{2}-\d{2}")
# Unit, then an optional apostrophe and a Turkish case ending (ms'den, ms'ye, ms'sinden).
_APOSTROPHE = r"['’‘ʼ´]"
_CASE_BUFFER = r"(?:s[ıiuü]n|n|y)?"
_FROM_CASE = rf"\s*{_APOSTROPHE}?\s*{_CASE_BUFFER}(?:dan|den|tan|ten)"
_TO_CASE = rf"\s*{_APOSTROPHE}?\s*{_CASE_BUFFER}(?:ya|ye|na|ne|[ae])"
_UNIT_TOKEN = r"(?:\s*[A-Za-zµμ°/%][\wµμ°/%^]*)?"
_TREND_FROM_TO = re.compile(
    rf"(?P<a>\d+(?:[.,]\d+)?)"
    rf"(?P<ua>{_UNIT_TOKEN})"
    rf"(?P<from>{_FROM_CASE})\s+"
    rf"(?P<b>\d+(?:[.,]\d+)?)"
    rf"(?P<ub>{_UNIT_TOKEN})"
    rf"(?P<to>{_TO_CASE})\s+"
    rf"(?P<verb>[A-Za-zÇĞİÖŞÜçğıöşü]+)",
    re.IGNORECASE,
)
_TREND_ARROW = re.compile(
    r"(?P<a>\d+(?:[.,]\d+)?)"
    r"(?P<ua>(?:\s*[A-Za-zµμ°/%][\wµμ°/%^]*)?)"
    r"\s*(?:→|->)\s*"
    r"(?P<b>\d+(?:[.,]\d+)?)"
    r"(?P<ub>(?:\s*[A-Za-zµμ°/%][\wµμ°/%^]*)?)",
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
    debug: list[dict] = field(default_factory=list)


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
_UNIT_CUT = re.compile(r"[\(\[;\n'’‘ʼ´]|\.(?!\d)")
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


def _parse_quantity(token: str) -> float | None:
    """A measured amount. Dates, clock times, and day.month fragments are not amounts."""
    if re.search(r"[:T]|^\d{4}-|^\d{1,2}\.\d{1,2}\.\d{4}", token):
        return None
    if re.fullmatch(r"\d{1,2}\.\d{1,2}|\d{1,2}\.\d{4}", token):
        return None
    if re.fullmatch(r"\d{1,3}(?:\.\d{3})+,\d+", token):
        return float(token.replace(".", "").replace(",", "."))
    if re.fullmatch(r"\d+,\d+", token):
        return float(token.replace(",", "."))
    if re.fullmatch(r"\d+", token):
        return float(token)
    return None


def _near_value(left: float, right: float) -> bool:
    scale = max(abs(left), abs(right), 1.0)
    return abs(left - right) <= max(_DERIVED_ABS, scale * _DERIVED_REL)


def _quantity_points(text: str) -> list[tuple[float, str | None]]:
    points: list[tuple[float, str | None]] = []
    for match in _numbers(text):
        if _skipped_number(text, match):
            continue
        value = _parse_quantity(match.group(0))
        if value is None:
            continue
        points.append((value, _unit_family(text, match)))
    return points


def _families_agree(answer: str | None, left: str | None, right: str | None) -> bool:
    if left and right and left != right:
        return False
    shared = left or right
    if answer is None:
        return True
    if left not in {None, answer} or right not in {None, answer}:
        return False
    if shared is None:
        return False
    return shared == answer


def _derived_quantity(claim: str, match: re.Match[str], blob: str) -> bool:
    """True when the amount is the sum or difference of two amounts in the cited text."""
    value = _parse_quantity(match.group(0))
    if value is None or not blob:
        return False
    family = _unit_family(claim, match)
    points = _quantity_points(blob)
    for index, (left, left_family) in enumerate(points):
        for right, right_family in points[index + 1:]:
            if not _families_agree(family, left_family, right_family):
                continue
            if _near_value(value, left + right):
                return True
            if _near_value(left, right):
                continue
            if _near_value(value, abs(left - right)):
                return True
    return False


def _key_tokens(text: str) -> list[str]:
    folded = normalize_text(text)
    return [
        token for token in re.findall(r"[a-z]{4,}", folded)
        if token not in _CAUSE_STOP and token not in _GENERIC_WORDS
    ]


def _prefix_lengths(text: str) -> list[int]:
    return [len(normalize_text(text[:index])) for index in range(len(text) + 1)]


def _origin_at(lengths: list[int], folded_pos: int) -> int:
    for index, length in enumerate(lengths):
        if length >= folded_pos:
            return index
    return len(lengths) - 1


def _cause_split(sentence: str) -> tuple[str, str, str] | None:
    folded = normalize_text(sentence)
    found: tuple[int, int] | None = None
    for marker in _CAUSE_MARKERS:
        start = folded.find(marker)
        while start >= 0:
            end = start + len(marker)
            before = start == 0 or not folded[start - 1].isalnum()
            after = end == len(folded) or not folded[end].isalnum()
            if before and after and (found is None or start < found[0] or (
                start == found[0] and end - start > found[1] - found[0]
            )):
                found = (start, end)
            start = folded.find(marker, start + 1)
    if found is None:
        return None
    lengths = _prefix_lengths(sentence)
    left = sentence[:_origin_at(lengths, found[0])].strip()
    right = sentence[_origin_at(lengths, found[1]):].strip()
    return left, sentence[_origin_at(lengths, found[0]):_origin_at(lengths, found[1])], right


def _finite_word(token: str) -> bool:
    for suffix in _FINITE_SUFFIXES:
        if not token.endswith(suffix):
            continue
        stem_len = len(token) - len(suffix)
        if stem_len < 2:
            continue
        if suffix in _PAST_SUFFIXES and token[stem_len - 1] in "aeiou":
            continue
        return True
    return False


def _can_stand_alone(text: str) -> bool:
    """A shortened clause needs a measurement or a finite verb, otherwise it is a fragment."""
    if any(_parse_quantity(match.group(0)) is not None for match in _numbers(text)):
        return True
    folded = normalize_text(text)
    return any(_finite_word(token) for token in re.findall(r"[a-z]{4,}", folded))


# These connectives put the cause after the marker. The others put it before.
_CAUSE_AFTER = frozenset({"cunku", "zira"})


def _cause_content(text: str) -> list[str]:
    return [token for token in _key_tokens(text) if token not in _EVENT_WORDS]


def _forms_match(left: str, right: str) -> bool:
    """Stem or a short inflection of the same word. A one-edit typo still matches."""
    if not left or not right:
        return False
    if left == right:
        return True
    short, long = (left, right) if len(left) <= len(right) else (right, left)
    if len(short) >= 4 and long.startswith(short) and len(long) - len(short) <= 4:
        return True
    shared = 0
    for char_a, char_b in zip(left, right, strict=False):
        if char_a != char_b:
            break
        shared += 1
    if shared >= 4 and len(left) - shared <= 4 and len(right) - shared <= 4:
        return True
    return (
        len(short) >= 6
        and abs(len(left) - len(right)) <= 2
        and fuzz.ratio(left, right) >= 90
    )


def _line_words(line: str) -> list[str]:
    return re.findall(r"[a-z]{3,}", normalize_text(line))


def _line_has_token(line: str, token: str) -> bool:
    return any(_forms_match(token, word) for word in _line_words(line))


def _cause_clause(left: str, marker: str, right: str) -> tuple[str, str]:
    """Return the cause text and the other clause. Marker direction is grammar."""
    key = normalize_text(marker).strip(" .,:;\"'")
    if key in _CAUSE_AFTER or any(key.startswith(item) for item in _CAUSE_AFTER):
        return right, left
    return left, right


def _cause_on_item_line(cause: str, other: str, blob: str) -> bool:
    """Cause tokens must match one line that also carries the sentence's item."""
    cause_tokens = _cause_content(cause)
    if not cause_tokens or not blob:
        return False
    lines = [line for line in blob.splitlines() if line.strip()] or [blob]
    other_tokens = _cause_content(other)
    other_numbers = [
        match.group(0) for match in _numbers(other)
        if _parse_quantity(match.group(0)) is not None
    ]

    def covers(line: str) -> bool:
        return all(_line_has_token(line, token) for token in cause_tokens)

    anchored = [
        line for line in lines
        if (other_tokens and any(_line_has_token(line, token) for token in other_tokens))
        or (other_numbers and any(number in line for number in other_numbers))
    ]
    if anchored:
        return any(covers(line) for line in anchored)
    # The item name sits in the cause clause. Use the line of its rarest token.
    ranked: list[tuple[int, int, list[str]]] = []
    for token in cause_tokens:
        matched = [line for line in lines if _line_has_token(line, token)]
        if matched:
            ranked.append((len(matched), -len(token), matched))
    if not ranked:
        return False
    ranked.sort()
    return any(covers(line) for line in ranked[0][2])


def _measured_amount(side: str, blob: str) -> bool:
    """A unit-bearing amount can support a factual side. A bare date cannot support a cause."""
    for match in _numbers(side):
        if _skipped_number(side, match) or _parse_quantity(match.group(0)) is None:
            continue
        if _unit_family(side, match) is None:
            continue
        if _token_in_text(match.group(0), blob) or _derived_quantity(side, match, blob):
            return True
    return False


def _side_supported(side: str, blob: str) -> bool:
    if not side.strip() or not blob:
        return False
    folded = normalize_text(blob)
    if any(token in folded for token in _cause_content(side)):
        return True
    return _measured_amount(side, blob)


def _without_unsupported_cause(sentence: str, blob: str) -> str | None:
    """Drop a cause clause whose words are not on the item's line.

    Stemming and a short inflection count. A date or a record-keeping event does
    not. When the cause matches, the whole sentence stays. When it does not, the
    other clause stays only if it is still a measurement or a finite clause.
    """
    parts = _cause_split(sentence)
    if parts is None:
        return sentence
    left, marker, right = parts
    cause, other = _cause_clause(left, marker, right)
    if _cause_on_item_line(cause, other, blob):
        return sentence
    # The cause did not match the item line. Keep a side that still stands and
    # is supported: the result clause first, then the cause clause when the
    # result is only an unsupported fragment.
    for candidate in (other, cause):
        kept = candidate.strip(" ,;:-")
        if not kept or not _can_stand_alone(kept) or not _clause_supported(kept, blob):
            continue
        if kept[-1] not in ".!?":
            kept += "."
        return kept
    return None


def _clause_supported(text: str, blob: str) -> bool:
    """A leftover clause needs a supported amount or a word the record actually uses."""
    if any(_parse_quantity(match.group(0)) is not None for match in _numbers(text)):
        return _unsupported_token(text, blob) == ""
    blob_tokens = set(re.findall(r"[a-z]{4,}", normalize_text(blob)))
    tokens = re.findall(r"[a-z]{4,}", normalize_text(text))
    return any(token in blob_tokens for token in tokens)


def _blank_sentence(text: str) -> bool:
    """A period or other punctuation with no word and no number is not a claim."""
    return re.search(r"[0-9A-Za-zÇĞİÖŞÜçğıöşü]", text) is None


def _parse_date(token: str) -> tuple[int, int, int] | None:
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", token):
        year, month, day = (int(part) for part in token.split("-"))
        return year, month, day
    if re.fullmatch(r"\d{1,2}\.\d{1,2}\.\d{4}", token):
        day, month, year = (int(part) for part in token.split("."))
        return year, month, day
    return None


def _date_near(blob: str, start: int) -> tuple[int, int, int] | None:
    window_start = max(0, start - 40)
    window = blob[window_start:start + 60]
    best: tuple[int, tuple[int, int, int]] | None = None
    for match in _DATE_TOKEN.finditer(window):
        parsed = _parse_date(match.group(0))
        if parsed is None:
            continue
        distance = abs((window_start + match.start()) - start)
        if best is None or distance < best[0]:
            best = (distance, parsed)
    return None if best is None else best[1]


_CLAUSE_SPLIT = re.compile(r"[,;\n]+|(?<=[.!?])\s+")


def _clause_bounds(blob: str, start: int) -> tuple[int, int]:
    """A comma, semicolon, or sentence break separates one labeled amount from the next."""
    left = 0
    for match in _CLAUSE_SPLIT.finditer(blob):
        if match.end() <= start:
            left = match.end()
            continue
        if match.start() >= start:
            return left, match.start()
        left = match.end()
    return left, len(blob)


def _role_hits(folded: str) -> list[tuple[int, str]]:
    hits: list[tuple[int, str]] = []
    for role, words in (("baseline", _BASELINE_WORDS), ("current", _CURRENT_WORDS)):
        for word in words:
            cursor = 0
            while True:
                found = folded.find(word, cursor)
                if found < 0:
                    break
                if found == 0 or not folded[found - 1].isalpha():
                    hits.append((found, role))
                cursor = found + len(word)
    return hits


def _nearest_role(blob: str, start: int) -> str | None:
    """Usual versus current label of this amount, inside its own clause.

    A label binds to the nearest amount in that clause. When two amounts are
    equally close, the one after the label wins, which is how a modifier is written.
    """
    left, right = _clause_bounds(blob, start)
    clause = blob[left:right]
    folded = normalize_text(clause)
    anchor = len(normalize_text(clause[: start - left]))
    amounts = [
        len(normalize_text(clause[: match.start()]))
        for match in _numbers(clause)
        if _parse_quantity(match.group(0)) is not None
    ]
    labels = _role_hits(folded)
    if not labels or not amounts:
        return None
    this_amount = min(amounts, key=lambda pos: (abs(pos - anchor), pos))

    def closest_amount(label_at: int) -> int:
        def rank(amount_at: int) -> tuple[int, int, int]:
            follows = 0 if amount_at >= label_at else 1
            return abs(amount_at - label_at), follows, amount_at
        return min(amounts, key=rank)

    claimed = [item for item in labels if closest_amount(item[0]) == this_amount]
    if not claimed:
        return None

    def rank_label(item: tuple[int, str]) -> tuple[int, int]:
        precedes = 0 if item[0] <= anchor else 1
        return abs(item[0] - anchor), precedes

    return min(claimed, key=rank_label)[1]


# "değer", "sonuç" and case crumbs are not the measurement's name.
_MEASURE_FILLERS = frozenset({
    "deger", "degeri", "degerin", "degerinin", "degerini", "degerinde",
    "seviye", "seviyesi", "seviyesinde", "sonuc", "sonucu", "sonucunda",
    "olcum", "olcumu", "olcumunde", "duzey", "duzeyi", "duzeyinde",
    "miktar", "miktari", "ortalama", "ortalamasi", "ortalamasinin",
})
_NAME_CRUMBS = frozenset({
    "nin", "nun", "dan", "den", "tan", "ten", "ya", "ye", "na", "ne",
    "icin", "gore", "kadar", "olan", "sonra", "once",
})


def _name_skip() -> frozenset[str]:
    skip = _CAUSE_STOP | _GENERIC_WORDS | _MEASURE_FILLERS | _NAME_CRUMBS
    for up, down in _UP_DOWN:
        skip = skip | {normalize_text(up), normalize_text(down)}
    return skip


def _measurement_words(sentence: str) -> list[str]:
    skip = _name_skip()
    folded = normalize_text(sentence)
    return [
        token for token in re.findall(r"[a-z]{3,}", folded)
        if token not in skip
    ]


def _trend_names(sentence: str, start: int) -> list[str]:
    """Name tokens a few words before the first amount. Fillers such as değeri drop out."""
    prefix = sentence[:start]
    tail = re.split(r"[.!?;\n]+", prefix)[-1]
    tokens = [
        token for token in re.findall(r"[a-z]{3,}", normalize_text(tail))
        if token not in _name_skip()
    ]
    return tokens[-3:]


def _keeps_period(blob: str, index: int) -> bool:
    """A period inside a date, a decimal, or an ordinal is not a sentence end."""
    if index <= 0 or not blob[index - 1].isdigit():
        return False
    stripped = blob[index + 1:].lstrip()
    if not stripped:
        return False
    return stripped[0].isdigit() or stripped[0].islower()


def _sentence_bounds(blob: str, start: int) -> tuple[int, int]:
    left = 0
    for match in re.finditer(r"[.!?](?:\s+|$)", blob):
        if blob[match.start()] == "." and _keeps_period(blob, match.start()):
            continue
        if match.end() <= start:
            left = match.end()
            continue
        if match.start() >= start:
            return left, match.start()
    return left, len(blob)


def _context_tokens(blob: str, start: int) -> set[str]:
    """Tokens of this sentence plus the section heading. A later sentence is not included."""
    left, right = _sentence_bounds(blob, start)
    chunks = [blob[left:right], blob.split("\n", 1)[0]]
    tokens: set[str] = set()
    for chunk in chunks:
        tokens.update(re.findall(r"[a-z0-9]{2,}", normalize_text(chunk)))
    return tokens


def _token_matches(name: str, tokens: set[str]) -> bool:
    if name in tokens:
        return True
    return any(_forms_match(name, token) for token in tokens)


def _words_near_span(blob: str, start: int, end: int, words: list[str]) -> bool:
    if not words:
        return True
    window = normalize_text(blob[max(0, start - 80):end + 80])
    tokens = set(re.findall(r"[a-z0-9]{2,}", window))
    return any(_token_matches(word, tokens) for word in words)


def _within(left: float, right: float, abs_limit: float, rel_limit: float) -> bool:
    scale = max(abs(left), abs(right), 1.0)
    return abs(left - right) <= max(abs_limit, scale * rel_limit)


def _snapped_hits(
    blob: str, value: float, family: str | None, words: list[str],
    names: list[str] | None = None,
) -> list[dict]:
    """Recorded amounts within a small tolerance, keeping only the nearest ones.

    When the claim names a measurement, only amounts in that measurement's
    sentence are eligible. An exact number in a later sentence does not win.
    """
    pool: list[dict] = []
    for match in _numbers(blob):
        parsed = _parse_quantity(match.group(0))
        if parsed is None or not _within(parsed, value, _SNAP_ABS, _SNAP_REL):
            continue
        other = _unit_family(blob, match)
        if family and other and family != other:
            continue
        pool.append({
            "value": parsed,
            "start": match.start(),
            "end": match.end(),
            "role": _nearest_role(blob, match.start()),
            "date": _date_near(blob, match.start()),
            "gap": abs(parsed - value),
        })
    if not pool:
        return []
    if names:
        chosen = [
            hit for hit in pool
            if any(_token_matches(name, _context_tokens(blob, hit["start"])) for name in names)
        ]
    else:
        near = [
            hit for hit in pool
            if _words_near_span(blob, hit["start"], hit["end"], words)
        ]
        chosen = near or pool
    if not chosen:
        return []
    closest = min(hit["gap"] for hit in chosen)
    return [hit for hit in chosen if hit["gap"] == closest]


def _ordered_endpoints(left: list[dict], right: list[dict]) -> tuple[float, float] | None:
    dated: list[tuple[dict, dict]] = []
    roled: list[tuple[dict, dict]] = []
    for first in left:
        for second in right:
            if first["start"] == second["start"]:
                continue
            if first["date"] and second["date"] and first["date"] != second["date"]:
                dated.append((first, second))
            elif (
                first["role"] and second["role"] and first["role"] != second["role"]
                and {first["role"], second["role"]} == {"baseline", "current"}
            ):
                roled.append((first, second))
    pair = dated[0] if dated else (roled[0] if roled else None)
    if pair is None:
        return None
    first, second = pair
    if first["date"] and second["date"]:
        earlier, later = (first, second) if first["date"] < second["date"] else (second, first)
        return earlier["value"], later["value"]
    earlier = first if first["role"] == "baseline" else second
    later = second if earlier is first else first
    return earlier["value"], later["value"]


def _verb_direction(verb: str) -> tuple[int, str] | None:
    folded = normalize_text(verb)
    for up, down in _UP_DOWN:
        if folded == normalize_text(up):
            return 1, down
        if folded == normalize_text(down):
            return -1, up
    return None


def _replace_spans(text: str, replacements: list[tuple[int, int, str]]) -> str:
    updated = text
    for start, end, new in sorted(replacements, key=lambda item: item[0], reverse=True):
        updated = updated[:start] + new + updated[end:]
    return updated


def _trend_match(sentence: str) -> re.Match[str] | None:
    match = _TREND_FROM_TO.search(sentence)
    if match and (match.start() == 0 or sentence[match.start() - 1] not in "0123456789."):
        return match
    arrow = _TREND_ARROW.search(sentence)
    if arrow and (arrow.start() == 0 or sentence[arrow.start() - 1] not in "0123456789."):
        return arrow
    return None


@dataclass
class _TrendResult:
    action: str
    text: str = ""
    issues: list[dict] = field(default_factory=list)


def _plain_number(value: float) -> str:
    rounded = round(value)
    if abs(value - rounded) <= 1e-9:
        return str(int(rounded))
    return f"{value:.4f}".rstrip("0").rstrip(".")


def _issue(check: str, sentence: str, number: str, nearest: str = "", detail: str = "") -> dict:
    return {
        "check": check,
        "number": number,
        "nearest": nearest,
        "detail": detail,
        "sentence": " ".join(sentence.split()),
    }


def _assess_trend(sentence: str, blob: str) -> _TrendResult:
    """Match a from-to claim to dated or usual-versus-current order.

    An off-by-a-little amount maps to the nearest recorded value and is a wrong
    value, not a corrected answer. Exact endpoints with a reversed direction are
    reordered.
    """
    match = _trend_match(sentence)
    if match is None or not blob:
        return _TrendResult("skip", sentence)
    start_value = _parse_quantity(match.group("a"))
    end_value = _parse_quantity(match.group("b"))
    if start_value is None or end_value is None:
        return _TrendResult("skip", sentence)
    words = _measurement_words(sentence)
    names = _trend_names(sentence, match.start())
    start_span = match.span("a")
    end_span = match.span("b")
    start_family = _match_family(_unit_window(sentence, start_span[1]))
    end_family = _match_family(_unit_window(sentence, end_span[1]))
    shared = start_family or end_family
    if start_family and end_family and start_family != end_family:
        return _TrendResult("fail", issues=[_issue(
            "unsupported_trend", sentence, match.group("a"), detail="unit_family",
        )])
    left = _snapped_hits(blob, start_value, shared, words, names)
    right = _snapped_hits(blob, end_value, shared, words, names)
    if not left or not right:
        missing = match.group("a") if not left else match.group("b")
        return _TrendResult("fail", issues=[_issue(
            "unsupported_trend", sentence, missing, detail="missing_endpoint",
        )])
    start_recorded = left[0]["value"]
    end_recorded = right[0]["value"]
    ordered = _ordered_endpoints(left, right)
    reversed_detail = ""
    exact = _near_value(start_value, start_recorded) and _near_value(end_value, end_recorded)
    if ordered is not None:
        earlier, later = ordered
        separated = not _near_value(earlier, later)
        true_up = later > earlier and separated
        forward = _within(start_value, earlier, _SNAP_ABS, _SNAP_REL) and _within(
            end_value, later, _SNAP_ABS, _SNAP_REL,
        )
        backward = _within(start_value, later, _SNAP_ABS, _SNAP_REL) and _within(
            end_value, earlier, _SNAP_ABS, _SNAP_REL,
        )
        verb = match.groupdict().get("verb")
        direction = _verb_direction(verb) if verb else None
        stated = None if direction is None else direction[0]
        true_dir = 1 if true_up else -1
        verb_wrong = stated is not None and stated != true_dir and separated and (forward or backward)
        if separated and (backward or verb_wrong):
            reversed_detail = "reversed_direction"
        if exact and separated and (forward or backward):
            replacements: list[tuple[int, int, str]] = []
            if backward:
                replacements.append((start_span[0], start_span[1], match.group("b")))
                replacements.append((end_span[0], end_span[1], match.group("a")))
            if verb and direction is not None and stated is not None and stated != true_dir:
                replacements.append((match.start("verb"), match.end("verb"), direction[1]))
            if replacements:
                return _TrendResult("rewrite", _replace_spans(sentence, replacements))
        if exact and separated and not forward and not backward:
            return _TrendResult("fail", issues=[_issue(
                "unsupported_trend", sentence, match.group("a"), detail="unordered",
            )])
    inexact: list[tuple[str, float]] = []
    if not _near_value(start_value, start_recorded):
        inexact.append((match.group("a"), start_recorded))
    if not _near_value(end_value, end_recorded):
        inexact.append((match.group("b"), end_recorded))
    if inexact:
        return _TrendResult("fail", issues=[
            _issue("wrong_value", sentence, token, _plain_number(recorded), reversed_detail)
            for token, recorded in inexact
        ])
    return _TrendResult("keep", sentence)


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


def _line_span(blob: str, start: int) -> tuple[int, int]:
    left = blob.rfind("\n", 0, start) + 1
    right = blob.find("\n", start)
    if right < 0:
        right = len(blob)
    return left, right


def _family_at(blob: str, match: re.Match[str]) -> str | None:
    """The unit on this amount, or the last explicit unit earlier on the same line."""
    explicit = _unit_family(blob, match)
    if explicit:
        return explicit
    left, right = _line_span(blob, match.start())
    line = blob[left:right]
    local = match.start() - left
    family = None
    for other in _numbers(line):
        if other.start() >= local:
            break
        found = _unit_family(line, other)
        if found:
            family = found
    return family


def _unit_attached(claim: str, match: re.Match[str], blob: str) -> bool:
    family = _unit_family(claim, match)
    if family is None:
        return True
    for other in _numbers(blob):
        if other.group(0) != match.group(0):
            continue
        if _family_at(blob, other) == family:
            return True
    return False


def _unsupported_token(claim: str, blob: str) -> str:
    """The first number or date the cited text does not support. Empty when all of them do."""
    if not _codes_supported(claim, blob):
        code = next(_ICD.finditer(claim), None)
        return "" if code is None else code.group(0)
    for match in _numbers(claim):
        if _skipped_number(claim, match):
            continue
        token = match.group(0)
        if _token_in_text(token, blob) and _unit_attached(claim, match, blob):
            prefixed = match.start() > 0 and claim[match.start() - 1] == "%"
            if prefixed and f"%{token}" not in blob and f"% {token}" not in blob:
                return token
            suffixed = match.end() < len(claim) and claim[match.end()] == "%"
            if suffixed and f"{token}%" not in blob and f"%{token}" not in blob:
                return token
            continue
        if _derived_quantity(claim, match, blob):
            continue
        return token

    def squash(value: str) -> str:
        return re.sub(r"\s+", " ", value.casefold())

    squashed = squash(blob)
    for match in _RELATIVE.finditer(claim):
        if squash(match.group(0)) not in squashed:
            return match.group(0)
    return ""


def _numbers_supported(claim: str, blob: str) -> bool:
    return _unsupported_token(claim, blob) == ""


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


def _ordinal_period(text: str, match: re.Match[str]) -> bool:
    """A digit, a period, and a lowercase word is an ordinal, not a sentence end."""
    if match.start() == 0 or text[match.start() - 1] != ".":
        return False
    if match.start() < 2 or not text[match.start() - 2].isdigit():
        return False
    after = text[match.end():]
    return bool(after[:1].islower())


def _sentences(text: str) -> list[str]:
    parts: list[str] = []
    start = 0
    for match in _SENTENCE_BREAK.finditer(text):
        if _ordinal_period(text, match):
            continue
        chunk = text[start:match.start()].strip()
        if chunk:
            parts.append(chunk)
        start = match.end()
    tail = text[start:].strip()
    if tail:
        parts.append(tail)
    return parts


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


def _cite(marker: str | None, record: TextRecord, index: int | None) -> str:
    if marker:
        return marker
    if index is not None:
        return record.sections[index].heading
    return ""


def _failure(
    errors: list[str], debug: list[dict], check: str, sentence: str,
    citation: str, number: str = "", nearest: str = "", detail: str = "",
) -> None:
    errors.append(check)
    debug.append({
        "check": check,
        "number": number,
        "nearest": nearest,
        "citation": citation,
        "sentence": " ".join(sentence.split()),
        "detail": detail,
    })


def _take_issues(
    errors: list[str], debug: list[dict], issues: list[dict], citation: str,
) -> None:
    for item in issues:
        _failure(
            errors, debug, item["check"], item["sentence"], citation,
            item.get("number", ""), item.get("nearest", ""), item.get("detail", ""),
        )


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
    debug: list[dict] = []
    saw_unsafe = False
    last_index: int | None = None
    sentence_count = 0
    hide_markers = False
    moved = False
    edited = False
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
            if _blank_sentence(sentence):
                edited = True
                continue
            sentence_count += 1
            if answer_has_unsafe_advice(sentence):
                saw_unsafe = True
                continue
            index = section_index if section_index is not None else last_index
            if index is None:
                index = _section_for_claim(sentence, record)
            blob = record.sections[index].text if index is not None else ""
            needs_number = bool(_numbers(sentence) or _RELATIVE.search(sentence))
            trend = _assess_trend(sentence, blob)
            if needs_number and (index is None or not _numbers_supported(sentence, blob)):
                if trend.action == "fail" and any(
                    item["check"] == "wrong_value" for item in trend.issues
                ):
                    _take_issues(errors, debug, trend.issues, _cite(marker, record, index))
                    continue
                softened = _without_unsupported_cause(sentence, blob) if blob else None
                if (
                    softened
                    and softened != sentence
                    and _numbers_supported(softened, blob)
                ):
                    sentence = softened
                    edited = True
                    trend = _assess_trend(sentence, blob)
                else:
                    relocated = _section_supporting_numbers(sentence, record)
                    if relocated is None and softened and softened != sentence:
                        relocated = _section_supporting_numbers(softened, record)
                        if relocated is not None:
                            sentence = softened
                            edited = True
                    if relocated is None:
                        if trend.action == "fail":
                            _take_issues(errors, debug, trend.issues, _cite(marker, record, index))
                        else:
                            _failure(
                                errors, debug, "unsupported_number", sentence,
                                _cite(marker, record, index),
                                _unsupported_token(sentence, blob),
                            )
                        continue
                    if index != relocated:
                        moved = True
                    index = relocated
                    blob = record.sections[index].text
                    trend = _assess_trend(sentence, blob)
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
                    trend = _assess_trend(sentence, blob)
            revised = _without_unsupported_cause(sentence, blob)
            if revised is None:
                _failure(
                    errors, debug, "unsupported_causal_link", sentence,
                    _cite(marker, record, index),
                )
                continue
            if revised != sentence:
                sentence = revised
                edited = True
                needs_number = bool(_numbers(sentence) or _RELATIVE.search(sentence))
                trend = _assess_trend(sentence, blob)
            if trend.action == "fail":
                _take_issues(errors, debug, trend.issues, _cite(marker, record, index))
                continue
            if trend.action == "rewrite":
                sentence = trend.text
                edited = True
            if _unit_conflict(sentence, record.text):
                _failure(
                    errors, debug, "incompatible_unit", sentence, _cite(marker, record, index),
                )
                continue
            if (
                not needs_number
                and not _states_absence(sentence)
                and (index is None or not _overlaps_section(sentence, blob))
            ):
                relocated = _section_overlapping(sentence, record)
                if relocated is None:
                    _failure(
                        errors, debug,
                        "unsupported_claim" if index is not None else "missing_citation",
                        sentence, _cite(marker, record, index),
                    )
                    continue
                if index != relocated:
                    moved = True
                index = relocated
                blob = record.sections[index].text
            if status == "answered" and index is None and not _states_absence(sentence):
                _failure(
                    errors, debug, "missing_citation", sentence, _cite(marker, record, index),
                )
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
            False, "validation_failed", "", [], errors[-1] if errors else "missing_citation", debug,
        )
    if status == "answered" and not citations:
        return GroundingResult(
            False, "validation_failed", "", [], "missing_citation",
            debug or [{
                "check": "missing_citation", "number": "", "nearest": "",
                "citation": "", "sentence": "", "detail": "",
            }],
        )
    answer = (
        body if len(kept) == sentence_count and not hide_markers and not moved and not edited
        else "\n".join(kept)
    )
    answer = answer.strip()
    if status == "answered" and _absence_only(answer):
        status = "insufficient_evidence"
    return GroundingResult(True, status, answer, citations)


_ABSENCE_PHRASES = (
    "bilinmiyor", "bilgi yok", "planlandi", "bulunmuyor", "bulunmamakta",
    "bulunamadi", "bulunmamaktadir", "gorulmuyor", "gorulmemekte", "gecmiyor",
    "yer almiyor", "soz edilmiyor", "kayitli degil", "mevcut degil",
    "kayitta yok", "kayitta bulunmuyor",
)
_ABSENCE_TOKENS = frozenset({"yok", "yoktur"})


def _states_absence(text: str) -> bool:
    """An absence sentence with no amount, date, or code adds no new chart fact."""
    if _numbers(text) or _DATE_TOKEN.search(text) or _ICD.search(text) or _RELATIVE.search(text):
        return False
    folded = normalize_text(text)
    if any(phrase in folded for phrase in _ABSENCE_PHRASES):
        return True
    tokens = set(re.findall(r"[a-z]+", folded))
    return bool(tokens & _ABSENCE_TOKENS)


def _absence_only(answer: str) -> bool:
    """A supported answer that only says the fact is missing or still planned."""
    return _states_absence(answer)
