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
    r"|\d{1,2}\.\d{4}"
    r"|\d{1,2}\.\d{1,2}"
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
# A repeated dose such as N x N. The letters are the schedule mark, not a drug name.
_SCHEDULE = re.compile(r"\d+(?:\s*[x×]\s*\d+)+", re.IGNORECASE)
_BULLET = re.compile(r"^(?:[-*•])\s+\S")
_SOFT_DROPS = frozenset({"unsupported_claim", "missing_citation"})
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
# Any other DURUM label is still a status line. The body is what gets checked.
_ANY_STATUS = re.compile(r"^durum:\s*\S+")


def _status_token(folded: str) -> str | None:
    """The DURUM prefix, including a label this server does not name."""
    for token, _status in _STATUS_PREFIXES:
        if folded == token or folded.startswith((token + " ", token + ".", token + ":")):
            return token
    match = _ANY_STATUS.match(folded)
    if match is None:
        return None
    return match.group(0)


def _status_of_line(line: str) -> str | None:
    folded = normalize_text(line).strip(" *_`#>")
    token = _status_token(folded)
    if token is None:
        return None
    for known, status in _STATUS_PREFIXES:
        if token == known:
            return status
    return "answered"


def _line_is_only_status(line: str) -> bool:
    folded = normalize_text(line).strip(" *_`#>")
    token = _status_token(folded)
    if token is None:
        return False
    return not folded[len(token):].strip(" :.-")


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
        body_lines = ([tail] if tail else []) + lines[index + 1 :]
        while body_lines and not body_lines[0].strip():
            body_lines.pop(0)
        while body_lines and not body_lines[-1].strip():
            body_lines.pop()
        return status or "answered", "\n".join(body_lines).strip()
    inferred = _infer_status(text)
    if inferred is None:
        return None
    return inferred, text.strip()


def _tail_after_status(line: str) -> str:
    """Keep the answer when a small model puts it on the DURUM line."""
    cleaned = line.strip().strip("*_`")
    folded = normalize_text(cleaned)
    token = _status_token(folded)
    if token is None:
        return ""
    # The folded prefix can be shorter than the original (İ → i).
    # Drop leading words until the folded remainder matches.
    words = cleaned.split()
    remainder = folded[len(token):].strip(" :.-")
    for count in range(1, len(words) + 1):
        rest = " ".join(words[count:])
        if normalize_text(rest) == remainder or (not remainder and not rest):
            return rest.strip(" :.-")
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


def _alt_decimal(token: str) -> str | None:
    """Comma and dot are the same decimal point. A day.month token is not a decimal.

    A dotted pair that cannot be a day and month matches the comma spelling.
    A day and month does not. A thousands group is not rewritten.
    """
    if re.fullmatch(r"\d+,\d+", token):
        swapped = token.replace(",", ".")
        if re.fullmatch(r"\d{1,2}\.\d{1,2}", swapped) and _parse_quantity(swapped) is None:
            return None
        return swapped
    if re.fullmatch(r"\d{1,2}\.\d{1,2}", token) and _parse_quantity(token) is not None:
        return token.replace(".", ",")
    return None


def _same_amount_token(left: str, right: str) -> bool:
    return left == right or _alt_decimal(left) == right


def _spelled_in(token: str, blob: str) -> bool:
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


def _token_in_text(token: str, blob: str) -> bool:
    if _spelled_in(token, blob):
        return True
    alt = _alt_decimal(token)
    return alt is not None and _spelled_in(alt, blob)


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
    ("celsius", ("celsius", "°c", "c")),
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
        if not _same_amount_token(token, match.group(0)):
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
    """A measured amount. Dates, clock times, and day.month fragments are not amounts.

    A dotted pair is a decimal when it cannot be a day and month: the day part is
    past 31, or the month part is past 12. A day and month stays a date. A month
    and year stays a month.
    """
    if re.search(r"[:T]|^\d{4}-|^\d{1,2}\.\d{1,2}\.\d{4}", token):
        return None
    if re.fullmatch(r"\d{1,2}\.\d{4}", token):
        return None
    dotted = re.fullmatch(r"(\d{1,2})\.(\d{1,2})", token)
    if dotted:
        whole = int(dotted.group(1))
        frac = int(dotted.group(2))
        if whole > 31 or frac > 12:
            return float(token)
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


# An apostrophe plus ı/i/u/ü is an object waiting for a verb ("mg'ı", "ad'ı").
_OBJECT_MARK = re.compile(r"['’‘ʼ´]\s*[ıiuü](?!\w)", re.IGNORECASE)


def _accusative_object(text: str) -> bool:
    return _OBJECT_MARK.search(text) is not None


def _has_finite_predicate(text: str) -> bool:
    folded = normalize_text(text)
    return any(_finite_word(token) for token in re.findall(r"[a-z]{4,}", folded))


def _capitalize_first(text: str) -> str:
    """Uppercase a sentence that starts with a letter. A second capital, as in eGFR, stays.

    A sentence that starts with a number, such as an ordinal, is left as written.
    Turkish i becomes İ. ı becomes I.
    """
    for index, char in enumerate(text):
        if char.isspace():
            continue
        if not char.isalpha():
            return text
        if not char.islower():
            return text
        nxt = text[index + 1] if index + 1 < len(text) else ""
        if nxt.isupper():
            return text
        if char == "i":
            upper = "İ"
        elif char == "ı":
            upper = "I"
        else:
            upper = char.upper()
        return text[:index] + upper + text[index + 1:]
    return text


def _side_words(text: str) -> list[str]:
    return [
        token for token in re.findall(r"[a-z]{4,}", normalize_text(text))
        if token not in _CAUSE_STOP and token not in _GENERIC_WORDS and token not in _EVENT_WORDS
    ]


def _measurement_only(text: str, blob: str) -> bool:
    """A unit-bearing amount whose other words sit on that amount's line.

    An object waiting for a verb is not a measurement, even when a dose is still there.
    """
    if _accusative_object(text):
        return False
    amounts: list[str] = []
    for match in _numbers(text):
        if _skipped_number(text, match) or _parse_quantity(match.group(0)) is None:
            continue
        if _unit_family(text, match) is None:
            continue
        token = match.group(0)
        if _token_in_text(token, blob) or _derived_quantity(text, match, blob):
            amounts.append(token)
    if not amounts:
        return False
    words = _side_words(text)
    if not words:
        return True
    lines = [line for line in blob.splitlines() if line.strip()] or [blob]
    for word in words:
        placed = any(
            _line_has_token(line, word) and any(_token_in_text(amount, line) for amount in amounts)
            for line in lines
        )
        if not placed:
            return False
    return True


def _finish_clause(text: str) -> str:
    kept = text.strip()
    if kept[-1:] not in ".!?":
        kept += "."
    return _capitalize_first(kept)


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
    """Cause tokens must match one line that also carries the sentence's item.

    A word the chart never uses does not veto a reason that does sit on the line.
    The item name alone is not that reason: if the missing words are at least as
    many as the words the chart contains, the cause is not supported.
    """
    raw = _cause_content(cause)
    if not raw or not blob:
        return False
    # A section title is not the item line, so it cannot supply a cause word.
    lines = [
        line for line in blob.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ] or [line for line in blob.splitlines() if line.strip()] or [blob]
    words = [word for line in lines for word in _line_words(line)]
    present = [token for token in raw if any(_forms_match(token, word) for word in words)]
    absent = len(raw) - len(present)
    if not present or absent >= len(present):
        return False
    cause_tokens = present
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
    not. When the cause matches, the whole sentence stays. When it does not, a
    side stays only if it still has a finite verb, or it is only the measurement
    on that amount's line. A noun phrase left after the verb is removed drops
    the whole sentence.
    """
    parts = _cause_split(sentence)
    if parts is None:
        return sentence
    left, marker, right = parts
    cause, other = _cause_clause(left, marker, right)
    if _cause_on_item_line(cause, other, blob):
        return sentence
    for candidate in (other, cause):
        kept = candidate.strip(" ,;:-")
        if not kept or not _clause_supported(kept, blob):
            continue
        if _has_finite_predicate(kept) or _measurement_only(kept, blob):
            return _finish_clause(kept)
    return None


def _clause_supported(text: str, blob: str) -> bool:
    """A leftover clause needs a supported amount or a word the record actually uses.

    A date or a record-keeping word is not that support. Sharing a prescription
    month with the chart does not make a stop claim true.
    """
    measured = [
        match for match in _numbers(text)
        if not _skipped_number(text, match) and _parse_quantity(match.group(0)) is not None
    ]
    if measured:
        return _unsupported_token(text, blob) == ""
    blob_tokens = set(re.findall(r"[a-z]{4,}", normalize_text(blob)))
    tokens = [
        token for token in re.findall(r"[a-z]{4,}", normalize_text(text))
        if token not in _EVENT_WORDS
    ]
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


def _iter_clauses(blob: str, start: int, end: int):
    cursor = start
    for match in _CLAUSE_SPLIT.finditer(blob, start, end):
        if match.start() > cursor:
            yield cursor, match.start()
        cursor = match.end()
    if cursor < end:
        yield cursor, end


def _amount_at(blob: str, start: int) -> re.Match[str] | None:
    for match in _numbers(blob):
        if match.start() == start and _parse_quantity(match.group(0)) is not None:
            return match
    return None


def _families_compatible(left: str | None, right: str | None) -> bool:
    return not (left and right and left != right)


def _reading_role(blob: str, start: int) -> str | None:
    """Clause label, or the unit-bearing reading written before the usual value.

    A comma-separated daily list is not that reading: its amounts carry no unit
    of their own, and they do not sit before the usual value.
    """
    direct = _nearest_role(blob, start)
    if direct:
        return direct
    sent_l, sent_r = _sentence_bounds(blob, start)
    singles: list[dict] = []
    for left, right in _iter_clauses(blob, sent_l, sent_r):
        amounts = [
            left + match.start()
            for match in _numbers(blob[left:right])
            if _parse_quantity(match.group(0)) is not None
        ]
        if len(amounts) != 1:
            continue
        found = _amount_at(blob, amounts[0])
        explicit = None if found is None else _unit_family(blob, found)
        inherited = None if found is None else _family_at(blob, found)
        singles.append({
            "start": amounts[0],
            "role": _nearest_role(blob, amounts[0]),
            "explicit": explicit,
            "family": explicit or inherited,
        })
    baselines = [item for item in singles if item["role"] == "baseline"]
    currents = [item for item in singles if item["role"] == "current"]
    if len(baselines) != 1 or currents:
        return None
    base = baselines[0]
    before = [
        item for item in singles
        if not item["role"]
        and item["explicit"]
        and item["start"] < base["start"]
        and _families_compatible(item["explicit"], base["family"])
    ]
    if not before:
        return None
    nearest = min(before, key=lambda item: base["start"] - item["start"])
    if nearest["start"] == start:
        return "current"
    return None


def _sentence_has_endpoint_pair(blob: str, start: int) -> bool:
    """True when this sentence already states both a usual and a current amount."""
    sent_l, sent_r = _sentence_bounds(blob, start)
    roles: set[str] = set()
    for match in _numbers(blob):
        if match.start() < sent_l or match.start() >= sent_r:
            continue
        if _parse_quantity(match.group(0)) is None:
            continue
        role = _reading_role(blob, match.start())
        if role:
            roles.add(role)
    return "baseline" in roles and "current" in roles


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
            "role": _reading_role(blob, match.start()),
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
    # A usual/current pair is the from-to. A same-sentence daily list is not an endpoint,
    # even when one of its numbers is exact and the labeled amount is one off.
    if any(_sentence_has_endpoint_pair(blob, hit["start"]) for hit in chosen):
        chosen = [
            hit for hit in chosen
            if hit["role"] or not _sentence_has_endpoint_pair(blob, hit["start"])
        ]
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
        if not _same_amount_token(match.group(0), other.group(0)):
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
        return sum(
            1 for match in _numbers(claim + " " + marker)
            if _token_in_text(match.group(0), section.text)
        )

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
        score = sum(3 for token in claim_numbers if _token_in_text(token, line.text))
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
        score = sum(1 for match in numbers if _token_in_text(match.group(0), section.text))
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


def _is_bullet(line: str) -> bool:
    return _BULLET.match(line.lstrip()) is not None


def _continues_line(prev: str, line: str) -> bool:
    """A wrapped line stays in the sentence. A bullet or a new paragraph does not."""
    if not prev.strip() or not line.strip():
        return False
    nxt = line.lstrip()
    if _is_bullet(nxt):
        return False
    if nxt[0] in ",;":
        return True
    if nxt[0].islower():
        return True
    tail = prev.rstrip()
    if len(tail) >= 2 and tail[-1] in ",." and tail[-2].isdigit() and nxt[:1].isdigit():
        return True
    return bool(tail[-1:].isdigit() and len(nxt) > 1 and nxt[0] in ",." and nxt[1].isdigit())


def _glue_line(prev: str, line: str) -> str:
    left = prev.rstrip()
    right = line.lstrip()
    if left[-1].isdigit() and right[0] in ",.":
        return left + right
    if left[-1] in ",." and right[:1].isdigit():
        return left + right
    if right[0] in ",;":
        return left + right
    return left + " " + right


def _normalize_layout(text: str) -> str:
    """Join a line break that falls inside a sentence or a decimal number."""
    merged: list[str] = []
    for line in text.split("\n"):
        if merged and _continues_line(merged[-1], line):
            merged[-1] = _glue_line(merged[-1], line)
        else:
            merged.append(line)
    return "\n".join(merged)


def _ordinal_period(text: str, match: re.Match[str]) -> bool:
    """A digit, a period, and a lowercase word is an ordinal, not a sentence end."""
    if match.start() == 0 or text[match.start() - 1] != ".":
        return False
    if match.start() < 2 or not text[match.start() - 2].isdigit():
        return False
    after = text[match.end():]
    return bool(after[:1].islower())


def _decimal_boundary(text: str, match: re.Match[str]) -> bool:
    """A dot between digits is a decimal point, not a sentence end."""
    punct_at = match.start() - 1
    if punct_at < 1 or text[punct_at] not in ".,":
        return False
    if not text[punct_at - 1].isdigit():
        return False
    return text[match.end():].lstrip()[:1].isdigit()


def _keeps_together(text: str, match: re.Match[str]) -> bool:
    if _ordinal_period(text, match) or _decimal_boundary(text, match):
        return True
    return text[match.end():].lstrip()[:1] in ",;"


def _sentences(text: str) -> list[str]:
    text = _normalize_layout(text)
    parts: list[str] = []
    start = 0
    for match in _SENTENCE_BREAK.finditer(text):
        if _keeps_together(text, match):
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


# Lab flags and the phrases that name a side of a reference range. Not a chart value.
_FLAG_PHRASES = (
    ("normalin ustunde", "high"),
    ("normalin uzerinde", "high"),
    ("araligin ustunde", "high"),
    ("araligin uzerinde", "high"),
    ("referansin ustunde", "high"),
    ("referansin uzerinde", "high"),
    ("normalin altinda", "low"),
    ("araligin altinda", "low"),
    ("referansin altinda", "low"),
)
_HIGH_STEM = "yuksek"
_LOW_STEM = "dusuk"
_WITHIN_STEM = "normal"
_FLAG_SURFACE = re.compile(
    r"(?i)(?:normalin\s+(?:üstünde|üzerinde|altında)"
    r"|(?:aralığın|referansın)\s+(?:üstünde|üzerinde|altında)"
    r"|\byüksek\w*"
    r"|\bdüşük\w*"
    r"|\bnormal\w*)"
)
_RANGE_PAIR = re.compile(r"(\d+(?:[.,]\d+)?)\s*[-–—]\s*(\d+(?:[.,]\d+)?)")
_FLAG_WORD = {"high": "yüksek", "low": "düşük", "within": "normal"}


def _end_sentence(text: str) -> str:
    """A displayed sentence ends with punctuation. This is formatting, not a check."""
    kept = text.strip()
    if not kept or kept[-1] in ".!?:":
        return kept
    return kept + "."


def _pole_of_token(token: str) -> str | None:
    """A flag stem plus a short ending. A longer word that merely shares the stem is not a flag."""
    if token.startswith(_HIGH_STEM) and len(token) - len(_HIGH_STEM) <= 6:
        return "high"
    if token.startswith(_LOW_STEM) and len(token) - len(_LOW_STEM) <= 6:
        return "low"
    if token.startswith(_WITHIN_STEM) and len(token) - len(_WITHIN_STEM) <= 6:
        return "within"
    return None


def _polarities(text: str) -> set[str]:
    """High, low, or within. A superlative such as 'en yüksek' is an extreme, not a flag."""
    folded = normalize_text(text)
    folded = re.sub(r"\ben yuksek\b", " ", folded)
    folded = re.sub(r"\ben dusuk\b", " ", folded)
    found: set[str] = set()
    for phrase, pole in _FLAG_PHRASES:
        if phrase in folded:
            found.add(pole)
            folded = folded.replace(phrase, " ")
    for token in re.findall(r"[a-z]+", folded):
        pole = _pole_of_token(token)
        if pole:
            found.add(pole)
    return found


def _bound_value(token: str) -> float | None:
    if re.fullmatch(r"\d+,\d+", token):
        return float(token.replace(",", "."))
    if re.fullmatch(r"\d+", token):
        return float(token)
    if _parse_quantity(token) is not None and re.fullmatch(r"\d{1,2}\.\d{1,2}", token):
        return _parse_quantity(token)
    return None


def _range_bounds(text: str) -> tuple[float, float] | None:
    for match in _RANGE_PAIR.finditer(text):
        low = _bound_value(match.group(1))
        high = _bound_value(match.group(2))
        if low is None or high is None or low > high:
            continue
        return low, high
    return None


def _primary_amount(line: str) -> float | None:
    """The result the line labels. A date is not that result."""
    for match in _numbers(line):
        if _skipped_number(line, match):
            continue
        value = _parse_quantity(match.group(0))
        if value is not None:
            return value
    return None


def _explicit_polarity(line: str) -> str | None:
    """The flag written after the result. The range's own words are not a second flag."""
    primary_at = None
    for match in _numbers(line):
        if _skipped_number(line, match):
            continue
        if _parse_quantity(match.group(0)) is not None:
            primary_at = match.end()
            break
    tail = line[primary_at:] if primary_at is not None else line
    poles = _polarities(tail)
    if "high" in poles and "low" in poles:
        return None
    if "high" in poles:
        return "high"
    if "low" in poles:
        return "low"
    if "within" in poles:
        return "within"
    return None


def _sentence_amounts(sentence: str) -> list[float]:
    found: list[float] = []
    for match in _numbers(sentence):
        if _skipped_number(sentence, match):
            continue
        value = _parse_quantity(match.group(0))
        if value is not None:
            found.append(value)
    return found


def _same_amount(left: float, right: float) -> bool:
    return abs(left - right) <= 1e-6


def _flag_line(sentence: str, blob: str) -> str:
    lines = [
        line for line in blob.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    amounts = [
        match.group(0) for match in _numbers(sentence)
        if not _skipped_number(sentence, match) and _parse_quantity(match.group(0)) is not None
    ]
    words = content_words(sentence)
    best = ""
    best_score = 0
    for line in lines:
        score = sum(3 for amount in amounts if _token_in_text(amount, line))
        folded = normalize_text(line)
        score += sum(1 for word in words if word in folded)
        if score > best_score:
            best_score = score
            best = line
    return best


def _expected_polarity(sentence: str, line: str) -> str | None:
    """The record's flag, or the side of its range the labeled result sits on."""
    if not line:
        return None
    primary = _primary_amount(line)
    amounts = _sentence_amounts(sentence)
    if primary is None or not any(_same_amount(primary, value) for value in amounts):
        return None
    explicit = _explicit_polarity(line)
    if explicit:
        return explicit
    bounds = _range_bounds(line)
    if bounds is None:
        return None
    low, high = bounds
    if primary < low:
        return "low"
    if primary > high:
        return "high"
    return "within"


def _keep_superlative(text: str, match: re.Match[str]) -> bool:
    prefix = normalize_text(text[max(0, match.start() - 4):match.start()])
    return prefix.endswith("en ")


def _rewrite_polarity(text: str, pole: str) -> str:
    target = _FLAG_WORD[pole]

    def repl(match: re.Match[str]) -> str:
        if _keep_superlative(text, match):
            return match.group(0)
        return target

    cleaned = _FLAG_SURFACE.sub(repl, text)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    cleaned = re.sub(r"\s+([,.!?])", r"\1", cleaned)
    return cleaned.strip()


def _align_flag(sentence: str, blob: str) -> str | None:
    """Keep a high/low/normal claim when the record's flag or range agrees.

    A claim with no such word is left alone. The opposite of the record's flag
    or range is replaced by that direction. When the record does not say, the
    sentence stays: deleting the word leaves a broken sentence.
    """
    if not _polarities(sentence) or not _sentence_amounts(sentence):
        return sentence
    expected = _expected_polarity(sentence, _flag_line(sentence, blob))
    if expected is None or _polarities(sentence) == {expected}:
        return sentence
    return _rewrite_polarity(sentence, expected)


def _present_line(line: str) -> str:
    shown = line.strip()
    if not shown or _blank_sentence(shown):
        return ""
    return _end_sentence(shown)


def _tidy_blocks(blocks: list[str]) -> str:
    """Keep paragraph breaks and bullet lines. Drop runs of empty lines."""
    lines: list[str] = []
    blank = False
    for block in blocks:
        if block == "":
            if lines and not blank:
                lines.append("")
                blank = True
            continue
        lines.append(block)
        blank = False
    return "\n".join(lines).strip()


def _render_prose(body: str) -> list[tuple[str, str | None]]:
    """Layout lines and the citation marker that covered each one."""
    rows: list[tuple[str, str | None]] = []
    for claim, marker in _pairs(body):
        laid = _normalize_layout(claim)
        if not laid.strip():
            continue
        for line in laid.split("\n"):
            if not line.strip():
                rows.append(("", marker))
                continue
            shown = _present_line(line)
            if shown:
                rows.append((shown, marker))
    return rows


def format_model_text(raw: str, record: TextRecord) -> tuple[str, str, list[dict]]:
    """Strip the status line and move citation tags out of the prose.

    Paragraph breaks and bullet lines stay. This does not check numbers,
    dates, trends, causes, or flags.
    """
    cleaned = strip_model_wrappers(raw)
    parsed = _status_and_body(cleaned)
    if parsed is None:
        status, body = "answered", cleaned.strip()
    else:
        status, body = parsed
    citations: list[dict] = []
    seen: set[tuple] = set()
    blocks: list[str] = []
    last_index: int | None = None
    for shown, marker in _render_prose(body):
        if not shown:
            blocks.append("")
            continue
        index = _resolve_section(marker, shown, record) if marker else None
        if index is None:
            index = last_index
        blocks.append(shown)
        if index is not None:
            last_index = index
            _remember(citations, seen, record, index, shown, marker)
    return status, _tidy_blocks(blocks), citations


def _schedules(text: str) -> list[str]:
    return _SCHEDULE.findall(text)


def _schedule_in(token: str, blob: str) -> bool:
    """A dose schedule matches with or without spaces around the mark."""

    def squash(value: str) -> str:
        return re.sub(r"\s+", "", value.casefold().replace("×", "x"))

    needle = squash(token)
    hay = squash(blob)
    start = 0
    while True:
        found = hay.find(needle, start)
        if found < 0:
            return False
        before = hay[found - 1] if found else ""
        after = hay[found + len(needle):found + len(needle) + 1]
        if not before.isdigit() and not after.isdigit():
            return True
        start = found + 1


def _chart_tokens_ok(sentence: str, blob: str) -> bool:
    """Every number, date, and dose schedule in the sentence is in the cited text."""
    numbers = _numbers(sentence)
    schedules = _schedules(sentence)
    if not numbers and not schedules:
        return False
    if numbers and not _numbers_supported(sentence, blob):
        return False
    return all(_schedule_in(item, blob) for item in schedules)


def _cited_blob(body: str, record: TextRecord) -> str:
    chunks: list[str] = []
    for claim, marker in _pairs(body):
        if not marker:
            continue
        index = _resolve_section(marker, claim, record)
        if index is not None:
            chunks.append(record.sections[index].text)
    return "\n".join(chunks) if chunks else record.text


def _support_plan(body: str, record: TextRecord) -> list[tuple[int | None, bool]]:
    """Each sentence's cited section, and whether that section holds its amounts."""
    plan: list[tuple[int | None, bool]] = []
    last_index: int | None = None
    for claim, marker in _pairs(body):
        section_index: int | None = None
        if marker and _marker_supported(marker, record):
            section_index = _resolve_section(marker, claim, record)
            if section_index is not None:
                last_index = section_index
        pieces = _sentences(claim) or ([claim.strip()] if claim.strip() else [])
        for sentence in pieces:
            if _blank_sentence(sentence):
                continue
            index = section_index if section_index is not None else last_index
            if index is None:
                index = _section_for_claim(sentence, record)
            blob = record.sections[index].text if index is not None else ""
            supported = index is not None and _chart_tokens_ok(sentence, blob)
            plan.append((index, supported))
    return plan


def _inherits_section_support(plan: list[tuple[int | None, bool]], ordinal: int) -> bool:
    """A numberless sentence may lean on a supported one in the same section run.

    The run is the sentences that cite that section with no other section between
    them. A paraphrase of a finding stays. A sentence about another section does not.
    """
    if ordinal < 0 or ordinal >= len(plan):
        return False
    index, supported = plan[ordinal]
    if index is None or supported:
        return False
    for step in (-1, 1):
        cursor = ordinal + step
        while 0 <= cursor < len(plan) and plan[cursor][0] == index:
            if plan[cursor][1]:
                return True
            cursor += step
    return False


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
    plan = _support_plan(body, record)
    ordinal = 0
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
            inherits = _inherits_section_support(plan, ordinal)
            ordinal += 1
            sentence_count += 1
            if answer_has_unsafe_advice(sentence):
                saw_unsafe = True
                continue
            index = section_index if section_index is not None else last_index
            if index is None:
                index = _section_for_claim(sentence, record)
            blob = record.sections[index].text if index is not None else ""
            schedules = _schedules(sentence)
            if schedules and not all(_schedule_in(item, blob) for item in schedules):
                missing = next(item for item in schedules if not _schedule_in(item, blob))
                _failure(
                    errors, debug, "unsupported_number", sentence,
                    _cite(marker, record, index), missing,
                )
                continue
            anchored = _chart_tokens_ok(sentence, blob)
            needs_number = bool(_numbers(sentence) or _RELATIVE.search(sentence) or schedules)
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
                sentence = _capitalize_first(trend.text)
                edited = True
            if _unit_conflict(sentence, record.text):
                _failure(
                    errors, debug, "incompatible_unit", sentence, _cite(marker, record, index),
                )
                continue
            if (
                not inherits
                and not anchored
                and not needs_number
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
            aligned = _align_flag(sentence, blob)
            if aligned is None:
                _failure(
                    errors, debug, "unsupported_flag", sentence, _cite(marker, record, index),
                )
                continue
            if aligned != sentence:
                sentence = aligned
                edited = True
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
        # A grounded answer is shown. Soft overlap misses must not blank it.
        if (
            errors
            and all(item in _SOFT_DROPS for item in errors)
            and _chart_tokens_ok(body, _cited_blob(body, record))
        ):
            rendered_status, rendered, rendered_cites = format_model_text(raw, record)
            if rendered.strip():
                return GroundingResult(
                    True, rendered_status, rendered, rendered_cites, "", debug,
                )
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
    changed = len(kept) != sentence_count or hide_markers or moved or edited
    if changed:
        answer = "\n".join(
            _end_sentence(_capitalize_first(part)) for part in kept
        ).strip()
        answer = _capitalize_first(answer)
    else:
        answer = _tidy_blocks([line for line, _marker in _render_prose(body)])
    if status == "answered" and _absence_only(answer):
        status = "insufficient_evidence"
    return GroundingResult(True, status, answer, citations, "", debug if errors else [])


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
