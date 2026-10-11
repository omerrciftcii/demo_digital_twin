"""Choose chart rows by section and by the question's own words.

No chart value, diagnosis code, or hand-picked brand list lives here.
Drug groups come from the ATC-style file. A row is relevant when its words
overlap the question (or the same drug group, when the question is about
continued use). Symptom-to-diagnosis mapping is left to the model.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

from .index import normalize_text
from .text_document import TextRecord

_DATA = Path(__file__).resolve().parent / "data" / "drug_classes.json"
_WORD = re.compile(r"[a-z0-9]{3,}", re.IGNORECASE)
_FULL_DATE = re.compile(r"\d{1,2}\.\d{1,2}\.\d{4}")
_UNIT = re.compile(
    r"\d+(?:[.,]\d+)?\s*(?:mg/dL|g/dL|pg/mL|ng/mL|mIU/L|U/L|mmHg|µg|mcg|kg|mg|mL|mm|cm)\b",
    re.IGNORECASE,
)
_FLAG = re.compile(r",\s*(yüksek|düşük)\b", re.IGNORECASE)
_RX_MARK = re.compile(r"TABLET|KAPSÜL|KAPSUL|DAMLA|İNHALER|INHALER|KALEM|SPREY", re.IGNORECASE)
_EXAM = re.compile(
    r"(?i)\b(?:ŞİKAYET|Şikayet|Şikâyet|HİKAYE|Hikaye|Hikâye|HİKÂYE|"
    r"BULGULAR|Bulgular|ÖZGEÇMİŞ|Özgeçmiş|Epikriz|EPİKRİZ)\s*:"
)
_SUFFIXES = (
    "larimizdan", "lerimizden", "larimiz", "lerimiz", "larimdan", "lerimden",
    "imizden", "imizdan", "undan", "unden", "indan", "inden",
    "nin", "nın", "nun", "nün", "yi", "yı", "yu", "yü",
    "den", "dan", "ten", "tan", "ler", "lar", "lik", "lık", "luk", "lük",
    "imiz", "ımız", "umuz", "ümüz",
    "iyor", "uyor", "yor", "yon", "yo",
    "si", "sı", "su", "sü",
    "im", "ım", "um", "üm", "in", "ın", "un", "ün",
    "de", "da", "te", "ta",
    "di", "du", "ti", "tu", "mis", "mus",
)
_STOP = frozenset({
    "bana", "benim", "bugun", "deger", "degeri", "durum", "hakkinda", "hala",
    "halen", "icin", "ilac", "ilaci", "iyi", "kadar", "kayit", "kayitta",
    "kullan", "miyim", "musun", "nasil", "neden", "nedir", "normal", "olabilir",
    "olarak", "oldu", "once", "recete", "sonra", "sonuc", "soyle", "tahlil",
    "tahliller", "var", "yok", "kotu", "seyir", "seyret", "degis", "degisti",
    "ilgili", "ilgil", "hastalik", "hastasi", "eder", "ediyor", "gibi", "daha",
    "gore", "baska", "hepsi", "tum", "butun", "simdi", "bugunku", "laboratuvar",
    "son", "olcum", "olcu", "kac", "cikti", "cikt", "cok", "sey", "bir", "bu",
    "veri", "gun", "gece", "zaman", "diye", "ile", "ne", "mi", "mu",
})
_WEARABLE_TERMS = frozenset({"nabiz", "hrv", "adim", "giyilebilir", "wearable"})
_LAB_WORDS = ("tahlil", "laboratuvar")
_PANEL_WORDS = ("nasil", "iyi", "kotu", "normal", "durum")
_CONTINUATION = (
    "hala ", "halen", "devam ediyor", "kullanmam gerekiyor", "kullaniyor muyum",
    "kullanmaya devam", "kullaniyor muydum",
)
_VOWELS = frozenset("aeiou")


def fold(text: str) -> str:
    return normalize_text(text or "")


def _skeleton(stem: str) -> str:
    return "".join(char for char in stem if char not in _VOWELS)


def stem(token: str) -> str:
    """Strip one trailing Turkish ending. The stem stays at least four letters."""
    current = fold(token)
    for suffix in _SUFFIXES:
        if len(current) - len(suffix) >= 4 and current.endswith(suffix):
            return current[: -len(suffix)]
    return current


def _terms_in(text: str) -> list[str]:
    found: list[str] = []
    for raw in _WORD.findall(fold(text)):
        token = stem(raw)
        if len(token) < 3 or token in _STOP or token in found:
            continue
        found.append(token)
    return found


def question_terms(question: str) -> list[str]:
    return [token for token in _terms_in(question) if token not in _STOP]


def _same(left: str, right: str) -> bool:
    if not left or not right:
        return False
    if left == right:
        return True
    short, long = (left, right) if len(left) <= len(right) else (right, left)
    # A shared prefix counts when the extra letters are a short ending.
    if len(short) >= 5 and long.startswith(short) and len(long) - len(short) <= 3:
        return True
    if len(short) >= 5 and short in long and len(long) - len(short) <= 4:
        return True
    return bool(len(short) >= 4 and _skeleton(left) == _skeleton(right) and len(_skeleton(left)) >= 3)


def overlaps(text: str, terms: list[str] | set[str]) -> bool:
    if not terms:
        return False
    hay = _terms_in(text)
    folded = fold(text)
    for term in terms:
        if len(term) <= 3:
            if re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", folded):
                window = _window(folded, term)
                if window and (re.search(r"\d", window) or _UNIT.search(window)):
                    return True
            continue
        if any(_same(term, token) for token in hay):
            return True
    return False


def _window(folded: str, term: str) -> str:
    match = re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", folded)
    if not match:
        return ""
    return folded[max(0, match.start() - 8): match.end() + 24]


def score_text(text: str, terms: list[str] | set[str]) -> int:
    if not terms:
        return 0
    hay = _terms_in(text)
    folded = fold(text)
    score = 0
    for term in terms:
        if len(term) <= 3:
            if re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", folded):
                window = _window(folded, term)
                if window and (re.search(r"\d", window) or _UNIT.search(window)):
                    score += 3
            continue
        if any(_same(term, token) for token in hay):
            score += 2 if len(term) >= 6 else 1
    return score


@lru_cache(maxsize=1)
def _entries() -> tuple[dict, ...]:
    payload = json.loads(_DATA.read_text(encoding="utf-8"))
    rows = []
    for entry in payload.get("entries", []):
        names = tuple(dict.fromkeys(fold(name) for name in entry.get("names", []) if name))
        if names:
            rows.append({"group": fold(entry.get("group", "")), "atc": entry.get("atc", ""), "names": names})
    return tuple(rows)


def matched_entries(question: str) -> list[dict]:
    folded = fold(question)
    tokens = _terms_in(question)
    hits = []
    for entry in _entries():
        if any(_name_in(name, folded, tokens) for name in entry["names"]):
            hits.append(entry)
    return hits


def _name_in(name: str, folded: str, tokens: list[str]) -> bool:
    if len(name) <= 3:
        return re.search(rf"(?<![a-z0-9]){re.escape(name)}(?![a-z0-9])", folded) is not None
    if re.search(rf"(?<![a-z0-9]){re.escape(name)}(?![a-z])", folded):
        return True
    return any(_same(name, token) for token in tokens)


def drug_names(question: str, *, expand_group: bool) -> set[str]:
    """Brand and active ingredient. Group mates only when continued use is the question."""
    hits = matched_entries(question)
    names: set[str] = set()
    groups = {entry["group"] for entry in hits if entry["group"]}
    for entry in hits:
        names.update(entry["names"])
    if expand_group and groups:
        for entry in _entries():
            if entry["group"] in groups:
                names.update(entry["names"])
    return {name for name in names if len(name) >= 3}


def section_kind(heading: str) -> str:
    folded = fold(heading)
    if any(token in folded for token in ("laboratuvar", "biyokimya", "hemogram")):
        return "lab"
    if "tani" in folded:
        return "diagnosis"
    if any(token in folded for token in ("recete", "ilac", "kulland")):
        return "med"
    if any(token in folded for token in ("patoloji", "sitoloji", "biyopsi", "smear")):
        return "pathology"
    if any(token in folded for token in ("giyilebilir", "wearable")):
        return "wearable"
    if "alerji" in folded:
        return "allergy"
    if any(token in folded for token in ("basvuru", "muayene", "poliklinik", "epikriz")):
        return "visit"
    return "other"


def asks_labs(question: str) -> bool:
    folded = fold(question)
    return any(token in folded for token in _LAB_WORDS)


def asks_panel(question: str) -> bool:
    """A lab question that names no measurement of its own."""
    if not asks_labs(question):
        return False
    return not question_terms(question)


def is_continuation(question: str) -> bool:
    folded = fold(question)
    return any(phrase in folded for phrase in _CONTINUATION)


def _date_key(text: str) -> tuple[int, int, int]:
    dates = _FULL_DATE.findall(text)
    if not dates:
        return (0, 0, 0)
    day, month, year = (int(part) for part in dates[-1].split("."))
    return (year, month, day)


def _clauses(text: str) -> list[str]:
    """Split sentences. A period after a digit stays put (5. yıl, 3. gün)."""
    parts = [
        part.strip(" -")
        for part in re.split(r"(?<!\d[.;])(?<=[.;])\s+", text)
        if part.strip(" -")
    ]
    return parts or [text.strip()]


def focus_clause(text: str, terms: list[str] | set[str]) -> str:
    """Keep the clause that shares the question's words. Drop exam-header blobs."""
    raw = " ".join(text.strip().lstrip("-").split())
    if not raw:
        return ""
    pieces = _clauses(raw)
    hits = [piece for piece in pieces if overlaps(piece, terms)]
    if _EXAM.search(raw):
        preferred = [piece for piece in hits if not _EXAM.search(piece)]
        hits = preferred or hits
    chosen = hits or ([raw] if overlaps(raw, terms) else [])
    if not chosen:
        return ""
    piece = max(chosen, key=lambda item: (score_text(item, terms), -len(item)))
    if piece in pieces:
        index = pieces.index(piece)
        bare = _FULL_DATE.sub(" ", piece)
        if not re.search(r"\d", bare) and index + 1 < len(pieces):
            nxt = pieces[index + 1]
            if re.search(r"\d", _FULL_DATE.sub(" ", nxt)):
                piece = f"{piece}; {nxt}"
    if len(piece) > 280:
        piece = piece[:280].rsplit(" ", 1)[0].rstrip(" ;,")
    if _FULL_DATE.search(raw) and not _FULL_DATE.search(piece):
        piece = f"{_FULL_DATE.findall(raw)[-1]} {piece}"
    return piece.strip()


def _abnormal_labs(record: TextRecord) -> list[str]:
    found = []
    for line in record.lines:
        if line.text.startswith("##"):
            continue
        if section_kind(record.sections[line.section_index].heading) != "lab":
            continue
        if not _FLAG.search(line.text):
            continue
        text = " ".join(line.text.strip().lstrip("-").split())
        if text and text not in found:
            found.append(text)
        if len(found) == 3:
            break
    return found


def select_rows(record: TextRecord, question: str, limit: int = 4) -> list[str]:
    """Rows whose section and words fit the question. Empty when nothing overlaps."""
    terms = question_terms(question)
    names = drug_names(question, expand_group=is_continuation(question))
    folded_names = list(names)
    wearable_ok = any(token in _WEARABLE_TERMS for token in terms) or any(
        token in fold(question) for token in _WEARABLE_TERMS
    )
    labs_only = asks_labs(question)
    search_terms = list(dict.fromkeys([*terms, *folded_names]))
    if labs_only and asks_panel(question):
        return _abnormal_labs(record)[:limit]

    ranked: list[tuple[int, tuple[int, int, int], str, str]] = []
    for line in record.lines:
        if line.text.startswith("##") or not line.text.strip():
            continue
        kind = section_kind(record.sections[line.section_index].heading)
        if kind == "wearable" and not wearable_ok:
            continue
        if labs_only and kind != "lab":
            continue
        if not labs_only and kind == "lab" and not overlaps(line.text, search_terms):
            continue
        row_score = score_text(line.text, search_terms)
        if names and any(_name_in(name, fold(line.text), _terms_in(line.text)) for name in names):
            row_score += 2
        if row_score <= 0:
            continue
        focused = focus_clause(line.text, search_terms) or " ".join(line.text.strip().lstrip("-").split())
        if not focused:
            continue
        if re.search(r"\d", _FULL_DATE.sub(" ", focused)):
            row_score += 1
        ranked.append((row_score, _date_key(line.text), kind, focused))

    if not ranked:
        return []
    visits = [row for row in ranked if row[2] == "visit"]
    others = [row for row in ranked if row[2] != "visit"]
    kept = list(others)
    if visits:
        best = max(row[0] for row in visits)
        close = [row for row in visits if row[0] >= best]
        kept.append(max(close, key=lambda row: row[1]))
    kept.sort(key=lambda row: (row[0], row[1]), reverse=True)
    unique: list[str] = []
    for _score, _date, _kind, text in kept:
        if text not in unique:
            unique.append(text)
        if len(unique) == limit:
            break
    return unique


def lab_rows(record: TextRecord) -> list[str]:
    rows = []
    for line in record.lines:
        if line.text.startswith("##"):
            continue
        if section_kind(record.sections[line.section_index].heading) != "lab":
            continue
        text = " ".join(line.text.strip().lstrip("-").split())
        if text:
            rows.append(text)
    return rows


def matched_lab_rows(record: TextRecord, question: str) -> list[str]:
    """Lab rows the question's words actually touch. A bare panel question uses abnormal flags."""
    if asks_panel(question):
        return _abnormal_labs(record)
    selected = select_rows(record, question, limit=4)
    labs = [fold(line) for line in lab_rows(record)]
    found = []
    for row in selected:
        folded = fold(row)
        if any(folded in line or line in folded for line in labs):
            found.append(row)
    return found
