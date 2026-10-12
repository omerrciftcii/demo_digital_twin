"""Choose chart rows by section and by the question's own words.

No chart value, diagnosis code, or question-type list lives here.
A row is relevant when its words share a stem with the question.
A generated drug dictionary (data/drug_classes.json, built by
scripts/build_drug_dictionary.py) links the other names on the same ATC
entry. Names are resolved through a startup trigram index, not a full scan.
Section order is lexical overlap fused with a local embedding when one is
on disk. Line vectors add a section when the best line clears a similarity
floor, and the latest question outweighs earlier turns. Symptom-to-diagnosis
mapping is left to the model.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

from snowballstemmer import stemmer

from .index import normalize_text
from .text_document import TextRecord
from .text_drug_index import drug_names, matched_entries
from .text_embed import semantic_line_scores, semantic_section_scores
from .text_llm import env_number

__all__ = [
    "drug_names",
    "matched_entries",
    "prompt_plan",
    "question_terms",
    "relevant_context",
    "relevant_sections",
]

_STEMMER = stemmer("turkish")
_STOP_PATH = Path(__file__).resolve().parent / "data" / "stopwords-tr.txt"
_WORD = re.compile(r"[a-z0-9]{3,}", re.IGNORECASE)
# A chart abbreviation is an all-caps token of a few letters. It is a metric name
# even when no digit sits beside it. Longer surfaces stay ordinary words.
_ABBR = re.compile(
    r"(?<![A-Za-zÇĞİÖŞÜçğıöşü])[A-ZÇĞİÖŞÜ]{2,5}(?![A-Za-zÇĞİÖŞÜçğıöşü])"
)
_FULL_DATE = re.compile(r"\d{1,2}\.\d{1,2}\.\d{4}")
_UNIT = re.compile(
    r"\d+(?:[.,]\d+)?\s*(?:mg/dL|g/dL|pg/mL|ng/mL|mIU/L|U/L|mmHg|µg|mcg|kg|mg|mL|mm|cm)\b",
    re.IGNORECASE,
)
_EXAM = re.compile(
    r"(?i)\b(?:ŞİKAYET|Şikayet|Şikâyet|HİKAYE|Hikaye|Hikâye|HİKÂYE|"
    r"BULGULAR|Bulgular|ÖZGEÇMİŞ|Özgeçmiş|Epikriz|EPİKRİZ)\s*:"
)


def fold(text: str) -> str:
    return normalize_text(text or "")


def stem(token: str) -> str:
    """Turkish stem (Snowball). Stems shorter than three letters stay whole."""
    folded = fold(token)
    if len(folded) < 3:
        return folded
    stemmed = fold(_STEMMER.stemWord(folded))
    if len(stemmed) < 3:
        return folded
    return stemmed


def _load_stop() -> frozenset[str]:
    """Lucene Turkish stopwords, plus the stem of each word."""
    found: set[str] = set()
    for line in _STOP_PATH.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#"):
            continue
        word = fold(raw)
        if len(word) >= 3:
            found.add(word)
        stemmed = stem(word)
        if len(stemmed) >= 3:
            found.add(stemmed)
    return frozenset(found)


_STOP = _load_stop()


def _terms_in(text: str) -> list[str]:
    return list(_cached_terms(text))


@lru_cache(maxsize=256)
def _cached_terms(text: str) -> tuple[str, ...]:
    found: list[str] = []
    for raw in _WORD.findall(fold(text)):
        token = stem(raw)
        if len(token) < 3 or token in _STOP or token in found:
            continue
        found.append(token)
    return tuple(found)


def question_terms(question: str) -> list[str]:
    return [token for token in _terms_in(question) if token not in _STOP]


def _same(left: str, right: str) -> bool:
    """Equal stems, or one stem is a short extension of the other."""
    if not left or not right:
        return False
    if left == right:
        return True
    short, long = (left, right) if len(left) <= len(right) else (right, left)
    return bool(len(short) >= 5 and long.startswith(short) and len(long) - len(short) <= 3)


def _abbreviations(text: str) -> set[str]:
    return {fold(match.group(0)) for match in _ABBR.finditer(text)}


def overlaps(text: str, terms: list[str] | set[str]) -> bool:
    if not terms:
        return False
    hay = _terms_in(text)
    folded = fold(text)
    abbrs = _abbreviations(text)
    for term in terms:
        if len(term) <= 3:
            if term in abbrs:
                return True
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
    abbrs = _abbreviations(text)
    score = 0
    for term in terms:
        if len(term) <= 3:
            if term in abbrs:
                score += 3
                continue
            if re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", folded):
                window = _window(folded, term)
                if window and (re.search(r"\d", window) or _UNIT.search(window)):
                    score += 3
            continue
        if any(_same(term, token) for token in hay):
            score += 2 if len(term) >= 6 else 1
    return score


def _alias_stems(question: str) -> set[str]:
    """Stems of dictionary names that share an entry with the question."""
    stems = {stem(name) for name in drug_names(question)}
    return {token for token in stems if len(token) >= 5}


def _lexical_score(text: str, terms: list[str], aliases: set[str]) -> int:
    score = score_text(text, terms)
    if not aliases:
        return score
    if set(_terms_in(text)) & aliases:
        score += 2
    return score


# Reciprocal-rank fusion constant. Not a clinical cutoff.
_RRF_K = 60
_SEMANTIC_MARGIN = 0.05
_SEMANTIC_ONLY_CAP = 2
# The latest question outweighs earlier turns. These are rank weights.
_QUESTION_WEIGHT = 3
_HISTORY_WEIGHT = 1
_EMBED_TOP_K = 4
_EMBED_FLOOR = 0.82


def _fuse_sections(lexical: dict[int, int], semantic: dict[int, float]) -> list[int]:
    """Lexical hits, reordered with embeddings when they exist.

    A section with no lexical overlap is added only when its cosine stands
    above the middle of the record and close to the best section.
    """
    fused: dict[int, float] = {}
    lexical_order = sorted(lexical, key=lambda index: (-lexical[index], index))
    for rank, index in enumerate(lexical_order, start=1):
        fused[index] = 1.0 / (_RRF_K + rank)
    semantic_only: list[int] = []
    if semantic:
        semantic_order = sorted(semantic, key=lambda index: (-semantic[index], index))
        for rank, index in enumerate(semantic_order, start=1):
            if index in fused:
                fused[index] += 1.0 / (_RRF_K + rank)
        ordered = sorted(semantic.values())
        median = ordered[len(ordered) // 2]
        best = ordered[-1]
        for index in semantic_order:
            if lexical.get(index, 0) > 0:
                continue
            score = semantic[index]
            if score < best - _SEMANTIC_MARGIN or score < median + _SEMANTIC_MARGIN:
                continue
            fused[index] = fused.get(index, 0.0) + 1.0 / (_RRF_K + 1)
            semantic_only.append(index)
            if len(semantic_only) == _SEMANTIC_ONLY_CAP:
                break
    return [
        index
        for index, _score in sorted(fused.items(), key=lambda item: (-item[1], item[0]))
        if lexical.get(index, 0) > 0 or index in semantic_only
    ]


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


def select_rows(record: TextRecord, question: str, limit: int = 4) -> list[str]:
    """Rows whose words overlap the question. Empty when nothing overlaps."""
    terms = question_terms(question)
    aliases = _alias_stems(question)
    search_terms = list(dict.fromkeys([*terms, *aliases]))
    ranked: list[tuple[int, tuple[int, int, int], str, str]] = []
    for line in record.lines:
        if line.text.startswith("##") or not line.text.strip():
            continue
        kind = section_kind(record.sections[line.section_index].heading)
        row_score = _lexical_score(line.text, terms, aliases)
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


def embed_top_k() -> int:
    """How many embedding-only sections may join the lexical hits."""
    return int(env_number("TEXT_SECTION_TOP_K", _EMBED_TOP_K, 1, 32, integer=True))


def embed_floor() -> float:
    """Minimum cosine for an embedding-only section. Below this, lexical wins."""
    return env_number("TEXT_SECTION_SIMILARITY", _EMBED_FLOOR, 0.0, 1.0, integer=False)


def _topk_above_floor(scores: dict[int, float]) -> list[int]:
    floor = embed_floor()
    ordered = sorted(
        (index for index, score in scores.items() if score >= floor),
        key=lambda index: (-scores[index], index),
    )
    return ordered[:embed_top_k()]


def _lexical_map(record: TextRecord, question: str, prior: str) -> dict[int, int]:
    """Question terms count more than terms that appear only in earlier turns."""
    q_terms = question_terms(question)
    q_alias = _alias_stems(question)
    p_terms = [token for token in question_terms(prior) if token not in q_terms] if prior.strip() else []
    p_alias = (_alias_stems(prior) - q_alias) if prior.strip() else set()
    scores: dict[int, int] = {}
    for index, section in enumerate(record.sections):
        score = _lexical_score(section.text, q_terms, q_alias) * _QUESTION_WEIGHT
        if p_terms or p_alias:
            score += _lexical_score(section.text, p_terms, p_alias) * _HISTORY_WEIGHT
        if score:
            scores[index] = score
    return scores


def _selection(
    record: TextRecord, question: str, prior: str = "",
) -> tuple[dict[int, float], list[int]]:
    """Rank of every section, and the relevant indexes, best first.

    Lexical overlap always qualifies. Section and line embeddings add a
    section only when it is in the top scores and at or above the floor.
    Embeddings use the latest question. Earlier turns add lexical weight.
    """
    lexical = _lexical_map(record, question, prior)
    semantic = semantic_section_scores(record, question)
    line_scores = semantic_line_scores(record, question)
    if prior.strip():
        history_semantic = semantic_section_scores(record, prior)
        scale = _HISTORY_WEIGHT / _QUESTION_WEIGHT
        for index, score in history_semantic.items():
            discounted = score * scale
            current = max(semantic.get(index, 0.0), line_scores.get(index, 0.0))
            if discounted > current:
                semantic = {**semantic, index: discounted}
    order = _fuse_sections(lexical, semantic)
    for index in _topk_above_floor(semantic):
        if index not in order:
            order.append(index)
    for index in _topk_above_floor(line_scores):
        if index not in order:
            order.append(index)
    rank: dict[int, float] = {}
    for index in range(len(record.sections)):
        cosine = max(semantic.get(index, 0.0), line_scores.get(index, 0.0))
        rank[index] = float(lexical.get(index, 0)) + cosine * _QUESTION_WEIGHT
    return rank, order


def prompt_plan(
    record: TextRecord, question: str, *, prior: str = "",
) -> tuple[dict[int, float], list[int]]:
    """Section scores (high is more relevant) and the relevant indexes."""
    return _selection(record, question, prior)


def relevant_sections(record: TextRecord, question: str, *, prior: str = "") -> list:
    """Whole sections that overlap the question, highest fused score first.

    Lexical overlap always qualifies. A local embedding may add a section
    that stands above the rest of the record, or whose best line clears the
    similarity floor. The latest question outweighs earlier turns.
    """
    _rank, order = _selection(record, question, prior)
    return [record.sections[index] for index in order]


def relevant_context(record: TextRecord, question: str) -> str:
    """Full text of every overlapping section. Empty when nothing overlaps."""
    parts = [section.text.strip() for section in relevant_sections(record, question) if section.text.strip()]
    return "\n\n".join(parts)


def matched_lab_rows(record: TextRecord, question: str) -> list[str]:
    """Lab rows the question's words actually touch."""
    selected = select_rows(record, question, limit=4)
    labs = [fold(line) for line in lab_rows(record)]
    found = []
    for row in selected:
        folded = fold(row)
        if any(folded in line or line in folded for line in labs):
            found.append(row)
    return found
