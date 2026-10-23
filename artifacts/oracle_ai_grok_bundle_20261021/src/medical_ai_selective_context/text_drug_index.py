"""Startup index for the generated drug dictionary.

Names are folded (Turkish case, dotless ı, circumflex) and stemmed once.
Lookup is an exact stem map plus a character-trigram posting list. Fuzzy
candidates are scored with a short edit distance. The full table is not
scanned per question.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

from rapidfuzz.distance import Levenshtein
from snowballstemmer import stemmer

from .index import normalize_text

_STEMMER = stemmer("turkish")
_WORD = re.compile(r"[a-z0-9]{3,}", re.IGNORECASE)
_STOP_PATH = Path(__file__).resolve().parent / "data" / "stopwords-tr.txt"
_MAX_FUZZY_ENTRIES = 3


def fold(text: str) -> str:
    """Turkish casefold: İ/I/ı and circumflex accents land on the same letters."""
    return normalize_text(text or "")


def stem(token: str) -> str:
    folded = fold(token)
    if len(folded) < 3:
        return folded
    stemmed = fold(_STEMMER.stemWord(folded))
    if len(stemmed) < 3:
        return folded
    return stemmed


def _load_stop() -> frozenset[str]:
    found: set[str] = set()
    if not _STOP_PATH.is_file():
        return frozenset()
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


def _trigrams(token: str) -> tuple[str, ...]:
    if len(token) < 3:
        return (token,) if token else ()
    return tuple(token[index:index + 3] for index in range(len(token) - 2))


def _close(query: str, candidate: str) -> int | None:
    """Edit distance when the forms are the same stem, a short suffix, or a typo."""
    if not query or not candidate:
        return None
    if query == candidate:
        return 0
    short, long = (query, candidate) if len(query) <= len(candidate) else (candidate, query)
    if len(short) >= 5 and long.startswith(short) and len(long) - len(short) <= 3:
        return 0
    if len(short) < 6:
        return None
    limit = 1 if min(len(query), len(candidate)) < 8 else 2
    if abs(len(query) - len(candidate)) > limit:
        return None
    distance = Levenshtein.distance(query, candidate, score_cutoff=limit)
    if distance > limit:
        return None
    return distance


def _dictionary_path() -> Path | None:
    override = os.environ.get("DRUG_DICTIONARY_PATH", "").strip()
    if override:
        path = Path(override)
        return path if path.is_file() else None
    candidate = Path(__file__).resolve().parents[2] / "data" / "drug_classes.json"
    return candidate if candidate.is_file() else None


class DrugIndex:
    def __init__(self, entries: tuple[dict, ...]) -> None:
        self.entries = entries
        self._forms: list[tuple[int, str]] = []
        self._exact: dict[str, list[int]] = defaultdict(list)
        self._postings: dict[str, list[int]] = defaultdict(list)
        seen_exact: dict[str, set[int]] = defaultdict(set)
        for entry_index, entry in enumerate(entries):
            for name in entry["names"]:
                folded = fold(name)
                if len(folded) < 3:
                    continue
                form_index = len(self._forms)
                stemmed = stem(folded)
                self._forms.append((entry_index, folded))
                for key in {folded, stemmed}:
                    if len(key) < 3 or entry_index in seen_exact[key]:
                        continue
                    seen_exact[key].add(entry_index)
                    self._exact[key].append(entry_index)
                grams = set(_trigrams(folded))
                if stemmed != folded:
                    grams.update(_trigrams(stemmed))
                for gram in grams:
                    self._postings[gram].append(form_index)

    def _query_forms(self, question: str) -> list[str]:
        found: list[str] = []
        seen: set[str] = set()
        for raw in _WORD.findall(fold(question)):
            if len(raw) < 4 or raw in _STOP:
                continue
            stemmed = stem(raw)
            for form in (raw, stemmed):
                if len(form) < 4 or form in _STOP or form in seen:
                    continue
                seen.add(form)
                found.append(form)
        return found

    def _fuzzy_entries(self, token: str) -> list[int]:
        grams = _trigrams(token)
        if not grams:
            return []
        ordered = sorted(grams, key=lambda gram: len(self._postings.get(gram, ())))
        usable = [gram for gram in ordered if self._postings.get(gram)]
        counts: Counter[int] = Counter()
        for gram in usable[:4]:
            for form_index in self._postings.get(gram, ()):
                counts[form_index] += 1
        if not counts:
            return []
        best: list[tuple[int, int]] = []
        for form_index, _shared in counts.most_common(48):
            _entry_index, form = self._forms[form_index]
            distance = _close(token, form)
            if distance is None:
                continue
            best.append((distance, _entry_index))
        if not best:
            return []
        closest = min(distance for distance, _entry in best)
        chosen: list[int] = []
        for distance, entry_index in best:
            if distance != closest or entry_index in chosen:
                continue
            chosen.append(entry_index)
            if len(chosen) == _MAX_FUZZY_ENTRIES:
                break
        return chosen

    def match(self, question: str) -> list[dict]:
        if not self.entries:
            return []
        hit: list[int] = []
        seen: set[int] = set()
        for form in self._query_forms(question):
            exact = self._exact.get(form, ())
            found = list(exact) if exact else self._fuzzy_entries(form)
            for entry_index in found:
                if entry_index in seen:
                    continue
                seen.add(entry_index)
                hit.append(entry_index)
        return [self.entries[index] for index in hit]


def _load_entries() -> tuple[dict, ...]:
    path = _dictionary_path()
    if path is None:
        return ()
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for entry in payload.get("entries", []):
        names = tuple(dict.fromkeys(fold(name) for name in entry.get("names", []) if name))
        if names:
            rows.append({
                "group": fold(str(entry.get("group", ""))),
                "atc": str(entry.get("atc", "")),
                "names": names,
            })
    return tuple(rows)


@lru_cache(maxsize=1)
def drug_index() -> DrugIndex:
    return DrugIndex(_load_entries())


def matched_entries(question: str) -> list[dict]:
    return drug_index().match(question)


def drug_names(question: str) -> set[str]:
    """Names on the same dictionary entry as a name in the question."""
    names: set[str] = set()
    for entry in matched_entries(question):
        names.update(entry["names"])
    return {name for name in names if len(name) >= 3}
