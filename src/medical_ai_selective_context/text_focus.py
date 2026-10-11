"""Keep answer sentences that share the question's words, and add those chart rows.

Rows come from lexical overlap. The number, date, and flag word stay verbatim.
A prescription date is not rewritten into a claim about current use.
"""

from __future__ import annotations

import re

from .index import normalize_text
from .text_document import TextRecord
from .text_grounding import GroundingResult, content_words, ground_answer
from .text_relevance import (
    drug_names,
    matched_lab_rows,
    overlaps,
    question_terms,
    select_rows,
)

_CLAUSE = re.compile(r"(?<!\d[.;])(?<=[.;])\s+")


def _folded(text: str) -> str:
    return normalize_text(text)


def _parts(text: str) -> list[str]:
    return [part.strip(" -") for part in _CLAUSE.split(text) if part.strip(" -")]


def _from_lines(record: TextRecord, status: str, lines: list[str]) -> GroundingResult | None:
    body = "\n".join(line.strip() for line in lines if line and line.strip())
    if not body:
        return None
    token = "DURUM: YANIT" if status == "answered" else "DURUM: BİLGİ_YOK"
    grounded = ground_answer(f"{token}\n{body}", record)
    return grounded if grounded.ok else None


def _stack(record: TextRecord, status: str, parts: list[str]) -> GroundingResult | None:
    current: GroundingResult | None = None
    built: list[str] = []
    for part in parts:
        cleaned = part.strip()
        if not cleaned or cleaned in built or any(_same_fact(cleaned, kept) for kept in built):
            continue
        trial = _from_lines(record, status, [*built, cleaned])
        if trial is None:
            continue
        built.append(cleaned)
        current = trial
    return current


def _sentences(answer: str) -> list[str]:
    return [part.strip() for part in re.split(r"\n+|(?<=[.!?])\s+", answer) if part.strip()]


def _drop_echo(answer: str, question: str) -> str:
    folded_q = _folded(question).strip(" ?.")
    kept = []
    for sentence in _sentences(answer):
        folded = _folded(sentence).strip(" ?.")
        echo = folded == folded_q or (folded_q in folded and len(content_words(sentence)) <= 4)
        if echo:
            continue
        kept.append(sentence)
    return "\n".join(kept)


def _generic(answer: str) -> bool:
    folded = _folded(answer)
    return "bu bilgi kayitta yok" in folded or (
        "kayitta yok" in folded and len(content_words(answer)) <= 2
    )


_FULL_DATE = re.compile(r"\d{1,2}\.\d{1,2}\.\d{4}")
_DUMP_MARK = (
    "epikriz", "sikayet", "hikaye", "bulgular", "ozgecmis", "ilk recete",
    "bilinmeyen cumle", "seyrindi", "kayittaki cumle", "kayittaki ilgili satir",
)
_BROKEN = ("bilinmeyen cumle", "seyrindi", "kayittaki cumle", "kayittaki ilgili satir")
_WARM = ("yaziyor", "goruyorum", "kaydinda", "recetende", "hekiminle")
_RX_RE = re.compile(
    r"([A-ZÇĞİÖŞÜ][A-ZÇĞİÖŞÜ0-9](?:[A-ZÇĞİÖŞÜ0-9 .'/,+-]{0,60}?)"
    r"(?:TABLET|KAPSÜL|DAMLA))"
)
_SON_RE = re.compile(r"son reçete\s+(\d{1,2}\.\d{1,2}\.\d{4}|\d{1,2}\.\d{4})")
_FLAG_RE = re.compile(r",\s*(yüksek|düşük|normal)\b", re.IGNORECASE)
_HEAD_RE = re.compile(r"([^\n]{0,140}?\(\d{1,2}\.\d{1,2}\.\d{4}\))")
_HEADER_RE = re.compile(
    r"\b(?:Şikâyet|Şikayet|ŞİKAYET|Hikâye|Hikaye|HİKAYE|HİKÂYE|Bulgular|BULGULAR|"
    r"Epikriz|Rapor|Özgeçmiş|ÖZGEÇMİŞ)\s*:\s*"
)
_EXAM_HEAD = re.compile(
    r"(?i)\b(?:ŞİKAYET|Şikayet|Şikâyet|HİKAYE|Hikaye|Hikâye|HİKÂYE|"
    r"BULGULAR|Bulgular|ÖZGEÇMİŞ|Özgeçmiş)\s*:"
)
_GENERAL = ("genellikle", "cogu hastada", "tipik olarak", "herkeste", "bilimsel olarak")
def _chartish(text: str) -> bool:
    folded = _folded(text)
    if any(token in folded for token in _DUMP_MARK):
        return True
    stripped = text.strip()
    if stripped.startswith("-"):
        return True
    if any(token in folded for token in _WARM):
        return False
    if any(token in text for token in ("TABLET", "KAPSÜL", "DAMLA")):
        return True
    if "beyan" in folded or "son recete" in folded or "—" in text:
        return True
    if _FLAG_RE.search(text):
        return True
    if "önceki" in text and _FULL_DATE.search(text):
        return True
    return len(stripped) > 180 and len(re.findall(r"\d", stripped)) >= 8


def _drop_broken(answer: str) -> str:
    kept = []
    for sentence in _sentences(answer):
        folded = _folded(sentence)
        if any(token in folded for token in _BROKEN):
            continue
        kept.append(sentence)
    return "\n".join(kept)


def _deheader(text: str) -> str:
    return " ".join(_HEADER_RE.sub("", text).split())


def _exam_focus(text: str) -> str:
    """Keep the complaint sentence. Drop the hikaye, bulgular, and preamble blobs."""
    if not _EXAM_HEAD.search(text):
        return text
    spans = list(_EXAM_HEAD.finditer(text))
    segments: list[tuple[str, str]] = []
    for index, match in enumerate(spans):
        end = spans[index + 1].start() if index + 1 < len(spans) else len(text)
        body = " ".join(text[match.end():end].split()).strip(" .;")
        segments.append((_folded(match.group(0)), body))
    preferred = [body for label, body in segments if "sikayet" in label and body]
    if not preferred:
        preferred = [body for label, body in segments if "hikaye" in label and body]
    chosen = preferred[0] if preferred else next((body for _label, body in segments if body), text)
    if len(chosen) > 180:
        chosen = chosen[:180].rsplit(" ", 1)[0]
    return chosen


def _drop_invented_stop(answer: str, record: TextRecord) -> str:
    """'İlaç kesildi' stays only when the same chart line says that drug was stopped."""
    kept = []
    for sentence in _sentences(answer):
        folded = _folded(sentence)
        stopped = any(token in folded for token in ("kesil", "birakildi", "birakti"))
        denied = "kullanmiyor" in folded or "kullanmiyorum" in folded
        if not stopped and not denied:
            kept.append(sentence)
            continue
        words = [
            word for word in content_words(sentence)
            if len(word) >= 5 and not word.startswith("kesil") and not word.startswith("birak")
            and not word.startswith("kullanmi")
        ]
        backed = False
        for line in record.lines:
            line_folded = _folded(line.text)
            if stopped and "kesil" not in line_folded and "birak" not in line_folded:
                continue
            if denied and "kullanmiyor" not in line_folded and "kullanmiyorum" not in line_folded:
                continue
            if not words or any(word[:6] in line_folded for word in words):
                backed = True
                break
        if backed:
            kept.append(sentence)
    return "\n".join(kept)


def _drop_general(answer: str, record: TextRecord) -> str:
    chart = _folded(record.text)
    kept = []
    for sentence in _sentences(answer):
        folded = _folded(sentence)
        if any(phrase in folded and phrase not in chart for phrase in _GENERAL):
            continue
        kept.append(sentence)
    return "\n".join(kept)


def _dated_head(text: str) -> str:
    match = _HEAD_RE.search(text.lstrip("- "))
    if not match:
        return ""
    return " ".join(match.group(1).split())


def _prev_span(text: str) -> str:
    match = re.search(r"önceki\s+.+?(?=;|\n|$)", text)
    if not match:
        return ""
    return " ".join(match.group(0).split()).rstrip(".")


def _flag_word(text: str) -> str:
    match = _FLAG_RE.search(text)
    return match.group(1) if match else ""


def _shrink_clause(text: str) -> str:
    parts = _parts(text) or [text]
    best = max(parts, key=lambda part: (len(re.findall(r"\d", part)), -len(part)))
    if not _FULL_DATE.search(best):
        index = parts.index(best)
        if index > 0 and _FULL_DATE.search(parts[index - 1]):
            best = f"{parts[index - 1].rstrip('.')} {best}"
    return best


def _speak_rx(text: str) -> str:
    if not any(token in text for token in ("TABLET", "KAPSÜL", "DAMLA")):
        return ""
    match = _RX_RE.search(text)
    if not match:
        return ""
    name = " ".join(match.group(1).split())
    son = _SON_RE.search(text)
    sentence = f"Reçetende {name} görünüyor"
    if son:
        sentence += f"; son reçete {son.group(1)}"
    return sentence + "."


def _speak_lab(text: str) -> str:
    flag = _flag_word(text)
    prev = _prev_span(text)
    if not flag and not (prev and _FULL_DATE.search(text)):
        return ""
    head = _dated_head(text)
    if not head or len(head) > 180 or "önceki" in head:
        return ""
    first = f"{head} kaydında {flag} diye geçiyor." if flag else f"Kayıtta {head} yazıyor."
    hedef = re.search(r"hedef\s+[^.;)]{0,48}", text)
    if hedef:
        goal = " ".join(hedef.group(0).split())
        if goal not in first:
            first = f"{first[:-1]}; kayıtta {goal} diyor."
    if not prev:
        return first
    return f"{first} Kayıtta ayrıca şu da yazıyor: {prev}."


def _speak_clause(text: str) -> str:
    raw = " ".join(_exam_focus(text).strip().lstrip("-").split())
    if not raw:
        return ""
    if not _chartish(raw) and len(raw) <= 360:
        return raw if raw[-1] in ".!?" else f"{raw}."
    spoken = _speak_rx(raw) or _speak_lab(raw)
    if spoken:
        return spoken
    clause = raw if len(raw) <= 320 else _shrink_clause(raw)
    return _frame(clause)


def _clip(piece: str, limit: int = 220) -> str:
    if len(piece) <= limit:
        return piece
    window = piece[:limit]
    for separator in (". ", "; "):
        at = window.rfind(separator)
        if at >= 40:
            return window[:at].rstrip(" ;,")
    return window.rsplit(" ", 1)[0].rstrip(" ;,")


def _frame(piece: str) -> str:
    piece = _deheader(piece).strip().rstrip(";").strip()
    piece = _clip(piece)
    if not piece:
        return ""
    if piece[-1] not in ".!?":
        piece += "."
    return piece


def _strip_template(answer: str) -> str:
    return "\n".join(" ".join(part.split()) for part in answer.splitlines() if part.strip())


def _normal_backed(sentence: str, record: TextRecord) -> bool:
    numbers = re.findall(r"\d+(?:,\d+)?", sentence)
    if not numbers:
        return False
    for line in record.lines:
        if "normal" not in _folded(line.text):
            continue
        if any(number in line.text for number in numbers):
            return True
    return False


def _drop_false_normal(answer: str, record: TextRecord) -> str:
    kept = []
    for sentence in _sentences(answer):
        folded = _folded(sentence)
        if "normal" not in folded:
            kept.append(sentence)
            continue
        if _normal_backed(sentence, record):
            kept.append(sentence)
            continue
        cleaned = re.sub(r"(?i)\s*,?\s*normal\b", "", sentence)
        cleaned = " ".join(cleaned.split()).strip(" ,;")
        if not cleaned:
            continue
        if cleaned[-1] not in ".!?":
            cleaned += "."
        kept.append(cleaned)
    return "\n".join(kept)


def _on_topic(sentence: str, question: str, selected: list[str]) -> bool:
    terms = question_terms(question)
    names = drug_names(question)
    if overlaps(sentence, [*terms, *names]):
        return True
    for row in selected:
        row_terms = [token for token in question_terms(row) if len(token) >= 5]
        if row_terms and overlaps(sentence, row_terms):
            return True
    return False


def _drop_off_topic(answer: str, question: str, selected: list[str]) -> str:
    kept = [sentence for sentence in _sentences(answer) if _on_topic(sentence, question, selected)]
    return "\n".join(kept)


def _covered(answer: str, row: str) -> bool:
    if not answer.strip():
        return False
    terms = [token for token in question_terms(row) if len(token) >= 5]
    if terms and not overlaps(answer, terms):
        return False
    numbers = [token for token in re.findall(r"\d+(?:,\d+)?", row) if not re.fullmatch(r"20\d{2}", token)]
    if numbers and not any(token in answer for token in numbers):
        return False
    return bool(terms or numbers)


def _short_lab(line: str) -> str:
    text = " ".join(line.strip().lstrip("-").split())
    text = re.sub(r"\((?:referans|ref)[^)]*\)", "", text, flags=re.IGNORECASE)
    head = text.split(";")[0].strip().rstrip(".")
    if not head:
        return ""
    return head if head[-1] in ".!?" else f"{head}."


def lab_fallback(record: TextRecord, question: str) -> GroundingResult | None:
    """A short sentence from the lab rows the question's words match."""
    rows = matched_lab_rows(record, question)
    if rows:
        sentences = [spoken for row in rows if (spoken := _short_lab(row))]
        stacked = _stack(record, "answered", sentences)
        if stacked is not None:
            return stacked
    return None


def _same_fact(left: str, right: str) -> bool:
    folded_left, folded_right = _folded(left), _folded(right)
    return folded_left in folded_right or folded_right in folded_left


def _collect_rows(record: TextRecord, question: str, base: str) -> list[str]:
    extra: list[str] = []
    for row in select_rows(record, question):
        if row not in extra and not _covered(base, row):
            extra.append(row)
    unique: list[str] = []
    for line in extra:
        if line and not any(_same_fact(line, kept) for kept in unique):
            unique.append(line)
    return unique[:4]


def align_answer(record: TextRecord, question: str, grounded: GroundingResult) -> GroundingResult:
    """Keep sentences that share the question's words, and add overlapping chart rows."""
    selected = select_rows(record, question)
    base = _strip_template(_drop_false_normal(
        _drop_off_topic(
            _drop_general(
                _drop_invented_stop(
                    _drop_broken(_drop_echo(grounded.answer if grounded.ok else "", question)),
                    record,
                ),
                record,
            ),
            question,
            selected,
        ),
        record,
    ))
    extra = _collect_rows(record, question, base)
    original = grounded.answer if grounded.ok else ""
    if not extra and not base.strip():
        fallback = lab_fallback(record, question)
        if fallback is not None:
            return fallback
        if not original.strip():
            return grounded
        return GroundingResult(False, "insufficient_evidence", "", [], "off_topic")
    if not extra and base.strip() == original.strip() and not _chartish(original):
        if _generic(original) or not grounded.ok:
            fallback = lab_fallback(record, question)
            if fallback is not None:
                return fallback
        return grounded
    parts = [
        _speak_clause(base) if base.strip() and _chartish(base) else base,
        *(_speak_clause(line) for line in extra),
    ]
    merged = _stack(record, "answered", parts)
    if merged is not None:
        return merged
    raw = _stack(record, "answered", [base, *extra] if base.strip() else extra)
    if raw is not None:
        return raw
    fallback = lab_fallback(record, question)
    if fallback is not None:
        return fallback
    return grounded
