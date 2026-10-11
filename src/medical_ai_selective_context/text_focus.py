"""Say the chart row the question's own words point at.

Rows are chosen by section and by lexical overlap, not by a list of chart
values. The number, date, and flag word stay verbatim. A missing lab is said
to be missing. A last prescription is not described as current use.
"""

from __future__ import annotations

import re

from .index import normalize_text
from .text_document import TextRecord
from .text_grounding import GroundingResult, content_words, ground_answer
from .text_relevance import (
    asks_labs,
    asks_panel,
    drug_names,
    is_continuation,
    lab_rows,
    matched_lab_rows,
    overlaps,
    question_terms,
    select_rows,
)

_CLAUSE = re.compile(r"(?<!\d[.;])(?<=[.;])\s+")
_EFFECT = ("azaldi mi", "artti mi", "ise yaradi", "etkisi oldu", "dusurdu mu", "fayda etti")
_UNKNOWN_Q = ("yapildi mi", "geldi mi", "basladi mi", "basladim mi", "uygulandi mi", "dustu mu")
_GAP = (
    "bilinmiyor", "kayitta yok", "kaydi yok", "henuz",
    "yapilip yapilmad", "uygulanip uygulanmad",
)
_WHY = (
    "neden olabilir", "neden agriyor", "neden oluyor", "neden ola",
    "niye", "sebeb",
)


def _folded(text: str) -> str:
    return normalize_text(text)


def _parts(text: str) -> list[str]:
    return [part.strip(" -") for part in _CLAUSE.split(text) if part.strip(" -")]


def _line_has(line: str, needles: tuple[str, ...]) -> bool:
    folded = _folded(line)
    return all(needle in folded or needle in line for needle in needles)


def _piece(line: str, needles: tuple[str, ...]) -> str:
    text = line.strip().lstrip("-").strip()
    if len(text) <= 180 and _line_has(text, needles):
        return text
    parts = _parts(text)
    hits = [part for part in parts if _line_has(part, needles)]
    if not hits:
        for index in range(1, len(parts)):
            window = f"{parts[index - 1]} {parts[index]}"
            if _line_has(window, needles):
                hits.append(window)
    if not hits:
        return text if _line_has(text, needles) else ""
    return min(hits, key=len)


def _question_hits_line(question: str, line: str) -> bool:
    if overlaps(line, question_terms(question)):
        return True
    names = drug_names(question, expand_group=False)
    return bool(names) and overlaps(line, names)


def _unknown_line(record: TextRecord, question: str, *, prefer_gap: bool) -> str | None:
    words = [word for word in content_words(question) if len(word) >= 6]
    best: tuple[int, int, str] | None = None
    for line in record.lines:
        if line.text.startswith("##"):
            continue
        folded = _folded(line.text)
        section = _folded(record.sections[line.section_index].heading)
        gap = "bilinmeyen" in section or any(token in folded for token in _GAP)
        planned = "planlandi" in folded
        if prefer_gap and not gap:
            continue
        if not gap and not planned:
            continue
        if not _question_hits_line(question, line.text):
            continue
        score = sum(1 for word in words if word[:6] in folded)
        if overlaps(line.text, drug_names(question, expand_group=False)):
            score += 2
        if gap:
            score += 1
        if score <= 0:
            continue
        text = line.text.strip().lstrip("-").strip()
        if len(text) > 240:
            text = _piece(text, tuple(word[:6] for word in words[:2])) or text
        ranked = (score, -len(text), text)
        if best is None or ranked > best:
            best = ranked
    return None if best is None else best[2]


def _same_date(date: str, answer: str) -> bool:
    if date in answer:
        return True
    match = re.fullmatch(r"0?(\d{1,2})\.0?(\d{1,2})\.(\d{4})", date)
    if not match:
        return date in _folded(answer)
    day, month, year = (int(part) for part in match.groups())
    return re.search(rf"0?{day}\.0?{month}\.{year}", answer) is not None


def _has_all(answer: str, needles: tuple[str, ...]) -> bool:
    folded = _folded(answer)
    for needle in needles:
        if not needle:
            continue
        if needle in answer or needle in folded or _same_date(needle, answer):
            continue
        return False
    return True


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


def _asks(question: str, phrases: tuple[str, ...]) -> bool:
    folded = _folded(question)
    return any(phrase in folded for phrase in phrases)


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
_FALSE_NORMAL = (
    "diger tahliller normal", "digerleri normal", "diger sonuclar normal",
    "kalan tahliller normal", "hepsi normal",
)
_NOTE = "Hekiminle konuşman iyi olur."
_BANNED_OPENER = "Kayıtta şunu görüyorum"


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


def _stale_use_sentence(record: TextRecord, question: str) -> str | None:
    """A last prescription date is not proof the medicine is still being taken."""
    if not is_continuation(question):
        return None
    names = drug_names(question, expand_group=False)
    if not names:
        return None
    rows = select_rows(record, question)
    if any(token in _folded(row) for row in rows for token in ("kesil", "birak", "kullanmad", "kullaniyor", "halen")):
        return None
    rows = [row for row in rows if _SON_RE.search(row) or _RX_RE.search(row)]
    if not rows:
        return None
    blob = rows[0]
    folded = _folded(blob)
    if any(token in folded for token in ("kesil", "birak", "kullanmad", "kullaniyor", "halen")):
        return None
    match = _SON_RE.search(blob) or _FULL_DATE.search(blob)
    if not match:
        return None
    date = match.group(1) if match.lastindex else match.group(0)
    branded = _RX_RE.search(blob)
    product = branded.group(1).split()[0] if branded else ""
    asked = max(names, key=len)
    product_fold = _folded(product)
    if product and asked not in product_fold and product_fold not in asked:
        shown = asked.upper() if len(asked) <= 4 else asked
        label = f"{shown} ilacının son reçetesi {date} ({product})"
    else:
        label = f"{product or asked} son reçetesi {date}"
    return f"Kayıtta {label}; güncel kullanımı kayıtta görünmüyor."


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


def _prefix(text: str, token: str) -> bool:
    return re.search(rf"(?<![a-z]){re.escape(token)}", text) is not None


def _date_key(text: str) -> tuple[int, int, int]:
    dates = _FULL_DATE.findall(text)
    if not dates:
        return (0, 0, 0)
    day, month, year = (int(part) for part in dates[-1].split("."))
    return (year, month, day)


def _followup_line(record: TextRecord, question: str) -> str | None:
    folded = _folded(question)
    if "ne zaman" not in folded:
        return None
    if "kontrol" not in folded and "randevu" not in folded:
        return None
    ranked: list[tuple[int, int, str]] = []
    for order, line in enumerate(record.lines):
        if line.text.startswith("##"):
            continue
        text = line.text.strip().lstrip("-").strip()
        folded_line = _folded(text)
        if "kontrol" not in folded_line and not _prefix(folded_line, "randevu"):
            continue
        if not _FULL_DATE.search(text):
            continue
        if _prefix(folded_line, "randevu") or _prefix(folded_line, "hatirlatma"):
            rank = 0
        elif "sonra" in folded_line and "kontrol" in folded_line:
            rank = 1
        else:
            rank = 2
        ranked.append((rank, order, text))
    if not ranked:
        return None
    best = min(item[0] for item in ranked)
    pool = [text for rank, _order, text in ranked if rank == best]
    text = max(pool, key=_date_key)
    return _follow_piece(text)


def _follow_piece(text: str) -> str:
    if len(text) <= 200:
        return _frame(text)
    parts = _parts(text) or [text]
    folded_parts = [(part, _folded(part)) for part in parts]
    preferred = [
        part for part, folded in folded_parts if "sonra" in folded and "kontrol" in folded
    ]
    if not preferred:
        preferred = [
            part for part, folded in folded_parts
            if _prefix(folded, "randevu") or "hatirlatma" in folded
        ]
    if not preferred:
        preferred = [
            part for part, folded in folded_parts if _prefix(folded, "kontrol")
        ]
    piece = min(preferred or parts, key=len)
    if not _FULL_DATE.search(piece):
        date = _FULL_DATE.findall(text)
        if date:
            piece = f"{date[0]} {piece}"
    return _frame(piece)


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
    piece = piece.replace(_BANNED_OPENER + ":", "").replace(_BANNED_OPENER, "").strip()
    piece = _clip(piece)
    if not piece:
        return ""
    if piece[-1] not in ".!?":
        piece += "."
    return piece


def clinician_note(question: str) -> str:
    """A gentle sen-form nudge. Appended after grounding, so it adds no number."""
    if _asks(question, _WHY):
        return _NOTE
    return ""


def _prefer_spoken(record: TextRecord, status: str, raw_line: str) -> GroundingResult | None:
    spoken = _speak_clause(raw_line)
    forced = _from_lines(record, status, [spoken])
    anchor = raw_line.strip().lstrip("-").strip()
    if forced is not None and (len(anchor) > 80 or anchor.rstrip(".") in forced.answer or anchor[:40] in forced.answer):
        return forced
    return _from_lines(record, status, [anchor])


def _strip_template(answer: str) -> str:
    cleaned = answer.replace(_BANNED_OPENER + ":", " ").replace(_BANNED_OPENER, " ")
    return "\n".join(" ".join(part.split()) for part in cleaned.splitlines() if part.strip())


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
    abnormal = any(
        _FLAG_RE.search(line) and _flag_word(line) != "normal" for line in lab_rows(record)
    )
    kept = []
    for sentence in _sentences(answer):
        folded = _folded(sentence)
        if "normal" not in folded:
            kept.append(sentence)
            continue
        if abnormal and any(phrase in folded for phrase in _FALSE_NORMAL):
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
    names = drug_names(question, expand_group=is_continuation(question))
    if overlaps(sentence, [*terms, *names]):
        return True
    for row in selected:
        row_terms = [token for token in question_terms(row) if len(token) >= 5]
        if row_terms and overlaps(sentence, row_terms):
            return True
    return False


def _drop_off_topic(answer: str, question: str, selected: list[str]) -> str:
    if not selected and not question_terms(question):
        return answer
    kept = [sentence for sentence in _sentences(answer) if _on_topic(sentence, question, selected)]
    return "\n".join(kept)


def _drop_outside_labs(answer: str, record: TextRecord) -> str:
    blob = "\n".join(lab_rows(record))
    folded_blob = _folded(blob)
    kept = []
    for sentence in _sentences(answer):
        folded = _folded(sentence)
        if any(token in folded and token not in folded_blob for token in ("hrv", "nabiz", "giyilebilir")):
            continue
        numbers = re.findall(r"\d+(?:,\d+)?", sentence)
        if numbers and any(token not in blob for token in numbers):
            continue
        kept.append(sentence)
    return "\n".join(kept)


def _absent_labs(record: TextRecord, question: str) -> GroundingResult | None:
    if not asks_labs(question) or asks_panel(question) or matched_lab_rows(record, question):
        return None
    sentence = "Laboratuvar bölümünde bu soruyla ilgili bir tahlil sonucu yok."
    citations = []
    for section in record.sections:
        if "laboratuvar" in _folded(section.heading):
            citations.append({
                "section": section.heading,
                "startLine": section.start_line,
                "endLine": section.start_line,
                "quote": section.heading,
                "marker": section.heading,
            })
            break
    return GroundingResult(True, "answered", sentence, citations, "")


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
    return _absent_labs(record, question)


def _same_fact(left: str, right: str) -> bool:
    folded_left, folded_right = _folded(left), _folded(right)
    return folded_left in folded_right or folded_right in folded_left


def _collect_rows(record: TextRecord, question: str, base: str) -> list[str]:
    extra: list[str] = []
    for row in select_rows(record, question):
        if row not in extra and not _covered(base, row):
            extra.append(row)
    follow = _followup_line(record, question)
    if follow and follow not in extra and not _covered(base, follow):
        extra.append(follow)
    stale = _stale_use_sentence(record, question)
    if stale and stale not in extra and "güncel kullanımı kayıtta görünmüyor" not in base:
        extra.append(stale)
    unique: list[str] = []
    for line in extra:
        if line and not any(_same_fact(line, kept) for kept in unique):
            unique.append(line)
    return unique[:4]


def align_answer(record: TextRecord, question: str, grounded: GroundingResult) -> GroundingResult:
    """Keep the sentences that match the question, and add the overlapping chart rows."""
    effect = _asks(question, _EFFECT)
    unknown_q = _asks(question, _UNKNOWN_Q)
    if effect or unknown_q:
        unknown = _unknown_line(record, question, prefer_gap=True)
        if unknown and _question_hits_line(question, unknown):
            forced = _prefer_spoken(record, "insufficient_evidence", unknown)
            if forced is not None:
                return forced
            return grounded

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
    if asks_labs(question):
        base = _drop_outside_labs(base, record)
    absent = _absent_labs(record, question)
    if absent is not None and not base.strip():
        return absent
    extra = _collect_rows(record, question, base)
    original = grounded.answer if grounded.ok else ""
    if not extra and base.strip() == original.strip() and not _chartish(original):
        if _generic(original) or not grounded.ok:
            unknown = _unknown_line(record, question, prefer_gap=False)
            if unknown and _question_hits_line(question, unknown):
                forced = _prefer_spoken(record, "insufficient_evidence", unknown)
                if forced is not None:
                    return forced
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
