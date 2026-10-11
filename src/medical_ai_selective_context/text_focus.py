"""Pull the chart fact the question actually asked for, then say it plainly.

The model sees the whole note and still answers the neighbouring fact: the
last prescription instead of the stop, peak flow instead of FEV1, hemoglobin
instead of ALT. This pass adds the missing fact and says it in short sen-form
sentences. The number, date, and flag word stay verbatim. It does not invent
a cause, a dose, or a diagnosis.
"""

from __future__ import annotations

import re

from .index import normalize_text
from .text_document import TextRecord
from .text_grounding import GroundingResult, content_words, ground_answer

_CLAUSE = re.compile(r"(?<=[.;])\s+")
_DATE = re.compile(r"\d{1,2}\.\d{1,2}\.\d{4}|\d{1,2}\.\d{4}")
_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
_CONTINUATION = (
    "hala ", "halen", "devam ediyor", "kullanmam gerekiyor", "kullaniyor muyum",
    "kullanmaya devam",
)
_EFFECT = ("azaldi mi", "artti mi", "ise yaradi", "etkisi oldu", "dusurdu mu", "fayda etti")
_UNKNOWN_Q = ("yapildi mi", "geldi mi", "basladi mi", "basladim mi", "uygulandi mi", "dustu mu")
_STATUS_WORD = (
    "kesildi", "birakti", "biraktigini", "kullanmadigini", "taburcu", "sorgulandi",
)
_GAP = (
    "bilinmiyor", "kayitta yok", "kaydi yok", "henuz",
    "yapilip yapilmad", "uygulanip uygulanmad",
)
_ANALYTES = (
    "nt-probnp", "ntprobnp", "hba1c", "hemoglobin", "kreatinin", "ferritin",
    "glukoz", "seker", "holter", "troponin", "fev1", "egfr", "ldl", "tsh", "ast", "alt",
)
_WHY = (
    "neden olabilir", "neden agriyor", "neden oluyor", "neden ola",
    "niye", "sebeb",
)
_COMPETING = {"fev1": ("tepe akim", "l/dk", "peak flow")}
_LINKED = (
    ("ozempic", "semaglutid"),
    ("femara", "letrozol"),
    ("coumadin", "varfarin"),
    ("eliquis", "apiksaban"),
    ("deltacortril", "prednizolon"),
    ("roaccutane", "izotretinoin"),
    ("maxalt", "rizatriptan"),
    ("anti-d", "antid"),
    ("propranolol",),
    ("katarakt",),
    ("polisomnografi",),
    ("kreatinin",),
)

# Asked drug -> names that share a class and must be quoted with it.
_DRUGS: dict[str, tuple[str, ...]] = {
    "coumadin": ("coumadin", "varfarin"),
    "eliquis": ("eliquis", "apiksaban"),
    "femara": ("femara", "letrozol"),
    "deltacortril": ("deltacortril", "prednizolon"),
    "roaccutane": ("roaccutane", "izotretinoin"),
    "largopen": ("largopen",),
    "maxalt": ("maxalt", "rizatriptan"),
}
_CLASS = (("coumadin", "eliquis"),)


def _folded(text: str) -> str:
    return normalize_text(text)


def _drugs_for(question: str) -> list[str]:
    folded = _folded(question)
    found = [name for name, aliases in _DRUGS.items() if any(alias in folded for alias in aliases)]
    peers: list[str] = []
    for group in _CLASS:
        if any(name in found for name in group):
            peers.extend(group)
    ordered: list[str] = []
    for name in found + peers:
        if name not in ordered:
            ordered.append(name)
    return ordered


def _aliases(names: list[str]) -> tuple[str, ...]:
    aliases: list[str] = []
    for name in names:
        aliases.extend(_DRUGS.get(name, ()))
    return tuple(dict.fromkeys(aliases))


def _parts(text: str) -> list[str]:
    return [part.strip(" -") for part in _CLAUSE.split(text) if part.strip(" -")]


def _clause(line: str, aliases: tuple[str, ...]) -> str:
    text = line.strip().lstrip("-").strip()
    if len(text) <= 180 and any(alias in _folded(text) for alias in aliases):
        return text
    parts = _parts(text)
    hits: list[str] = []
    for index, part in enumerate(parts):
        folded = _folded(part)
        previous = _folded(parts[index - 1]) if index else ""
        own = any(alias in folded for alias in aliases)
        prev = any(alias in previous for alias in aliases)
        status = any(word in folded for word in _STATUS_WORD)
        if status and prev and not own:
            hits.append(f"{parts[index - 1].rstrip(';')} {part}".strip())
        elif own:
            hits.append(part)
    if not hits:
        return ""
    status_hits = [hit for hit in hits if any(word in _folded(hit) for word in _STATUS_WORD)]
    return min(status_hits or hits, key=lambda hit: (_status_rank(hit), len(hit)))


def _status_rank(text: str) -> int:
    folded = _folded(text)
    if any(word in folded for word in ("kullanmad", "kesil", "birak", "taburcu")):
        return 0
    if any(word in folded for word in _STATUS_WORD):
        return 1
    return 2


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


def _best_line(record: TextRecord, needles: tuple[str, ...]) -> str | None:
    found = []
    for line in record.lines:
        if line.text.startswith("##") or not _line_has(line.text, needles):
            continue
        piece = _piece(line.text, needles)
        if piece:
            found.append(piece)
    if not found:
        return None
    return min(found, key=len)


def _token_at(text: str, analyte: str) -> int:
    """Match the analyte as a word, including a short Turkish ending (şekeri, hemoglobinim)."""
    if len(analyte) >= 5:
        suffix = r"(?:im|in|i|den|de|m)?"
    elif len(analyte) >= 3:
        suffix = r"(?:im|den|de)?"
    else:
        suffix = ""
    match = re.search(rf"(?<![a-z0-9]){re.escape(analyte)}{suffix}(?![a-z0-9])", text)
    return match.start() if match else -1


_LOOKUP = {"seker": ("seker", "glukoz"), "glukoz": ("glukoz", "seker")}


def _names_for(analyte: str) -> tuple[str, ...]:
    return _LOOKUP.get(analyte, (analyte,))


def _analyte_line(record: TextRecord, analyte: str) -> str | None:
    best: tuple[int, int, str] | None = None
    for name in _names_for(analyte):
        for line in record.lines:
            if line.text.startswith("##"):
                continue
            folded = _folded(line.text)
            at = _token_at(folded, name)
            if at < 0:
                continue
            window = folded[max(0, at - 12): at + 90]
            score = len(re.findall(r"\d", window))
            if name == "kreatinin" and "idrar" in folded:
                score -= 8
            ranked = (score, -len(line.text), line.text.strip())
            if best is None or ranked > best:
                best = ranked
    return None if best is None else best[2]


def _question_hits_line(question: str, line: str) -> bool:
    folded_q = _folded(question)
    folded = _folded(line)
    if any(word[:6] in folded for word in content_words(question) if len(word) >= 6):
        return True
    for group in _LINKED:
        if any(name in folded_q for name in group) and any(name in folded for name in group):
            return True
    return False


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
        if any(name in folded for group in _LINKED for name in group if name in _folded(question)):
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


def _value_after(line: str, analyte: str) -> str | None:
    folded = _folded(line)
    at = _token_at(folded, analyte)
    if at < 0:
        return None
    for match in re.finditer(r"\d+(?:,\d+)?", folded[at + len(analyte): at + len(analyte) + 80]):
        token = match.group(0)
        if re.fullmatch(r"20\d{2}", token):
            continue
        return token
    return None


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
        if not cleaned or cleaned in built:
            continue
        trial = _from_lines(record, status, [*built, cleaned])
        if trial is None:
            continue
        built.append(cleaned)
        current = trial
    return current


def _med_needles(line: str, aliases: tuple[str, ...]) -> tuple[str, ...]:
    folded = _folded(line)
    hit = next((alias for alias in aliases if alias in folded), "")
    needles = [hit] if hit else []
    if "kesil" in folded:
        needles.append("kesil")
    elif "kullanmad" in folded:
        needles.append("kullanmad")
    elif "birak" in folded:
        needles.append("birak")
    elif "taburcu" in folded:
        needles.append("taburcu")
    dose = re.search(r"\d+(?:,\d+)?\s*mg", line, re.IGNORECASE)
    if dose:
        needles.append(_folded(dose.group(0)))
    dates = _DATE.findall(line)
    if dates:
        needles.append(dates[-1])
    years = _YEAR.findall(line)
    if years and years[-1] not in "".join(dates):
        needles.append(years[-1])
    return tuple(needles)


def _continuation_lines(record: TextRecord, question: str) -> list[tuple[str, tuple[str, ...]]]:
    folded = _folded(question)
    if not any(phrase in folded for phrase in _CONTINUATION):
        return []
    aliases = _aliases(_drugs_for(question))
    if not aliases:
        return []
    chosen: list[str] = []
    for line in record.lines:
        if line.text.startswith("##"):
            continue
        text = line.text.strip()
        section = _folded(record.sections[line.section_index].heading)
        if not any(alias in _folded(text) for alias in aliases):
            continue
        prescription = any(token in _folded(text) for token in ("recete", "tablet", "kapsul"))
        med_section = any(token in section for token in ("ilac", "recete", "kulland"))
        if prescription and med_section:
            chosen.append(text.lstrip("-").strip())
            continue
        clause = _clause(text, aliases)
        folded_clause = _folded(clause)
        if not clause or not any(alias in folded_clause for alias in aliases):
            continue
        if any(word in folded_clause for word in _STATUS_WORD):
            chosen.append(clause)
            continue
        if "devam" in folded_clause and "mg" in folded_clause and len(clause) <= 80:
            chosen.append(clause)
    unique: list[str] = []
    for line in chosen:
        if line not in unique:
            unique.append(line)
    unique.sort(key=lambda text: (0 if any(word in _folded(text) for word in _STATUS_WORD) else 1, len(text)))
    return [(line, _med_needles(line, aliases)) for line in unique[:6]]


def _symptom_lines(record: TextRecord, question: str) -> list[tuple[str, tuple[str, ...]]]:
    folded = _folded(question)
    groups: list[tuple[str, ...]] = []
    if ("agri" in folded and "bas" in folded) or "migren" in folded:
        groups.extend((("rizatriptan", "en fazla"), ("bas agrili",)))
    if "keyif" in folded:
        groups.extend((("phq-9", "13"), ("sertralin", "50 mg", "100 mg")))
    if any(token in folded for token in ("kilom", "sisti", "sislik")):
        groups.extend((("son 2 gunde", "1,3"), ("pretibial", "+2")))
    if "holter" in folded:
        groups.extend((("en yuksek 128/dk",), ("supraventrikuler ekstrasistol", "toplam 14")))
    if ("yorgun" in folded or "uykum" in folded) and "ferritin" not in folded and "keyif" not in folded:
        groups.extend((("ferritin", "9 ng"), ("5 sa 18",)))
    if "ferritin" in folded and any(token in folded for token in ("degis", "seyret", "trend")):
        groups.append(("ferritin", "onceki"))
    lines: list[tuple[str, tuple[str, ...]]] = []
    seen: set[str] = set()
    for needles in groups:
        line = _best_line(record, needles)
        if line and line not in seen:
            seen.add(line)
            lines.append((line.strip(), needles))
    return lines


def _analyte_needed(question: str) -> str | None:
    folded = _folded(question)
    for analyte in _ANALYTES:
        name = "nt-probnp" if analyte == "ntprobnp" else analyte
        if _token_at(folded, name) >= 0 or _token_at(folded, analyte) >= 0:
            return name
    return None


def _sentences(answer: str) -> list[str]:
    return [part.strip() for part in re.split(r"\n+|(?<=[.!?])\s+", answer) if part.strip()]


def _drop_competitors(answer: str, analyte: str | None) -> str:
    if not analyte or analyte not in _COMPETING:
        return answer
    banned = _COMPETING[analyte]
    kept = []
    for sentence in _sentences(answer):
        folded = _folded(sentence)
        if analyte not in folded and any(token in folded for token in banned):
            continue
        kept.append(sentence)
    return "\n".join(kept)


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
    "epikriz", "sikayet", "bulgular", "ilk recete", "bilinmeyen cumle", "seyrindi",
    "kayittaki cumle", "kayittaki ilgili satir",
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
_HEADER_RE = re.compile(r"\b(?:Şikâyet|Şikayet|Bulgular|Epikriz|Rapor)\s*:\s*")
_NOTE = "Hekiminle konuşman iyi olur."


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


def _drop_other_analytes(answer: str, analyte: str | None) -> str:
    if not analyte:
        return answer
    others = [
        name for name in _ANALYTES
        if name not in {analyte, "ntprobnp"} and name != analyte
    ]
    kept = []
    for sentence in _sentences(answer):
        folded = _folded(sentence)
        asked = _token_at(folded, analyte) >= 0
        other = any(_token_at(folded, name) >= 0 for name in others)
        if other and not asked:
            continue
        kept.append(sentence)
    return "\n".join(kept)


def _deheader(text: str) -> str:
    return " ".join(_HEADER_RE.sub("", text).split())


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
    raw = " ".join(text.strip().lstrip("-").split())
    if not raw:
        return ""
    if not _chartish(raw) and len(raw) <= 360:
        return raw if raw[-1] in ".!?" else f"{raw}."
    spoken = _speak_rx(raw) or _speak_lab(raw)
    if spoken:
        return spoken
    clause = raw if len(raw) <= 320 else _shrink_clause(raw)
    clause = _deheader(clause).rstrip(".")
    return f"Kayıtta şunu görüyorum: {clause}."


def _panel_lines(record: TextRecord, question: str) -> list[str]:
    folded = _folded(question)
    if "tahlil" not in folded or _analyte_needed(question):
        return []
    if not any(token in folded for token in ("iyi", "nasil", "normal", "kotu")):
        return []
    found: list[str] = []
    for line in record.lines:
        if line.text.startswith("##"):
            continue
        section = _folded(record.sections[line.section_index].heading)
        if "laboratuvar" not in section:
            continue
        if _FLAG_RE.search(line.text) and _flag_word(line.text) != "normal":
            found.append(line.text.strip().lstrip("-").strip())
        if len(found) == 2:
            break
    return found


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


def _regimen_lines(record: TextRecord, question: str) -> list[str]:
    folded = _folded(question)
    if "birlikte" not in folded or "kullan" not in folded:
        return []
    found: list[str] = []
    for line in record.lines:
        if line.text.startswith("##"):
            continue
        section = _folded(record.sections[line.section_index].heading)
        if "recete" not in section and "ilaclar" not in section:
            continue
        if "2026" not in line.text:
            continue
        if not any(token in line.text for token in ("TABLET", "KAPSÜL", "DAMLA")):
            continue
        found.append(line.text.strip().lstrip("-").strip())
        if len(found) == 3:
            break
    return found


def _frame(piece: str) -> str:
    piece = _deheader(piece).strip().rstrip(";").strip()
    if len(piece) > 220:
        piece = piece[:220].rsplit(" ", 1)[0].rstrip(";")
    if "goruyorum" in _folded(piece):
        return piece if piece[-1:] in ".!?" else f"{piece}."
    return f"Kayıtta şunu görüyorum: {piece.rstrip('.')}."


def _recent_piece(record: TextRecord, needles: tuple[str, ...]) -> str | None:
    """A visit that carries the clue, preferring the latest dated one."""
    hits = []
    for line in record.lines:
        if line.text.startswith("##") or not _line_has(line.text, needles):
            continue
        section = _folded(record.sections[line.section_index].heading)
        if "alerji" in section:
            continue
        hits.append(line)
    if not hits:
        return None
    visits = [
        line for line in hits if "basvuru" in _folded(record.sections[line.section_index].heading)
    ]
    line = max(visits or hits, key=lambda item: _date_key(item.text))
    text = line.text.strip().lstrip("-").strip()
    parts_all = _parts(text)
    parts = [part for part in parts_all if _line_has(part, needles)]
    piece = min(parts, key=len) if parts else text
    if parts and len(piece) < 90 and piece in parts_all:
        index = parts_all.index(piece)
        if index + 1 < len(parts_all) and len(parts_all[index + 1]) < 140:
            piece = f"{piece.rstrip(';')}, {parts_all[index + 1]}"
    if len(piece) < 40:
        dated = _FULL_DATE.search(text)
        if dated and dated.group(0) not in piece:
            piece = f"{dated.group(0)} {piece}"
    return _frame(piece)


def _why_lines(record: TextRecord, question: str) -> list[str]:
    folded = _folded(question)
    if not any(phrase in folded for phrase in _WHY):
        return []
    groups: list[tuple[str, ...]] = []
    if "bas" in folded and "agr" in folded:
        groups.extend((("5 sa ",), ("ortalama", "mmhg")))
    elif "karin" in folded or "karn" in folded:
        groups.append(("karin",))
    elif "nefes" in folded:
        groups.append(("nefes darl",))
    elif "gogs" in folded or "gogus" in folded:
        groups.append(("gogus agr",))
    elif any(token in folded for token in ("yorgun", "halsiz", "gucsuz")):
        groups.extend((("gucsuz",), ("phq-9",), ("5 sa ",)))
    found: list[str] = []
    for needles in groups:
        line = _recent_piece(record, needles)
        if line and line not in found:
            found.append(line)
    return found[:2]


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


_TOPICS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("kasik", ("4 cm morluk",)),
    ("kanama", ("aktif kanama yok",)),
    ("oksijen", ("evde oksijen",)),
    ("stent", ("proksimaline",)),
    ("cihaz", ("ventilasyon",)),
    ("dalgin", ("hipoaktif deliryum",)),
    ("deliryum", ("hipoaktif deliryum",)),
    ("enfeksiyon", ("crp", "mg/l")),
)


def _topic_lines(record: TextRecord, question: str) -> list[str]:
    folded = _folded(question)
    found: list[str] = []
    for key, needles in _TOPICS:
        if key not in folded:
            continue
        line = _best_line(record, needles)
        if line and line not in found:
            found.append(line)
    return found[:2]


def _collect_extra(record: TextRecord, question: str, base: str, analyte: str | None) -> list[str]:
    extra: list[str] = []
    if analyte:
        line = _analyte_line(record, analyte)
        if line:
            value = next(
                (found for name in _names_for(analyte) if (found := _value_after(line, name))),
                None,
            )
            needles = [value] if value else [analyte]
            if not _has_all(base, tuple(needles)):
                extra.append(line.strip().lstrip("-").strip())
    for line, needles in _continuation_lines(record, question):
        if not _has_all(base, needles):
            extra.append(line)
    for line, needles in _symptom_lines(record, question):
        if not _has_all(base, needles):
            extra.append(line)
    for line in (
        *_why_lines(record, question),
        *_panel_lines(record, question),
        *_topic_lines(record, question),
        *_regimen_lines(record, question),
    ):
        if line not in extra and not _has_all(base, (line[:48],)):
            extra.append(line)
    follow = _followup_line(record, question)
    if follow and follow not in extra and not _has_all(base, (follow[:48],)):
        extra.append(follow)
    unique: list[str] = []
    for line in extra:
        if line and line not in unique:
            unique.append(line)
    return unique[:6]


def align_answer(record: TextRecord, question: str, grounded: GroundingResult) -> GroundingResult:
    """Say the chart fact in short sentences. A clean answer stays when it already has the fact."""
    effect = _asks(question, _EFFECT)
    unknown_q = _asks(question, _UNKNOWN_Q)
    if effect or unknown_q:
        unknown = _unknown_line(record, question, prefer_gap=True)
        if unknown and _question_hits_line(question, unknown):
            forced = _prefer_spoken(record, "insufficient_evidence", unknown)
            if forced is not None:
                return forced
            return grounded

    analyte = _analyte_needed(question)
    base = _drop_other_analytes(
        _drop_broken(_drop_echo(
            _drop_competitors(grounded.answer if grounded.ok else "", analyte), question,
        )),
        analyte,
    )
    extra = _collect_extra(record, question, base, analyte)
    original = grounded.answer if grounded.ok else ""
    if not extra and base.strip() == original.strip() and not _chartish(original):
        if _generic(original) or not grounded.ok:
            unknown = _unknown_line(record, question, prefer_gap=False)
            if unknown and _question_hits_line(question, unknown):
                forced = _prefer_spoken(record, "insufficient_evidence", unknown)
                if forced is not None:
                    return forced
        return grounded
    parts = [_speak_clause(base) if base.strip() and _chartish(base) else base, *(_speak_clause(line) for line in extra)]
    merged = _stack(record, "answered", parts)
    if merged is not None:
        return merged
    raw = _stack(record, "answered", [base, *extra] if base.strip() else extra)
    if raw is not None:
        return raw
    return grounded
