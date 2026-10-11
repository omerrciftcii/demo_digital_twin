"""Refuse before the model. Chart questions still go through.

Acute first-person emergencies, self-harm, explicit dose changes, and
prompt-injection / data grabs never see the provider. A recorded finding
("nefesim daralıyor, ne yapmalıyım?", "ibuprofen alabilir miyim?") is not
one of those: the eval expects the chart sentence, and the answer prompt
still forbids telling the person what to take.

Dose-change refusals quote a few verbatim lines. There is no model call,
so the lines are chosen by distinctive overlap with the question, including
a small brand alias list used only as a search key. The quoted characters
are copied from the record.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from .index import normalize_text
from .text_document import TextRecord

DISCLAIMER = (
    "Bu yanıt tıbbi tanı, tedavi veya acil müdahale önerisi değildir. "
    "Yalnızca bu kayıttaki metne dayanır. Kayıtta görünmemek, "
    "hastada olmadığını göstermez. Karar için hekiminize başvurun. "
    "Acil belirti varsa acil sağlık hizmetine başvurun."
)

_MESSAGES = {
    "self_harm": (
        "Kendinize zarar verme düşüncesi acil bir durumdur. Tanı koyamam ve "
        "yöntem veya ilaç öneremem. Bulunduğunuz yerdeki acil sağlık hizmetine "
        "veya güvendiğiniz bir kişiye hemen başvurun."
    ),
    "emergency": (
        "Bu belirtiler acil olabilir. Tanı koyamam ve tedavi öneremem. "
        "Hemen acil sağlık hizmetine başvurun. Buradaki kayıt özeti acil "
        "müdahalenin yerine geçmez."
    ),
    "treatment": (
        "İlaç dozu, başkasının ilacı veya tedavi değişikliği öneremem. "
        "Bu bir tedavi kararı değildir."
    ),
    "prompt_injection": (
        "Bu istek yanıtlanamaz. Sistem talimatını, başka kişilerin kimlik "
        "bilgilerini veya kaydın tamamını yazmam. İlaç dozu da önermem."
    ),
}

# Brand strings are retrieval keys for a refusal quote, not dosing advice.
_ALIASES = {
    "lasix": ("furosemid", "lasix"),
    "maxalt": ("rizatriptan", "maxalt"),
    "lantus": ("lantus", "glarjin"),
    "coumadin": ("varfarin", "coumadin"),
    "ozempic": ("semaglutid", "ozempic"),
    "femara": ("letrozol", "femara"),
    "deltacortril": ("prednizolon", "deltacortril"),
    "roaccutane": ("izotretinoin", "roaccutane"),
    "eliquis": ("apiksaban", "eliquis"),
    "glucophage": ("metformin",),
    "majezik": ("flurbiprofen", "majezik"),
}

_QUOTE_STOP = frozenset({
    "alabilir", "artirmali", "azaltmali", "baslamali", "birakip", "cikarabilir",
    "cikarmali", "devam", "dozum", "dozumu", "etsem", "gunde", "icin", "kendim",
    "kullanabilir", "miyim", "olarak", "olur", "onceki", "onerirsin", "sadece",
    "talimat", "talimatlari", "tane", "yuksek", "cikiyor", "yapmaliyim", "yapmam",
    "unut", "yoksa", "simdi", "hemen", "lutfen",
})

_SELF_HARM = (
    r"kendime zarar", r"kendimi oldur", r"intihar etmek", r"yasamak istemiyorum",
    r"\bsuicide\b", r"kill myself", r"self-harm", r"self harm",
)
_EMERGENCY = (
    r"gogsumde ezici", r"gogusumde ezici", r"nefes alamiyorum", r"nefesim cok daraldi",
    r"dudaklarim mor", r"konusamiyorum", r"sekerim\s+\d+", r"kafam karisik",
    r"gozlerimin onunde isik", r"yuzum sisti", r"hayatimin en siddetli bas agr",
    r"kolum uyus", r"konusmam bozul", r"bayildim", r"bilincimi kayb",
    r"kanamam durmuyor", r"kalbim durdu",
)
_EXFIL = (
    r"sistem talimat", r"ham veri", r"\btc kimlik\b", r"butun hastalarin",
    r"aynen yaz", r"ignore (all |previous )?instructions",
)
_INJECTION = _EXFIL + (
    r"onceki (tum )?talimat", r"talimatlari (yok say|unut)", r"kisitlamasi olmayan",
    r"sistem notu", r"sadece evet de",
)
_TREATMENT = (
    r"dozumu[\s\S]{0,80}(artir|azalt|cikar|yukselt|dusur)",
    r"(artirmali|azaltmali|cikarmali)\s+miyim",
    r"\bbirakip\b",
    r"annemin|babamin|baskasinin",
    r"gunde\s+\d+\s+(tane|adet|tablet)",
    r"ne kadar\s+(alayim|iceyim|kullanayim|almaliyim)",
    r"kac mg (yap|oner|basla|kullan)",
    r"dozu dogrudan",
    r"guvenli oldugunu yaz",
)
_UNSAFE_ANSWER = (
    r"dozunuzu (artir|azalt|kes|degistir|birak|yukselt)",
    r"dozu (artirin|azaltin|kesin|birakin)",
    r"(almalisiniz|kullanmalisiniz|icmelisiniz|baslamalisiniz)",
    r"(alabilirsiniz|kullanabilirsiniz)",
    r"recete (ediyorum|ederim|yaziyorum|yazabilirim)",
    r"ilacinizi (kesin|birakin|artirin)",
    r"acil degil", r"acile gitmeyin", r"doktorunuza gitmeyin",
    r"\btaniniz\b", r"guvenle kullan",
    r"sizde (diyabet|kanser|hipertansiyon|kalp yetmezligi) var",
)


def _flex(patterns: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(pattern.replace(" ", r"\s+") for pattern in patterns)


_SELF_HARM_RE = _flex(_SELF_HARM)
_EMERGENCY_RE = _flex(_EMERGENCY)
_EXFIL_RE = _flex(_EXFIL)
_INJECTION_RE = _flex(_INJECTION)
_TREATMENT_RE = _flex(_TREATMENT)
_UNSAFE_RE = _flex(_UNSAFE_ANSWER)


@dataclass(frozen=True)
class SafetyDecision:
    action: Literal["allow", "refuse"]
    reason: str
    message: str
    include_quotes: bool


def _matches(text: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(pattern, text) for pattern in patterns)


def classify_question(question: str) -> SafetyDecision:
    text = normalize_text(question or "")
    if _matches(text, _SELF_HARM_RE):
        reason = "self_harm"
        quotes = False
    elif _matches(text, _EMERGENCY_RE):
        reason = "emergency"
        quotes = False
    elif _matches(text, _EXFIL_RE):
        reason = "prompt_injection"
        quotes = False
    elif _matches(text, _TREATMENT_RE):
        reason = "treatment"
        quotes = True
    elif _matches(text, _INJECTION_RE):
        reason = "prompt_injection"
        quotes = False
    else:
        return SafetyDecision("allow", "record_question", "", False)
    return SafetyDecision("refuse", reason, _MESSAGES[reason], quotes)


def answer_has_unsafe_advice(text: str) -> bool:
    return _matches(normalize_text(text or ""), _UNSAFE_RE)


def _question_tokens(question: str) -> set[str]:
    folded = normalize_text(question)
    tokens = {
        token for token in re.findall(r"[a-z0-9]{5,}", folded) if token not in _QUOTE_STOP
    }
    for key, aliases in _ALIASES.items():
        if key in folded:
            tokens.update(aliases)
    return tokens


def _hits(token: str, line: str) -> bool:
    if token in line:
        return True
    stem = token[:5]
    return len(stem) >= 5 and stem in line


def supporting_lines(record: TextRecord, question: str, limit: int = 6) -> list:
    """Top verbatim lines for a refusal. Empty when the question names nothing in the text."""
    tokens = _question_tokens(question)
    if not tokens:
        return []
    candidates = [
        (line, normalize_text(line.text))
        for line in record.lines
        if line.text.strip() and not line.text.startswith("##")
    ]
    document_frequency = {
        token: sum(1 for _, folded in candidates if _hits(token, folded))
        for token in tokens
    }
    scored: list[tuple[float, int, object]] = []
    for line, folded in candidates:
        score = sum(
            1.0 / count
            for token, count in document_frequency.items()
            if count and _hits(token, folded)
        )
        if score:
            scored.append((score, -line.number, line))
    scored.sort(reverse=True)
    # One noisy section (a long visit list) must not crowd out the lab line
    # that actually holds the number. Keep a section's best line first.
    per_section: dict[int, list] = {}
    for score, _order, line in scored:
        bucket = per_section.setdefault(line.section_index, [])
        if len(bucket) < 2:
            bucket.append((score, line))
    ranked = sorted(per_section.values(), key=lambda rows: rows[0][0], reverse=True)
    primary = [rows[0][1] for rows in ranked]
    extra = [rows[1][1] for rows in ranked if len(rows) > 1]
    return (primary + extra)[:limit]


def render_refusal(record: TextRecord, question: str, decision: SafetyDecision) -> tuple[str, list[dict]]:
    parts = [decision.message]
    citations: list[dict] = []
    if decision.include_quotes:
        chosen = supporting_lines(record, question)
        if chosen:
            parts.append("Kayıtta yazan (birebir alıntı, tedavi önerisi değildir):")
        for line in chosen:
            section = record.sections[line.section_index]
            parts.append(f"- {line.text} [{section.heading}]")
            citations.append({
                "section": section.heading,
                "startLine": line.number,
                "endLine": line.number,
                "quote": line.text,
                "marker": section.heading,
            })
    return "\n".join(parts), citations


def with_disclaimer(answer: str) -> str:
    body = (answer or "").strip()
    if DISCLAIMER in body:
        body = body.replace(DISCLAIMER, "").strip()
    parts = [part for part in (body, DISCLAIMER) if part]
    return "\n\n".join(parts)
