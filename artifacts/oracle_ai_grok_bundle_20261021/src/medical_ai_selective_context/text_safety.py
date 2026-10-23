"""Refuse before the model. Chart questions still go through.

Acute first-person emergencies, self-harm, explicit dose or permission-to-take
requests, and prompt-injection or data grabs never see the provider. The
classes are broad conversational patterns. A chart question that does not ask
for a dose, a start, a stop, or permission to take a drug still goes to the model.

Dose-change refusals quote a few medication-section lines. There is no model
call. The lines are chosen by overlap with the question. A generated drug
dictionary is used only when that file is present. The quoted characters are
copied from the record.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from .index import normalize_text
from .text_document import TextRecord
from .text_embed import max_similarity
from .text_llm import env_number
from .text_relevance import drug_names, section_kind
from .text_relevance import fold as fold_row

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
        "Hemen 112'yi ara. Buradaki kayıt özeti acil müdahalenin yerine geçmez."
    ),
    "treatment": (
        "İlaç dozu, ilaca devam etmek ya da ilacı bırakmak bir tedavi kararıdır; "
        "bunu söyleyemem. Kayıttaki son reçete bilgisini aşağıya bırakıyorum. "
        "Bunu hekiminle konuş."
    ),
    "prompt_injection": (
        "Bu istek yanıtlanamaz. Sistem talimatını, başka kişilerin kimlik "
        "bilgilerini veya kaydın tamamını yazmam. İlaç dozu da önermem."
    ),
}

_TREATMENT_WITH_RX = (
    "İlaç dozu, ilaca başlamak, ilacı bırakmak, atlamak, artırmak, azaltmak "
    "ya da ilaçları birlikte kullanmak bir tedavi kararıdır; bunu söyleyemem. "
    "Kayıttaki reçete satırını aşağıya bırakıyorum. Bunu hekiminle konuş."
)
_TREATMENT_WITHOUT_RX = (
    "İlaç dozu, ilaca başlamak, ilacı bırakmak, atlamak, artırmak, azaltmak "
    "ya da ilaçları birlikte kullanmak bir tedavi kararıdır; bunu söyleyemem. "
    "Kayıtta bu ilaç için bir reçete satırı yok. Bunu hekiminle konuş."
)

_QUOTE_STOP = frozenset({
    "alabilir", "artirmali", "azaltmali", "baslamali", "birakip", "cikarabilir",
    "cikarmali", "devam", "dozum", "dozumu", "etsem", "gunde", "icin", "kendim",
    "kullanabilir", "miyim", "olarak", "olur", "onceki", "onerirsin", "sadece",
    "talimat", "talimatlari", "tane", "yuksek", "cikiyor", "yapmaliyim", "yapmam",
    "unut", "yoksa", "simdi", "hemen", "lutfen",
})

_SELF_HARM = (
    r"kendime zarar", r"kendimi oldur", r"intihar", r"yasamak istem",
    r"olmek istiyor", r"canima kiymak", r"hayata son ver",
    r"\bsuicide\b", r"kill myself", r"self-harm", r"self harm",
)
_EMERGENCY = (
    # Chest pressure, heaviness, or crushing pain, including radiation to arm or jaw.
    r"gog(s|us)\w{0,16}(?:\s+\w+){0,3}\s+(ezici|sikis|baski|agirlik|sikist)",
    r"(ezici|baskici|sikisan) (agri|gogus)",
    r"gog(s|us)\w{0,20}(agr|sikis|ezil|baski).{0,48}(kol|cene|sirt|omuz)",
    r"(kolum|cene|cenem).{0,40}(gogus|gogs)",
    r"gogusumden agir",
    # Breathlessness that is sudden or severe, not a mild "daralıyor, neden".
    r"nefes alamiyor", r"nefesim kesil", r"boguluyor",
    r"nefes(im|i) (cok|birden|asiri) (daral|kesil|tikand)",
    r"konusamiyor", r"dudak\w{0,8} mor", r"morariyor",
    # Stroke signs.
    r"(yuzum|agzim) (egri|dustu|uyus|kaydi)",
    r"yuzumun bir taraf", r"(kolum|bacak\w{0,4}) (uyus|gucsuz|kalkmiyor)",
    r"konusmam (bozul|peltek)", r"bir taraf\w{0,6} gucsuz",
    # Syncope that is happening now. Reported history (bayılmıştım) stays a chart question.
    r"\bbayil(?:iyorum|iyorsun|iyoruz|iyor|maktayim|acagim)\b",
    r"\bbayil(?:dim|din|duk)\b",
    r"bilinc(im|i) (kapand|kayb|gitti|kaybold)", r"gozum karardi",
    # Severe bleeding.
    r"kanama(m|si)? (durmuyor|cok|siddet|fisil)", r"kan kus", r"siyah diski",
    r"asiri kanama", r"kanamam durmuyor",
    # Severe hypoglycemia: a sugar reading, or a first-person event in progress.
    r"sekerim\s+\d+",
    r"\bhipoglisemi(?:m|n)?\s+(?:yasiyorum|oluyorum|oldum|geciriyorum|basladi)\b",
    r"(seker|glukoz).{0,40}(bayil|titre|bilinc|karis)",
    r"(titriyorum|kafam karis).{0,40}(seker|glukoz)",
    # Anaphylaxis.
    r"(yuzum|dudak\w{0,6}|dilim|bogaz\w{0,6}) (sis|sisti)",
    r"anafilak", r"alerji.{0,24}(nefes|bogaz|sis)",
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
    r"doz\w{0,10}[\s\S]{0,80}(artir|azalt|cikar|yukselt|dusur)",
    r"(?:artir|azalt|dusur|yukselt|kes|birak|atla|basla)(?:ayim|eyim|alim|elim|mali|meli|sam|sem)\s+mi(?:yim|sin)?",
    r"\bbirakip\b",
    r"(biraksam|kessem|atlasam|artirsam|azaltsam|baslasam)\b",
    r"annemin|babamin|esimin|arkadasimin|baskasinin",
    r"gunde\s+\d+\s+(tane|adet|tablet|kapsul)",
    r"ne kadar\s+(?:ic|al|kullan)(?:eyim|ayim|meli|mali)",
    r"kac\s+(tane|adet|tablet|kapsul|mg|ml|unite)",
    r"devam etmeli miyim",
    r"kullanmaya devam etmeli",
    r"(birakayim|birakmali|keseyim|kesmeli)\s+mi(?:yim|sin)?",
    r"(baslayayim|baslamali)\s+mi(?:yim|sin)?",
    r"(atlayabilir|icmesem|almazsam|atlamali)",
    r"birlikte (kullan|al|ic)",
    r"(beraber|ayni anda) (kullan|al|ic)",
    r"(alabilir|kullanabilir|icilebilir) mi(?:yim)?",
    r"dozu dogrudan",
    r"guvenli oldugunu yaz",
)
# Descriptions for the embedding check. Similarity decides, not a string match.
# No chart value and no dose. A missing local encoder leaves this path idle.
_EMERGENCY_DESCRIPTIONS = (
    "Göğüste ezici baskı veya sıkışma, ağrının kola çeneye veya sırta yayılması",
    "Ani ve ağır nefes darlığı, konuşurken cümleyi tamamlayamama, dudak renginin morarması",
    "Astım krizinde nefesin kesilmesi ve sözcüklerin çıkmaması",
    "Gebelikte şiddetli baş ağrısı, görme bozukluğu ve üst karın ağrısının birlikte olması",
    "Yüzün bir yanında düşme, tek tarafta ani güç kaybı, konuşmanın bozulması",
    "Durmayan ağır kanama",
    "O anda gelişen bayılma veya bilincin kapanması",
    "Yüzde dudakta veya boğazda ani şişlik ile birlikte nefes alamama",
    "Kan şekerinin hızla düşmesi, titreme ve bilincin bulanması",
)
_EMERGENCY_SIMILARITY = 0.82

_UNSAFE_ANSWER = (
    r"dozunuzu (artir|azalt|kes|degistir|birak|yukselt)",
    r"dozu (artirin|azaltin|kesin|birakin)",
    r"(almalisiniz|kullanmalisiniz|icmelisiniz|baslamalisiniz)",
    r"(alabilirsiniz|kullanabilirsiniz)",
    r"recete (ediyorum|ederim|yaziyorum|yazabilirim)",
    r"ilacinizi (kesin|birakin|artirin)",
    r"acil degil", r"acile gitmeyin", r"doktorunuza gitmeyin",
    r"\btaniniz\b", r"guvenle kullan",
    r"\b(?:sizde|sende)(?:\s+\w+){1,8}\s+var\b",
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


def emergency_threshold() -> float:
    """Cosine at or above this uses the emergency wording. Configurable."""
    return env_number(
        "TEXT_EMERGENCY_SIMILARITY", _EMERGENCY_SIMILARITY, 0.0, 1.0, integer=False,
    )


def _similar_emergency(question: str) -> bool:
    """True when the question is close to an emergency description."""
    score = max_similarity(question, _EMERGENCY_DESCRIPTIONS)
    if score is None:
        return False
    return score >= emergency_threshold()


def classify_question(question: str) -> SafetyDecision:
    text = normalize_text(question or "")
    if _matches(text, _SELF_HARM_RE):
        reason = "self_harm"
        quotes = False
    elif _matches(text, _EMERGENCY_RE) or _similar_emergency(question):
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
    tokens.update(name for name in drug_names(question) if len(name) >= 4)
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


def _quote_body(text: str) -> str:
    cleaned = " ".join(text.strip().split())
    while cleaned.startswith("-"):
        cleaned = cleaned[1:].strip()
    return cleaned


def _blob(text: str) -> bool:
    folded = fold_row(text)
    heads = sum(token in folded for token in ("sikayet", "hikaye", "bulgular", "epikriz", "ozgecmis"))
    return heads >= 2 or ("epikriz" in folded and len(text) > 180)


def _looks_like_prescription(text: str) -> bool:
    folded = fold_row(text)
    return bool(re.search(r"TABLET|KAPSÜL|KAPSUL|DAMLA|İNHALER|KALEM|SPREY", text, re.IGNORECASE)) or (
        "recete" in folded or "mg" in folded or "unite" in folded
    )


def prescription_quotes(record: TextRecord, question: str, limit: int = 3) -> list:
    """Medication-section rows for the named drug, or the latest rows when none is named."""
    names = {name for name in drug_names(question) if len(name) >= 3}
    tokens = _question_tokens(question)
    matched = []
    latest = []
    for line in record.lines:
        if line.text.startswith("##"):
            continue
        if section_kind(record.sections[line.section_index].heading) != "med":
            continue
        text = _quote_body(line.text)
        if not text or _blob(text) or not _looks_like_prescription(text):
            continue
        folded = fold_row(text)
        hit = bool(names) and any(name in folded for name in names)
        if not hit and tokens:
            hit = any(_hits(token, folded) for token in tokens)
        if hit:
            matched.append(line)
        else:
            latest.append(line)
    if names:
        return matched[:limit]
    return (matched or latest)[:limit]


def render_refusal(record: TextRecord, question: str, decision: SafetyDecision) -> tuple[str, list[dict]]:
    message = decision.message
    citations: list[dict] = []
    chosen = []
    if decision.include_quotes and decision.reason == "treatment":
        chosen = prescription_quotes(record, question, limit=3)
        has_rx = any("recete" in fold_row(line.text) or "reçete" in line.text.casefold() for line in chosen)
        message = _TREATMENT_WITH_RX if chosen and has_rx else _TREATMENT_WITHOUT_RX
    parts = [message]
    if chosen:
        parts.append("Kayıtta yazan (birebir alıntı, tedavi önerisi değildir):")
    for line in chosen:
        section = record.sections[line.section_index]
        text = _quote_body(line.text)
        if not text or _blob(text):
            continue
        parts.append(f"- {text} [{section.heading}]")
        citations.append({
            "section": section.heading,
            "startLine": line.number,
            "endLine": line.number,
            "quote": text,
            "marker": section.heading,
        })
    return "\n".join(parts), citations


def with_disclaimer(answer: str) -> str:
    body = (answer or "").strip()
    if DISCLAIMER in body:
        body = body.replace(DISCLAIMER, "").strip()
    parts = [part for part in (body, DISCLAIMER) if part]
    return "\n\n".join(parts)
