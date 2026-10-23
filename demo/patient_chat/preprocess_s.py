"""İsteğe bağlı S-serisi ön işleme modları (bench S5 ve S6b'nin demo uyarlaması).

raw  : değişiklik yok (ham kayıt metni).
s5   : cco_pre (tarih/yaş, güncel-geçmiş ayrımı) + S4b kalıcılık etiketi + S5 seçici ve özet satırı sahipliği.
s6b  : s5 + düz S4b katalog, ICD 3 karakter kökünde olası tekrar notu, 6 sınırı dışında kanıt paketleri.

Kurallar yalnızca CCO yapısına dayanır; hastalık/soru/hasta listesi yoktur. Ağ erişimi yoktur: model çağrıları
dışarıdan verilen `call` fonksiyonuyla (demo'nun seçili modeli) yapılır. Hasta tarafına kanıt ID'si veya JSON gitmez;
seçici/etiket çıktıları yalnızca bağlam kurmak için kullanılır."""
from __future__ import annotations

import asyncio
import datetime as dt
from collections.abc import Awaitable, Callable
from typing import Any, Literal

from .s_series import cco_pre, s5frame, s6frame

PreprocessMode = Literal["raw", "s5", "s6b"]
MODE_LABELS = {"raw": "Ham kayıt", "s5": "S5 ön işleme", "s6b": "S6b ön işleme"}
CCO_HEADER = "Klinik Bağlam (CCO):\n"

# bench sys_S5.txt madde 3-4'ün özü; yalnızca S modlarında sistem istemine eklenir.
S_CONTEXT_NOTE = (
    "Kayıt ön işlenmiştir: \"Zaman özeti\" bölümünü ve her kalemin yanındaki \"[... önce]\" yaşını esas al; tarihleri "
    "kendin yeniden hesaplama. Bağlamda son dönemin kayıtları vardır; soruyla ilgili olabileceği için seçilmiş bazı eski "
    "kayıtlar \"Geçmiş kayıtlar (bugünkü durumu yansıtmaz, ilgili olabilir)\" başlığı altında yaşlarıyla verilir. Bu eski "
    "kayıtları güncel veya aktif diye sunma; anarsan tarihini veya yaşını söyle ve arka plan bilgisi olarak sun. Bir eski "
    "kaydın neden gösterildiğine dair yorum yapma. \"Daha eski diğer kayıtlar\" satırındaki gösterilmeyen kayıtlar hakkında "
    "tahmin yürütme. \"ÖN TANI\" olarak işaretli hiçbir şeyi kesin tanı gibi sunma."
)

# call(messages, max_tokens, response_format|None) -> model metni
Call = Callable[[list[dict[str, str]], int, "dict[str, Any] | None"], Awaitable[str]]

_LABEL_CACHE: dict[tuple[str, str, tuple[str, ...]], dict[str, str]] = {}


def preprocessed_record(text: str, today: dt.date) -> str:
    """Ham CCO metni -> cco_pre ile yaş/güncel-geçmiş işaretli metin (Hasta Sorusu bloğu olmadan)."""
    return CCO_HEADER + cco_pre.process(text, today)


async def _structured(call: Call, messages, max_tokens: int, schema) -> str | None:
    for fmt in (schema, None):
        try:
            return await call(messages, max_tokens, fmt)
        except Exception:  # şema desteklenmezse şemasız dene; ikinci hata da yedeğe düşer
            continue
    return None


async def build_context(
    mode: PreprocessMode,
    *,
    text: str,
    question: str,
    today: dt.date,
    call: Call,
    cache_key: str,
) -> tuple[str, dict[str, Any]]:
    """-> (modele gidecek kayıt metni, iç günlük). Seçici başarısız olursa tüm ön işlenmiş kayıt gider (bench yedeği)."""
    if mode == "raw":
        return text, {"mode": "raw"}
    S = s6frame if mode == "s6b" else s5frame
    user = preprocessed_record(text, today)
    items = S.annotate(user)[4]
    info: dict[str, Any] = {"mode": mode, "catalog_size": len(items)}

    targets = S.label_targets(items)
    if targets:
        ids = [i["id"] for i in targets]
        key = (mode, cache_key, tuple(ids))
        labels = _LABEL_CACHE.get(key)
        if labels is None:
            txt = await _structured(call, S.label_messages(targets), 1500, S.label_schema(ids))
            labels = S.parse_labels(txt, set(ids)) if txt else None
            if labels is not None:
                _LABEL_CACHE[key] = labels
            info["label_status"] = "ok" if labels is not None else "fail"
        else:
            info["label_status"] = "cached"
        S.apply_labels(items, labels or {})

    sel = []
    if items:
        ids = [i["id"] for i in items]
        txt = await _structured(call, S.selector_messages(question, S.catalog(items)), 1000, S.selector_schema(ids))
        sel = S.parse_selector(txt, set(ids)) if txt else None
    info["selector_status"] = "ok" if sel is not None else "fallback"
    if sel is not None:
        sel, dropped = S.enforce(sel, items)
        info["dropped"] = len(dropped)
        if mode == "s6b":
            packs = s6frame.evidence_packs(question, items, {k for k, _ in sel})
            info["packs"] = len(packs)
            sel = sel + packs
        info["selected"] = len(sel)
    ctx, cinfo = S.build_context(user, sel)
    info.update({"fallback": cinfo.get("fallback"), "ctx_chars": len(ctx), "raw_chars": len(text)})
    return ctx, info


def run_sync(coro):
    return asyncio.run(coro)
