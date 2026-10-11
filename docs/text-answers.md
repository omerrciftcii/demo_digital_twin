# cco-text/1 yanıtları

Üst akış artık fact/insight grafiği değil, tek bir Türkçe markdown özeti gönderiyor. Zarf:

```json
{
  "format": "cco-text/1",
  "id": "uuid",
  "subjectId": "uuid",
  "asOf": "2026-10-11T12:00:00+03:00",
  "language": "tr",
  "text": "## Laboratuvar (hastane)\n..."
}
```

`text` en fazla 48 KB. Tarih zaman dilimi olmadan gelirse kayıt reddedilir; eksik alan uydurulmaz. `tests/fixtures/text/example-facts.json` ile `example-text.md` aynı metnin iki yüzü. `patients/patient-01.json` … `patient-05.json` sentetik hastalardır (yaklaşık 20–27 KB).

## Neden parça parça arama yok

20–30 KB, ucuz bir 32k bağlamına sığar. Bölüm seçmek ikinci bir model çağrısı ya da bir kelime kaçırınca yanlış `BİLGİ_YOK` demektir. Metnin tamamı ilk kullanıcı iletisindedir; soru son iletidir. Sistem yönergesi ve kayıt, aynı hasta için sorudan soruya değişmez — ön ek önbelleği buna göre durur. Atıf satır numarası modelden istenmez; sunucu `[Bölüm]` veya `[Bölüm 12.09.2026]` işaretini bölüme ve satıra çözer.

## Yanıt biçimi

JSON şeması / `response_format` yok. Model düz metin yazar, böylece akış ve vLLM bozulmaz.

```text
DURUM: YANIT
Son eGFR 41 mL/dk/1,73 m² (18.09.2026). [Laboratuvar (hastane) 18.09.2026]
```

`DURUM: BİLGİ_YOK` kayıttaki "yok / bilinmiyor / planlandı" cümlesi içindir. Sayı, tarih ve birim, atıf yapılan bölümde birebir yoksa yanıt gösterilmez (`validation_failed`). Kısaltılmış `04.08`, kaynakta yalnız `04.08.2026` diye geçiyorsa kabul edilmez. Kayıtta olmayan "nedeniyle" veya "3 yıl önce" de düşer.

Acil belirti, kendine zarar, açık doz/tedavi değişikliği ve kayıt veya sistem metnini döktürme istekleri modele gitmez. Doz reddi, soruyla örtüşen satırları kayıttan birebir ekler; bu satırlar öneri değildir. "İbuprofen alabilir miyim?" ve "nefesim daralıyor, ne yapmalıyım?" kayıt sorusudur: modelden alıntı beklenir, doz veya talimat değil.

Her klinik yanıta sabit, rakamsız bir uyarı eklenir. `clinicalMeaningVerified` her zaman false'tur; kontrol karakter kontrolüdür.

Gerçek (fixture hash'i tutmayan) kayıt, `ALLOW_EXTERNAL_REAL_DATA=true` olmadan dışarı çıkmaz. Gönderilen beş hasta ve `example-facts.json` bu hash ile sentetik sayılır. İstemci kendine sentetik diyemez.

## Uçlar

- `POST /api/v1/text-records` — zarfın kendisi
- `GET /api/v1/text-records` ve `GET /api/v1/text-records/{recordId}`
- `POST /api/v1/text/answer` — `{ "recordId", "question", "stream": false }`
- `stream: true` → `text/event-stream`: `event: token`, sonra `event: done`

`done` gövdesi yetkilidir. Token olayları modelin ham akışıdır; sayı kontrolü bitince `done.answer` onları geçersiz kılabilir. `metrics.modelCalls`, `metrics.timeToFirstTokenMs` (sağlayıcıda ilk görünen token) ve `metrics.totalMs` döner.

İstek gövdesinde `response_format` yoktur. `reasoning` yalnız `LLM_REASONING` doluysa gider. `provider.zdr` yalnız `openrouter.ai` için gider. ZDR açıkken başka host reddedilir (`zdr_requires_openrouter`).

## Eval

Anahtar bu süreçte yoktur; çağrıyı çalışan sunucu yapar. Önce sunucu, sonra harness.

OpenRouter:

```bash
export LLM_API_KEY="$OPENROUTER_API_KEY"   # veya yalnız OPENROUTER_API_KEY
export LLM_MODEL="qwen/qwen3.5-9b"
export LLM_BASE_URL="https://openrouter.ai/api/v1"
export OPENROUTER_REQUIRE_ZDR=true
export ALLOW_EXTERNAL_REAL_DATA=false
# LLM_REASONING boş kalsın. Sunucu alanı tanımıyorsa göndermeyin.
unset LLM_REASONING
export TEXT_ANSWER_MAX_TOKENS=800
export LLM_TIMEOUT_SECONDS=90

python -m uvicorn medical_ai_selective_context.main:app --host 127.0.0.1 --port 8091
```

Başka bir kabukta:

```bash
python scripts/evaluate_text.py --base-url http://127.0.0.1:8091
```

Sorular `tests/fixtures/text/eval_questions.jsonl` (40). Çıkış kodu: kritik hata varsa 2, harness hatası 1, değilse 0. Özet stdout'a, tam rapor `data/evaluation/text-run-*/report.json` altına yazılır. Rapor soru kimliği, durum, eksik alt dizge, sunucu ve istemci gecikmesi, model çağrı sayısı tutar.

Yerel vLLM (48 GB L40S dahil, OpenAI uyumlu):

```bash
export LLM_BASE_URL="http://127.0.0.1:8000/v1"
export LLM_MODEL="Qwen/Qwen2.5-7B-Instruct"
export LLM_API_KEY=local
export OPENROUTER_REQUIRE_ZDR=false
unset LLM_REASONING
python -m uvicorn medical_ai_selective_context.main:app --host 127.0.0.1 --port 8091
python scripts/evaluate_text.py --base-url http://127.0.0.1:8091
```

`LLM_MODEL` ne ise o gider. Sağlayıcıya özel alan eklenmez. Düşünen modeller `<think>` bloğunu yanıtın dışında bırakır; yine de TTFT'yi şişirmemek için düşünmeyi sunucu tarafında kapatmak daha ucuzdur. Bunu ancak sunucu tanıyorsa `LLM_REASONING=false` ile isteyin.

Mock sınamalar anahtarsızdır:

```bash
python -m pytest -q tests/test_text_answer.py
python -m ruff check src/medical_ai_selective_context/text_document.py \
  src/medical_ai_selective_context/text_safety.py \
  src/medical_ai_selective_context/text_grounding.py \
  src/medical_ai_selective_context/text_llm.py \
  src/medical_ai_selective_context/text_qa.py \
  src/medical_ai_selective_context/text_eval.py \
  scripts/evaluate_text.py tests/test_text_answer.py
```
