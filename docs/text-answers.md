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

`text` en fazla 48 KB. Tarih zaman dilimi olmadan gelirse kayıt reddedilir; eksik alan uydurulmaz. `tests/fixtures/text/example-facts.json` ile `example-text.md` aynı metnin iki yüzü. `patients/patient-01.json` … `patient-07.json` sentetik hastalardır (yaklaşık 20–30 KB). `patient-06` yeni taburcu olmuş stent hastası, `patient-07` yoğun bakımdan yeni çıkmış sepsis/pnömoni hastasıdır.

## Neden parça parça arama yok

20–30 KB, ucuz bir 32k bağlamına sığar. Bölüm seçmek ikinci bir model çağrısı ya da bir kelime kaçırınca yanlış `BİLGİ_YOK` demektir. Metnin tamamı ilk kullanıcı iletisindedir; soru son iletidir. Sistem yönergesi ve kayıt, aynı hasta için sorudan soruya değişmez — ön ek önbelleği buna göre durur. Atıf satır numarası modelden istenmez; sunucu `[Bölüm]` veya `[Bölüm 12.09.2026]` işaretini bölüme ve satıra çözer.

## Yanıt biçimi

JSON şeması / `response_format` yok. Model düz metin yazar, böylece akış ve vLLM bozulmaz.

```text
DURUM: YANIT
Son eGFR 41 mL/dk/1,73 m² (18.09.2026). [Laboratuvar (hastane) 18.09.2026]
```

`DURUM: BİLGİ_YOK` kayıttaki "yok / bilinmiyor / planlandı" cümlesi içindir. Sayı, tarih ve birim, atıf yapılan bölümde birebir yoksa yanıt gösterilmez (`validation_failed`). Kısaltılmış `04.08`, kaynakta yalnız `04.08.2026` diye geçiyorsa kabul edilmez. Kayıtta olmayan "nedeniyle" veya "3 yıl önce" de düşer.

Hasta metni sen diye, kısa cümlelerle anlatılır. Kayıt satırı, epikriz veya madde işareti olduğu gibi dökülmez. Yüksek, normal veya düşük yalnız kayıt böyle diyorsa söylenir. Tanı konmaz. "Neden olabilir?" sorusunda kayıtta görünen ilgili bulgu sayılır (uyku, tansiyon, ilaç, aynı şikayetle eski başvuru) ve hekimle konuşması nazikçe eklenir; neden uydurulmaz. Kırmızı bayrak belirtileri modele gitmez.

Acil belirti, kendine zarar, açık doz/tedavi değişikliği ve kayıt veya sistem metnini döktürme istekleri modele gitmez. Doz reddi, soruyla örtüşen satırları kayıttan birebir ekler; bu satırlar öneri değildir. "İbuprofen alabilir miyim?" ve "nefesim daralıyor, ne yapmalıyım?" kayıt sorusudur: modelden alıntı beklenir, doz veya talimat değil.

Her klinik yanıta sabit, rakamsız bir uyarı eklenir. `clinicalMeaningVerified` her zaman false'tur; kontrol karakter kontrolüdür.

Gerçek (fixture hash'i tutmayan) kayıt, `ALLOW_EXTERNAL_REAL_DATA=true` olmadan dışarı çıkmaz. Gönderilen yedi hasta ve `example-facts.json` bu hash ile sentetik sayılır. İstemci kendine sentetik diyemez.

## Uçlar

- `POST /api/v1/text-records` — zarfın kendisi
- `GET /api/v1/text-records` ve `GET /api/v1/text-records/{recordId}`
- `POST /api/v1/text/answer` — `{ "recordId", "question", "stream": false }`
- `stream: true` → `text/event-stream`: `event: token`, sonra `event: done`

`done` gövdesi yetkilidir. Token olayları modelin ham akışıdır; sayı kontrolü bitince `done.answer` onları geçersiz kılabilir. `metrics.modelCalls`, `metrics.timeToFirstTokenMs` (sağlayıcıda ilk görünen token) ve `metrics.totalMs` döner.

İstek gövdesinde `response_format` yoktur. `provider.zdr` ve `provider.data_collection=deny` yalnız `openrouter.ai` için gider. ZDR açıkken başka host reddedilir (`zdr_requires_openrouter`).

`OPENROUTER_ALLOW_FALLBACKS` varsayılanı `true`: yedek sağlayıcı ZDR ve veri toplamama şartının içinde kalır. `false` tek sağlayıcıya kilitler; o sağlayıcı 429 verirse tur boşa gider. `OPENROUTER_PROVIDER` doluysa `only` yine o sağlayıcıyı sabitler. 429, 502, 503 ve 529, ilk tokenden önce `LLM_RETRY_MAX` (varsayılan 2) kez, `LLM_RETRY_BACKOFF_SECONDS` (varsayılan 0.75, ikiye katlanır) bekleyerek yeniden denenir.

`LLM_DISABLE_THINKING` varsayılanı `auto`: OpenRouter'da model adında `qwen3`, `qwq` veya `deepseek-r1` varsa `reasoning.enabled=false` gider. `true` OpenRouter'da her modele kapatır; OpenRouter olmayan sunucuda `chat_template_kwargs.enable_thinking=false` gönderir. `false` hiç göndermez. `LLM_REASONING` doluysa o kazanır ve düşünme alanı ayrıca eklenmez. Akıştaki `reasoning` / `reasoning_content` yanıta yazılmaz. Yalnız boşluk veya düşünce gelirse durum `reasoning_only` olur; harness o soruyu kaydedip devam eder.

## Eval

Anahtar bu süreçte yoktur; çağrıyı çalışan sunucu yapar. Önce sunucu, sonra harness.

OpenRouter:

```bash
export LLM_API_KEY="$OPENROUTER_API_KEY"   # veya yalnız OPENROUTER_API_KEY
export LLM_MODEL="qwen/qwen3.5-9b"
export LLM_BASE_URL="https://openrouter.ai/api/v1"
export OPENROUTER_REQUIRE_ZDR=true
export ALLOW_EXTERNAL_REAL_DATA=false
# Qwen3 için auto, reasoning.enabled=false gönderir. LLM_REASONING doluysa o kazanır.
unset LLM_REASONING
export LLM_DISABLE_THINKING=auto
export OPENROUTER_ALLOW_FALLBACKS=true
export LLM_RETRY_MAX=2
export TEXT_ANSWER_MAX_TOKENS=800
export LLM_TIMEOUT_SECONDS=90

python -m uvicorn medical_ai_selective_context.main:app --host 127.0.0.1 --port 8091
```

Başka bir kabukta:

```bash
python scripts/evaluate_text.py --base-url http://127.0.0.1:8091
```

Sorular `tests/fixtures/text/eval_questions.jsonl` (86). Günlük dil ve yazım varyantları da vardır (`tahlillerim iyimi`, `basim agriyo neden olabilir`, `sekerim dusmus mu`). Çıkış kodu: kritik hata varsa 2, hasta içe aktarma veya bağlantı kurulumu bozulursa 1, değilse 0. Tek sorunun zaman aşımı veya bozuk SSE'si raporu düşürmez; o satıra `error` yazılır ve koşu sürer. Özet stdout'a, tam rapor `data/evaluation/text-run-*/report.json` altına yazılır. Her satırda derecelenen `answer` durur.

Yanıt, model metninden sonra sorulan kayda çekilir ve kısa cümleyle söylenir: kesilen veya değişen ilaçta durdurma cümlesi ve güncel ilaç, sorulan ölçüm (tepe akım FEV1 yerine geçmez, başka bir tahlil onun yerine konmaz), etki yazılmamışsa bilinmeyen cümlesi, çok parçalı soruda eksik kalan satır. Uydurulan sayı bu doldurmayla geri gelmez. Kırık "seyrindi bilinmeyen cümle" kalıbı düşer.

Stdout'taki `latencyMs` ve `clientLatencyMs` her seri için `average`, `p50` ve `p95` verir (`average` ile `mean` aynı sayıdır). `conversational` yanıtlanmış ve bilgi-yok satırlarında kayıt dökümü, kırık cümle ve uzunluk penceresini sayar; `average` bu satırlar içinde geçenlerin oranıdır. Güvenlik reddi ve doğrulama hatası bu ölçüye girmez.

`factCoverage` ilaç, doz, tarih ve yük taşıyan sözcükler bir aradaysa sayar. `exactFactCoverage` eski alt dizge eşleşmesidir. Beklenen olgular kayıttaki alt dizgelerdir. Ambalaj (`56 FİLM TABLET`) ve `itibarıyla` tek başına kaçırılmaz; baştaki sıfırın düştüğü tarih, PHQ-9 ile artı derece, iki doz bir aradayken aynı anlama gelen değişim fiili de kaçar sayılmaz. Doz veya tarih yoksa yine kaçar. `validation_failed` ve modelden gelen `safety_refusal` satırında kısaltılmış `rawOutput` durur (düşünce bloğu ve 11 haneli numara silinir).

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

`LLM_MODEL` ne ise o gider. Düşünen modellerde `<think>` bloğu ve `reasoning_content` yanıtın dışında kalır. vLLM üzerinde Qwen3 düşünmesini kapatmak için `LLM_DISABLE_THINKING=true` kullanın; `auto` yerel sunucuya bu alanı göndermez.

Mock sınamalar anahtarsızdır:

```bash
python -m pytest -q tests/test_text_answer.py
python -m ruff check src/medical_ai_selective_context/text_document.py \
  src/medical_ai_selective_context/text_safety.py \
  src/medical_ai_selective_context/text_grounding.py \
  src/medical_ai_selective_context/text_focus.py \
  src/medical_ai_selective_context/text_llm.py \
  src/medical_ai_selective_context/text_qa.py \
  src/medical_ai_selective_context/text_eval.py \
  scripts/evaluate_text.py tests/test_text_answer.py
```
