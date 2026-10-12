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

`text` en fazla 48 KB. Tarih zaman dilimi olmadan gelirse kayıt reddedilir; eksik alan uydurulmaz. `tests/fixtures/text/example-facts.json` ile `example-text.md` aynı metnin iki yüzü. `patients/patient-01.json` … `patient-07.json` sentetik hastalardır (yaklaşık 20–30 KB). `patient-06` yeni taburcu olmuş stent hastası, `patient-07` yoğun bakımdan yeni çıkmış sepsis/pnömoni hastasıdır. `patient-08` aynı üslupta kısa bir sentetik kayıttır: tanı, anemi, karın yakınması, karaciğer ve pıhtı soruları. Gerçek hasta metni yoktur.

## Bağlam

20–30 KB, 32k bağlamına sığar. İlk kullanıcı iletisi kaydın tamamıdır ve aynı hasta için önbellekte kalır. Soru iletisi, soruyla örtüşen her bölümü bütünüyle tekrarlar; bölümün içi kesilmez. Örtüşme, Snowball Türkçe kökü ve karakter trigramıdır. Marka ile etken madde, süreç açılışında bir kez indekslenen `data/drug_classes.json` ATC grubundan gelir. Bu dosyayı `scripts/build_drug_dictionary.py` açık ürün listesinden üretir; elle ad eklenmez ve dosya `src/` altında durmaz. Yerel bir gömme modeli bölümleri yeniden sıralayabilir (`TEXT_EMBEDDING_MODEL`, varsayılan `intfloat/multilingual-e5-small`, `TEXT_EMBEDDING_DEVICE`, `TEXT_EMBEDDING_DISABLE`). Ağırlık yoksa yalnız sözcük örtüşmesi kalır; model indirilmez. `TEXT_STARTUP_WARMUP` varsayılanı açıktır: süreç açılışında ilaç indeksi ve yerel gömme modeli bir kez hazırlanır. `false` bu adımı atlar. Kaydın tamamı ilk kullanıcı iletisindedir. Örtüşme yoksa model kaydın tamamına bakar. Atıf satır numarası modelden istenmez; sunucu `[Bölüm]` işaretini bölüme ve satıra çözer. Model adı kodda sabit değildir; `LLM_MODEL` yeter.

## Yanıt biçimi

JSON şeması / `response_format` yok. Model düz metin yazar, böylece akış ve vLLM bozulmaz.

```text
DURUM: YANIT
Son eGFR 41 mL/dk/1,73 m² (18.09.2026). [Laboratuvar (hastane) 18.09.2026]
```

`DURUM: BİLGİ_YOK` kayıttaki "yok / bilinmiyor / planlandı" cümlesi içindir. Sayı, tarih ve birim, atıf yapılan bölümde birebir yoksa yanıt gösterilmez (`validation_failed`). Kısaltılmış `04.08`, kaynakta yalnız `04.08.2026` diye geçiyorsa kabul edilmez. Kayıtta olmayan "3 yıl önce" düşer. İki kayıtlı miktarın toplamı veya farkı, birim ailesi aynıysa, cevapta kabul edilir. Aynı satırda birim yalnız ilk miktarda yazılmışsa sonraki çıplak miktar o birimi devralır. Neden bağlacının sözcükleri, kök ve kısa çekim farkıyla, o öğenin satırında duruyorsa cümle bütünüyle kalır. Durmuyorsa desteklenen yan tutulur; kalan parça ölçüm ya da çekimli fiil değilse, ya da nesne hâlinde fiilsiz kaldıysa, cümle düşer. Yalnız tarih ya da reçete/kayıt olayı neden sayılmaz ve o cümleyi ayakta tutmaz. Ay.yıl tek tarih belirtecidir. Nokta, virgül gibi içi boş bir cümle iddia sayılmaz. Sayı, tarih veya kod içermeyen bir yokluk cümlesi ("kayıtta yok", "bulunmuyor") yeni olgu sayılmaz ve örtüşme aranmadan kalır. "A'dan B'ye arttı" kalıbı birimin üstündeki kesme ve hâl ekini (`ms'den`, `ms'ye`, `ms'sinden`) birimden ayırır. Ölçüm adı, miktardan birkaç sözcük önce durabilir; `değeri` gibi genel sözcük ad sayılmaz. Uçlar, o ölçüm cümlesindeki olağan ve güncel çifte bağlanır. Olağan etiketin yanındaki etiketsiz tek okuma güncel sayılır. Aynı cümledeki günlük liste bu çiftin yerine geçmez. Sonraki cümledeki birebir sayı bu eşi seçemez. Payın dışındaki fark yanlış değerdir ve cümle düşer. Sıra tarih ya da olağan/güncel sırasına uyuyorsa ve uçlar kayıtla aynıysa yön düzeltilir. "1. saat" gibi sıra bildiren nokta cümle sonu sayılmaz. Doğrulama düşerse kullanıcıya genel red gider; `validation.debug` hangi sayı, atıf ve kontrolün düşürdüğünü tutar. Kayıttaki bir sayının birimi ile cevapta o sayıya takılan birim aynı ailede değilse o cümle düşer. Aynı cümlenin yakın kopyası da düşer.

Hasta metni sen diye, sıcak ve doğal Türkçe ile anlatılır. Cevap soruyu bütünüyle karşılar: satırdaki ad, tarih, uzmanlık dalı ve hekimin yazdığı öneri; kayıttaki terimin sade Türkçesi; sonuç ile hedef ayrı. Kalıp giriş cümlesi kurulmaz ve cevap tek cümleye sıkıştırılmaz. Kayıt satırı, epikriz veya madde işareti olduğu gibi dökülmez. Yüksek, normal veya düşük yalnız kayıt böyle diyorsa söylenir. Tanı konmaz. Neden, doz ve tedavi uydurulmaz. Kırmızı bayrak belirtileri modele gitmez.

Acil belirti (göğüste sıkışma, kola vuran ağrı dahil) modele gitmez; yanıt 112 çağrısıdır. Doz ("kaç tane içeyim", "ne kadar içeyim"), ilaca devam ya da bırakma sorusu da modele gitmez; hekime yönlendirir ve kayıttaki son reçete satırını bırakır. "Hâlâ kullanıyor muyum" kayıt sorusudur. Şikayet, hikaye ve bulgular başlıkları cevapta kopyalanmaz. Tanı sorusu tanı satırındaki kod ve tarihi söyler. Doğrulama geçmezse uydurulan sayı gösterilmez. Sunucu model cümlesinin yerine başka bir satır yazmaz. `TEXT_ANSWER_MAX_TOKENS` varsayılanı 2000'dür; model adı `LLM_MODEL` ile değişir. `factCoverage` karşılaştırması büyük-küçük harfe bakmaz. Doz reddi, soruyla örtüşen satırları kayıttan birebir ekler; bu satırlar öneri değildir. "Alabilir miyim" bir tedavi reddidir. "Nefesim daralıyor, ne yapmalıyım?" kayıt sorusudur.

Her klinik yanıta sabit, rakamsız bir uyarı eklenir. `clinicalMeaningVerified` her zaman false'tur; kontrol karakter kontrolüdür.

Gerçek (fixture hash'i tutmayan) kayıt varsayılan olarak modele gitmez. `ALLOW_EXTERNAL_REAL_DATA=true` yapılandırılmış her hosta izin verir; bu, `openrouter.ai` dahil dışarı çıkış demektir. Yerel veya özel ağdaki vLLM için ayrı bayrak vardır: `LLM_LOCAL_INFERENCE=true`. Bayrak açıkken gerçek kayıt yalnız güvenilen hosta gider: `localhost`, `127.0.0.1`, `::1`, özel adres (10/8, 172.16/12, 192.168/16), link-local ve IPv6 ULA. DNS çözümüne bakılmaz. `LLM_LOCAL_HOSTS` ek makine adı ekler (`gpu.local`). `openrouter.ai` ve genel bir IP bu listeye yazılırsa süreç ayarı reddeder. Bayrak açık olsa bile host güvenilmiyorsa gerçek kayıt yine `external_real_data_blocked` olur. `/health` içindeki `textLlm.realDataMode` etkin izni söyler: `blocked`, `local` veya `external`. İki bayrak birden açıksa mod `external` kalır, çünkü adres sonradan buluta çevrilse de gerçek kayıt gider. Gönderilen sekiz hasta ve `example-facts.json` bu hash ile sentetik sayılır. `patient-08` tanı, anemi, karın yakınması, karaciğer ve pıhtı sorularının sentetik karşılığıdır; gerçek kayıt içermez. İstemci kendine sentetik diyemez.

## Uçlar

- `POST /api/v1/text-records` — zarfın kendisi
- `GET /api/v1/text-records` ve `GET /api/v1/text-records/{recordId}`
- `POST /api/v1/text/answer` — `{ "recordId", "question", "stream": false, "history": [] }`
- `stream: true` → `text/event-stream`: `event: token`, sonra `event: done`

`done` gövdesi yetkilidir. Token olayları modelin ham akışıdır; sayı kontrolü bitince `done.answer` onları geçersiz kılabilir. `metrics.modelCalls`, `metrics.timeToFirstTokenMs` (sağlayıcıda ilk görünen token) ve `metrics.totalMs` döner.

İstek `history` alabilir: en fazla altı `{role, content}` turu. Kayıt iletisi önbellek öneki olarak durur; önceki turlar ondan sonra, soru en sonda gelir. Erişim, son soru ile bu turların sözcüklerini birlikte kullanır. Sayı kontrolü yine kayda bakar.

İstek gövdesinde `response_format` yoktur. `provider.zdr` ve `provider.data_collection=deny` yalnız `openrouter.ai` için gider. ZDR açıkken başka host reddedilir (`zdr_requires_openrouter`).

`OPENROUTER_ALLOW_FALLBACKS` varsayılanı `true`: yedek sağlayıcı ZDR ve veri toplamama şartının içinde kalır. `false` tek sağlayıcıya kilitler; o sağlayıcı 429 verirse tur boşa gider. `OPENROUTER_PROVIDER` doluysa `only` yine o sağlayıcıyı sabitler. 429, 502, 503 ve 529, ilk tokenden önce `LLM_RETRY_MAX` (varsayılan 2) kez, `LLM_RETRY_BACKOFF_SECONDS` (varsayılan 0.75, ikiye katlanır) bekleyerek yeniden denenir.

`LLM_DISABLE_THINKING` varsayılanı `true`: OpenRouter'a `reasoning.enabled=false` gider; OpenRouter olmayan sunucuya (vLLM dahil) `chat_template_kwargs.enable_thinking=false` gider. `auto` bu OpenRouter alanını yalnız model adında `qwen3`, `qwq` veya `deepseek-r1` varsa gönderir ve yerel sunucuya alan koymaz. `false` hiç göndermez. `LLM_REASONING` doluysa o kazanır ve düşünme alanı ayrıca eklenmez. Akıştaki `reasoning` / `reasoning_content` yanıta yazılmaz. `<think>` bloğu, kapanmamış olsa bile, yanıttan silinir. Yalnız boşluk veya düşünce gelirse durum `reasoning_only` olur; harness o soruyu kaydedip devam eder.

## Eval

Anahtar bu süreçte yoktur; çağrıyı çalışan sunucu yapar. Önce sunucu, sonra harness.

OpenRouter:

```bash
export LLM_API_KEY="$OPENROUTER_API_KEY"   # veya yalnız OPENROUTER_API_KEY
export LLM_MODEL="qwen/qwen3.5-9b"
export LLM_BASE_URL="https://openrouter.ai/api/v1"
export OPENROUTER_REQUIRE_ZDR=true
export ALLOW_EXTERNAL_REAL_DATA=false
# Varsayılan true, Qwen3 düşünmesini kapatır. LLM_REASONING doluysa o kazanır.
unset LLM_REASONING
export LLM_DISABLE_THINKING=true
export OPENROUTER_ALLOW_FALLBACKS=true
export LLM_RETRY_MAX=2
export TEXT_ANSWER_MAX_TOKENS=2000
export LLM_TIMEOUT_SECONDS=90

python -m uvicorn medical_ai_selective_context.main:app --host 127.0.0.1 --port 8091
```

Başka bir kabukta:

```bash
python scripts/evaluate_text.py --base-url http://127.0.0.1:8091
```

Sorular `tests/fixtures/text/eval_questions.jsonl` (100). Günlük dil ve yazım varyantları da vardır (`tahlillerim iyimi`, `basim agriyo neden olabilir`, `sekerim dusmus mu`). Çıkış kodu: kritik hata varsa 2, hasta içe aktarma veya bağlantı kurulumu bozulursa 1, değilse 0. Tek sorunun zaman aşımı veya bozuk SSE'si raporu düşürmez; o satıra `error` yazılır ve koşu sürer. Özet stdout'a, tam rapor `data/evaluation/text-run-*/report.json` altına yazılır. Her satırda derecelenen `answer` durur.

Soru iletisinde, soruyla örtüşen her bölüm bütünüyle durur. İlk kullanıcı iletisi kaydın tamamıdır. Model cümleleri, durum satırı ve atıflar doğrulamadan sonra durur. Uydurulan sayı düşer. Kırık "seyrindi bilinmeyen cümle" kalıbı konuşma ölçüsünde düşer.

Stdout'taki `latencyMs` ve `clientLatencyMs` her seri için `average`, `p50` ve `p95` verir (`average` ile `mean` aynı sayıdır). `conversational` yanıtlanmış ve bilgi-yok satırlarında kayıt dökümü, kırık cümle ve uzunluk penceresini sayar; `average` bu satırlar içinde geçenlerin oranıdır. Güvenlik reddi ve doğrulama hatası bu ölçüye girmez.

`factCoverage` ilaç, doz, tarih ve yük taşıyan sözcükler bir aradaysa sayar. `exactFactCoverage` eski alt dizge eşleşmesidir. Beklenen olgular kayıttaki alt dizgelerdir. Ambalaj (`56 FİLM TABLET`) ve `itibarıyla` tek başına kaçırılmaz; baştaki sıfırın düştüğü tarih, PHQ-9 ile artı derece, iki doz bir aradayken aynı anlama gelen değişim fiili de kaçar sayılmaz. Doz veya tarih yoksa yine kaçar. `validation_failed` ve modelden gelen `safety_refusal` satırında kısaltılmış `rawOutput` durur (düşünce bloğu ve 11 haneli numara silinir).

Yerel vLLM (48 GB L40S dahil, OpenAI uyumlu):

```bash
export LLM_BASE_URL="http://127.0.0.1:8000/v1"
export LLM_MODEL="Qwen/Qwen2.5-14B-Instruct-AWQ"
export LLM_API_KEY=local
export OPENROUTER_REQUIRE_ZDR=false
# Gerçek kayıt için. Kapalıyken localhost bile gerçek kaydı dışarı vermez.
# Özel ağ: LLM_BASE_URL=http://192.168.1.20:8000/v1
# Ad ile: LLM_LOCAL_HOSTS=gpu.local
export LLM_LOCAL_INFERENCE=true
unset LLM_REASONING
python -m uvicorn medical_ai_selective_context.main:app --host 127.0.0.1 --port 8091
python scripts/evaluate_text.py --base-url http://127.0.0.1:8091
```

`LLM_MODEL` ne ise o gider. Düşünen modellerde `<think>` bloğu ve `reasoning_content` yanıtın dışında kalır. vLLM üzerinde Qwen3 düşünmesi varsayılan olarak kapalıdır (`chat_template_kwargs.enable_thinking=false`). Alanı göndermemek için `LLM_DISABLE_THINKING=false` kullanın.

Mock sınamalar anahtarsızdır:

```bash
python -m pytest -q tests/test_text_answer.py
python -m ruff check src/medical_ai_selective_context/text_document.py \
  src/medical_ai_selective_context/text_safety.py \
  src/medical_ai_selective_context/text_grounding.py \
  src/medical_ai_selective_context/text_relevance.py \
  src/medical_ai_selective_context/text_llm.py \
  src/medical_ai_selective_context/text_qa.py \
  src/medical_ai_selective_context/text_eval.py \
  scripts/evaluate_text.py tests/test_text_answer.py
```
