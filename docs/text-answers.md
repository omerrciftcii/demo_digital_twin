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

Kayıt ilk kullanıcı iletisinde bir kez durur ve aynı hasta için önbellekte kalır. Soru iletisi bölüm metnini tekrarlamaz; öne çıkan bölüm başlıklarını ipucu olarak yazar ve kaydın diğer bölümlerinin de kullanılabileceğini söyler. Örtüşme, Snowball Türkçe kökü ve karakter trigramıdır. Son soru, önceki turlardan daha ağır basar. Yerel gömme varsa bölüm vektörü ve satır vektörü eklenir: benzerliği tabanın üstündeki ilk `TEXT_SECTION_TOP_K` bölüm (varsayılan 4, taban `TEXT_SECTION_SIMILARITY`, varsayılan 0.82). Ağırlık yoksa yalnız sözcük örtüşmesi kalır. İstem `QA_CONTEXT_TOKENS` içinde kalır (varsayılan 16384). Sayım tutumlu bir karakter tahminidir. Sınır aşılırsa en az ilgili bölüm önce çıkar; son bölümün satırları sondan budanır. Marka ile etken madde, süreç açılışında bir kez indekslenen `data/drug_classes.json` ATC grubundan gelir. Bu dosyayı `scripts/build_drug_dictionary.py` açık ürün listesinden üretir; elle ad eklenmez ve dosya `src/` altında durmaz. Yerel gömme modeli `TEXT_EMBEDDING_MODEL` (varsayılan `intfloat/multilingual-e5-small`), `TEXT_EMBEDDING_DEVICE`, `TEXT_EMBEDDING_DISABLE`. Model indirilmez. `TEXT_STARTUP_WARMUP` varsayılanı açıktır: süreç açılışında ilaç indeksi ve yerel gömme modeli bir kez hazırlanır. `false` bu adımı atlar. Atıf satır numarası modelden istenmez; sunucu `[Bölüm]` işaretini bölüme ve satıra çözer. Model adı kodda sabit değildir; `LLM_MODEL` yeter.

## Yanıt biçimi

JSON şeması / `response_format` yok. Model düz metin yazar, böylece akış ve vLLM bozulmaz.

```text
DURUM: YANIT
Son eGFR 41 mL/dk/1,73 m² (18.09.2026). [Laboratuvar (hastane) 18.09.2026]
```

`DURUM: BİLGİ_YOK` kayıttaki "yok / bilinmiyor / planlandı" cümlesi içindir. Sayı, tarih ve birim, atıf yapılan bölümde birebir yoksa yanıt gösterilmez (`validation_failed`). Kısaltılmış `04.08`, kaynakta yalnız `04.08.2026` diye geçiyorsa kabul edilmez. Kayıtta olmayan "3 yıl önce" düşer. İki kayıtlı miktarın toplamı veya farkı, birim ailesi aynıysa, cevapta kabul edilir. Aynı satırda birim yalnız ilk miktarda yazılmışsa sonraki çıplak miktar o birimi devralır. Neden bağlacının sözcükleri, kök ve kısa çekim farkıyla, o öğenin satırında duruyorsa cümle bütünüyle kalır. Kayıtta hiç geçmeyen bir sözcük, satırda duran nedeni tek başına düşürmez; eksik sözcükler, kayıtta bulunanlar kadar veya onlardan fazlaysa neden desteklenmiş sayılmaz. Desteklenmiyorsa çekimli fiili olan yan tutulur. Fiil yoksa yalnız miktarın kendi satırındaki ölçüm kalır. Geriye nesne hâlinde ya da yarım bir ad tamlaması kalırsa cümlenin tamamı düşer, kırık parça yazılmaz. Virgül ile nokta aynı ondalık miktardır; °C ile C aynı sıcaklık ailesidir. Gün.ay ve ay.yıl miktar değildir. Bir cümle düşüp bir cümle kalırsa kalan gösterilir ve kısa bir uyarı eklenir. Hiçbir doğrulanabilir cümle kalmazsa yanıtın tamamı gösterilmez. Budamadan sonra cümlenin ve yanıtın ilk harfi büyür; ikinci harfi zaten büyük olan kısaltma (eGFR) olduğu gibi kalır, Türkçe i harfi İ olur. Yalnız tarih ya da reçete/kayıt olayı neden sayılmaz ve o cümleyi ayakta tutmaz. Ay.yıl tek tarih belirtecidir. Nokta, virgül gibi içi boş bir cümle iddia sayılmaz. Sayı, tarih veya kod içermeyen bir yokluk cümlesi ("kayıtta yok", "bulunmuyor") yeni olgu sayılmaz ve örtüşme aranmadan kalır. "A'dan B'ye arttı" kalıbı birimin üstündeki kesme ve hâl ekini (`ms'den`, `ms'ye`, `ms'sinden`) birimden ayırır. Ölçüm adı, miktardan birkaç sözcük önce durabilir; `değeri` gibi genel sözcük ad sayılmaz. Uçlar, o ölçüm cümlesindeki olağan ve güncel çifte bağlanır. Olağan etiketin yanındaki etiketsiz tek okuma güncel sayılır. Aynı cümledeki günlük liste bu çiftin yerine geçmez. Sonraki cümledeki birebir sayı bu eşi seçemez. Payın dışındaki fark yanlış değerdir ve cümle düşer. Sıra tarih ya da olağan/güncel sırasına uyuyorsa ve uçlar kayıtla aynıysa yön düzeltilir. "1. saat" gibi sıra bildiren nokta cümle sonu sayılmaz. Doğrulama düşerse kullanıcıya genel red gider; `validation.debug` hangi sayı, atıf ve kontrolün düşürdüğünü tutar. Kayıttaki bir sayının birimi ile cevapta o sayıya takılan birim aynı ailede değilse o cümle düşer. Aynı cümlenin yakın kopyası da düşer.

Hasta metni sen diye, sıcak ve konuşur gibi anlatılır. İlk cevap soruyu tekrarlamaz. Yakınma varsa kısa bir geçmiş olsun konabilir. Tıbbi terim, kullanıcı anlamını sormadıkça açılmaz. Uzunluk soruya göredir; geniş bir soru birkaç paragraf sürebilir. Soruya değen sayılar cümle içinde verilir. Aynı ölçümün daha önceki değerleri de kayıtta duruyorsa onlar da söylenir; ilgili satır kısalık için atlanmaz. Birkaç öneri tire ile liste olur. Kayıt bir yakınmayı yanıtlamıyorsa ilaç adı ve doz olmadan genel güvenli öneri yazılır ve ne zaman hekime başvuracağı söylenir. Gerektiğinde `Dikkat:` paragrafı, ne zaman yardım isteyeceğini söyler. Soru yalnız eksik ya da belirsiz durumda sorulur. Hekimden en fazla bir kez söz edilir. Doz söylenmez. Sonuç ile hedef, sorulursa ayrılır. Yüksek, normal veya düşük yalnız kayıt böyle diyorsa söylenir. Tanı konmaz. Neden ve tedavi uydurulmaz. Kırmızı bayrak belirtileri modele gitmez.

Acil belirti modele gitmez; yanıt 112 çağrısıdır. Kalıp eşleşmesine ek olarak soru, acil durum betimlemelerine gömme benzerliği ile bakılır (`TEXT_EMERGENCY_SIMILARITY`, varsayılan 0.82). Yerel gömme yoksa bu yol boş kalır ve kalıp eşleşmesi durur. Doz ("kaç tane içeyim", "ne kadar içeyim"), ilaca devam ya da bırakma sorusu da modele gitmez; hekime yönlendirir ve kayıttaki son reçete satırını bırakır. "Hâlâ kullanıyor muyum" kayıt sorusudur. Şikayet, hikaye ve bulgular başlıkları cevapta kopyalanmaz. Tanı sorusu tanı satırındaki kod ve tarihi söyler. Doğrulama geçmezse uydurulan sayı gösterilmez. Sunucu model cümlesinin yerine başka bir satır yazmaz. `TEXT_ANSWER_MAX_TOKENS` varsayılanı 2000'dür; model adı `LLM_MODEL` ile değişir. `factCoverage` karşılaştırması büyük-küçük harfe bakmaz. Doz reddi, soruyla örtüşen satırları kayıttan birebir ekler; bu satırlar öneri değildir. "Alabilir miyim" bir tedavi reddidir. "Nefesim daralıyor, ne yapmalıyım?" kayıt sorusudur.

Her klinik yanıta sabit, rakamsız bir uyarı eklenir. `clinicalMeaningVerified` her zaman false'tur; kontrol karakter kontrolüdür.

Gerçek (fixture hash'i tutmayan) kayıt varsayılan olarak modele gitmez. `ALLOW_EXTERNAL_REAL_DATA=true` yapılandırılmış her hosta izin verir; bu, `openrouter.ai` dahil dışarı çıkış demektir. Yerel veya özel ağdaki vLLM için ayrı bayrak vardır: `LLM_LOCAL_INFERENCE=true`. Bayrak açıkken gerçek kayıt yalnız güvenilen hosta gider: `localhost`, `127.0.0.1`, `::1`, özel adres (10/8, 172.16/12, 192.168/16), link-local ve IPv6 ULA. DNS çözümüne bakılmaz. `LLM_LOCAL_HOSTS` ek makine adı ekler (`gpu.local`). `openrouter.ai` ve genel bir IP bu listeye yazılırsa süreç ayarı reddeder. Bayrak açık olsa bile host güvenilmiyorsa gerçek kayıt yine `external_real_data_blocked` olur. `/health` içindeki `textLlm.realDataMode` etkin izni söyler: `blocked`, `local` veya `external`. İki bayrak birden açıksa mod `external` kalır, çünkü adres sonradan buluta çevrilse de gerçek kayıt gider. Gönderilen sekiz hasta ve `example-facts.json` bu hash ile sentetik sayılır. `patient-08` tanı, anemi, karın yakınması, karaciğer ve pıhtı sorularının sentetik karşılığıdır; gerçek kayıt içermez. İstemci kendine sentetik diyemez.

## Uçlar

- `POST /api/v1/text-records` — zarfın kendisi
- `GET /api/v1/text-records` ve `GET /api/v1/text-records/{recordId}`
- `POST /api/v1/text/answer` — `{ "recordId", "question", "stream": false, "history": [], "validation": true }`
- `stream: true` → `text/event-stream`: `event: token`, sonra `event: done`

`done` gövdesi yetkilidir. Token olayları modelin ham akışıdır; sayı kontrolü bitince `done.answer` onları geçersiz kılabilir. `metrics.modelCalls`, `metrics.timeToFirstTokenMs` (sağlayıcıda ilk görünen token) ve `metrics.totalMs` döner.

`validation` isteğe bağlıdır. Yoksa `QA_VALIDATION_ENABLED` geçerlidir ve varsayılanı açıktır. `false` aynı istemi ve aynı kanıtı modele gönderir. Sayı, tarih, yön, neden, bayrak, cümle budama, büyük harf ve klinik uyarı uygulanmaz. Durum satırı ve köşeli atıf yine ayrılır: durum satırı yanıtta durmaz, atıf `citations` alanına gider. Bu biçim ayrıştırması doğrulama kapalıyken de çalışır. Gövde `validation: {"enabled": false}` olur. Acil belirti ve doz/tedavi reddi modelden önce çalışır ve bu anahtar kapalıyken de açıktır. Açıkken `validation.enabled` true olur. Bir cümle düşerse uyarı kendi satırında durur. Yanıt konuşma biçimindedir: kısa bir giriş, doğrudan cevap, gerektiğinde tire listesi ve `Dikkat:` paragrafı. Paragraf ve madde satırları durur; köşeli atıf yine `citations` alanına gider. Satır kaydırınca ondalık virgül ya da nokta cümle sonu sayılmaz. Atıf bölümünde duran sayı, tarih veya doz düzeni cümleyi düşürmez. Böyle bir cevap boş reddedilmez. Evet, Hayır, Kısmen ya da Kayıtta yeterli bilgi yok, soru bunu istiyorsa söylenir. Yüksek, düşük ve normal kayıttaki bayrak veya aralıkla uyuşmuyorsa o yöne çekilir. Kayıt bir yön söylemiyorsa sıfat cümlede kalır. Sayı içermeyen bir açıklama, aynı bölümü aralıksız gösteren ve sayısı kayıtta duran bir cümleden destek alır. `DURUM:` ile başlayan her satır durum satırıdır. Bölümdeki iki ile beş harflik büyük harf kısaltma, soruda aynı sözcükse, yanında rakam olmasa da bölüm kanıta girer.

İstek `history` alabilir: en fazla altı `{role, content}` turu. Kayıt iletisi önbellek öneki olarak durur; önceki turlar ondan sonra, soru en sonda gelir. Erişim, son soru ile bu turların sözcüklerini birlikte kullanır. Sayı kontrolü yine kayda bakar.

İstek gövdesinde `response_format` yoktur. `provider.zdr` ve `provider.data_collection=deny` yalnız `openrouter.ai` için gider. ZDR açıkken başka host reddedilir (`zdr_requires_openrouter`).

`OPENROUTER_ALLOW_FALLBACKS` varsayılanı `true`: yedek sağlayıcı ZDR ve veri toplamama şartının içinde kalır. `false` tek sağlayıcıya kilitler; o sağlayıcı 429 verirse tur boşa gider. `OPENROUTER_PROVIDER` doluysa `only` yine o sağlayıcıyı sabitler. 429, 502, 503 ve 529, ilk tokenden önce `LLM_RETRY_MAX` (varsayılan 2) kez, `LLM_RETRY_BACKOFF_SECONDS` (varsayılan 0.75, ikiye katlanır) bekleyerek yeniden denenir.

`LLM_DISABLE_THINKING` varsayılanı `true`: OpenRouter'a `reasoning.enabled=false` gider; OpenRouter olmayan sunucuya (vLLM dahil) `chat_template_kwargs.enable_thinking=false` gider. `auto` bu OpenRouter alanını yalnız model adında `qwen3`, `qwq` veya `deepseek-r1` varsa gönderir ve yerel sunucuya alan koymaz. `false` OpenRouter'a düşünme alanı koymaz; OpenRouter olmayan sunucuya `chat_template_kwargs.enable_thinking=true` gider. `LLM_REASONING` doluysa o kazanır ve düşünme alanı ayrıca eklenmez. `TEXT_ANSWER_MAX_TOKENS` üst sınırı 16000'dır; varsayılan 2000 kalır. Akıştaki `reasoning` / `reasoning_content` yanıta yazılmaz. `<think>` bloğu, kapanmamış olsa bile, yanıttan silinir. Yalnız boşluk veya düşünce gelirse durum `reasoning_only` olur; harness o soruyu kaydedip devam eder.

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

`LLM_MODEL` ne ise o gider. Düşünen modellerde `<think>` bloğu ve `reasoning_content` yanıtın dışında kalır. vLLM üzerinde Qwen3 düşünmesi varsayılan olarak kapalıdır (`chat_template_kwargs.enable_thinking=false`). Düşünmeyi açmak için `LLM_DISABLE_THINKING=false` kullanın; bu, OpenRouter olmayan sunucuya `enable_thinking=true` gönderir.

`DETERMINISTIC_EVIDENCE_VERIFICATION_ENABLED` varsayılanı kapalıdır. Açıkken model iç JSON üretir, her olgusal cümle kanıt kimliğine bağlanır ve sayı, tarih, tanı durumu, ilaç aktivitesi, nedensellik ve güncellik kodla denetlenir. vLLM'e `response_format` JSON şeması gider; sunucu 400 dönerse şemasız bir kez daha istenir. Katalog, çıktı token tavanı ve bir pay bırakılarak kısılır. Hatalı cümle düşer; sonuç reddedilmediyse ve kalan metin anlamlıysa yeniden üretim olmaz. Yeniden üretim yalnız sonuç reddedilirse ya da geçen cümle kalmazsa bir kez yapılır. O da yetmezse geçen cümleler ve soruya değen kısa kanıt özeti döner. JSON okunamazsa serbest metin yoluna düşülür ve yalnız sayı kontrolü çalışır. Kapalıyken bu katman çalışmaz; mevcut doğrulama aynen durur.

`QA_FULL_CONTEXT_ENABLED` varsayılanı kapalıdır ve yalnız doğrulamalı yolda etkilidir. Açıkken `cco-text/1` `text` alanı değiştirilmeden ve bütün olgular tek katalogda gider; bölüm ipucu ve kısma yoktur. İleti sırası sistem, kayıt ve katalog, onay, geçmiş, sorudur; ilk üçü sorudan bağımsızdır, önek önbelleği bunu kullanır. Token sayısı yerel vLLM `/tokenize` ucundan alınır (`LLM_TOKENIZE_URL`, varsayılan `LLM_BASE_URL` eksi `/v1`). Uç yoksa ya da yerel değilse karakter tahmini kullanılır ve `tokenCountSource` bunu söyler. Sınır `QA_CONTEXT_TOKENS` ile sunucunun `max_model_len` değerinin küçüğüdür; istem, çıktı tavanı ve 128 token pay sığmazsa model çağrılmaz, durum `context_overflow` olur. Yeniden üretim istemi sığmazsa yeniden üretim atlanır (`retry_skipped`). `validation.context` gönderilen kaydın hash'ini, olgu sayısını, geçmiş kesimlerini ve token dökümünü taşır. Model ağ geçidi `/tokenize` ucunu taşımaz.

`scripts/run_full_context_eval.py` 33 soruyu Full ve Oracle kollarında çalıştırır. Oracle kolu yalnız altın olguların kaynak satırlarını ve o olguları gönderir. `--dry-run` model çağırmadan istem ve sığma raporu üretir; `--tokenize` sayımı yerel sunucudan alır. Çıktı `data/evaluation` altında kalmalıdır.

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
