# CCO V2 training data

Üretim modeli için esas giriş biçimi `{id, subjectId, asOf, text}` text CCO'dur.
Eski sistemden kalan raw fact export'u `legacy_audit_only` sınıfındadır. Text
CCO ile aynı snapshot veya renderer sürümüne ait olduğu varsayılmaz. Gold
etiket, klinik statü ya da aktif ilaç durumu üretmez. Gerçek sağlık verileri bu
dizine veya Git'e kopyalanmaz.

## Kaynak denetimi

```powershell
python scripts/audit_training_sources.py `
  --cco-dir C:\DigitalTwinData\cco `
  --facts-dir C:\DigitalTwinData\facts `
  --output output\training-source-audit.json
```

Rapor yalnız toplu sayaçlar ve corpus hash'leri içerir. Hasta kimliği, klinik
metin, tanı adı veya ölçüm değeri yazmaz. Şunları kontrol eder:

- text CCO ile fact dosyalarının hasta bazında birebir eşleşmesi;
- JSON kimlikleri, zaman damgaları, fact kimlikleri ve tarih aralıkları;
- fact tanımlarının katalogda bulunması;
- legacy fact ile text arasındaki zaman ve statü farklarının yalnız tanısal
  sayaç olarak kaydedilmesi;
- fact export'unda bulunmayan hasta beyanı, alerji, aile öyküsü ve demografi;
- tarihsel rekonstrüksiyon, eski günler ve gelecekte bitiyormuş gibi görünen
  günlük ölçüm pencereleri.

`legacy_audit_only`, fact farklarının otomatik CCO hatası olmadığı ve bu
fact'lerin eğitim gold'una kaynak olamayacağı anlamına gelir.

Eğitim anotasyonu değiştirilmeyen text CCO'ya `snapshotId`, `contentHash`,
`evidenceId` ve satır span'ı ile bağlanır. `factId` opsiyonel legacy izidir;
birincil kanıt değildir. `evidenceId`, snapshot + content hash + satır konumu +
span hash ile üretildiği için kaynak metin değişirse eski anotasyon geçersiz
olur.

## Ana politika

Kişisel bağlam önceliklidir. İlgili kayıt varsa `required` veya `helpful`
etiketlenir. `not_supported`, Gemma'nın mevcut “kayıtta bulamadım” davranışını
koruyan küçük bir regresyon sınıfıdır; veri setinin ana hedefi değildir. İzin
verilen üst oran manifestte belirtilir ve validator tarafından uygulanır.

Seed veri yalnız şema ve araç smoke testidir. Klinik olarak gözden geçirilmiş
gold veri değildir ve doğrudan eğitime verilmemelidir.

## Doğrulama

```powershell
python scripts/validate_training_dataset.py `
  --dataset data/training_v2/seed/annotations.jsonl `
  --manifest data/training_v2/seed/manifest.json
```

Validator şunları kontrol eder:

- katı alan ve enum sözleşmesi;
- CCO dosyasının yalnız `data/` altında bulunması;
- `snapshotId` ile CCO kimliğinin eşleşmesi;
- `contentHash`, `evidenceId`, bölüm, satır aralığı ve span hash eşleşmesi;
- yalnız metinde açıkça bulunan assertion/aktiflik/iyileşme/nedensellik
  işaretlerinin anotasyona alınması;
- aynı kök senaryonun farklı split'lere sızmaması;
- kişiselleştirme kararı ile pozitif kanıtların tutarlılığı;
- `not_supported` oranının manifest sınırını aşmaması;
- safety bundle, nedensellik ve ilaç statü kontrolü gibi yapısal kurallar.

Validator klinik doğruluk veya cevap kalitesini onaylamaz. Raporlara soru,
hasta kimliği ya da klinik değer yazmaz.

## Hibrit sentetik üretim

Ana sentetik hat iki parçalıdır:

- OpenAI Responses API yalnız tamamen kurgusal kök taslakları, çeşitlilik
  adayları ve üç doğal Türkçe soru varyasyonu üretir. Çıktı strict JSON Schema
  ile alınır.
- Python kök split'ini varyantlardan önce belirler; olay grafiğini kurar;
  dört varyantı üretir; text CCO'yu render eder; `evidenceId` ve `sourceSpan`
  bağlarını doğrular.

Modelin çeşitlilik ve klinik ilişki fikirleri doğru kabul edilmez. İnsan/klinik
incelemesine kadar cevap `candidateAnswer` alanındadır, `goldAnswer=null`,
`reviewStatus=unreviewed` ve `trainable=false` kalır.

Her taslak `eventType`, `recordSection` ve `subjectScope` taşır. Bu alanların
template'e göre izin verilen üçlüsü `cco-hybrid-taxonomy/1.0.0` sözleşmesiyle
kapalıdır. Uyumsuz üçlüler olay grafiği veya kaynak CCO yazılmadan reddedilir.
Uyku dışı wearable kavramlarının `Uyku (yüzük)` bölümüne, aile öyküsü/alerji/
yaşam tarzı kayıtlarının hastanın kalıcı tanısına dönüşmesi ayrıca semantik
kurallarla engellenir. Renderer sonrası gerçek bölüm başlığı yeniden doğrulanır.

OpenAI ile 20 kök / 80 varyant pilotu:

```powershell
$env:OPENAI_API_KEY="..."
$env:OPENAI_SYNTHETIC_MODEL="gpt-6-luna"
uv run python scripts/generate_hybrid_synthetic_pilot.py `
  --workspace output\hybrid-pilot-workspace `
  --provider openai
```

Aynı strict JSON Schema isteği OpenRouter üzerinden de çalışır:

```powershell
$env:OPENROUTER_API_KEY="..."
$env:OPENROUTER_SYNTHETIC_MODEL="openai/gpt-6-luna"
uv run python scripts/generate_hybrid_synthetic_pilot.py `
  --workspace output\hybrid-pilot-workspace `
  --provider openrouter
```

API anahtarı olmadan aynı sözleşmeyi ve Python derleyicisini test eden yerel
fixture pilotu:

```powershell
uv run python scripts/generate_hybrid_synthetic_pilot.py `
  --workspace output\hybrid-pilot-workspace `
  --provider fixture
```

`hybrid-report.json` çeşitlilik, yapısal doğruluk, kaynak bütünlüğü, dış isteğe
giren veri sınıfları, taksonomi hata sayıları, inceleme durumu ve eğitime
uygunluk sayılarını içerir. `taxonomyErrors=0` ve
`taxonomyRenderMismatches=0` sağlanmadan kök sayısı artırılmaz.
Pilot üretimi LoRA eğitimini başlatmaz.

## Eski deterministik pilot ve gold inceleme kuyruğu

```powershell
python scripts/prepare_training_pilots.py `
  --workspace output\training-pilot-workspace `
  --synthetic-count 480 `
  --cco-dir C:\DigitalTwinData\cco `
  --questions C:\Users\omercftc\Documents\Projects\doktorsitesi-scraper\processed\cco_benchmark_questions.jsonl `
  --gold-output output\gold-review-60 `
  --gold-count 60
```

Bu komut geriye dönük deterministik smoke paketi ve gerçek CCO'lar için yerel
inceleme kuyruğu üretir. Sentetik örnekler önce kontrollü olay grafı olarak oluşturulur, sonra gerçek
dört alanlı text CCO biçimine render edilir. Gold kanıtlar graph event'lerinden
render edilmiş `evidenceId` span'larına yeniden çözülür. Eşleşmeyen event için
örnek üretilemez.

60 gerçek kayıt için oluşturulan kuyruk yalnız yerel, metadatalı ve
`reviewStatus=pending` durumundadır. İnsan incelemesi yapılmadan bunlar gold
sayılmaz. Araç dış model çağırmaz ve LoRA eğitimi başlatmaz.

