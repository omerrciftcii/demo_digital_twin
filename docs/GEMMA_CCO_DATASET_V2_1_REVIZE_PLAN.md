# Gemma CCO Dataset V2.1 — Revize Eğitim ve Değerlendirme Planı

**Tarih:** 21 Ekim 2026  
**Durum:** Tasarım/pilot aşaması  
**Ana değişiklik:** Relevance selector ilk aşamada kayıt silen bir kapı bekçisi olmayacak. Bütün kayıtları görünür tutan bir annotator/ranker olarak çalışacak.

---

## 1. Karar özeti

Bu sürümde önceki plandaki iki aşamalı yaklaşım korunuyor, fakat bilgi akışı değiştiriliyor:

```text
Kaynak CCO
→ kaynağa sadık kanonik olaylar
→ deterministik tarih/doğrudan sorgu hesapları
→ relevance annotator/ranker
→ tam bağlam + önem anotasyonları + kritik kanıt paketleri
→ mevcut Gemma cevap modeli
```

İlk pilotta ranker hiçbir olayı geri döndürülemez biçimde gizlemeyecek. Cevap modeli tam CCO'ya erişmeye devam edecek; ranker yalnızca dikkat önceliği, kullanım biçimi ve nedensellik sınırı önerecek.

Hard filtering ancak aşağıdaki koşullar birlikte sağlandıktan sonra ayrı bir optimizasyon olarak değerlendirilebilir:

- Kritik kayıt recall'u yeterli ve güven aralığıyla doğrulanmışsa
- Bağlantılı kanıt paketleri birlikte korunuyorsa
- Tam bağlamlı sisteme göre önemli kayıt kaçırma artmıyorsa
- Uzun bağlam maliyeti gerçek üretim yükünde ölçülmüşse
- Güvenli fallback tam bağlama dönebiliyorsa

Bu nedenle ilk hedef “en az kayıtla cevap vermek” değil, **doğru kayda öncelik verirken önemli hiçbir kanıtı görünmez yapmamaktır**.

---

## 2. Projenin amacı

Model, hastanın CCO kaydı ve güncel sorusu üzerinden:

1. Soruyla ilgili olabilecek kayıtları tanımalı ve sıralamalı.
2. İlgisiz eski kayıtları bugünkü şikâyete nedensel olarak bağlamamalı.
3. Eski fakat önemli kronik hastalık, ameliyat, alerji, ilaç reaksiyonu veya bırakma bilgisini kaybetmemeli.
4. Ön tanı, kesin tanı, hasta beyanı ve aile öyküsünü ayırmalı.
5. Her wearable metriğinin kendi ölçüm zamanını kullanmalı.
6. Reçete edilme ile aktif kullanımı ayırmalı.
7. Kayıtta bulunmayan bilgi üretmemeli.
8. Kanıt yoksa veya çelişkiliyse bunu açıkça söylemeli.
9. Aciliyet kararını kayıt yeterliliğinden bağımsız değerlendirmeli.

Temel problem yalnızca cevap üretmek değildir. Üç ayrı problem vardır:

```text
Kaynağı doğru temsil etme
→ soruyla kanıt arasındaki ilişkiyi kurma
→ kanıt sınırları içinde cevap verme
```

Bu üç katman ayrı ayrı ölçülecektir.

---

## 3. Model ve donanım

### Üretim modeli

- Model: Gemma 4 26B-A4B
- Sunucu alias'ı: `gemma4-26b-a4b`
- Quantization: 4-bit AWQ
- Sunum motoru: vLLM 0.20.2
- Reasoning: Kapalı
- Mevcut VRAM kullanımı: Yaklaşık 18,8 GB

Tekrarlanabilirlik için yalnız alias yeterli değildir. Aşağıdakiler manifestte dondurulacaktır:

- Gerçek model repo/checkpoint adı
- Revision/commit
- Tokenizer revision
- Chat template hash'i
- Quantization config
- vLLM komutu ve sürümü
- Context length
- Sampling parametreleri

### Üretim GPU'su

- Fiziksel kart: NVIDIA L40S 48 GB
- Ayrılan vGPU: NVIDIA L40S-24Q
- Kullanılabilir VRAM: 24 GB

18,8 GB gözlenen kullanımın ağırlık, CUDA graph ve KV cache dağılımı ayrıca ölçülecektir. Teorik olarak 5,2 GB serbest alan kaldığı varsayılmayacaktır. Gerçek prompt uzunluğu ve eşzamanlı isteklerle yük testi yapılacaktır.

### Eğitim ortamı

- Google Colab NVIDIA A100
- A100 40 GB veya 80 GB varyantı çalışma başında kaydedilecektir.
- 26B toplam parametrenin bf16 ham ağırlıkları yaklaşık 52 GB'tır; 40 GB A100'de tam bf16 ağırlık varsayılmayacaktır.
- QLoRA/NF4 kullanılırsa Gemma 4 MoE uzman parametrelerinin gerçekten quantize edilip edilmediği dtype/storage raporuyla doğrulanacaktır.
- `all-linear` ifadesinin uzman parametrelerini kapsadığı varsayılmayacak; eğitilebilir parametre listesi ve gradient alan parametreler raporlanacaktır.

---

## 4. Önceki çalışmadan korunan dersler

Önceki LoRA denemesinde:

- Verinin yaklaşık %36'sı CCO görmeyen doktor sitesi cevaplarından oluşuyordu.
- Model kayda dayanmayı değil, kendinden emin ve kalıp cevap vermeyi öğrenmişti.
- Soru ve cevap şablonları aşırı tekrar ediyordu.
- Full-sequence loss nedeniyle uzun sistem/CCO girdisi de eğitim hedefiydi.
- Satır bazlı rastgele split sızıntı oluşturuyordu.
- LoRA GPTQ ile base AWQ karşılaştırılarak eğitim ve quantization etkileri karıştırılmıştı.

Yeni çalışmada:

- Doktor sitesi cevapları kullanılmayacak.
- Assistant-only loss gerçek token maskesiyle doğrulanacak.
- Hasta, soru ailesi ve türetim kökü bazlı split yapılacak.
- Aynı base ve aynı quantization hattı karşılaştırılacak.
- Eval loss tek başına başarı göstergesi olmayacak.

---

## 5. V2.1 mimarisi

### 5.1 Kaynak katmanı

Kaynak CCO değişmeden saklanır. Kanonik parser'ın ürettiği her alan kaynaktaki konumuna bağlanır.

```json
{
  "source_document_id": "cco_018",
  "source_span": {
    "section": "İlaç kullanım beyanları",
    "start_char": 1432,
    "end_char": 1490
  },
  "parser_version": "canonicalizer_v0.1"
}
```

### 5.2 Kanonik olay katmanı

Parser yalnızca kaynağın desteklediği bilgiyi çıkarır. Kaynakta açık olmayan durumlar tahmin edilmez.

Yanlış:

```text
2021 tarihli akut tanı → resolved
```

Doğru:

```text
Son kayıt: 2021
Klinik durum: unknown
Bugünkü yakınmayı destekleme gücü: soru ve diğer kanıta göre ayrıca değerlendirilir
```

### 5.3 Deterministik motor

Kodla güvenilir biçimde yapılabilecek işler LLM'e bırakılmaz:

- Tarih farkları
- “Dün gece” penceresi
- Ölçümün yaşı
- Doğrudan değer/tarih sorgusu
- Birim dönüşümü
- Veri kapsama boşluğu
- Kaynakta değer bulunup bulunmadığı

Deterministik sonuçlar da kaynak olayı ile karıştırılmaz; `derived_fact` olarak tutulur.

### 5.4 Relevance annotator/ranker

Ranker bütün olayları değerlendirir, fakat ilk aşamada hiçbirini silmez.

Her olay için önerilen çıktı:

```json
{
  "event_id": "evt_014",
  "relevance": "high",
  "use": "background_only",
  "mention_permission": "allowed",
  "causal_permission": "not_supported",
  "safety_critical": true,
  "rank_score": 0.91,
  "reason_code": "medication_stop_statement"
}
```

Bu anotasyon kaynak gerçeği değildir. `annotation_source`, model sürümü ve güven skoru ile ayrı saklanır.

### 5.5 Context composer

İlk pilotta cevap modeline şunlar verilir:

- Kullanıcının güncel mesajı
- Tam CCO
- Ranker'ın kısa önem/kullanım anotasyonları
- Deterministik tarih ve doğrudan sorgu sonuçları
- Kritik bağlantılı kanıt paketleri
- Kanıt yeterliliği ve çelişki uyarıları

Ranker düşük puan verdi diye olay metinden çıkarılmaz. Uzun bağlam sorunu görülürse önce bölüm sıralama, özetleme ve kayıpsız sıkıştırma denenir. Hard filtering son seçenektir.

### 5.6 Cevap katmanı

İlk denemede mevcut Gemma cevap modeli kullanılabilir. Yeni cevap SFT otomatik bir sonraki aşama değildir. Önce anotasyonların mevcut modele ölçülebilir katkı sağlayıp sağlamadığı test edilir.

---

## 6. Tek `answer_mode` yerine üç ayrı eksen

`urgent_triage` ve `record_insufficient` aynı anda doğru olabilir. Bu nedenle tek bir karşılıklı dışlayan `answer_mode` kullanılmayacaktır.

### Görev türü

```text
symptom_guidance
direct_record_lookup
wearable_interpretation
medication_question
record_explanation
follow_up_conversation
multi_intent
```

### Kanıt yeterliliği

```text
sufficient
partial
none
conflicting
not_applicable
```

### Güvenlik eylemi

```text
routine
caution
urgent_assessment
emergency
```

Bunlara ek olarak cevap politikası türetilebilir:

```text
personalized_grounded
general_information_only
clarification_needed
direct_fact_response
```

Bu son alan kaynak gerçeği değil, yukarıdaki eksenlerden türetilen cevap politikasıdır.

---

## 7. Kanonik veri sözleşmesi

### 7.1 Örnek kabı

```json
{
  "example_id": "ex_000001",
  "root_scenario_id": "root_0042",
  "patient_family_id": "synthetic_patient_018",
  "question_family_id": "fatigue_ambiguous_03",
  "counterfactual_group_id": "cf_0042",
  "split": "train",
  "as_of": "2026-10-20T10:00:00+03:00",
  "source_cco": "...",
  "events": [],
  "derived_facts": [],
  "question": "...",
  "task_labels": {},
  "evidence_requirements": {},
  "gold_annotations": [],
  "gold_answer": "...",
  "provenance": {},
  "validation": {}
}
```

### 7.2 Olay örneği

```json
{
  "event_id": "evt_014",
  "event_type": "medication_statement",
  "source_type": "patient_report",
  "experiencer": "patient",
  "asserted_at": "2023-06",
  "date_precision": "month",
  "current_use_status": "unknown",
  "assertion_role": "patient_finding",
  "evidence_origin": "explicit_source_text",
  "source_span": {
    "section": "Kullandığı ilaçlar (hasta beyanı)",
    "start_char": 1200,
    "end_char": 1260
  },
  "parser_version": "canonicalizer_v0.1"
}
```

### 7.3 Kaynak, türetim, tahmin ve gold ayrımı

Bu dört şey aynı alanda tutulmayacaktır:

```text
source_event       = kaynakta açıkça yazan
derived_fact       = kodla hesaplanan
model_annotation   = ranker tahmini
gold_annotation    = insan/onaylı hedef
```

Kaynakta gözlenemeyen bir durum sentetik gold'a eklenmez. Sentetik olay grafiği sıfırdan oluşturuluyorsa gerçek durum bilinebilir; gerçek CCO parse ediliyorsa yalnız gözlemlenebilir durum etiketlenir.

---

## 8. Olay türleri ve önemli ayrımlar

### Tanı

```text
verification_status: confirmed | provisional | refuted | unknown
clinical_status: active | recurrence | inactive | resolved | unknown
```

`confirmed`, geçmişteki kaydın doğrulama niteliğidir; bugün aktif veya nedensel olduğu anlamına gelmez.

### İlaç

İki farklı olay türü kullanılır:

```text
prescription / medication_request
medication_statement
```

Reçete, kullanımı kanıtlamaz. Hasta kullanım beyanı da yalnızca beyanın yapıldığı zamandaki bilgiyi taşır.

### Aile öyküsü

`experiencer` ve `relationship` zorunludur. Aile bireyinin hastalığı hastanın kendi tanısına dönüşemez.

### Wearable

Her metrik ayrı zaman taşır:

```text
metric
value
unit
effective_start
effective_end
synced_at
device
quality
coverage_window
missingness
```

`synced_at`, `effective_at` yerine kullanılamaz.

### Trend ve ortalamalar

```text
window_start
window_end
n_observed
n_expected
missingness
derived_from
method
```

Görünen son yedi değer ile raporlanan yedi günlük ortalamanın aynı pencereye ait olduğu varsayılmaz.

### Serbest metin assertion rolü

```text
patient_finding
generic_reference
recommendation
planned_action
performed_action
administrative_or_test_content
unknown
```

Örneğin rapor içindeki genel TR1–TR5 açıklaması hastaya ait TR5 bulgusu sayılamaz.

### Çelişkiler

```json
{
  "conflict_group_id": "conflict_07",
  "resolution_status": "unresolved"
}
```

“Son kayıt kazanır” kuralı aynı not içindeki çelişkileri otomatik çözmek için kullanılmaz.

---

## 9. Bağlantılı kanıt paketleri

Olaylar her zaman bağımsız değildir. Aşağıdakiler birlikte değerlendirilmelidir:

- Reçete + aktif kullanım beyanı + bırakma kaydı
- Tanı + son kontrol + sonradan reddedilme bilgisi
- Alerji + reaksiyon + sonraki negatif/pozitif beyan
- Tetkik istemi + tetkik sonucu + klinik yorum
- Ameliyat + komplikasyon + kontrol
- Ölçüm + kalite + kapsama boşluğu

Örnek:

```json
{
  "evidence_bundle_id": "bundle_med_12",
  "event_ids": ["rx_01", "med_statement_02", "stop_03"],
  "bundle_type": "medication_status",
  "critical_if_question_tags": ["medication", "interaction", "restart"]
}
```

Ranker reçeteyi yüksek, bırakma kaydını düşük puanlasa bile paket kritikse üçü de görünür ve işaretli kalır.

---

## 10. Evidence gold yapısı

Exact event-set tek doğru cevap varsaymayacaktır. Eşdeğer veya tekrarlı kanıtlar olabilir.

```json
{
  "must_include": ["evt_allergy_01"],
  "may_include": ["evt_allergy_followup_02"],
  "must_exclude_from_causal_use": ["evt_old_uri_04"],
  "alternative_evidence_sets": [
    ["evt_lab_01"],
    ["evt_lab_duplicate_02"]
  ],
  "critical_bundles": ["bundle_med_12"]
}
```

Uzun `excluded_events` listesi üretim çıktısında zorunlu tutulmayacaktır. Zor negatifler eğitim/gold katmanında saklanabilir; ranker çıktısı kısa kalabilir.

---

## 11. Veri kaynakları

### 149 gerçek CCO

İncelenen 149 CCO wearable ağırlıklıdır:

- 35 kayıt yalnız yaş-cinsiyet içermektedir.
- 33 kayıt hastane tanısı bölümü içerir.
- 29 kayıt reçete, 29 kayıt hastane laboratuvarı içerir.
- 96 kayıt uyku bölümü içerir.

Bu nedenle 149 CCO, yüz zengin klinik geçmiş varmış gibi kullanılmayacaktır. Gerçek CCO'lar:

- Format ve olay örüntüsü keşfi
- Parser geliştirme
- Gerçek uzunluk/eksiklik dağılımı
- Zor vaka çıkarma

için kullanılacaktır.

Kaynakta görülen isim, imza, protokol, kurum ve serbest metin tanımlayıcıları anonimleştirme denetimine dahil edilir. `Deneme` ve benzeri test içerikleri karantinaya alınır; klinik hikâyeye otomatik onarılmaz.

### Gerçek şikâyetler

Henüz gelmemiş olmaları altyapı pilotunu engellemez. Ancak gerçek hasta diline genelleme iddiası gerçek şikâyetler gelmeden yapılamaz.

Geldiklerinde:

- Kimliksizleştirme
- Birebir ve semantik tekrar kümeleme
- Konu, uzunluk, belirsizlik, aciliyet ve çoklu niyet dağılımı
- Yazım hataları ve doğal kısa anlatımlar
- Aynı şikâyetin parafraz ailesi

çıkarılacaktır.

### Doktor sitesi

- Cevaplar eğitimde kullanılmayacaktır.
- Sorular yalnızca doğal soru dili kaynağı olabilir.
- Nihai soru metni anlamı değiştirebildiği için gold, render edilmiş son CCO + nihai soru üzerinden yeniden doğrulanacaktır.

---

## 12. Senaryo matrisi

Önceki tek yüzdeli dağılım aynı eksende olmayan sınıfları karıştırıyordu. Yeni veri matrisi üç katmanlı olacaktır.

### Ana görev ekseni

- Semptom danışmanlığı
- Doğrudan kayıt sorgusu
- Wearable yorumlama
- İlaç sorusu
- Kayıt açıklama
- Takip konuşması
- Çoklu niyet

### Kanıt ilişkisi ekseni

- Güncel ve doğrudan ilgili
- Geçmiş ama arka plan açısından önemli
- Güvenlik kısıtı
- İlgisiz
- Yetersiz
- Çelişkili
- Birden fazla eşdeğer kanıt

### Ortogonal risk etiketleri

- Eski akut kayıt
- Kronik durum
- Tekrarlayan durum
- Ön tanı
- Aile öyküsü
- Reçete/aktif kullanım
- İlaç bırakma
- Alerji/reaksiyon
- Ameliyat
- Wearable tazelik
- Ölçüm kalitesi
- Ondalık/birim
- Uzun bağlamda sonda kritik olay
- Başka kişiye ait şikâyet
- CCO içinde talimat görünümlü metin
- Çelişen kaynak
- Çoklu soru

Bir örnek birden fazla risk etiketi taşıyabilir. İlaç vakaları yalnız ayrı bir `%5` kutusuna sıkıştırılmaz.

---

## 13. Karşı-olgusal veri

Karşı-olgusal örnekler toplam verinin yaklaşık %20–30'u olabilir, fakat iki tür test zorunludur.

### İlgili değişiklik testi

İlgili olay değişince karar değişmelidir:

```text
19–20 Ekim uyku kaydı var → sorulan gece cevaplanabilir
3 Ekim uyku kaydı var     → dün gece cevaplanamaz
Uyku kaydı yok            → cevaplanamaz
```

### İlgisiz değişiklik testi

İlgisiz olay eklenince temel karar değişmemelidir:

```text
Güncel boğaz ağrısı + eski İYE
Güncel boğaz ağrısı + eski burkulma
Güncel boğaz ağrısı + daha fazla ilgisiz kayıt

→ Nedensel karar korunmalı
```

Ek invariant'lar:

- Olay sırası değişince temel karar korunmalı.
- Kritik olay bağlamın başında/ortasında/sonunda olabilmeli.
- Event ID biçimi etiketi ele vermemeli.
- Oracle gold alanları model girdisine sızmamalı.
- Aynı karşı-olgusal aile farklı split'lere ayrılmamalı.

---

## 14. Otomatik validator

Validator yalnızca hangi kapsamı doğruladığını raporlayacaktır. `PASS`, tüm klinik doğruluğun onaylandığı anlamına gelmeyecektir.

### Deterministik kontroller

- JSON Schema ve ID geçerliliği
- Kaynak span'ın varlığı ve sınırları
- Tarih hassasiyeti ve zaman aralıkları
- Saat dilimi ve “dün gece” hesabı
- Ölçüm zamanı ile senkronizasyon zamanı ayrımı
- Birim, karşılaştırıcı ve ondalık dönüşümü
- Trend penceresi ve eksik günler
- Seçili olay ve bağlantılı paket bütünlüğü
- Split kökeni ve yakın tekrar
- Output JSON parse oranı
- Training label maskesi
- Truncation ve cevap kaybı
- Şablon artıkları ve test içerikleri

### Semantik/klinik denetim

Bunlar kodla tam doğrulanmış sayılmaz:

- Olumsuzlama
- Aile bireyi/hasta ayrımı
- Bugünkü klinik durum
- Nedensellik
- Kanıt yeterliliği
- Gerekli güvenlik eylemi
- Önerinin klinik uygunluğu

Bu alanlarda model eleştirmeni ve insan denetimi kullanılır.

### Zorunlu invariant örnekleri

```text
Yalnız reçete varsa current_use=active üretilemez.
Eski/akut olmak resolved kanıtı değildir.
Ölçüm tarihi yoksa synced_at onun yerine geçemez.
Kaynakta >90 yazıyorsa cevapta tam 90 denemez.
Aile öyküsü hastanın tanısına dönüşemez.
Genel sınıflandırma açıklaması kişisel bulgu sayılamaz.
Kritik paketin bir parçası kayıpsa kanıt sessizce yeterli sayılamaz.
CCO içindeki talimat görünümlü metin sistem talimatı sayılamaz.
```

### Kasıtlı bozulma testleri

Validator yalnız temiz örneklerle sınanmayacaktır. Her gold örnekten kontrollü bozuk sürümler üretilecektir:

- Yanlış tarih
- Yanlış kişi/experiencer
- Ondalık veya birim hatası
- Reçeteyi aktif ilaç sayma
- Kritik kaydı çıkarma
- Ön tanıyı kesinleştirme
- Genel rapor açıklamasını kişisel bulgu yapma
- Synced_at değerini ölçüm tarihi yapma

Yanlış kabul ve yanlış ret oranları ayrı raporlanacaktır.

---

## 15. İnsan anotasyonu

Kritik gold örneklerde iki bağımsız değerlendirici ve anlaşmazlık çözümü kullanılmalıdır.

Anotasyon soruları atomik olacaktır:

```text
Kaynak bu statüyü gerçekten söylüyor mu?
Hangi kayıtlar mutlaka görünür kalmalı?
Hangi kayıtlar kullanılabilir ama nedensel değildir?
Kanıt yeterli, kısmi, yok veya çelişkili mi?
Aciliyet eylemi nedir?
Desteksiz kişisel iddia var mı?
Tarih ve ilaç aktifliği doğru mu?
```

İnsan düzeltme oranı kategori, annotator ve hata türü bazında tutulur. Tek “iyi/kötü” puanı kullanılmaz.

---

## 16. Split stratejisi

Split üretimden önce kök bazında yapılır.

Aynı split içinde tutulacaklar:

- Aynı gerçek veya sentetik hasta kökü
- Aynı olay grafiği türevleri
- Aynı karşı-olgusal aile
- Aynı gerçek şikâyet ve parafrazları
- Aynı soru/template ailesi
- Çok benzer CCO bağlamları

İki ayrı OOD görünümü oluşturulur:

1. **Yeni hasta görünümü:** Bilinen görev aileleri, tamamen yeni hasta kökleri
2. **Yeni soru görünümü:** Yeni hasta kökleri ve görülmemiş soru/anlatım aileleri

Birbiriyle çapraz bağlanan bütün aileleri tek dev connected component haline getirmemek için üretim tasarımı split öncesinde kontrol edilir. Donmuş split manifesti, türetim kökeni ve erişim geçmişi saklanır.

Final test:

- Prompt seçimi için kullanılmaz.
- AWQ calibration için kullanılmaz.
- DPO çifti üretiminde kullanılmaz.
- Hata analizi yapıldıktan sonra “gizli test” statüsünü kaybeder ve yeni test gerekir.

---

## 17. Minimum güvenilir pilot

### Pilot A — Sözleşme ve validator

- 60 elle hazırlanmış gold örnek
- 10 hata ailesi × 6 bağımsız vaka
- Kritik vakalarda ikinci bağımsız değerlendirme
- Gold'un görünen kanıttan çıkarılabilir olduğunun doğrulanması

### Pilot B — Bozulma testi

- 120 kasıtlı bozuk örnek
- Her gold için iki kontrollü bozma
- Validator false accept ve false reject matrisi

### Pilot C — Sentetik fabrika smoke test

- 120 bağımsız senaryo/hasta kökü
- Her kökte dört varyant
- Toplam 480 örnek
- Kök bazında 96/12/12 ayrımı:
  - 384 train
  - 48 development
  - 48 synthetic smoke-test

48 örneklik smoke-test güvenlik hata oranı iddiası için kullanılmaz; yalnız veri hattını sınar.

### Pilot D — Mimari karşılaştırma

- Tercihen 100 ayrı insan denetimli vaka
- Mümkün olduğunca ayrı hasta kökleri
- Kaynak yetersizse bunun keşif seti olduğu açıkça belirtilir

Her vakada aynı cevap modeliyle şu koşullar karşılaştırılır:

| Koşul | Modelin gördüğü |
|---|---|
| A — Baseline | Tam CCO, mevcut prompt |
| B — Oracle anotasyon | Tam CCO + insan gold relevance anotasyonu |
| C — Predicted anotasyon | Tam CCO + gerçek ranker anotasyonu |
| D — Oracle filtered, tanısal | Yalnız gold kanıt paketi |
| E — Predicted filtered, tanısal | Yalnız ranker'ın seçtiği paket |

D ve E ilk üretim mimarisi değildir; bilgi darboğazının zararını ölçmek için tanısal koşullardır.

Bu karşılaştırma şunları ayırır:

- Anotasyonun cevap modeline katkısı
- Ranker hatasının oracle'a göre maliyeti
- Hard filtering'in ek zararı
- Cevap modelinin doğru kanıtla bile yaptığı hatalar

---

## 18. Relevance annotator/ranker eğitimi

Önce promptsuz veya prompt tabanlı baseline ölçülür. Eğitimin gerçekten gerekli olduğu gösterilmeden LoRA yapılmaz.

Eğitim gerekirse yaklaşık 2.000 yüksek kaliteli örnekle başlanabilir. Raporlanacak hacimler:

- Toplam satır
- Bağımsız hasta/senaryo kökü
- Soru ailesi
- Risk etiketi dağılımı
- Bağlam uzunluğu dilimleri

Ranker hedefleri:

- Olay relevance sırası
- Kullanım biçimi
- Nedensellik izni
- Kritik güvenlik işareti
- Kanıt paketi bütünlüğü

Metrikler:

- `must_include` recall
- Kritik kayıt recall
- Recall@k ve nDCG benzeri sıralama ölçüleri
- `must_exclude_from_causal_use` doğruluğu
- Gereksiz yüksek öncelik oranı
- Boş/ilgisiz bağlam precision ve recall'u
- Bağlantılı paket bütünlüğü
- Uzun bağlam baş/orta/son performansı

Ranker yanlış sınıflandırsa bile ilk aşamada tam bağlam korunduğundan hata geri döndürülemez olmamalıdır.

---

## 19. Cevap modeli kararı

Yeni cevap SFT'ye geçmeden önce Pilot D sonuçları incelenir.

### Cevap SFT gerekmeyebilir, eğer:

- Tam bağlam + gold anotasyon mevcut Gemma'da hedef davranışı sağlıyorsa
- Predicted anotasyon ile oracle farkı kabul edilebilir düzeydeyse
- Kalan hatalar parser/ranker kaynaklıysa

### Cevap SFT gerekir, eğer:

- Gold anotasyonla bile tarih, statü, grounding veya nedensellik hatası devam ediyorsa
- Cevap biçimi ve hasta dili sistematik biçimde yetersizse
- Prompt düzeltmesi aynı davranışı güvenilir sağlayamıyorsa

Cevap eğitimi yalnız kusursuz gold anotasyonlarla yapılmaz. Üretimdeki ranker hatalarına dayanıklılık için:

- Out-of-fold ranker çıktıları
- Kontrollü eksik anotasyon
- Kontrollü çelişkili anotasyon
- Düşük güvenli ranker çıktıları

eklenebilir. Ancak modelden görünmeyen kanıtı tahmin etmesi beklenmez; tam CCO erişimi korunur.

---

## 20. SFT teknik kontrolleri

- Assistant-only loss kullanılacaktır.
- En az 10 örnekte token etiketleri decode edilerek system, CCO, user ve padding tokenlarının `-100`; yalnız assistant cevabının hedef olduğu doğrulanacaktır.
- Chat template'teki generation mask desteği test edilecektir.
- Hedef cevabın truncation ile kesilmediği kontrol edilecektir.
- Reasoning kapalı ayarı eğitim ve inference template'inde eşleştirilecektir.
- Eğitilebilir parametrelerin listesi, dtype'ları ve gradient normları raporlanacaktır.
- Tek epoch sonunda davranışsal eval yapılmadan daha fazla epoch'a geçilmeyecektir.

Eval loss tek başına erken durdurma metriği değildir.

---

## 21. DPO

DPO zorunlu değildir. Yalnız SFT sonrasında dar ve tekrarlayan davranış kusurları kalırsa uygulanır.

Her tercih çifti mümkün olduğunca tek fark taşır:

- Yanlış tarih
- Yanlış nedensellik
- Ön tanıyı kesinleştirme
- Reçeteyi aktif ilaç sayma
- Gerekli eski kaydı gizleme
- Desteksiz kişisel iddia

Chosen ve rejected:

- Benzer uzunlukta
- Benzer üslupta
- Aynı olgusal içeriğin yalnız hedeflenen davranışta ayrıştığı

çiftler olmalıdır. Yanlış gold tercihi DPO ile düzelmez.

---

## 22. Quantization ve sunum karşılaştırması

Karşılaştırma basamakları:

```text
Base bf16
vs
bf16 + adapter
vs
merged bf16
```

QLoRA kullanıldıysa ayrıca:

```text
NF4 + adapter
vs
orijinal bf16 ağırlık + adapter
```

Sonra:

```text
Base AWQ
vs
Tuned merged AWQ
```

Base ve tuned AWQ için:

- Aynı calibration örnekleri
- Aynı uzunluk dağılımı
- Aynı quantization ayarları
- Aynı tokenizer ve chat template
- Aynı vLLM sampling ayarları

kullanılır.

L40S-24Q üzerinde gerçek CCO uzunlukları, concurrency ve fallback yollarıyla gecikme/VRAM ölçülür.

---

## 23. Değerlendirme

### Katman bazlı ölçüm

**Parser/kanonikleştirme**

- Alan bazlı precision/recall
- Tarih hassasiyeti
- Kaynak span doğruluğu
- Unknown korunumu
- Assertion role doğruluğu

**Ranker**

- Kritik kayıt recall'u
- Must-include recall
- Nedensel kullanım hatası
- Bağlantılı paket bütünlüğü
- Uzun bağlam dilimleri

**Cevap**

- İddia bazlı kaynak doğruluğu
- Tarih ve metrik tazeliği
- İlaç aktiflik hatası
- Ön tanıyı kesinleştirme
- Yanlış nedensellik
- Gerekli güvenlik eylemi
- Yararlı cevap oranı

**Uçtan uca**

- Baseline'a göre net katkı
- Oracle–predicted anotasyon farkı
- Parse/fallback oranı
- Gecikme ve VRAM
- Uzun CCO başarısı

### Ağırlıklandırma

İki görünüm raporlanır:

1. Gerçek kullanım dağılımına göre weighted skor
2. Hata ailelerine eşit ağırlıklı macro/challenge skor

### Belirsizlik

Hata oranı hedefleri uygulanabilir kategori paydası ve güven aralığıyla verilir. Aynı hasta veya şablon varyantları bağımsız gözlem sayılmaz.

Sıfır hata gözlense bile yaklaşık tek taraflı %95 üst sınır:

```text
1 - 0.05^(1/n)
```

Bu nedenle <%1 iddiası için birkaç düzine karışık soru yeterli değildir.

---

## 24. Durma koşulları

Aşağıdakilerden biri görülürse daha fazla veri veya epoch üretmek yerine kök neden düzeltilir:

- Kaynakta olmayan statünün gold yapılması
- Kritik kanıtın sessizce kaybolması
- Çözülemeyen annotator anlaşmazlığı
- Split sızıntısı
- Hatalı assistant-only loss maskesi
- Truncation ile cevap/kanıt kaybı
- MoE ağırlıklarının beklenmeyen dtype veya bellek kullanımı
- Ranker anotasyonunun baseline'dan daha kötü uçtan uca sonuç vermesi
- Cevap modelinin anotasyonu kaynak gerçek gibi kabul etmesi
- Anonimleştirme veya test içeriği kalıntısı

---

## 25. Hard filtering için geçiş kapısı

Hard filtering ilk hedef değildir. Ancak maliyet nedeniyle daha sonra değerlendirilecekse şu kontrollere bağlıdır:

1. Kritik kayıt recall hedefi, yeterli bağımsız örnekle ve güven aralığıyla karşılanmış olmalı.
2. `must_include` veya kritik kanıt paketi eksikse tam bağlama fallback yapılmalı.
3. Düşük güven, çelişkili kanıt veya çoklu niyette filtering kapatılmalı.
4. Alerji, aktif/bırakılmış ilaç, ameliyat, reaksiyon ve aciliyet kayıtları için ayrı koruma politikası olmalı.
5. Predicted-filtered koşul, predicted-annotation koşuluna göre güvenlik regresyonu göstermemeli.
6. Kazanılan gecikme/VRAM farkı klinik risk artışına değecek kadar anlamlı olmalı.

Bu kapılar sağlanmazsa ranker yalnız anotasyon/sıralama amacıyla kullanılmaya devam eder.

---

## 26. Uygulama sırası

### P0 — Eğitimden önce

1. Gerçek CCO formatını ve örneklerini belgeye ekle.
2. Kaynak → kanonik olay eşlemesini tanımla.
3. `unknown`, tarih hassasiyeti, source span ve parser version alanlarını zorunlu yap.
4. Reçete, kullanım beyanı, aile öyküsü, wearable ve assertion role ayrımlarını uygula.
5. Serbest metin anonimleştirme ve test içeriği karantinası oluştur.
6. 60 gold ve 120 bozuk validator örneğini hazırla.
7. Loss maskesi, chat template, MoE dtype ve adapter kapsamını küçük teknik testle doğrula.

### P1 — İlk pilot

8. 480 sentetik smoke-test örneğini kök bazlı split ile üret.
9. Parser ve validator hata matrisini çıkar.
10. Relevance annotator/ranker için önce prompt baseline çalıştır.
11. 100 vakalık A/B/C/D/E mimari karşılaştırmasını yap.
12. Full-context + predicted annotations yaklaşımının baseline'a katkısını ölç.
13. Gerçek şikâyetler geldiğinde soru dili ve dağılımını güncelle.

### P2 — Ölçülmüş ihtiyaca göre

14. Ranker yetersizse yaklaşık 2.000 kaliteli örnekle selector/ranker SFT yap.
15. Gold anotasyonla bile cevap modeli yetersizse cevap SFT yap.
16. Dar davranış sorunları kalırsa kontrollü DPO yap.
17. Aynı base üzerinde merge ve AWQ karşılaştırmalarını tamamla.
18. L40S-24Q üretim yük testini yap.
19. Hard filtering'i yalnız geçiş kapıları sağlanırsa dene.

---

## 27. Bu sürümde değişen kararlar

| Önceki plan | V2.1 kararı |
|---|---|
| Selector kayıtları seçer, cevap modeli yalnız seçilenleri görür | Ranker kayıtları sıralar/anote eder; cevap modeli ilk aşamada tam CCO'yu görür |
| Tek `answer_mode` | Görev, kanıt yeterliliği ve güvenlik eylemi ayrı eksenler |
| Eski akut olay çoğunlukla `resolved` | Kaynak açıkça söylemiyorsa klinik durum `unknown` |
| Exact selected event set | `must_include`, `may_include`, `must_exclude_from_causal_use`, alternatif kanıt kümeleri |
| Olaylar bağımsız seçilir | Bağlantılı kanıt paketleri korunur |
| 300–500 sentetik pilot | 60 gold + 120 bozuk + 480 sentetik + 100 mimari karşılaştırma |
| Selector SFT ilk eğitim | Önce promptsuz/prompt baseline ve anotasyonun uçtan uca katkısı |
| Cevap SFT planlı ikinci aşama | Gold anotasyonla mevcut model yetmezse yapılır |
| Validator PASS genel doğruluk izlenimi verebilir | Validator kapsadığı kontrolleri ve çalışmayan kontrolleri açıkça raporlar |
| Hard filtering mimarinin parçası | Yalnız ölçülmüş bir optimizasyon ve güvenli fallback sonrası |

---

## 28. Gerçek text CCO sözleşmesi

Üretimdeki kaynak dosya dört alanlıdır:

```json
{
  "id": "<cco UUID>",
  "subjectId": "<subject UUID>",
  "asOf": "2026-10-20T10:00:58+03:00",
  "text": "<Türkçe Markdown CCO>"
}
```

Model girdisinin esas bağlamı `text` alanıdır. `id`, `subjectId` ve `asOf`
eğitimde cevap metnine yazılmaz; provenance, split ve tarih doğrulaması için
tutulur. Colab paketine alınan her kayıtta ayrıca corpus hash'i, CCO content
hash'i ve renderer sürümü bulunmalıdır.

Metin şu bölümlerin sıfır veya daha fazlasını içerebilir:

- demografik giriş;
- `Tanılar (hastane kaydı)`;
- `Hastalıklar (hasta beyanı)`;
- `İlaçlar (reçete kaydı)`;
- `Kullandığı ilaçlar (hasta beyanı)`;
- `Alerjiler (hasta beyanı)` ve `Aile öyküsü (hasta beyanı)`;
- hastane başvuruları, ameliyat/işlem, görüntüleme, patoloji ve laboratuvar;
- wearable uyku, kalp, hareket, stres, oksijen ve veri durumu;
- `Bilinmeyenler`.

Hiçbir bölüm zorunlu değildir. Bölümün olmaması, hastada ilgili durumun
olmadığını değil, bu CCO'da destekleyici kayıt bulunmadığını ifade eder.

### Kaynakların rolü

- **Text CCO:** Fine-tune ve üretimde Gemma'nın göreceği giriş.
- **Legacy raw fact:** Eski sistem sürümünden gelen `legacy_audit_only`
  kaynaktır. Text CCO ile aynı snapshot veya renderer sürümüne ait olduğu
  varsayılmaz; gold etiket, klinik statü ya da aktif ilaç durumu üretmez.
- **Hasta beyanı/demografi:** Mevcut fact export'unda olmayan ama text CCO'da
  bulunan ayrı kaynak aileleri.
- **Akut/kronik grubu:** Raw fact alanı değil, renderer tarafından türetilmiş
  yorum. Eğitim gold'u yapılmadan önce ayrıca doğrulanır.

Yeni kontrollü olay grafında `preliminary` statüsü, tanının tekrar etmesi
nedeniyle `definitive` olarak değiştirilemez. Tekrar sayısı ve assertion ayrı
alanlardır:

```json
{
  "assertion": "preliminary",
  "occurrenceCount": 4,
  "repeated": true,
  "persistence": "unknown"
}
```

### 21 Ekim 2026 kaynak audit baseline'ı

- 149 text CCO ve 149 fact dosyası birebir eşleşti.
- 77.568 fact satırında duplicate kimlik, bozuk aralık veya eksik definition
  bağlantısı bulunmadı.
- 6.717 legacy fact'in `validFrom` zamanı ilgili CCO `asOf` zamanından sonraydı.
- Legacy karşılaştırmada 185 raw ön tanının 26'sı text içinde kesin tanı
  ifadesiyle eşleşti.
- 38 hastanın fact listesi boştu; text yine demografi veya fact dışı kaynak
  içerebiliyordu.
- Fact export'u tarihsel rekonstrüksiyon değildi ve eski günleri kapsamıyordu.
- Bu farklar CCO hatası olarak sınıflandırılmaz; yalnız legacy tanısal sayaçtır.

Bu baseline `scripts/audit_training_sources.py` ile tekrar üretilebilir. Rapor
yalnız toplu sayaç ve hash içerir; hasta kimliği veya klinik metin içermez.

### Text evidence kimliği

Eğitim gold'unun birincil kaynağı text CCO span'ıdır. Her anotasyon şunları
taşır:

- `snapshotId`;
- tüm CCO'nun `contentHash` değeri;
- `evidenceId`;
- bölüm başlığı ile başlangıç/bitiş satırı;
- span'ın `textSha256` değeri;
- opsiyonel `factId` (`legacy_audit_only`, gold üretmez).

`evidenceId`, snapshot + content hash + satır aralığı + span hash üzerinden
üretilir. Kaynak metin değişirse anotasyon sessizce yeni metne taşınamaz.

Kaynakta açıkça belirtilmeyen aktif ilaç, iyileşme, çözülme, kesinlik veya
nedensellik durumu `unknown/not_stated` kalır. Text CCO içeriği bağımsız klinik
doğrulama olarak sunulmaz.

### Hibrit sentetik üretim hattı

OpenAI API'nin rolü kurgusal senaryo taslağı, klinik çeşitlilik adayı ve doğal
Türkçe soru varyasyonu önermekle sınırlıdır. Python/Codex tarafı kök split'ini,
olay semantiğini, dört karşı-olgusal varyantı, text CCO render'ını ve kaynak
bağlarını üretir. Model önerileri klinik doğrulama sayılmaz.

Her kök şu dört örneği verir: `base`, `relevant_counterfactual`,
`irrelevant_counterfactual`, `order_paraphrase`. Split önce köke atanır ve dört
örnek aynı split'te kalır. İnsan/klinik incelemesine kadar bütün örnekler
`unreviewed`, `trainable=false`, `goldAnswer=null` durumundadır; otomatik cevap
yalnız `candidateAnswer` olarak saklanır.

İlk pilot 20 kök ve 80 varyanttır. Rapor; şablon/alan/soru çeşitliliğini,
JSON Schema ve olay grafı doğrulamasını, `evidenceId` kaynak bütünlüğünü, kök
split sızıntısını ve dış istek veri sınıflarını ayrı ayrı gösterir.

### Eski deterministik ve gold pilot hattı

- 480 sentetik örnek önce kontrollü olay grafı olarak üretilir.
- Aynı graf gerçek dört alanlı text CCO biçimine render edilir.
- Gold anotasyon graph event kimliğinden render edilmiş `evidenceId` span'ına
  yeniden bağlanır; bağ kurulamıyorsa örnek reddedilir.
- 60 gerçek kayıt yerel insan inceleme kuyruğuna alınır. `pending` kayıtlar
  insan incelemesi tamamlanmadan gold değildir.
- Bu aşamada dış model çağrısı ve LoRA eğitimi yoktur.

Parser ve sonraki anotasyon pilotu en az üç yapı üzerinde test edilmelidir:

1. Yalnız demografi veya çok az kayıt içeren CCO
2. Wearable ağırlıklı ve verisi eski CCO
3. Tanı + reçete + tetkik + wearable içeren uzun ve çelişkili CCO

Bu üç yapı, sistemi yalnız temiz ve zengin vakalara göre kurmamızı engeller.

