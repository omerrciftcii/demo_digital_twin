# Yerel kanıt seçimi değerlendirmesi

Bu araç CCO servisini, Neo4j'yi veya model API'sini çağırmadan gerçek 0.7 sözleşmesini doğrular; kaynak kanıtları indeksler, soruyla arar ve 8.000 tahmini token bütçeli seçiciyi ölçer. Üretim verisi dışarı gönderilmez. Model planlayıcısı, terminoloji genişletmesi ve yanıt üretimi bu ölçümün parçası değildir: sonuç **çevrimdışı retrieval taban çizgisi**, uçtan uca sohbet başarısı değildir.

## Çalıştırma

Proje kökünde, kurulu sanal ortamla:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate.py docs\evaluation-cases.example.jsonl
.\.venv\Scripts\python.exe scripts\evaluate.py data\evaluation\cases.jsonl --require-acceptance
```

`--semantic` yalnız önceden önbelleğe alınmış yerel `intfloat/multilingual-e5-small` modelini kullanır; otomatik indirme yoktur. Model veya opsiyonel bağımlılıklar yoksa exact/lexical arama çalışır ve rapordaki `index.semantic` bunu açıkça gösterir. Bu nedenle semantik karşılaştırmada yalnız bayrağa değil bu duruma bakın. `--token-budget 8000` varsayılandır. Performans ölçümü ilk indekslemeyi ayrı tutar; `selectionLatencyMs` arama ve paketlemeyi içerir, model süresini içermez. Token sayıları gerçek sağlayıcı tokenizer ölçümü değildir.

Her koşu `data/evaluation/run-<UTC zaman>/` altında `report.json` ve yerel SQLite indeksi üretir. Bu dizin Git tarafından dışlanır; **yine de cihazda sağlık verisi içerir ve erişimi korunmalıdır.** Konsola yalnız toplu sayılar yazılır; hasta kimlikleri, sorular, kaynak değerler ve dosya adları yazılmaz. Rapor vaka kodları, source pointer'lar ve ölçümler içerir; klinik soru/metin ve subjectId içermez. Standart hata çıktısı kaynak içeriğini içermez.

Normal koşuda çıkış kodu `0` rapor üretildi demektir, kabul edildi demek değildir. `--require-acceptance` ile kabul başarısızsa `2`, geçersiz girdi veya koşu hatasında `1` döner. Girdi hatası olan satır sessizce atlanmaz; koşu başarısız olur.

## JSONL etiketleme

Her satır bağımsız JSON nesnesidir; ek alanlar ve yinelenen anahtarlar reddedilir:

```json
{"caseId":"case-001","ccoFile":"data/cco/snapshot-a.json","question":"Kreatinin sonuçlarım zaman içinde nasıl değişmiş?","requiredSourcePointers":["/sections/lab/facts/0","/sections/lab/facts/1"],"critical":true,"reviewStatus":"unreviewed","failureOrigin":"unknown"}
```

Örnekteki pointer'lar şablondur: kendi snapshot'ınızda ekranın kaynak panelinden gerçek adresleri kopyalayın. `ccoFile`, proje `data/` dizini altındaki bir dosyaya proje kökünden veya JSONL dosyası yanındaki dosyaya göreli yoldur. Tam yol, URL, `..`, kök dışına çıkan symlink ve belirsiz iki aday kabul edilmez. Dışa aktarılmış CcoDto `document`/`Document` zarfı kabul edilir; pointer'lar zarf içindeki CCO'nun köküne göredir.

| Alan | Anlam |
|---|---|
| `caseId` | Benzersiz, kişisel bilgi içermeyen vaka kodu. Hasta adı/kimlik numarası yazmayın. |
| `question` | Kullanıcının doğal sorusu; gerçek değerlendirmede fact kodunu ipucu olarak eklemeyin. |
| `requiredSourcePointers` | Cevapta mutlaka gerekli **kanıt kökleri**: `/sections/<section>/facts/<n>`, `/insights/<n>` veya `/insights/<n>/facts/<n>`. `/value/amount` gibi alt alanlar değil. |
| `critical` | Bu soruda gerekli kanıtın kaçması güvenlik açısından kritik mi? CCO insight severity'sinden ayrı, insanın belirlediği değerlendirme etiketi. |
| `reviewStatus` | `unreviewed` taslak; insan kaynakları ve gerekliliği kontrol ettikten sonra `reviewed`. |
| `failureOrigin` | Bilinen kaçırma nedeni `retrieval` veya `missing_in_cco`; henüz bilinmiyorsa `unknown`. Bu etiket ölçülen sonucu değiştirmez. |

İndeks numaraları değişen dosyalar etiketleri geçersiz kılabilir. Etiketlemeden sonra CCO dosyasını değiştirmeyin, yeni snapshot için yeni vaka oluşturun. Aynı factId'nin section kaydıyla insight içi donmuş kaydı farklı kanıtlardır: gerekli olan her iki pointer'ı ayrı işaretleyin. Bir değerin yeni tarihli insight'ta bulunması ölçüm tarihinin yeni olduğunu kanıtlamaz.

## 15–20 hasta ve en az 100 soru nasıl hazırlanır?

1. Yetkili olarak alınan snapshot'ları `data/cco/` içinde saklayın. Ayrı hastaları gerçekten ayrı `subjectId` değerleri temsil etmeli; sentetik dosyanın kimliğini değiştirerek hasta sayısını artırmayın.
2. Her hasta için 5–8 doğal soru hazırlayın. Eski önemli kayıt, sonraki takip/karşı kanıt, reçete ile güncel kullanım ayrımı, laboratuvar sayısı ile kaynak yorumu ayrımı, wearable kapsamı ve kişisel baz çizgisi örneklerini dağıtın. Kritik soruları ayrıca işaretleyin.
3. **Seçicinin çıktısına bakmadan** ham snapshot üzerinden gerekli kanıtları işaretleyin. Klinik önem ve güvenlik gerektiren etiketleri hekim incelesin. Tanım kodlu sorular yalnız mühendislik smoke testi olarak kalsın.
4. İlk etiketler `unreviewed` olsun. İnsan incelemesi biten satırlar `reviewed` olur. Tercihen geliştirme için kullanılan sorularla son kabul sorularını ayırın.
5. Koşu raporunda her kaçırmayı inceleyin. Kaynak gerçekten var ama seçilmediyse retrieval sorunudur. Gerekli olay eldeki CCO'da hiç yoksa CCO-generation/backfill eksikliğidir; eski anormalliği sistemin uydurması beklenmez. Sahte/yanlış yazılmış bir pointer'ı gerçek veri eksiği sanmamak için ham JSON'u kontrol edin.
6. CCO'da bulunmayan olaya ait geçerli bir pointer üretilemez. Henüz var olmayan olay için boş gold kümesi ve `failureOrigin: missing_in_cco` ile bir inceleme vakası tutulabilir; bu vaka puanlanamaz ve kabulü geçirmez. Olayın kaynağını/CCO eksiğini yerel inceleme notunuzda belgeleyin. Bir kaynak adresi taşınmış ya da silinmişse mevcut etiketin çözülemeyen pointer'ı `missingInCco` olarak raporlanır.

Tam klinik değerlendirme ayrıca yanıtın anlamı, aktif/eski durum ayrımı, uydurulmuş neden-sonuç ilişkisi, güvenlik uyarıları ve doğru kaynağa dayalı doğru yorum için insan incelemesi gerektirir. Bu CLI bu kontrolleri yaptığı iddiasında bulunmaz. İstenen tüm kanıtlar seçilse bile bunlar yanlış yorumlanabilir.

## Metrikler ve kabul

`requiredEvidenceRecall`, **CCO'da gerçekten bulunan gerekli kanıtların** ne kadarının seçildiğidir. CCO'da bulunmayan pointer'lar ayrı `missingInCco` sayılır; retrieval paydasını bozamaz. Yanlış kanıt kökü `invalidAnnotations`, seçilen kaynak/value/ref/snapshot uyuşmazlığı `invalidSelectedCitations` olur. Soru düzeyindeki boş altın küme yüzde 100 sayılmaz. Mikro ortalama kanıtlar üzerinden hesaplanır; çok kanıt gerektiren soru daha çok ağırlık taşır.

Rapor hem bütün satırları hem yalnız `reviewed` satırları verir. `retrievalAcceptance.passed` için birlikte gerekenler:

- En az 100 insan tarafından incelenmiş soru, bu sorularda en az 15 farklı subjectId.
- İncelenmiş sorularda en az %95 gerekli kanıt yakalama; en az bir kritik vaka ve sıfır kritik retrieval kaçırması.
- Sıfır çözümlenmemiş CCO veri eksiği, yanlış kaynak atfı, geçersiz etiket veya boş altın küme.

Bu eşikler baştan belirlenmiş mühendislik kabul kapısıdır; `clinicalProductionValidated` ve `answerQualityEvaluated` her zaman `false` kalır. İncelenmiş etiketlerin dürüstlüğünü araç doğrulayamaz; aynı hastaya farklı ID verilmesini klinik olarak tespit edemez. Veri ve inceleme süreçleri ayrıca yönetilmelidir.

Örnek 15 vaka **tek sentetik hasta**, **tamamı incelenmemiş**, çoğunlukla tanım kodlu mühendislik vakalarıdır. Bunlar tek başına klinik doğruluk, Türkçe arama başarısı veya 15 hasta kapsaması sağlamaz; kabul kapısını geçemez. Ek mevcut unit testleri eski/yeni aynı factId çakışmasını, kritik insight bağımlılıklarını, yetkiyle saklı veriyi, yetersiz bütçeyi ve snapshot karışmasını ayrıca kontrol eder.
