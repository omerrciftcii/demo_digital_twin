# MedicalAI Selective Context V2

ABP lisansı veya çalışan CCO/MedicalAI servisi gerektirmeyen, Python/FastAPI tabanlı yerel geliştirme alanı. **Mevcut CCO generation değiştirilmez.** Kaynak JSON değişmeden saklanır; model girdisi ayrı bir `ContextPacket` olur.

Metin özeti (`cco-text/1`) ayrı bir yoldur: tüm markdown tek çağrıda kalır, yanıt düz Türkçedir, atıf köşeli parantezdir. Ayrıntı ve eval komutları: [docs/text-answers.md](docs/text-answers.md).

## Uygulanan akış

1. **Snapshot:** yalnız CCO **0.7.0** sözleşmesi; ham belge ve `document`/`Document` zarfı. Şema hataları reddedilir, eksik tarih veya metadata uydurulmaz. Aynı hastanın birden fazla snapshot'ı içerik hash'iyle ayrı saklanır.
2. **Kanıt indeksi:** section fact'leri, insight'lar ve insight içindeki donmuş fact değerleri. SQLite FTS5 + kod/tanım eşleşmesi + yerel `intfloat/multilingual-e5-small`. Her kanıt snapshot'a ve gerçek JSON Pointer'a bağlıdır. Aynı factId'nin farklı değerleri birleştirilmez.
3. **Neo4j:** yalnız paketle gelen, hash'i doğrulanmış glossary'den Family/Definition/Rule metadata'sı. Hasta değeri, narrative veya hasta dosyasından gelen yeni glossary metni ortak graph'a yazılmaz. Önceki beş hastalıklı terminology dosyası artık seçimde kullanılmaz.
4. **Seçim:** soru adayları + ilgili tanımın eski/yeni kayıtları + insight bağımlılıkları birlikte taşınır. Sabit 12 fact / 2 family sınırı ve tam CCO fallback yoktur. Kritik insight ve bağımlılıkları zorunludur; bütçeye sığmıyorsa yanıta izin verilmez. Warning index kaynakta bildirilen uyarıları korur; kendi tıbbi eşiklerini üretmez.
5. **Yanıt:** model planlayıcı ve kaynaklı yanıt için en fazla iki generation çağrısı. Sınırlı plan, sabit model, strict JSON, atıf/değer kontrolleri; geçersiz çıktı kullanıcıya klinik yanıt olarak gösterilmez. Kontroller klinik anlamın doğruluğunu kanıtlamaz.

**Önemli sınırlar:** CCO'da yer almayan beş yıl önceki bir ölçüm kurtarılamaz. Kaynak eksikliği, kısıtlı erişim, truncation ve tarih belirsizliği görünürdür; “kayıt yok” hiçbir zaman “hastada yok” anlamına gelmez. Wearable kayıtları da aynı kanıt yolundan geçer; kaynağın vermediği kişisel baseline/anormallik üretilmez.

Semantik arama aday bulur; cosine ve RRF skorları güven yüzdesi değildir. İnsan etiketli veriyle kalibrasyon yapılana kadar **yalnız semantik benzerlik**, bağımsız bir yanıt kanıtı olarak kabul edilmez. Sözcük/kod desteği veya seçili kanıtın zaman/bağımlılık bağlantısı gerekir. Bu bilinçli sınır özellikle örtük sorularda recall'ı düşürebilir; arayüzde elenme gerekçesi görünür. Lexical eşleşme de klinik ilgililiğin garantisi değildir.

## Çalıştırma

Python 3.12+ gerekir. Neo4j için Docker gerekir; Docker/Neo4j yoksa yerel arama çalışır ve health `degraded` bildirir. Docker bu bilgisayarda henüz kurulu değildir.

```powershell
cd C:\Users\omerr\Documents\Projects\MedicalAI.SelectiveContextV2
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev,semantic]"
# Yalnız ilk kurulumda, genel model dosyalarını indirir; hasta verisi göndermez.
.\.venv\Scripts\python.exe scripts\prepare_embeddings.py
# .env mevcutsa üzerine yazmayın.
Copy-Item .env.example .env
docker compose up -d neo4j
.\scripts\start.ps1
```

### Hasta sohbeti sunum demosu

Üç özel hasta kaydını ham `text` alanıyla Gemma'ya gönderen, retrieval/fact/verifier
kullanmayan sade localhost demosunu başlatmak için:

```powershell
.\scripts\start_patient_chat_demo.ps1
```

Ardından `http://127.0.0.1:8092` adresini açın. Başlatıcı mevcut SSH anahtarıyla
GPU sunucusuna yerel bir tünel kurar; uygulama ve hasta seçimi yalnızca localhost'ta
çalışır. Modelin yanıt sonunda ürettiği `A)`, `B)`, `C)` seçenekleri arayüzde
tıklanabilir hızlı yanıtlara dönüşür. Çıkmak ve açılan SSH tünelini kapatmak için
terminalde `Ctrl+C` kullanın.

Mevcut sanal ortam ve embedding önbelleği hazırlanmıştır. Sonraki çalıştırmalarda `scripts\start.ps1` yeterlidir; Neo4j konteyneri ayrıca açık olmalıdır. İlk model yükleme/indeksleme zaman alabilir. Runtime otomatik model indirmez.

- Arayüz: [http://127.0.0.1:8091](http://127.0.0.1:8091)
- Swagger: [http://127.0.0.1:8091/docs](http://127.0.0.1:8091/docs)
- Model geçidi (ayrı süreç, varsayılan `127.0.0.1:8095`): `docs/model-gateway.md`
- Neo4j Browser: [http://127.0.0.1:7474](http://127.0.0.1:7474)

Neo4j sonradan açılırsa `POST /api/v1/terminology/reload` çağırın. Yerel geliştirme parolası `.env.example` içindedir; bu ayarlarla uygulamayı internete açmayın. Servis sadece loopback'e bağlanır, kimlik doğrulamalı çok kullanıcılı bir ürün değildir.

## Model ve veri güvenliği

Varsayılan bağlantı OpenRouter, model `qwen/qwen3.5-9b`. Anahtar yokken **Kanıtları incele** tamamen yereldir. `.env` içine `OPENROUTER_API_KEY` ekleyip yeniden başlatınca sentetik örnek üzerinde yanıt üretilebilir. Anahtarı sohbetlere veya Git'e koymayın. Gerçek sağlayıcıyla yanıt kalitesi, gecikme ve maliyet henüz ölçülmemiştir; anahtar olmadan ücretli çağrı yapılmadı.

`ENFORCE_ANSWER_VALIDATION=false` yalnız geliştirme içindir: atıf/değer kontrolünden geçmeyen yanıt gizlenmez, `answered_unvalidated` durumu ve hata koduyla doğrulanmamış olarak gösterilir. Şema dışı veya yarım kalmış çıktı yine reddedilir. `LLM_ANSWER_MAX_TOKENS` (varsayılan 4000) yanıt çağrısının üst sınırıdır; 1800 gerçek sorularda JSON'u yarıda kesiyordu.

İstekler model değiştirme/retry yapmaz; OpenRouter için structured-output desteği, data-collection deny ve varsayılan ZDR istenir. Bunlara uyan sağlayıcı yoksa çağrı başarısız olur; gizlilik koşulları sessizce gevşetilmez. Tek sağlayıcıyla karşılaştırma için `OPENROUTER_PROVIDER` kullanılabilir.

**İçe aktarılan her dosya gerçek veri sayılır.** Yalnız paketle gelen doğrulanmış sentetik örnek otomatik sentetik sayılır. `ALLOW_EXTERNAL_REAL_DATA=false` iken gerçek snapshot için planlayıcı dahil dış model çağrısı engellenir. `/context/select` model çağırmaz. Sentetik modda bile soru ve sohbet metni dış modele gidebilir: **gerçek hasta bilgilerini soru/sohbet metnine yazmayın.** Bu bayrak serbest metni otomatik anonimleştirmez. Dahili kimliklerin çıkarılması da narrative'leri anonimleştirmez.

Gerçek veri aktarımını ancak alıcı, veri işleme koşulları ve yetki/onam değerlendirilip açıkça onaylandıktan sonra açın. ZDR tek başına hukuki uygunluk garantisi değildir. Bu oturumda gerçek veri aktarımı açılmadı.

`data/cco/`, `data/runtime/`, `data/evaluation/` ve `.env` Git dışında tutulur. **JSON arşivleri, SQLite/FTS ve embedding'ler yerel sağlık verisi içerir; şifreli kasa değildir.** Cihaz erişimi, disk şifreleme, yedek ve silme politikası ayrıca yönetilmelidir. Soru/yanıt gövdeleri uygulama tarafından loglanmaz veya kalıcı sohbet geçmişine yazılmaz.

## CCO ekleme ve toplu çekim

Arayüzde **CCO ekle** ile JSON seçin veya dosyaları `data/cco/` altına koyup **Yenile**'ye basın. Aynı hastanın snapshot'larını tarih ve snapshotId ile seçebilirsiniz. Eski snapshot üzerine yazılmaz.

Repo/controller ve gateway eşlemesinden belirlenen endpoint (bu uygulama prod'u kendiliğinden çağırmaz):

```http
POST https://wag.kendineiyibak.app/api/clinical-insight-service/cco
Authorization: Bearer <yetkili-token>
Content-Type: application/json

{
  "subjectId": "00000000-0000-0000-0000-000000000000",
  "purpose": "user-chat",
  "correlationId": "selective-v2-capture-001"
}
```

[Prod Swagger](https://wag.kendineiyibak.app/clinical-insight-service/swagger/index.html). `asOf` opsiyoneldir. Servis yetki, subject access ve consent kontrolü yapar. İncelenen sözleşmede batch endpoint yoktur; script her subject için ayrı POST atar. Dağıtılmış prod servisi bu oturumda çağrılmadı.

```powershell
Copy-Item .\data\subject-ids.example.txt .\data\subject-ids.txt
# Dosyaya satır başına bir subject UUID ekleyin.
.\scripts\fetch_prod_ccos.ps1 -SubjectIdsPath .\data\subject-ids.txt
```

Script token'ı maskeli sorar, her çekimi benzersiz dosyaya kaydeder. Ardından arayüzde **Yenile** veya `POST /api/v1/ccos/reload`.

## API kullanımı

Önce `GET /api/v1/ccos` ile **snapshotId** alın:

```http
POST /api/v1/context/select
Content-Type: application/json

{
  "snapshotId": "<listeden-alınan-snapshotId>",
  "question": "Kreatinin sonuçlarım zaman içinde nasıl değişmiş?",
  "tokenBudget": 8000,
  "history": []
}
```

Yanıt ayrı bir `ContextPacket`: `evidence`, `excluded`, `coverage`, `warningIndex`, `blockers`, `modelContext`. Token tahmini UTF-8 boyutuna dayanır; gerçek model tokenizer sayımı değildir. Kaynak JSON: `GET /api/v1/ccos/{snapshotId}`. `POST /api/v1/context/select-document` dosyayı kaydetmeden yalnız exact/lexical inceleme yapar.

`POST /api/v1/chat` aynı istek gövdesini kullanır. Her yanıtta paket, atıflar, model ve ölçümler döner. History sadece referans çözümleyen güvenilmeyen sohbet bağlamıdır; tıbbi kaynak değildir. API istemcisi history'yi aynı snapshot'a ait tutmalıdır; arayüz snapshot değişince temizler.

## Test ve değerlendirme

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\ruff.exe check .
.\.venv\Scripts\python.exe scripts\evaluate.py docs\evaluation-cases.example.jsonl
.\.venv\Scripts\python.exe scripts\evaluate.py data\evaluation\cases.jsonl --semantic --require-acceptance
```

[Etiketleme ve kabul kılavuzu](docs/evaluation.md): 15–20 gerçek hasta, en az 100 insan tarafından incelenmiş soru; ≥%95 gerekli kanıt yakalama ve sıfır kritik kaçırma. Mevcut 15 örnek vaka tek sentetik hastaya aittir, klinik doğrulama veya kabul başarısı değildir. Retrieval ve CCO-generation kaynaklı eksikler ayrı raporlanır.

[API sözleşmesi](openapi.yaml) · [Tarihsel branch incelemesi](docs/branch-review.md)

