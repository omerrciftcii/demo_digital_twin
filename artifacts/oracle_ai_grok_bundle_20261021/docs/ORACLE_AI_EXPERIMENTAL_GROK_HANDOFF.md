# MedicalAIEngine — deneysel A/B_AI/C Grok handoff

## Amaç ve statü

Bu paket mevcut 20 Oracle V3 vakasını kullanır. İlk 3 review GPT-6 Astra,
kalan 17 review GPT-6.1 Sol (`reasoning=medium`) ile üretilmiştir. Model; ilgili
ve ilgisiz evidence span'larını, tarih/klinik statüleri, `must_use|helpful`
rollerini, yasak çıkarımları ve güvenlik uyarılarını önerir. Python yalnız mevcut
`evidenceId` seçimlerini kabul eder, `sourceSpan` ve açık klinik işaretleri kaynak
CCO'dan yeniden üretir ve çelişen taslağı reddeder. Her review kendi gerçek model,
provider, request hash ve usage kaydını taşır; model karışımı manifestte açıktır.

AI review statüsü daima:

```text
reviewStatus=ai_provisional
humanVerified=false
gold=false
trainable=false
experimentalOnly=true
```

Orijinal `oracle_evidence_reviews.jsonl` değiştirilmez. Resmî B kolunun insan
kapısı korunur. İnsan onaysız kolun adı yalnız `B_AI`'dır.

## Taşınabilir paket

`artifacts/oracle_ai_grok_bundle_20261021.zip` sunucuya aktarılacak bağımsız
pakettir. Arşiv açıldığında komutlar paket kökünden çalıştırılır. Paket şunları
içerir:

- 20 kaynak CCO snapshot'ı (`cco/`);
- `data/evaluation/oracle_v3/` altındaki vakalar, resmî review dosyası, ayrı AI
  review/manifest/validation dosyaları ve yerel dry-run kanıtı;
- gerekli `src/`, runner/analyzer/validator scriptleri, prompt şablonları,
  `pyproject.toml` ve `uv.lock`;
- bu handoff belgesi.

Arşiv içindeki CCO yolu kullanılırken aşağıdaki örneklerdeki `/secure/cco`
yerine `./cco` verilir. Gerçek koşudan önce paket doğrulama komutu mutlaka
yeniden çalıştırılır.

## AI review üretimi

Kesilen bir koşuya devam etmek güvenlidir:

```bash
uv run python scripts/generate_ai_oracle_reviews.py \
  --cases data/evaluation/oracle_v3/oracle_cases.jsonl \
  --base-manifest data/evaluation/oracle_v3/oracle_run_manifest.json \
  --cco-dir /secure/cco \
  --out-dir data/evaluation/oracle_v3/ai_oracle \
  --model openai/gpt-6.1-sol \
  --reasoning-effort medium \
  --workers 2 \
  --retries 2 \
  --resume
```

`medium`, reasoning'i açık tutar ve bu anotasyon işi için maliyet/gecikmeyi
sınırlar. Astra'nın `:batch` rotası katı JSON Schema kullanan chat completion
istekleriyle uyumlu endpoint vermediği için kullanılmamıştır. Üretim yarıda
kesilirse `--resume`, yalnız eksik veya kaynak doğrulamasından geçmeyen vakaları
yeniden çalıştırır.

Üretilen dosyalar:

- `oracle_ai_reviews.jsonl`: kaynak doğrulamalı AI-provisional review'lar;
- `oracle_ai_drafts.jsonl`: modellerin ham structured-output taslakları;
- `oracle_ai_errors.jsonl`: nihai olarak reddedilen vaka listesi (tam pakette boş);
- `oracle_ai_manifest.json`: ayrı hash/provenance manifesti;
- `oracle_ai_generation_report.json`: tamamlanma durumu;
- `oracle_ai_validation.json`: bağımsız bütünlük raporu.

## Paket doğrulaması

```bash
uv run python scripts/validate_ai_oracle_package.py \
  --cases data/evaluation/oracle_v3/oracle_cases.jsonl \
  --base-manifest data/evaluation/oracle_v3/oracle_run_manifest.json \
  --ai-reviews data/evaluation/oracle_v3/ai_oracle/oracle_ai_reviews.jsonl \
  --ai-manifest data/evaluation/oracle_v3/ai_oracle/oracle_ai_manifest.json \
  --cco-dir /secure/cco \
  --output data/evaluation/oracle_v3/ai_oracle/oracle_ai_validation.json
```

Mevcut doğrulama sonucu `valid=true`, `cases=20`, `ai_provisional=20`, ilgili
evidence sayısı 82, ilgisiz evidence sayısı 105 ve hata sayısı 0'dır. Bu kapı
geçilmeden Gemma koşusuna geçilmez.

## Çağrısız Grok provası

```bash
uv run python scripts/run_oracle_abc.py \
  --cases data/evaluation/oracle_v3/oracle_cases.jsonl \
  --reviews data/evaluation/oracle_v3/oracle_evidence_reviews.jsonl \
  --manifest data/evaluation/oracle_v3/oracle_run_manifest.json \
  --ai-reviews data/evaluation/oracle_v3/ai_oracle/oracle_ai_reviews.jsonl \
  --ai-manifest data/evaluation/oracle_v3/ai_oracle/oracle_ai_manifest.json \
  --experimental-ai-oracle \
  --cco-dir /secure/cco \
  --out-dir runs/oracle_ai_dry \
  --base-url http://127.0.0.1:8000/v1 \
  --arms A B_AI C \
  --repeats 1 \
  --dry-run
```

## Grok'un gerçek Gemma koşusu

```bash
export LLM_API_KEY='sunucu-gerektiriyorsa'
uv run python scripts/run_oracle_abc.py \
  --cases data/evaluation/oracle_v3/oracle_cases.jsonl \
  --reviews data/evaluation/oracle_v3/oracle_evidence_reviews.jsonl \
  --manifest data/evaluation/oracle_v3/oracle_run_manifest.json \
  --ai-reviews data/evaluation/oracle_v3/ai_oracle/oracle_ai_reviews.jsonl \
  --ai-manifest data/evaluation/oracle_v3/ai_oracle/oracle_ai_manifest.json \
  --experimental-ai-oracle \
  --cco-dir /secure/cco \
  --out-dir runs/oracle_ai_abc \
  --base-url http://127.0.0.1:8000/v1 \
  --model gemma4-26b-a4b \
  --arms A B_AI C \
  --repeats 1 \
  --seed 20261021 \
  --structured-output response-format \
  --disable-reasoning chat-template
```

Hedef yine Gemma 4 26B-A4B AWQ, vLLM 0.20.2, reasoning kapalı ve L40S-24Q
24 GB vGPU'dur. Toplam 60 completion oluşur. Üç kol aynı tam CCO, soru, ağırlık,
sampling ve cevap şemasını kullanır. Runner AI manifest/review hash'leri veya
source span eşleşmesi bozuksa Gemma çağrısından önce durur.

Bütün run/grading/summary kayıtları `experimental=true` taşır. Analiz P/K/F,
kritik hata, evidence precision/recall, hata kaynakları, latency ve eşleştirilmiş
`B_AI−A`, `C−A`, `C−B_AI` farklarını raporlar. Bu sonuçlar ön hata analizi için
kullanılabilir; eğitim verisi veya doğrulanmış gold değildir.

Yerel çağrısız prova 60/60 satırı üretmiş, 0 blok ve 0 CCO hash uyuşmazlığı
bulmuştur. `experimentalAiOracleReady=true`; resmî insan B kolu ise değiştirilmeden
`officialOracleBReady=false` durumunda kalır. Özet dosyası 20 vakanın tamamını
`pendingHumanReview` altında ayrıca listeler; bu bekleyiş deneysel A/B_AI/C koşusunu
bloke etmez.
