# MedicalAIEngine — Oracle A/B/C Grok handoff

## Ne hazır, ne hazır değil?

Bu paket 20 farklı anonim text CCO için 20 doğal Türkçe soru adayı içerir. Her
vaka source snapshot/content hash, `evidenceId` ve `sourceSpan` ile bağlıdır.
Kaynak CCO'lar değiştirilmedi. Eski fact export'ları kullanılmadı.

Sorular Luna taslaklarından dil açısından düzenlendi; klinik etiketleri Codex
tarafından doğrulanmadı. Bu nedenle bütün sorular ve evidence anotasyonları
`pending`, bütün vakalar `trainable=false` durumundadır. Şu anda resmi Oracle B
çalıştırılamaz. Yerel dry-run sonucu: A ve C için 40 istem kurulabildi, 20 B
satırı insan oracle incelemesi olmadığı için bloke edildi; model çağrısı sıfırdı.

Bu pakette S6 kodu varsayılmıyor. C kolu repoda bulunan lexical evidence
ranker'ı kullanır. S6 veya başka bir ranker sonradan bağlanırsa
`annotationSource` adı değiştirilerek ayrı bir kol/koşu olarak raporlanmalıdır.
Dry-run'da C kolu 010, 019 ve 020 için boş anotasyon üretti. 019 ve 020 kasıtlı
`not_supported` vakalarıdır; 010 ise tansiyon/nabız bağlamı bulunan gerçek bir
ranker kaçırması adayıdır. Bu vaka elle C'ye eklenmemeli; C'nin ölçülen davranışı
olarak korunmalıdır.

## Dosyalar

- `data/evaluation/oracle_v3/oracle_cases.jsonl`: soru, kaynak snapshot ve
  yalnız aday relevance bilgisi.
- `data/evaluation/oracle_v3/oracle_evidence_reviews.jsonl`: insan inceleme
  kuyruğu. Resmi B'nin kapısı budur.
- `data/evaluation/oracle_v3/oracle_run_manifest.json`: eşleştirilmiş deney,
  hedef model/GPU ve dosya hash'leri.
- `data/evaluation/oracle_v3/source-audit.json`: 149 CCO taraması; olası
  tanımlayıcı değerlerini değil yalnız tür, satır ve hash bilgisini taşır.
- `data/evaluation/oracle_v3/package-validation.json`: son bütünlük raporu.
- `oracle_prompts/`: A/B/C okunabilir istem sözleşmesi ve cevap JSON Schema'sı.
- `scripts/run_oracle_abc.py`: Grok'un GPU sunucusunda çalıştıracağı runner.
- `scripts/analyze_oracle_abc.py`: insan puanlarından deterministik rapor.
- `scripts/validate_oracle_package.py`: snapshot/evidence/hash kapıları.
- `scripts/finalize_oracle_reviews.py`: insan review'larını doğrulayıp manifesti
  yeniden mühürler.

## İnsan incelemesi

Her `oracle_evidence_reviews.jsonl` satırı tek bir uzman tarafından incelenir.
Soru klinik amaç bakımından uygunsa `questionReviewStatus=approved`, değilse
`rejected` yapılır ve vaka yeniden yazılır. Luna/Codex adayları otomatik doğru
kabul edilmez.

Onaylanan vaka için:

1. `verifiedRelevantEvidence` yalnız gerçekten gerekli veya yardımcı span'ları
   içerir. Cevap kayıtta desteklenmiyorsa boş liste geçerlidir.
2. `verifiedIrrelevantEvidence` cazip ama kullanılmaması gereken span'ları
   içerir.
3. Her ilgili anotasyon şu alanları taşır:
   `evidenceId`, değişmemiş `sourceSpan`, `importance` (`must_use|helpful`),
   `useRole` (`direct_fact|context_only|safety_constraint`),
   `causalUseAllowed`, `explicitClaims`, `forbiddenInferences` ve
   `annotationSource=human_oracle`.
4. `verifiedForbiddenInferences` doldurulur.
5. `temporalStatusReview`, `diagnosisStatusReview` ve
   `medicationActivityReview` alanları `pending` dışında açıklayıcı bir değere
   çekilir; uygulanmıyorsa `not_applicable` yazılır.
6. `reviewer`, ISO-8601 `reviewedAt`, gerekirse `reviewNotes` doldurulur ve
   `reviewStatus=verified` yapılır.

`gold=true` yalnız ayrıca adjudication yapıldıysa kullanılmalıdır. Resmi Oracle
B için insan doğrulaması zorunludur; `gold=true` zorunlu değildir.

Review dosyası tamamlandıktan sonra hash ve B anotasyon eşleşmelerini mühürle:

```bash
uv run python scripts/finalize_oracle_reviews.py \
  --cases data/evaluation/oracle_v3/oracle_cases.jsonl \
  --reviews data/evaluation/oracle_v3/oracle_evidence_reviews.jsonl \
  --manifest data/evaluation/oracle_v3/oracle_run_manifest.json \
  --cco-dir /secure/cco \
  --require-all
```

Bu adım bütün doğrulanmış B anotasyonlarından gerçek prompt planı kurar ve kısa
istem kimliği/kararlı evidence kimliği eşleşmesini de sınar. Runner manifestteki
vaka ve review hash'leri uyuşmazsa model çağrısından önce durur.

## Sunucuya taşıma

Repo ile birlikte `data/evaluation/oracle_v3`, `oracle_prompts`, `src` ve
`scripts` dizinlerini sunucuya taşı. Ayrıca pilotta referans verilen 20 kaynak
CCO'yu, dosya adları değişmeden ayrı bir CCO dizinine koy. API anahtarını dosyaya
yazma; gerekiyorsa yalnız ortam değişkeni kullan.

Sunucuda önce kaynak ve paket bütünlüğünü doğrula:

```bash
uv sync --extra dev
uv run python scripts/validate_oracle_package.py \
  --package-dir data/evaluation/oracle_v3 \
  --cco-dir /secure/cco \
  --output data/evaluation/oracle_v3/package-validation-server.json
```

`valid=true`, 20 vaka, 20 farklı CCO ve sıfır hata görülmeden devam etme.

## Çağrısız prova

Bu komut A/C istemlerini kurar, B'yi bekleyen insan incelemesi nedeniyle bloke
eder ve Gemma'ya istek atmaz:

```bash
uv run python scripts/run_oracle_abc.py \
  --cases data/evaluation/oracle_v3/oracle_cases.jsonl \
  --reviews data/evaluation/oracle_v3/oracle_evidence_reviews.jsonl \
  --manifest data/evaluation/oracle_v3/oracle_run_manifest.json \
  --cco-dir /secure/cco \
  --out-dir runs/oracle_abc_dry \
  --base-url http://127.0.0.1:8000/v1 \
  --repeats 1 \
  --allow-unreviewed-questions \
  --dry-run
```

`--allow-unreviewed-questions` yalnız dry-run/smoke için kullanılmalıdır.

## Resmi A/B/C koşusu

İnsan incelemesi tamamlandıktan sonra bu bayrağı kullanmadan çalıştır:

```bash
export LLM_API_KEY='sunucu-gerektiriyorsa-buraya'
uv run python scripts/run_oracle_abc.py \
  --cases data/evaluation/oracle_v3/oracle_cases.jsonl \
  --reviews data/evaluation/oracle_v3/oracle_evidence_reviews.jsonl \
  --manifest data/evaluation/oracle_v3/oracle_run_manifest.json \
  --cco-dir /secure/cco \
  --out-dir runs/oracle_abc_official \
  --base-url http://127.0.0.1:8000/v1 \
  --model gemma4-26b-a4b \
  --arms A B C \
  --repeats 3 \
  --seed 20261021 \
  --structured-output response-format \
  --disable-reasoning chat-template
```

Hedef koşul: Gemma 4 26B-A4B, AWQ 4-bit, vLLM 0.20.2, reasoning kapalı,
NVIDIA L40S-24Q 24 GB vGPU. Üç kolda aynı ağırlıklar, temperature=0, tam CCO,
soru, çıktı şeması ve token limiti kullanılır. Kol sırası vaka/tekrar bazında
döndürülür. `response_format` sunucu tarafından gerçekten desteklenmiyorsa
`--structured-output off` bütün kollar için aynı yeni koşuda kullanılmalı ve iki
koşu birbirine karıştırılmamalıdır.

`--resume` tamamlanmış `runKey` satırlarını korur ve mevcut insan puanlarını
silmez. Başarısız bir satır yeniden denenebilir. Çıktılar:

- `oracle_abc_runs.jsonl`: istem/request hash, latency, yanıt ve evidence id'leri.
- `oracle_grading_queue.jsonl`: insan P/K/F ve hata kaynağı kuyruğu.
- `oracle_run_summary.json`: çağrı/blok durumları ve gerçek koşu ayarları.

## Puanlama ve hata kaynağı

Her tamamlanmış satır için `oracle_grading_queue.jsonl` içinde:

- `verdict`: `pass`, `partial` veya `fail`;
- `criticalSafetyFailure`: boolean;
- `usedEvidenceIds`: yanıtın gerçekten kullandığı evidence kimlikleri;
- `usedPromptEvidenceIds`: modelin ham çıktısındaki kısa `E1/E2` kimlikleri;
- `errorSources`: aşağıdaki kapalı listeden sıfır veya daha fazla değer;
- `errorFlags`: yanlış tarih, eskiyi güncel sunma, ön tanıyı kesinleştirme,
  reçeteyi aktif kullanım sayma, uydurma ve desteksiz nedensellik;
- `grader`, `gradedAt`, `notes` doldurulur.

Kapalı hata kaynağı listesi:

```text
cco_source_renderer
evidence_extraction
evidence_selection_ranking
prompt_context_construction
llm_answer_generation
temporal_grounding
assertion_medication_state
unsupported_causal_inference
numeric_calculation
evaluation_judge_uncertainty
```

Analiz:

```bash
uv run python scripts/analyze_oracle_abc.py \
  --runs runs/oracle_abc_official/oracle_abc_runs.jsonl \
  --grades runs/oracle_abc_official/oracle_grading_queue.jsonl \
  --reviews data/evaluation/oracle_v3/oracle_evidence_reviews.jsonl \
  --output runs/oracle_abc_official/oracle_abc_analysis.json
```

Rapor P/K/F, kritik hata, evidence precision/recall, hata kaynakları, median/p95
latency ve eşleştirilmiş B−A, C−A, C−B kazanma/kaybetme/beraberlik sayılarını
üretir. İnsan puanları tamamlanmadan `interpretationReady=true` olmaz.

Runner modelin kullandığı kısa istem kimliklerini `promptEvidenceIdMap` üzerinden
kararlı `evidenceId` kimliklerine çevirir. Precision/recall kararlı kimliklerle
hesaplanır; iki kimlik uzayı doğrudan karşılaştırılmaz.

## Kapsam sınırı

Codex bu handoff sırasında Gemma'ya bağlanmadı, GPU sunucusuna erişmedi ve model
performansı sonucu üretmedi. LoRA/SFT/DPO başlamadı; sentetik üretim 120 köke
genişletilmedi. S4b/S5/S6 kıyasları Grok'un elindeki harici sonuçlar olarak ayrı
bir rapor alanına eklenmelidir; bu pakette olmayan sonuçlar uydurulmamalıdır.
