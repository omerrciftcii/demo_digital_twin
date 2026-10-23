# C — Tam CCO + otomatik ranker rehberi

A kolundaki tam CCO ve soru aynen korunur. Son kullanıcı iletisinin başına şu
rehber eklenir:

```text
KANIT REHBERİ:
{
  "annotationSource": "predicted_ranker",
  "rules": [
    "Tam CCO bağlamdır; anotasyonlar yalnız önem ve kullanım rehberidir.",
    "Anotasyonda olmayan kayıtlar görünmez veya yanlış kabul edilmez.",
    "Yasak çıkarımları yapma; kaynakta olmayan aktiflik, kesinlik ve nedenselliği ekleme.",
    "Yanıtın evidence_ids alanında yalnız promptEvidenceIds kimliklerini kullan."
  ],
  "evidenceAnnotations": {AUTOMATIC_RANKER_ANNOTATIONS}
}

{QUESTION}
```

Bu ilk paket S6'yı varsaymaz. `predicted_ranker`, bu repoda mevcut olan lexical
evidence ranker'dır; sonraki bir ranker aynı sözleşmeyle ayrıca adlandırılmalıdır.
Kararlı `evidenceId` ile istem içi kısa `promptEvidenceIds` runner tarafından
çift yönlü eşlenir.
