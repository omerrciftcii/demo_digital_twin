# B — Tam CCO + insan doğrulamalı oracle rehberi

A kolundaki tam CCO ve soru aynen korunur. Son kullanıcı iletisinin başına şu
rehber eklenir:

```text
KANIT REHBERİ:
{
  "annotationSource": "human_oracle",
  "rules": [
    "Tam CCO bağlamdır; anotasyonlar yalnız önem ve kullanım rehberidir.",
    "Anotasyonda olmayan kayıtlar görünmez veya yanlış kabul edilmez.",
    "Yasak çıkarımları yapma; kaynakta olmayan aktiflik, kesinlik ve nedenselliği ekleme.",
    "Yanıtın evidence_ids alanında yalnız promptEvidenceIds kimliklerini kullan."
  ],
  "evidenceAnnotations": {HUMAN_VERIFIED_ANNOTATIONS}
}

{QUESTION}
```

`reviewStatus=verified`, onaylı soru, insan kimliği/zamanı ve tamamlanmış klinik
statü incelemeleri yoksa bu kol çalıştırılmaz.

Kararlı `evidenceId` kaynak/audit kimliğidir. Runner bunu modelin cevap şemasında
kullandığı kısa `promptEvidenceIds` (`E1`, `E2`, …) ile eşler ve yanıttan sonra
yeniden kararlı kimliğe çevirir.
