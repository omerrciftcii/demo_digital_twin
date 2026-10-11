# `feature/terminology-context-v2-chat-integration` incelemesi

İncelenen uç commit: `60f6b5c1a9b3f566d9d7d0d9c37c855a49f997a6`  
Karşılaştırma tabanı: `development` üzerindeki `0825ab8`  
Fark: 13 commit, 55 dosya, yaklaşık 8.5K ek satır.

## Anlaşılan mimari

- ClinicalInsight hasta fact'lerinin, CCO'nun, consent ve provenance'ın sahibi.
- MedicalAI, CCO'yu kalıcılaştırmıyor; yalnızca istek sırasında bağlam üretiyor.
- Neo4j sınırı PHI içermiyor. Terminoloji release'leri, kavramlar, alias'lar,
  kavram ilişkileri ve fact-family/definition metadata'sı tutuluyor.
- Çözümleme sırası explicit code, exact term, full-text ve vector adaylarının RRF
  birleşimi; ardından kontrollü graph expansion.
- Planner; family, zaman niyeti, safety ihtiyacı ve full-context kararını veriyor.
  CCO filtreleme bundan sonra deterministik yapılıyor.
- Normal seçim 12 fact ile sınırlandırılıyor. Safety gerektiğinde başarılı tanı ve
  ilaç fact'leri bu limitin dışında ekleniyor.
- Terminoloji, metadata ya da planner güvenilir sonuç üretemezse tam CCO görünür bir
  fallback nedeni ile gönderiliyor.

## Bu çalışma alanında korunan kararlar

- CCO/subject/fact instance/prompt/chat Neo4j'ye yazılmıyor.
- Neo4j terminoloji ve fact-family metadata'sı için zorunlu çalışma bileşeni.
- Filtreleme deterministik, sonuç içinde strateji ve fallback nedeni görünür.
- CCO glossary'si doğrudan family eşleştirmesinde kullanılıyor.
- LLM'e gönderilecek içerik ayrıca `llmContext` alanı olarak hazırlanıyor; uygulama
  henüz dış LLM çağrısı yapmıyor.

## Bilinçli farklar ve açık işler

- Branch'in parser'ı 0.2/0.4 CCO kabul ediyor. ClinicalInsight güncel contract'ı
  0.7.0 olduğu için bu proje 0.4–0.7 aralığını kabul ediyor ve 0.7 fixture ile test
  ediliyor.
- Branch'teki otomatik üst-aday kabul bayrağı buraya taşınmadı. Yanlış klinik kavram
  eşleştirmesini sessizce kabul etmek yerine düşük güven full-context fallback olmalı.
- Başlangıç terminoloji paketi yalnızca iskelet veridir ve `review-required` olarak
  işaretlidir. Üretim SNOMED IPS release'i, lisans/review/hash bilgileriyle ayrıca
  import edilmelidir.
- Vector embedding ve kontrollü graph expansion bir sonraki katmandır. İlk sürüm exact
  code/alias + CCO glossary eşleşmesiyle gözlenebilir bir baseline oluşturur.
- Güncel CCO'da `glucose` ve `lab` section/family yapısı bulunduğundan branch'teki
  iki-family ağırlıklı SNOMED eşlemeleri gerçek hasta CCO'larıyla yeniden kalibre
  edilmelidir.

