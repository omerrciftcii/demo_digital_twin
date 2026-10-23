# B_AI — Tam CCO + AI-provisional oracle rehberi

Bu kol yalnız `--experimental-ai-oracle` modunda kullanılabilir. A kolundaki tam
CCO ve soru değişmeden korunur. Son soru iletisinin başına Astra tarafından
önerilmiş, Python tarafından `evidenceId/sourceSpan` bütünlüğü doğrulanmış rehber
eklenir.

Rehberde:

- `annotationSource=ai_provisional`;
- ilgili span'lar `must_use` veya `helpful`;
- distractor span'lar `importance=irrelevant` ve `useRole=do_not_use`;
- model çıktısında kullanılabilecek kısa kimlikler `promptEvidenceIds`;
- tarih, tanı statüsü, kalıcılık, ilaç aktifliği ve yasak çıkarımlar bulunur.

Bu anotasyonlar insan doğrulaması veya gold değildir. Vaka, run satırı, grading
kuyruğu, özet ve analiz çıktısı `experimental=true` taşır. Resmî B kolunun insan
inceleme kapısı bundan etkilenmez.
