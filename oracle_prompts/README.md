# Oracle A/B/C prompt sözleşmesi

Üç kol da aynı `VERIFIED_SYSTEM`, aynı tam CCO, aynı soru ve aynı cevap şemasını
kullanır. B ve C'deki rehber CCO'nun yerine geçmez; son kullanıcı iletisinin
başına eklenir. Çalıştırılan kesin istemler ve hash'leri `run_oracle_abc.py`
çıktısında saklanır.

- A: kanıt rehberi yok.
- B: yalnız insan doğrulamalı oracle rehberi.
- C: repodaki otomatik lexical ranker rehberi.

Bu klasördeki dosyalar okunabilir şablonlardır. Çalışma zamanı sözleşmesinin
kaynağı `build_oracle_abc_plan` ve `ANSWER_JSON_SCHEMA` kodudur.
