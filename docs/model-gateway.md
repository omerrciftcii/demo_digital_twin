# Model geçidi

8091'deki soru-cevap API'sine dokunmayan ayrı bir süreçtir. Başka arayüzler buradan yerel modeli dener, indirilmiş modelleri listeler ve vLLM sürecini değiştirir.

```bash
export GATEWAY_API_KEYS="yerel-anahtar"
export GATEWAY_MODELS_CONFIG=config/gateway-models.example.json
python3 scripts/run_model_gateway.py
```

Varsayılan bağ `127.0.0.1:8095`. `0.0.0.0` veya `::` yalnız `GATEWAY_ALLOW_PUBLIC_BIND=true` iken kabul edilir. Betik bu kontrolü süreç açılırken yapar.

## Anahtar

`Authorization: Bearer` veya `x-api-key`. Sunucu anahtarın SHA-256 özetini `GATEWAY_API_KEY_SHA256` (virgülle ayrılmış onaltılık) ve `GATEWAY_API_KEYS` (açılışta özetlenir) ile `hmac.compare_digest` üzerinden karşılaştırır. Düz anahtar loga yazılmaz. Hiç anahtar yoksa `/health` dışında her çağrı 401 döner. `/health` anahtarsızdır ve sır sızdırmaz.

## Uçlar

- `GET /health` — süreç ayakta mı, hangi model seçili.
- `GET /v1/models` ve `POST /v1/chat/completions` — `GATEWAY_VLLM_BASE_URL` (varsayılan `http://127.0.0.1:8000/v1`) üzerine vekil. `temperature` ve `max_tokens` olduğu gibi gider. `enable_thinking` `chat_template_kwargs.enable_thinking` olur. `system` sistem iletisini yazar veya başa ekler. Akış (`stream: true`) parçalar halinde iletilir.
- `GET /models` — yapılandırma dosyasındaki modeller: kimlik, ad, start/stop betiği, argümanlar. Dosya `src/` altında durmaz.
- `GET /models/status` — `idle`, `stopping`, `starting`, `ready`, `failed`; `progress`, `activeId`, `error`.
- `POST /models/switch` — `{ "id": "..." }`. Önce stop betiği, sonra start betiği ve argümanları. Betik proje kökünün dışına `..` ile çıkamaz. Süreç meşgulse ikinci çağrı 409 alır; kilit, bekleyen `await` olmadan alınır.
- `POST /qa` — gövdeyi `GATEWAY_QA_BASE_URL` (veya gövdedeki `qaBaseUrl`) altındaki `/api/v1/text/answer` adresine iletir. İletilen alanlar `recordId`, `question`, `history`, `stream`. 8091 şemasına yeni alan eklenmez.

`GATEWAY_LOG_CONTENT` varsayılanı kapalıdır. Log satırı yöntem, yol, durum ve `elapsed_ms` tutar. İleti metni yazılmaz; gerçek kayıt bu uçtan geçebilir. Betiğin standart çıktısı da loglanmaz, yalnız betik adı ve çıkış kodu yazılır.

## Betikler

`scripts/models/start-model.sh` ve `scripts/models/stop-vllm.sh` şablondur: ağırlık indirmez, vLLM başlatmaz, çıkış kodu 0 döner. Operatör bunları kendi indirdiği modelin başlatma komutuyla değiştirir. Zaman aşımı `GATEWAY_SCRIPT_TIMEOUT_SECONDS` (varsayılan 120).

Örnek yapılandırma `config/gateway-models.example.json`. Her kayıt `id`, `name`, `start`, `stop`, `args` taşır.
