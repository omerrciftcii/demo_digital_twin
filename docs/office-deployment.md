# Office-only deployment

This application handles clinical records. A private GitHub repository protects source access; it does not make a running web service office-only.

## Recommended topology

Run the UI/API container on an office host or a private VPN address. The browser talks only to that host. The server calls OpenRouter for every non-OpenAI model and calls OpenAI directly for OpenAI models.

```text
Doctor browser -> office LAN or VPN -> Digital Twin :8092 -> OpenRouter (Gemma/Qwen/Kimi/DeepSeek/Nemotron)
                                                   \-> OpenAI API (GPT-6 Luna)
```

Do not open port 8092 on a public interface. For a host outside the office, use the organization's existing private VPN instead.

## Required settings

Set these values in the shell or in a server-side `.env` file that is never committed:

```dotenv
DEMO_BIND_IP=10.40.0.25
DEMO_ALLOWED_HOSTS=demo.internal.example,10.40.0.25
DEMO_ALLOWED_NETWORKS=10.40.0.0/16

# No patient payload leaves the server until this explicit switch is enabled.
DEMO_ALLOW_EXTERNAL_MODELS=true

OPENROUTER_API_KEY=...
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
DEMO_OPENROUTER_REQUIRE_ZDR=true
DEMO_OPENROUTER_ALLOW_FALLBACKS=true

OPENAI_API_KEY=...
OPENAI_BASE_URL=https://api.openai.com/v1
DEMO_OPENAI_MODEL=gpt-6-luna
```

Use the narrowest real office/VPN CIDR. Also restrict TCP 8092 with the host firewall. If a reverse proxy is used, enforce the same CIDR there and add TLS plus organizational authentication; do not trust client-supplied forwarding headers for access control.

Start and stop the occasional demo with:

```bash
docker compose -f compose.office.yaml up -d --build
docker compose -f compose.office.yaml down
```

Place the ten anonymized JSON files in `data/cco/`. The directory is ignored by Git and mounted read-only. Files are listed by filename as `Hasta 1`, `Hasta 2`, and so on; the original identifiers are not displayed.

## External model routing

The browser never receives either API key. OpenRouter requests use `data_collection=deny` and request ZDR by default. Provider fallbacks stay within those routing requirements; set `DEMO_OPENROUTER_PROVIDER` when an evaluation must pin one OpenRouter provider.

The selected patient's full CCO text is sent to the selected external service. Keep `DEMO_ALLOW_EXTERNAL_MODELS=false` for any record that has not been approved for external transfer, even when it is described as anonymized.

OpenAI models never use OpenRouter. GPT-6 Luna goes directly to `api.openai.com` with `OPENAI_API_KEY`. All other demo choices share `OPENROUTER_API_KEY`:

- Gemma 4 26B-A4B: `google/gemma-4-26b-a4b-it`
- Qwen 3.5 9B: `qwen/qwen3.5-9b`
- Kimi K2.5: `moonshotai/kimi-k2.5`
- DeepSeek V4.1 Flash: `deepseek/deepseek-v4.1-flash`
- NVIDIA Nemotron 3 Nano 30B-A3B: `nvidia/nemotron-3-nano-30b-a3b`

Model slugs can be overridden with the corresponding `DEMO_*_MODEL` environment variables without changing the UI or response format.
