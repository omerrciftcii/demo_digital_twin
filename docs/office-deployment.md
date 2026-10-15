# Office-only deployment

This application handles clinical records. A private GitHub repository protects source access; it does not make a running web service office-only.

## Recommended topology

Run both the UI/API container and the active vLLM service on the GPU host. Expose the UI only on the host's office-LAN or VPN address. Keep vLLM on loopback or a private Docker network. The GPU host may stay off outside test sessions.

```text
Doctor browser -> office LAN or VPN -> Digital Twin :8092 -> local vLLM :18000
                                                   \-> OpenAI API (only when enabled)
```

For a host outside the office, use a private VPN address (for example the organization's existing WireGuard/Tailscale network) instead of opening port 8092 on the public interface.

## Required settings

Set these values in the shell or in a server-side `.env` file that is never committed:

```dotenv
DEMO_BIND_IP=10.40.0.25
DEMO_ALLOWED_HOSTS=demo.internal.example,10.40.0.25
DEMO_ALLOWED_NETWORKS=10.40.0.0/16
DEMO_VLLM_BASE_URL=http://host.docker.internal:18000
DEMO_GEMMA_MODEL=gemma4-26b-a4b
```

Use the narrowest real office/VPN CIDR. Also restrict TCP 8092 with the host firewall. If a reverse proxy is used, enforce the same CIDR there and add TLS plus organizational authentication; do not trust client-supplied forwarding headers for access control.

Start and stop the occasional demo with:

```bash
docker compose -f compose.office.yaml up -d --build
docker compose -f compose.office.yaml down
```

Place the ten anonymized JSON files in `data/cco/`. The directory is ignored by Git and mounted read-only. Files are listed by filename as `Hasta 1`, `Hasta 2`, and so on; the original identifiers are not displayed.

## OpenAI comparison

OpenAI is deliberately locked by default. Enable it only after the anonymization and data-processing decision is approved:

```dotenv
DEMO_ALLOW_EXTERNAL_MODELS=true
OPENAI_API_KEY=...
DEMO_OPENAI_MODEL=gpt-6-luna
```

The browser never receives the API key. The selected patient's full CCO text is sent to the chosen model, so this switch must remain off for records that are not approved for external transfer.

## GPU model services

Each optional local model can have its own OpenAI-compatible endpoint. Only running endpoints appear as ready in the UI. This keeps the interface simple while allowing one model service at a time on a 24 GB card.

- Gemma 4 26B-A4B: `DEMO_VLLM_BASE_URL`
- Qwen 3.5 9B: `DEMO_QWEN_BASE_URL`
- Kimi/Moonlight 16B-A3B: `DEMO_KIMI_BASE_URL`
- DeepSeek R1 Distill Qwen 14B: `DEMO_DEEPSEEK_BASE_URL`
- NVIDIA Nemotron Nano 9B v2: `DEMO_NEMOTRON_BASE_URL`

For the larger 14B–26B candidates, use the tested 4-bit checkpoint and leave VRAM headroom for the CCO context/KV cache. Do not expose a raw vLLM port to the public internet.
