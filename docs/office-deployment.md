# Office-only deployment

This application handles clinical records. A private GitHub repository protects source access; it does not make a running web service office-only.

## Recommended topology

For the first test, run the UI/API on the doctor's office computer. Keep Gemma 4 on the existing GPU server and reach its loopback-only vLLM port through SSH. The application calls OpenRouter only for Qwen, Kimi, DeepSeek and Nemotron, and calls OpenAI directly for GPT-6 Luna.

```text
Doctor browser -> office computer -> Digital Twin :8092 -> SSH tunnel -> GPU vLLM (Gemma 4)
                                                   \-> OpenRouter (Qwen/Kimi/DeepSeek/Nemotron)
                                                   \-> OpenAI API (GPT-6 Luna)
```

Do not open port 8092 on a public interface. For a host outside the office, use the organization's existing private VPN instead.

## First local test with the GPU server

Open the tunnel in the first PowerShell window. Type the SSH password only into SSH's password prompt, never into `.env`, Git, chat, or a command argument:

```powershell
ssh -N -L 18000:127.0.0.1:8000 arge-adm@185.48.24.103
```

Keep that window open. This command exposes the GPU server's private port `8000` only as `127.0.0.1:18000` on your computer. In a second PowerShell window:

```powershell
cd C:\Users\omercftc\Documents\Projects\demo_digital_twin
$env:DEMO_VLLM_BASE_URL = "http://127.0.0.1:18000"
.\scripts\start_patient_chat_demo.ps1 -CcoDirectory (Resolve-Path tests\fixtures\text\patients).Path
```

Open `http://127.0.0.1:8092` and choose **GPU Gemma**. This path does not require an OpenRouter key or `DEMO_ALLOW_EXTERNAL_MODELS=true`. If `/v1/models` is not served on remote port `8000`, replace the last port in the SSH command with the port used by the running vLLM service.

## Required office settings

Set these values in the shell or in a server-side `.env` file that is never committed:

```dotenv
DEMO_BIND_IP=10.40.0.25
DEMO_ALLOWED_HOSTS=demo.internal.example,10.40.0.25
DEMO_ALLOWED_NETWORKS=10.40.0.0/16

# Existing GPU Gemma service. For a local UI this is the SSH tunnel above.
DEMO_VLLM_BASE_URL=http://127.0.0.1:18000
DEMO_GEMMA_MODEL=gemma4-26b-a4b
DEMO_LOCAL_API_KEY=

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

The selected patient's full CCO text is sent to OpenRouter or OpenAI only when one of those models is selected. Keep `DEMO_ALLOW_EXTERNAL_MODELS=false` for any record that has not been approved for external transfer, even when it is described as anonymized. GPU Gemma is not controlled by this external-transfer switch because its traffic stays on the SSH/private path.

OpenAI models never use OpenRouter. GPT-6 Luna goes directly to `api.openai.com` with `OPENAI_API_KEY`. Gemma 4 goes to the configured vLLM service. The remaining demo choices share `OPENROUTER_API_KEY`:

- Qwen 3.5 9B: `qwen/qwen3.5-9b`
- Kimi K2.5: `moonshotai/kimi-k2.5`
- DeepSeek V4.1 Flash: `deepseek/deepseek-v4.1-flash`
- NVIDIA Nemotron 3 Nano 30B-A3B: `nvidia/nemotron-3-nano-30b-a3b`

Model slugs can be overridden with the corresponding `DEMO_*_MODEL` environment variables without changing the UI or response format.
