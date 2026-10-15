# Demo model shortlist

The UI keeps two models one click away: on-prem Gemma 4 and OpenAI GPT-6 Luna. The rest stay in the experiment list so model choice does not dominate the doctor's workflow.

| UI choice | Runtime target | 24 GB L40S-24Q position |
| --- | --- | --- |
| GPU Gemma | `gemma4-26b-a4b` | Primary. Use the tested Q4 checkpoint and a bounded context. Google's estimate is 14.4 GB for Q4 static loading, before runtime and KV cache. |
| OpenAI Luna | `gpt-6-luna` | External comparison. No local VRAM use; disabled until external transfer is explicitly enabled. |
| Qwen | `Qwen/Qwen3.5-9B` | Strong single-GPU candidate. Use text-only serving to avoid vision memory and cap context for the real CCO size. |
| Kimi | `moonshotai/Moonlight-16B-A3B-Instruct` | Experimental. Official weights are 16B total/3B active; use a validated 4-bit checkpoint because BF16 weights do not leave safe 24 GB headroom. |
| DeepSeek | `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B` | Experimental. Official weights are about 15B BF16; use a validated 4-bit checkpoint and measure Turkish clinical behavior. |
| Nemotron | `nvidia/NVIDIA-Nemotron-Nano-9B-v2` | Experimental NVIDIA-native candidate; still measure full-prompt peak VRAM and output quality. |

Primary references:

- [Google Gemma 4 model overview and memory table](https://ai.google.dev/gemma/docs/core)
- [OpenAI GPT-6 Luna model page](https://developers.openai.com/api/docs/models/gpt-6-luna)
- [Qwen 3.5 9B official model card](https://huggingface.co/Qwen/Qwen3.5-9B)
- [Moonshot Moonlight 16B-A3B official model card](https://huggingface.co/moonshotai/Moonlight-16B-A3B)
- [DeepSeek R1 Distill Qwen 14B official model card](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B)
- [NVIDIA Nemotron-H model coverage](https://docs.nvidia.com/nemo/automodel/latest/model-coverage/large-language-models/nemotron-h)

These are deployment candidates, not clinical recommendations. Compare all models on the same ten anonymized patients and score factual grounding, omission of critical findings, unsupported claims, latency, and peak VRAM before choosing a default.
