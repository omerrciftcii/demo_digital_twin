# Demo model shortlist

The UI keeps two models one click away: Gemma 4 running on the project's GPU server and GPT-6 Luna directly through OpenAI. GPT-5.4, o4-mini, GPT-6.1 Sol, and GPT-6 Astra are also direct OpenAI choices in the full model list. The OpenRouter models stay in that same list so model choice does not dominate the doctor's workflow.

| UI choice | Provider | API model id |
| --- | --- | --- |
| GPU Gemma 4 | Local vLLM on `185.48.24.103` | `gemma4-26b-a4b` |
| OpenAI Luna | OpenAI direct | `gpt-6-luna` |
| OpenAI GPT-5.4 | OpenAI direct | `gpt-5.4` |
| OpenAI o4-mini | OpenAI direct | `o4-mini` |
| OpenAI GPT-6.1 Sol | OpenAI direct | `gpt-6.1-sol` |
| OpenAI GPT-6 Astra | OpenAI direct | `gpt-6-astra` |
| Qwen | OpenRouter | `qwen/qwen3.5-9b` |
| Kimi | OpenRouter | `moonshotai/kimi-k2.5` |
| DeepSeek | OpenRouter | `deepseek/deepseek-v4.1-flash` |
| Nemotron | OpenRouter | `nvidia/nemotron-3-nano-30b-a3b` |

The OpenRouter ids above were checked against its live model catalog on 15 October 2026. They are configuration defaults rather than hard UI dependencies; each can be overridden with a `DEMO_*_MODEL` environment variable. Gemma uses `DEMO_VLLM_BASE_URL` and never goes through OpenRouter.

Primary references:

- [OpenRouter live model catalog](https://openrouter.ai/api/v1/models)
- [OpenRouter Chat Completions API](https://openrouter.ai/docs/api/api-reference/chat/create-a-chat-completion)
- [OpenAI GPT-6 Luna model page](https://developers.openai.com/api/docs/models/gpt-6-luna)
- [OpenAI GPT-5.4 model page](https://developers.openai.com/api/docs/models/gpt-5.4)
- [OpenAI o4-mini model page](https://developers.openai.com/api/docs/models/o4-mini)
- [OpenAI GPT-6.1 Sol model page](https://developers.openai.com/api/docs/models/gpt-6.1-sol)
- [OpenAI GPT-6 Astra model page](https://developers.openai.com/api/docs/models/gpt-6-astra)
- [vLLM OpenAI-compatible server](https://docs.vllm.ai/en/latest/serving/openai_compatible_server/)

These are evaluation candidates, not clinical recommendations. Compare all models on the same ten anonymized patients and score factual grounding, omission of critical findings, unsupported claims, latency, and cost before choosing a default.
