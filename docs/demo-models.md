# Demo model shortlist

The UI keeps two models one click away: Gemma 4 through OpenRouter and GPT-6 Luna directly through OpenAI. The rest stay in the experiment list so model choice does not dominate the doctor's workflow.

| UI choice | Provider | API model id |
| --- | --- | --- |
| Gemma 4 | OpenRouter | `google/gemma-4-26b-a4b-it` |
| OpenAI Luna | OpenAI direct | `gpt-6-luna` |
| Qwen | OpenRouter | `qwen/qwen3.5-9b` |
| Kimi | OpenRouter | `moonshotai/kimi-k2.5` |
| DeepSeek | OpenRouter | `deepseek/deepseek-v4.1-flash` |
| Nemotron | OpenRouter | `nvidia/nemotron-3-nano-30b-a3b` |

The OpenRouter ids above were checked against its live model catalog on 15 October 2026. They are configuration defaults rather than hard UI dependencies; each can be overridden with a `DEMO_*_MODEL` environment variable.

Primary references:

- [OpenRouter live model catalog](https://openrouter.ai/api/v1/models)
- [OpenRouter Chat Completions API](https://openrouter.ai/docs/api/api-reference/chat/create-a-chat-completion)
- [OpenAI GPT-6 Luna model page](https://developers.openai.com/api/docs/models/gpt-6-luna)

These are evaluation candidates, not clinical recommendations. Compare all models on the same ten anonymized patients and score factual grounding, omission of critical findings, unsupported claims, latency, and cost before choosing a default.
