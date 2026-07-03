# Token generation

Type a prompt, watch tokens come out. Providers are auto-detected and degrade
gracefully — using only the standard library for HTTP, so nothing extra is
required for local Ollama.

```bash
optimumai providers                       # what's available on this machine
optimumai generate "The math behind attention is" --max-tokens 32
optimumai generate "Explain softmax" --provider ollama --model llama3.2
```

```python
from optimumai import generate
print(generate("Attention is", max_tokens=32))
```

## Providers (tried in order)

| Provider | How to enable | Notes |
|---|---|---|
| **Ollama** | `ollama run llama3.2` | Local, zero keys. Auto-detected at `localhost:11434`. |
| **Hugging Face** | `export HF_TOKEN=...` | Serverless Inference API. |
| **Anthropic** | `pip install "optimumai[llm]"` + `ANTHROPIC_API_KEY` | Via the built-in tutor. |
| **toy** | always | A tiny built-in bigram sampler, so a demo always produces tokens offline. |

`generate_trace(...)` returns a `Trace` showing the provider, model, input/output
token counts, and timing — the autoregressive decode loop, made concrete.
