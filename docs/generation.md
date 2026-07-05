# Token generation

Type a prompt, watch tokens come out. Providers are auto-detected and degrade
gracefully — using only the standard library for HTTP, so nothing extra is
required for local Ollama.

```bash
optimumai providers                                    # what's available on this machine
optimumai generate "The math behind attention is"      # real tokens, streamed live
optimumai generate "Explain softmax" --provider ollama --model llama3.2
optimumai generate "What is a gradient?" --max-tokens 64 --temperature 0.7
```

```python
from optimumai import generate

text = generate("Attention is", max_tokens=32)
print(text)
```

---

## Providers

Tried in this order — the first one that works wins:

| Provider | How to enable | Notes |
|---|---|---|
| **Ollama** | `ollama run llama3.2` | Local, zero API keys. Auto-detected at `localhost:11434`. |
| **Hugging Face** | `export HF_TOKEN=hf_...` | Serverless Inference API. |
| **Anthropic** | `pip install "optimumai[llm]"` + `ANTHROPIC_API_KEY=...` | Via the built-in tutor. |
| **toy** | Always available | Built-in bigram sampler — always produces tokens offline. |

The toy fallback means `generate(...)` never raises an error due to missing
providers. A demo always runs.

---

## The `generate_trace` variant

```python
from optimumai.llm import generate_trace

trace = generate_trace("Attention is all you need", max_tokens=16)
trace.result        # the generated text
trace.provider      # which provider was used: 'ollama', 'huggingface', 'anthropic', 'toy'
trace.model         # model name
trace.input_tokens  # input token count
trace.output_tokens # output token count
trace.timing_ms     # wall-clock decode time
trace.render("engineer")  # prints a full structured breakdown
```

---

## The LLM tutor

The optional tutor wraps generation into a Q&A interface over AI/ML concepts:

```bash
optimumai ask "why LayerNorm after attention?"
optimumai ask "explain the difference between RoPE and sinusoidal positional encoding"
optimumai ask "what is the intuition behind DPO?"
```

```python
from optimumai.tutor import Tutor

tutor = Tutor()
print(tutor.ask("why does the transformer need positional encoding?"))
```

!!! note "Graceful degradation"
    Without `optimumai[llm]` and an API key, `Tutor().ask(...)` never raises —
    it returns a friendly message explaining exactly what's missing and reminds
    you that the core math works fully offline.

---

## Streaming (Ollama)

When Ollama is available, `generate` streams tokens to the terminal as they
arrive — you see the output grow token by token, just like a chat interface.

```bash
optimumai generate "Walk me through the scaled dot-product attention formula step by step"
```

---

## Using in notebooks

```python
from optimumai import generate

# Stream tokens in a Jupyter cell
for chunk in generate("Softmax turns logits into", stream=True, max_tokens=50):
    print(chunk, end="", flush=True)
```
