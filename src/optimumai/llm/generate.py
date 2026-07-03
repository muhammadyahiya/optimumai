"""Real token generation — type a prompt, watch tokens come out.

Providers are tried in order and degrade gracefully, using only the standard
library (``urllib``) for HTTP so nothing new is required:

* **Ollama** — a local model server (``ollama run llama3.2``); zero keys, fully
  local. Auto-detected at ``http://localhost:11434``.
* **Hugging Face** — the Inference API, if ``HF_TOKEN`` / ``HUGGINGFACE_API_KEY``
  is set.
* **Anthropic** — via the optional :class:`~optimumai.tutor.llm_tutor.Tutor`.
* **toy** — a tiny built-in bigram sampler so you *always* see tokens generated,
  even fully offline (clearly labelled as a toy).
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

from optimumai.core.trace import Trace

_OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
_HF_URL = "https://api-inference.huggingface.co/models/"

# A tiny corpus for the offline toy generator (so a demo always produces tokens).
_TOY_CORPUS = (
    "the model learns to predict the next token from the previous tokens . "
    "attention lets each token look at every other token . a transformer stacks "
    "attention and a feed forward network with residual connections . gradients "
    "flow backward through the network to update the weights . softmax turns the "
    "logits into a probability distribution over the vocabulary ."
).split()


def _hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_API_KEY")


def _ollama_models() -> list[str]:
    try:
        with urllib.request.urlopen(f"{_OLLAMA_HOST}/api/tags", timeout=2) as r:
            data = json.loads(r.read())
        return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []


def _ollama_available() -> bool:
    return bool(_ollama_models())


def available_providers() -> list[str]:
    """Which generation providers are usable right now (``toy`` is always last)."""
    providers = []
    if _ollama_available():
        providers.append("ollama")
    if _hf_token():
        providers.append("huggingface")
    from optimumai.tutor.llm_tutor import Tutor

    if Tutor().available:
        providers.append("anthropic")
    providers.append("toy")
    return providers


def _ollama_generate(prompt: str, model: str | None, max_tokens: int, temperature: float):
    models = _ollama_models()
    model = model or (models[0] if models else "llama3.2")
    body = json.dumps({
        "model": model, "prompt": prompt, "stream": False,
        "options": {"num_predict": max_tokens, "temperature": temperature},
    }).encode()
    req = urllib.request.Request(
        f"{_OLLAMA_HOST}/api/generate", data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=180) as r:
        data = json.loads(r.read())
    return data.get("response", ""), model, data.get("eval_count")


def _hf_generate(prompt: str, model: str | None, max_tokens: int, temperature: float):
    model = model or "google/gemma-2-2b-it"
    body = json.dumps({
        "inputs": prompt,
        "parameters": {"max_new_tokens": max_tokens, "temperature": max(temperature, 0.01),
                       "return_full_text": False},
    }).encode()
    req = urllib.request.Request(
        _HF_URL + model, data=body,
        headers={"Authorization": f"Bearer {_hf_token()}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        data = json.loads(r.read())
    text = data[0]["generated_text"] if isinstance(data, list) else data.get("generated_text", "")
    return text, model, None


def _toy_generate(prompt: str, max_tokens: int, seed: int = 0):
    """A tiny bigram sampler over a fixed corpus — offline, deterministic-ish."""
    import random

    rng = random.Random(seed)
    bigrams: dict[str, list[str]] = {}
    for a, b in zip(_TOY_CORPUS, _TOY_CORPUS[1:], strict=False):
        bigrams.setdefault(a, []).append(b)
    last = (prompt.split() or ["the"])[-1].lower()
    out = []
    for _ in range(min(max_tokens, 24)):
        choices = bigrams.get(last) or _TOY_CORPUS
        last = rng.choice(choices)
        out.append(last)
    return " ".join(out), "toy-bigram", len(out)


def generate(
    prompt: str, provider: str = "auto", model: str | None = None,
    max_tokens: int = 64, temperature: float = 0.7,
) -> str:
    """Generate a continuation of ``prompt`` and return the text.

    ``provider="auto"`` picks the best available (Ollama → HF → Anthropic → toy).
    """
    text, _model, _n, _provider = _generate_with_meta(
        prompt, provider, model, max_tokens, temperature
    )
    return text


def _generate_with_meta(prompt, provider, model, max_tokens, temperature):
    order = [provider] if provider != "auto" else available_providers()
    last_error = None
    for prov in order:
        try:
            if prov == "ollama":
                return (*_ollama_generate(prompt, model, max_tokens, temperature), prov)
            if prov == "huggingface":
                return (*_hf_generate(prompt, model, max_tokens, temperature), prov)
            if prov == "anthropic":
                from optimumai.tutor.llm_tutor import Tutor

                return Tutor(model=model).ask(prompt), model or "claude", None, prov
            if prov == "toy":
                return (*_toy_generate(prompt, max_tokens), prov)
        except (urllib.error.URLError, OSError, KeyError, TimeoutError) as exc:
            last_error = exc
            continue
    # Everything failed — fall back to the toy generator.
    text, m, n = _toy_generate(prompt, max_tokens)
    return text, f"{m} (fallback after: {last_error})", n, "toy"


def generate_trace(
    prompt: str, provider: str = "auto", model: str | None = None,
    max_tokens: int = 64, temperature: float = 0.7,
) -> Trace:
    """Generate and return a :class:`Trace` of input → output tokens."""
    t0 = time.time()
    text, used_model, n_out, used_provider = _generate_with_meta(
        prompt, provider, model, max_tokens, temperature
    )
    dt = time.time() - t0
    in_tokens = prompt.split()
    out_tokens = text.split()
    n_out = n_out if n_out is not None else len(out_tokens)

    t = Trace(
        op="generate",
        formula="prompt → tokenize → model → sample next token → repeat",
        complexity="autoregressive: one forward pass per generated token",
        why_ai=[
            "This is what an LLM does: predict the next token, append it, repeat",
            "Ollama runs it locally; the HF/Anthropic providers call a hosted model",
            "Temperature controls randomness — the softmax knob you learned earlier",
        ],
        meta={"provider": used_provider, "model": used_model,
              "input_tokens": len(in_tokens), "output_tokens": n_out,
              "seconds": round(dt, 3)},
    )
    t.add("Provider", f"{used_provider}  ·  model = {used_model}", None,
          detail=f"Chosen from available: {', '.join(available_providers())}")
    t.add("Input", f"{len(in_tokens)} tokens: {prompt!r}", None)
    t.add("Generate", f"{n_out} tokens in {dt:.2f}s "
          f"({n_out / dt:.1f} tok/s)" if dt > 0 else f"{n_out} tokens", None,
          detail="Autoregressive decoding: each token is sampled, appended, and fed back in.")
    t.add("Output", text.strip(), None)
    t.result = text.strip()
    return t


def demo() -> Trace:
    """Generate a short continuation of a fixed prompt with whatever's available."""
    return generate_trace("The math behind attention is", max_tokens=32)
