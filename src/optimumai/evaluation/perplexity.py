"""Perplexity — how surprised a language model is by the text it's scored on.

A language model assigns a probability to every token it sees, conditioned on
everything before it: ``p(token_i | token_1, ..., token_{i-1})``. Perplexity
turns that sequence of per-token probabilities into one number:

    H   = −(1/N) Σᵢ log p(tokenᵢ)      (average negative log-likelihood,
                                          i.e. the cross-entropy in nats)
    PPL = exp(H)

Each term ``−log p(tokenᵢ)`` is that token's *surprisal*: 0 if the model was
certain and correct, large if the model assigned the true token low
probability. Averaging surprisal gives the cross-entropy; exponentiating
undoes the log so the result lives back in "number of choices" units rather
than log-probability units.

Why "average branching factor" is the right intuition
-------------------------------------------------------
If a model were *perfectly uncertain* among ``k`` equally likely next tokens
at every step (``p = 1/k`` always), its perplexity would be exactly ``k``
(see the demo below). So a perplexity of, say, 20 means the model is, on
average, about as unsure as if it were guessing uniformly among 20 options at
each step — even though the real distribution is never actually uniform.
Lower perplexity means the model consistently placed more probability mass on
the token that actually came next, i.e. it modeled the data distribution
better. This is exactly the cross-entropy loss language models are trained
to minimize, just reported in an exponentiated, more interpretable form.

What perplexity does NOT tell you
-----------------------------------
* It is a property of a (model, text) pair, not of text alone — two models
  can rank two sentences' fluency in opposite orders.
* It is only comparable across models that share the same tokenizer and
  vocabulary; a byte-level tokenizer and a word-level tokenizer produce
  numbers on different scales because ``N`` (token count) differs.
* Low perplexity on a benchmark can reflect memorized training data rather
  than genuine generalization or "understanding."
* It says nothing about factuality, coherence at the discourse level, or
  whether the *generated* text (as opposed to scored, human-written text)
  is any good — a model can have low perplexity on real text while still
  generating repetitive or degenerate text under greedy/beam search.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

from optimumai.core._fmt import arr, num
from optimumai.core.trace import Trace


def _validate_probs(probs: list[float]) -> None:
    if not probs:
        raise ValueError("probs must contain at least one token probability")
    for p in probs:
        if not (0.0 < p <= 1.0):
            raise ValueError(f"each token probability must be in (0, 1], got {p}")


def perplexity_trace(probs: Iterable[float]) -> Trace:
    """Build the full trace: per-token surprisal → cross-entropy → perplexity.

    Args:
        probs: The probability the model assigned to the *actual* next token
            at each position (i.e. ``p(tokenᵢ | tokens_{<i})`` for the ground
            truth sequence), one value per token. Must be in ``(0, 1]``.
    """
    plist = [float(p) for p in probs]
    _validate_probs(plist)

    t = Trace(
        op="perplexity",
        formula="H = −(1/N) Σ log p(tokenᵢ);  PPL = exp(H)",
        complexity="O(N)",
        why_ai=[
            "The default intrinsic metric for language models — what pretraining "
            "cross-entropy loss looks like in interpretable units",
            "Lower perplexity on held-out text is the standard way to compare LM "
            "checkpoints before any downstream task is run",
            "'Average branching factor': a uniform guess among k options gives "
            "perplexity exactly k (see demo)",
        ],
        meta={"n_tokens": len(plist), "probs": plist},
    )

    surprisals = [-math.log(p) for p in plist]
    t.add(
        "Per-token surprisal",
        f"−log p(tokenᵢ) for p = {arr(plist)}\n→ {arr(surprisals)}",
        surprisals,
        detail="Surprisal is 0 when the model was fully confident and right; it grows "
        "without bound as the assigned probability shrinks toward 0.",
    )

    cross_entropy = sum(surprisals) / len(surprisals)
    t.add(
        "Cross-entropy (average surprisal)",
        f"H = (1/{len(plist)}) Σ = {num(cross_entropy)} nats/token",
        cross_entropy,
        detail="This is exactly the training loss of a language model, averaged over "
        "the sequence instead of a mini-batch.",
    )

    ppl = math.exp(cross_entropy)
    t.add(
        "Perplexity",
        f"PPL = exp({num(cross_entropy)}) = {num(ppl)}",
        ppl,
        detail=f"Equivalent to being as unsure as a uniform guess among ~{num(ppl)} options "
        "at each step, on average.",
    )
    t.result = ppl
    return t


def perplexity(probs: Iterable[float]) -> float:
    """Return the perplexity of a sequence of true-token probabilities."""
    return perplexity_trace(probs).result


def demo(seed: int = 0) -> Trace:
    """Perplexity of a 4-token sequence the model was reasonably confident about."""
    probs = [0.5, 0.25, 0.8, 0.4]
    return perplexity_trace(probs)
