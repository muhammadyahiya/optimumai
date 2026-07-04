"""Attention as differentiable memory access — the RNN-era idea, before "Transformer".

Distill's *Attention and Augmented Recurrent Neural Networks* (2016) frames
attention not as a transformer component but as a much older, simpler idea:
give a recurrent network a *memory* — a bank of vectors — and let it learn to
read from that memory with a soft, differentiable lookup instead of a hard
index.

The intuition: imagine an RNN translating a sentence. At every output step it
would love to "look back" at the input word that matters most right now. A
hard lookup (``memory[3]``) is not differentiable — you cannot backprop
through an integer index. So instead the network computes a *relevance score*
for every row of memory, turns those scores into a probability distribution
with softmax, and reads a **weighted blend** of every row:

    scores = M · q                      (how relevant is each row to the query?)
    weights = softmax(scores)           (turn relevance into a distribution)
    read  = Σᵢ weightsᵢ · Mᵢ             (blend the rows by their weight)

Every step is a matrix multiply or a softmax — both fully differentiable — so
the whole read operation can be trained end-to-end with backprop, exactly like
any other layer. Because every weight is >0, the network can never fully
"switch off" a memory row, but with enough training weights become close to
one-hot, approximating a hard lookup while staying trainable.

Where it led: Bahdanau et al. (2014) used exactly this mechanism to fix the
fixed-length bottleneck of sequence-to-sequence RNNs. Distill's article shows
the same read operation powering Neural Turing Machines (:mod:`ntm`) and
Adaptive Computation Time (:mod:`act`). Three years later "Attention Is All
You Need" removed the RNN entirely and kept only this memory-read idea,
scaled up into self-attention — see :mod:`optimumai.transformers.attention`
for that descendant. This module is the ancestor: a single query reading a
fixed external memory, no queries-attending-to-queries, no multi-head split.
"""

from __future__ import annotations

import numpy as np

from optimumai.core._fmt import arr, num
from optimumai.core.trace import Trace


def attention_read(query: np.ndarray, memory: np.ndarray) -> np.ndarray:
    """Read a weighted blend of ``memory`` rows using content-based attention.

    Args:
        query: A 1-D vector of shape ``(d,)`` — "what am I looking for?".
        memory: A 2-D array of shape ``(n, d)`` — ``n`` memory slots of
            dimension ``d`` — "what is stored?".

    Returns:
        The blended read vector, shape ``(d,)``.
    """
    q = np.asarray(query, dtype=float)
    mem = np.asarray(memory, dtype=float)
    if q.ndim != 1:
        raise ValueError(f"query must be 1-D, got shape {q.shape}")
    if mem.ndim != 2:
        raise ValueError(f"memory must be 2-D (slots × features), got shape {mem.shape}")
    if mem.shape[1] != q.shape[0]:
        raise ValueError(
            f"memory feature dim {mem.shape[1]} must match query dim {q.shape[0]}"
        )

    scores = mem @ q
    shifted = scores - np.max(scores)
    exps = np.exp(shifted)
    weights = exps / np.sum(exps)
    return weights @ mem


def attention_read_trace(query: np.ndarray, memory: np.ndarray) -> Trace:
    """Build the full trace of a content-based memory read.

    Shows the relevance scores, the softmax attention weights (which sum to
    1), and the resulting blended read vector.
    """
    q = np.asarray(query, dtype=float)
    mem = np.asarray(memory, dtype=float)
    if q.ndim != 1:
        raise ValueError(f"query must be 1-D, got shape {q.shape}")
    if mem.ndim != 2:
        raise ValueError(f"memory must be 2-D (slots × features), got shape {mem.shape}")
    if mem.shape[1] != q.shape[0]:
        raise ValueError(
            f"memory feature dim {mem.shape[1]} must match query dim {q.shape[0]}"
        )

    t = Trace(
        op="attention_read",
        formula="read = Σᵢ softmax(M·q)ᵢ · Mᵢ",
        complexity="O(n·d) for n memory slots of dimension d",
        why_ai=[
            "The original RNN-era attention: Bahdanau et al. (2014) used this "
            "to let a decoder look back at any encoder state",
            "A fully differentiable substitute for a hard, non-differentiable "
            "memory index — the whole read is trainable end-to-end",
            "The direct ancestor of transformer self-attention "
            "(see optimumai.transformers.attention) and of NTM/DNC memory reads",
        ],
        meta={"n_slots": mem.shape[0], "d_model": mem.shape[1]},
    )

    scores = mem @ q
    t.add(
        "Score each memory slot: M · q",
        f"relevance of every row of memory to the query\n{arr(scores)}",
        scores,
        detail="Entry i is the dot product of memory row i with the query — "
        "higher means more relevant.",
    )

    x_max = float(np.max(scores))
    shifted = scores - x_max
    exps = np.exp(shifted)
    denom = float(np.sum(exps))
    weights = exps / denom
    t.add(
        "Softmax the scores",
        f"weights = softmax(scores)  →  {arr(weights)}",
        weights,
        detail=f"Weights sum to {num(float(np.sum(weights)))} — a valid probability "
        "distribution over memory slots, and fully differentiable.",
    )

    read = weights @ mem
    t.add(
        "Blend memory rows by weight: read = Σ wᵢ Mᵢ",
        f"read vector  →  {arr(read)}",
        read,
        detail="A convex combination of every memory row — no hard indexing, "
        "so gradients flow back into both the query and the memory.",
    )
    t.result = read
    return t


def demo(seed: int = 0) -> Trace:
    """A tiny, reproducible 4-slot memory / d=3 example for docs and the CLI."""
    rng = np.random.default_rng(seed)
    memory = rng.normal(size=(4, 3)).round(2)
    query = rng.normal(size=3).round(2)
    return attention_read_trace(query, memory)
