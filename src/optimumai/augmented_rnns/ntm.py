"""Neural Turing Machines — giving a recurrent network an external memory tape.

Graves, Wayne & Danihelka (2014), covered in Distill's *Attention and
Augmented Recurrent Neural Networks*, asked: an RNN's hidden state is its only
memory, and it is squeezed through the same fixed-size vector at every step.
What if we gave the network a separate memory bank — like a Turing machine's
tape — that it could read from and write to with *learned, differentiable*
operations?

The key move is **content-based addressing**: instead of the network learning
"go to slot 7", it emits a *key* vector and the memory returns a soft address —
a distribution over slots — based on how similar each slot's content is to the
key. Similarity is measured with cosine similarity (scale-invariant, so the
network cares about direction, not magnitude) and sharpened with a learned
temperature ``β``:

    wᵢ = softmax(β · cosine(key, Mᵢ))

This ``w`` is exactly the attention-weight idea from :mod:`attention`, just
computed with cosine similarity instead of a raw dot product, and it is used
for two operations instead of one:

    read:  r = Σᵢ wᵢ Mᵢ
    write: Mᵢ ← Mᵢ · (1 − wᵢ · e) + wᵢ · a      (erase, then add)

The write is a soft, per-slot blend of "erase what's there" (scaled by an
*erase vector* ``e`` in [0, 1]) and "add something new" (an *add vector*
``a``) — both weighted by how strongly the address vector points at that slot.
Every operation is a matmul, a cosine similarity, or a softmax, so gradients
flow through addressing, reading, and writing alike.

Where it led: NTMs showed that a network can learn to copy, sort, and do
simple algorithmic tasks by learning *where* to read/write, not just *what* to
compute — external, differentiable memory, decoupled from the hidden state.
DeepMind's Differentiable Neural Computer (2016) extended this addressing
scheme with usage tracking and temporal links. The read half of this exact
mechanism — score by similarity, softmax, weighted sum — is precisely what
became transformer attention (:mod:`optimumai.transformers.attention`): NTMs
are attention over a *writable* memory, transformers are attention over a
*fixed* one (the other tokens).
"""

from __future__ import annotations

import numpy as np

from optimumai.core._fmt import arr, num
from optimumai.core.trace import Trace


def _cosine_similarity(key: np.ndarray, memory: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity between ``key`` and each row of ``memory``."""
    key_norm = np.linalg.norm(key)
    mem_norms = np.linalg.norm(memory, axis=1)
    denom = np.maximum(key_norm * mem_norms, 1e-8)
    return (memory @ key) / denom


def _content_weights(key: np.ndarray, memory: np.ndarray, beta: float) -> np.ndarray:
    """Content-based addressing weights ``softmax(beta * cosine(key, M))``."""
    sims = _cosine_similarity(key, memory)
    scaled = beta * sims
    shifted = scaled - np.max(scaled)
    exps = np.exp(shifted)
    return exps / np.sum(exps)


def ntm_read(key: np.ndarray, memory: np.ndarray, beta: float = 1.0) -> np.ndarray:
    """Content-addressed read: blend memory rows by cosine similarity to ``key``.

    Args:
        key: 1-D vector of shape ``(d,)`` used to address memory.
        memory: 2-D array of shape ``(n, d)`` — the memory bank.
        beta: Sharpness ("key strength"); larger values make addressing more
            peaked around the best-matching slot.

    Returns:
        The read vector, shape ``(d,)``.
    """
    key = np.asarray(key, dtype=float)
    memory = np.asarray(memory, dtype=float)
    _validate(key, memory, beta)
    weights = _content_weights(key, memory, beta)
    return weights @ memory


def ntm_write(
    memory: np.ndarray,
    key: np.ndarray,
    erase: np.ndarray,
    add: np.ndarray,
    beta: float = 1.0,
) -> np.ndarray:
    """Content-addressed write: erase then add, blended by addressing weights.

    ``Mᵢ ← Mᵢ · (1 − wᵢ · erase) + wᵢ · add`` for every slot ``i``.

    Args:
        memory: 2-D array of shape ``(n, d)`` — the memory bank *before* the write.
        key: 1-D vector of shape ``(d,)`` used to address memory.
        erase: 1-D vector of shape ``(d,)`` in ``[0, 1]`` — what to clear.
        add: 1-D vector of shape ``(d,)`` — what to write in.
        beta: Addressing sharpness, as in :func:`ntm_read`.

    Returns:
        The new memory bank, shape ``(n, d)``.
    """
    memory = np.asarray(memory, dtype=float)
    key = np.asarray(key, dtype=float)
    erase = np.asarray(erase, dtype=float)
    add = np.asarray(add, dtype=float)
    _validate(key, memory, beta)
    if erase.shape != key.shape or add.shape != key.shape:
        raise ValueError(
            f"erase and add must match key shape {key.shape}; "
            f"got erase={erase.shape}, add={add.shape}"
        )
    weights = _content_weights(key, memory, beta)
    erase_term = weights[:, None] * erase[None, :]
    add_term = weights[:, None] * add[None, :]
    return memory * (1.0 - erase_term) + add_term


def _validate(key: np.ndarray, memory: np.ndarray, beta: float) -> None:
    if key.ndim != 1:
        raise ValueError(f"key must be 1-D, got shape {key.shape}")
    if memory.ndim != 2:
        raise ValueError(f"memory must be 2-D (slots × features), got shape {memory.shape}")
    if memory.shape[1] != key.shape[0]:
        raise ValueError(
            f"memory feature dim {memory.shape[1]} must match key dim {key.shape[0]}"
        )
    if beta <= 0:
        raise ValueError(f"beta must be > 0, got {beta}")


class NTMMemory:
    """A tiny Neural Turing Machine memory bank with content-based addressing.

    Wraps :func:`ntm_read` / :func:`ntm_write` as stateful operations over a
    fixed-size memory matrix, mirroring how a real NTM controller would issue
    a read followed by a write each timestep.
    """

    def __init__(self, memory: np.ndarray, beta: float = 1.0):
        self.memory = np.asarray(memory, dtype=float)
        self.beta = beta

    def read(self, key: np.ndarray) -> np.ndarray:
        """Content-addressed read from the current memory state."""
        return ntm_read(key, self.memory, beta=self.beta)

    def write(self, key: np.ndarray, erase: np.ndarray, add: np.ndarray) -> np.ndarray:
        """Content-addressed write; updates and returns the new memory state."""
        self.memory = ntm_write(self.memory, key, erase, add, beta=self.beta)
        return self.memory


def ntm_trace(
    memory: np.ndarray,
    read_key: np.ndarray,
    write_key: np.ndarray,
    erase: np.ndarray,
    add: np.ndarray,
    beta: float = 1.0,
) -> Trace:
    """Build the full trace of an NTM read followed by a write.

    Shows the content-based addressing weights for both operations, the read
    vector, and the memory bank before/after the write.
    """
    memory = np.asarray(memory, dtype=float)
    read_key = np.asarray(read_key, dtype=float)
    write_key = np.asarray(write_key, dtype=float)
    erase = np.asarray(erase, dtype=float)
    add = np.asarray(add, dtype=float)
    _validate(read_key, memory, beta)
    _validate(write_key, memory, beta)
    if erase.shape != write_key.shape or add.shape != write_key.shape:
        raise ValueError(
            f"erase and add must match key shape {write_key.shape}; "
            f"got erase={erase.shape}, add={add.shape}"
        )

    t = Trace(
        op="ntm",
        formula="w=softmax(β·cos(k,Mᵢ)); read=Σwᵢ Mᵢ; write: Mᵢ←Mᵢ(1−wᵢe)+wᵢa",
        complexity="O(n·d) per read or write, for n slots of dimension d",
        why_ai=[
            "External, differentiable memory decouples 'how much to remember' "
            "from the RNN's fixed-size hidden state",
            "Content-based (cosine) addressing lets the controller find a slot "
            "by *what* it contains, not by a hardcoded index",
            "The read half of this mechanism is the direct ancestor of "
            "transformer attention (optimumai.transformers.attention)",
        ],
        meta={"n_slots": memory.shape[0], "d_model": memory.shape[1], "beta": beta},
    )

    t.add(
        "Memory bank before any operation",
        f"M, shape {memory.shape}\n{arr(memory)}",
        memory.copy(),
    )

    read_sims = _cosine_similarity(read_key, memory)
    t.add(
        "Read addressing: cosine(read_key, Mᵢ)",
        f"similarity of the read key to each slot\n{arr(read_sims)}",
        read_sims,
        detail="Cosine similarity is scale-invariant — only direction, not "
        "magnitude, determines the match.",
    )

    read_weights = _content_weights(read_key, memory, beta)
    t.add(
        "Sharpen and normalize: softmax(β · similarity)",
        f"read weights (sum to 1)  →  {arr(read_weights)}",
        read_weights,
        detail=f"β = {num(beta)} controls how peaked the addressing is; "
        f"weights sum to {num(float(np.sum(read_weights)))}.",
    )

    read_vec = read_weights @ memory
    t.add(
        "Read: r = Σ wᵢ Mᵢ",
        f"read vector  →  {arr(read_vec)}",
        read_vec,
        detail="A soft lookup — the controller receives a blend of the most "
        "relevant slots instead of one hard-indexed row.",
    )

    write_sims = _cosine_similarity(write_key, memory)
    write_weights = _content_weights(write_key, memory, beta)
    t.add(
        "Write addressing: softmax(β · cosine(write_key, Mᵢ))",
        f"write weights (sum to 1)  →  {arr(write_weights)}",
        write_weights,
        detail=f"Raw similarities were {arr(write_sims)} before sharpening.",
    )

    erase_term = write_weights[:, None] * erase[None, :]
    after_erase = memory * (1.0 - erase_term)
    t.add(
        "Erase: Mᵢ ← Mᵢ · (1 − wᵢ · erase)",
        f"memory after erase, before add\n{arr(after_erase)}",
        after_erase,
        detail="erase ∈ [0,1] per feature; wᵢ=0 slots are left untouched.",
    )

    add_term = write_weights[:, None] * add[None, :]
    new_memory = after_erase + add_term
    t.add(
        "Add: Mᵢ ← Mᵢ + wᵢ · add",
        f"memory after the full write\n{arr(new_memory)}",
        new_memory,
        detail="The most strongly addressed slots absorb the most of `add`.",
    )
    t.result = {"read": read_vec, "memory": new_memory}
    return t


def demo(seed: int = 0) -> Trace:
    """A tiny, reproducible 4-slot / d=3 read-then-write example."""
    rng = np.random.default_rng(seed)
    memory = rng.normal(size=(4, 3)).round(2)
    read_key = rng.normal(size=3).round(2)
    write_key = rng.normal(size=3).round(2)
    erase = np.clip(rng.uniform(size=3), 0.0, 1.0).round(2)
    add = rng.normal(size=3).round(2)
    return ntm_trace(memory, read_key, write_key, erase, add, beta=2.0)
