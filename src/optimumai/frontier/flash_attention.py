"""FlashAttention — IO-aware attention that is fast *and* exact.

Standard attention (see :mod:`optimumai.transformers.attention`) materializes
the full ``N×N`` score matrix ``S = Q·Kᵀ`` in slow, high-capacity HBM, runs a
softmax over it, then multiplies by ``V``. That N×N matrix is the bottleneck:
attention is *memory-bound*, not compute-bound, and reading/writing S from HBM
dominates the runtime for long sequences.

FlashAttention never builds S. It **tiles** Q, K and V into blocks small enough
to live in fast on-chip SRAM, streams the K/V blocks past each Q block, and
folds each block into the answer using the **online-softmax** recurrence:

    m  ← running row-max of the scores seen so far
    l  ← running Σ e^(score − m)     (the softmax denominator so far)
    O  ← running, un-normalized weighted sum of value vectors

When a new block arrives with its own block-max ``m_blk`` we pick a new running
max ``m_new = max(m_old, m_blk)`` and rescale the two accumulators by the
correction factor ``alpha = e^(m_old − m_new)`` so every term is expressed
relative to the *same* max. Because that rescaling is algebraically exact, the
final ``O / l`` equals ``softmax(Q·Kᵀ·scale)·V`` to floating-point precision —
FlashAttention is an EXACT algorithm, not an approximation. This module proves
that by computing ``max_abs_error`` against the standard formula and asserting it
is ~0.

Why the online-softmax algebra is exact
---------------------------------------
For scalars, ``e^(x−a) = e^(x−b)·e^(b−a)``. So when the running max changes from
``m_old`` to ``m_new`` every previously accumulated exponential ``e^(s−m_old)``
becomes ``e^(s−m_new)`` after multiplying by ``alpha = e^(m_old−m_new)``. Scaling
both ``l`` and ``O`` by the same ``alpha`` before adding the new block keeps the
whole running sum consistent — nothing is dropped or approximated.
"""

from __future__ import annotations

import numpy as np

from optimumai.core._fmt import arr, num
from optimumai.core.explain import ExplainLevel
from optimumai.core.trace import Trace


def _standard_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, scale: float):  # noqa: N803
    """Reference softmax(Q·Kᵀ·scale)·V used only to prove flash is exact."""
    scores = (Q @ K.T) * scale
    scores = scores - np.max(scores, axis=-1, keepdims=True)
    weights = np.exp(scores)
    weights = weights / np.sum(weights, axis=-1, keepdims=True)
    return weights @ V


def flash_attention_trace(Q, K, V, block_size: int = 2) -> Trace:  # noqa: N803
    """Build the full trace of tiled attention with online softmax.

    Args:
        Q: Query matrix ``(n_q × d)``.
        K: Key matrix ``(n_k × d)``.
        V: Value matrix ``(n_k × d_v)``.
        block_size: How many rows of K/V (and of Q) live in one SRAM tile.
    """
    Q = np.asarray(Q, dtype=float)
    K = np.asarray(K, dtype=float)
    V = np.asarray(V, dtype=float)
    for name, mat in ("Q", Q), ("K", K), ("V", V):
        if mat.ndim != 2:
            raise ValueError(f"{name} must be 2-D (tokens × features), got shape {mat.shape}")
    if Q.shape[1] != K.shape[1]:
        raise ValueError(
            f"Q and K must share the feature dim; got {Q.shape[1]} and {K.shape[1]}"
        )
    if K.shape[0] != V.shape[0]:
        raise ValueError(
            f"K and V must share the number of tokens; got {K.shape[0]} and {V.shape[0]}"
        )
    if block_size < 1:
        raise ValueError(f"block_size must be >= 1, got {block_size}")

    n_q, d = Q.shape
    n_k = K.shape[0]
    d_v = V.shape[1]
    scale = 1.0 / np.sqrt(d)

    q_blocks = int(np.ceil(n_q / block_size))
    kv_blocks = int(np.ceil(n_k / block_size))

    t = Trace(
        op="flash_attention",
        formula="O = softmax(Q·Kᵀ/√d)·V, computed block-by-block with online softmax",
        complexity=(
            "compute O(N²·d) (same as standard); HBM IO drops from O(N²) to "
            "O(N²/M) for SRAM of size M — attention is memory-bound, so this is the win"
        ),
        why_ai=[
            "Attention is memory-bound: reading/writing the N×N score matrix from "
            "HBM dominates runtime, not the matmuls",
            "Tiling keeps Q/K/V blocks in on-chip SRAM, which has ~10x the bandwidth "
            "of HBM, so the hot loop touches fast memory",
            "Online softmax means the full N×N matrix is never materialized → peak "
            "memory is linear in sequence length instead of quadratic",
            "~7.6x faster attention on GPT-2 and the reason 100k+ token context "
            "windows are trainable at all",
            "EXACT, not approximate: the running-max rescaling is algebraically "
            "identical to a global softmax (proven by max_abs_error ≈ 0 below)",
        ],
        meta={
            "block_size": block_size,
            "q_blocks": q_blocks,
            "kv_blocks": kv_blocks,
            "n_blocks": q_blocks * kv_blocks,
            "q_shape": Q.shape,
            "k_shape": K.shape,
            "v_shape": V.shape,
            "scale": float(scale),
            "io_note": (
                "Standard attention reads/writes an N×N matrix in HBM; flash streams "
                "block_size-row tiles through SRAM, so HBM traffic is O(N²/block_size)."
            ),
        },
    )

    t.add(
        "Tiling plan",
        f"scale = 1/√d = 1/√{d} = {num(scale)}; split into "
        f"{q_blocks} query block(s) × {kv_blocks} key/value block(s) of "
        f"up to {block_size} row(s) each",
        detail=(
            "Each tile fits in fast SRAM. We stream K/V tiles past each Q tile and "
            "fold them in on the fly — the full N×N score matrix is never stored."
        ),
    )

    output = np.zeros((n_q, d_v), dtype=float)

    # ---- Online-softmax block algorithm -------------------------------------
    for qi in range(q_blocks):
        q_lo = qi * block_size
        q_hi = min(q_lo + block_size, n_q)
        Q_blk = Q[q_lo:q_hi]  # noqa: N806 - block of queries

        rows = q_hi - q_lo
        m_i = np.full(rows, -np.inf)  # running row max
        l_i = np.zeros(rows)  # running Σ exp(scores − m_i)
        O_i = np.zeros((rows, d_v))  # running (un-normalized) output  # noqa: N806

        for kj in range(kv_blocks):
            k_lo = kj * block_size
            k_hi = min(k_lo + block_size, n_k)
            K_blk = K[k_lo:k_hi]  # noqa: N806
            V_blk = V[k_lo:k_hi]  # noqa: N806

            # Scores for just this tile (never the full N×N matrix).
            S = (Q_blk @ K_blk.T) * scale  # noqa: N806
            block_max = np.max(S, axis=-1)  # (rows,)

            m_old = m_i.copy()
            m_new = np.maximum(m_old, block_max)
            # Correction factor rescales the OLD accumulators onto the new max.
            alpha = np.exp(m_old - m_new)  # (rows,)
            # Exponentials of this block relative to the new running max.
            p = np.exp(S - m_new[:, None])  # (rows, tile)

            l_i = alpha * l_i + np.sum(p, axis=-1)
            O_i = alpha[:, None] * O_i + p @ V_blk  # noqa: N806
            m_i = m_new

            # Log a couple of the running (m, l) updates so the recurrence is visible.
            if qi == 0 and kj < 2:
                t.add(
                    f"Q-block {qi}, K/V-block {kj}: online update",
                    f"block scores S={arr(S)}; rowmax={arr(block_max)}\n"
                    f"m: {arr(m_old)} → {arr(m_new)}   "
                    f"alpha = e^(m_old−m_new) = {arr(alpha)}\n"
                    f"running l = {arr(l_i)}",
                    detail=(
                        "alpha rescales the old sum l and output O onto the new max "
                        "before this block is added — this keeps the streaming softmax "
                        "exact without ever storing the full score matrix."
                    ),
                )

        # Normalize once, at the very end, by the running denominator.
        O_i = O_i / l_i[:, None]  # noqa: N806
        output[q_lo:q_hi] = O_i

    # ---- Correctness: prove flash == standard attention ---------------------
    reference = _standard_attention(Q, K, V, scale)
    max_abs_error = float(np.max(np.abs(output - reference)))
    t.meta["max_abs_error"] = max_abs_error

    t.add(
        "Final output O = (running O) / (running l)",
        f"divide each accumulator by its softmax denominator\n{arr(output)}",
        output,
        detail="Each row is a convex combination of value vectors, identical to full softmax.",
    )
    t.add(
        "Correctness check vs standard attention",
        f"max |O_flash − O_standard| = {num(max_abs_error)}  (~0 ⇒ EXACT, not approximate)",
        max_abs_error,
        detail=(
            "softmax(Q·Kᵀ·scale)·V computed the naive way agrees to floating-point "
            "precision. FlashAttention trades memory-access patterns for speed while "
            "computing the very same function."
        ),
    )
    assert max_abs_error < 1e-9, (
        f"flash attention should be exact but max_abs_error={max_abs_error}"
    )

    t.result = output
    return t


def flash_attention(
    Q,  # noqa: N803
    K,  # noqa: N803
    V,  # noqa: N803
    block_size: int = 2,
    explain: bool = False,
    level: str | ExplainLevel = ExplainLevel.ENGINEER,
) -> np.ndarray:
    """Return flash attention output. Set ``explain=True`` to print the trace."""
    t = flash_attention_trace(Q, K, V, block_size=block_size)
    return t.render(level) if explain else t.result


def demo(seed: int = 0) -> Trace:
    """A tiny, reproducible 4×4 example with ``block_size=2`` for docs and the CLI."""
    rng = np.random.default_rng(seed)
    Q = rng.normal(size=(4, 4)).round(2)  # noqa: N806
    K = rng.normal(size=(4, 4)).round(2)  # noqa: N806
    V = rng.normal(size=(4, 4)).round(2)  # noqa: N806
    return flash_attention_trace(Q, K, V, block_size=2)
