"""Scaled dot-product attention — the operation that defines transformers.

    Attention(Q, K, V) = softmax( Q·Kᵀ / √dₖ ) · V

This is where every earlier idea converges: the dot products become the score
matrix (algebra), ``/√dₖ`` keeps gradients healthy (calculus), softmax turns
scores into weights (probability), and the final matmul mixes the values. Run it
with ``explain=True`` to watch all four stages.
"""

from __future__ import annotations

import numpy as np

from optimumai.core._fmt import arr, num
from optimumai.core.base_op import BaseOp
from optimumai.core.trace import Trace


def _softmax_rows(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over the last axis (per query row)."""
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exps = np.exp(shifted)
    return exps / np.sum(exps, axis=-1, keepdims=True)


class Attention(BaseOp):
    """Single-head scaled dot-product attention.

    Args:
        d_k: Key/query dimension used for the ``1/√dₖ`` scaling. If omitted it is
            inferred from ``Q`` at call time.
    """

    name = "attention"

    def __init__(self, d_k: int | None = None):
        self.d_k = d_k

    def trace(self, Q, K, V) -> Trace:  # noqa: N803 - Q/K/V are the standard names
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

        d_k = self.d_k or Q.shape[1]
        t = Trace(
            op="attention",
            formula="Attention(Q,K,V) = softmax(Q·Kᵀ / √dₖ) · V",
            complexity="O(n²·d) for n tokens of dimension d",
            why_ai=[
                "The core operation of every transformer (GPT, BERT, ViT, ...)",
                "Lets each token gather information from the tokens it finds relevant",
                "Q = what I'm looking for, K = what I offer, V = what I pass on",
            ],
            meta={"d_k": d_k, "q_shape": Q.shape, "k_shape": K.shape, "v_shape": V.shape},
        )

        scores = Q @ K.T
        t.add(
            "Score: Q · Kᵀ",
            f"raw relevance of every query to every key\n{arr(scores)}",
            scores,
            detail="Entry [i,j] is the dot product of query i with key j.",
        )

        scale = np.sqrt(d_k)
        scaled = scores / scale
        t.add(
            f"Scale by 1/√dₖ = 1/√{d_k}",
            f"scores / {num(scale)}\n{arr(scaled)}",
            scaled,
            detail="Scaling keeps the softmax out of its saturated, low-gradient region.",
        )

        weights = _softmax_rows(scaled)
        row_sums = np.sum(weights, axis=-1)
        t.add(
            "Softmax over each row",
            f"attention weights (each row sums to 1)\n{arr(weights)}",
            weights,
            detail=f"Row sums = {arr(row_sums)} — every query distributes 100% of its attention.",
        )

        output = weights @ V
        t.add(
            "Weighted sum: weights · V",
            f"blend value vectors by attention weight\n{arr(output)}",
            output,
            detail="Each output row is a convex combination of the value vectors.",
        )
        t.result = output
        return t

    def forward(self, Q, K, V, explain: bool = False, level="intermediate"):  # noqa: N803
        """Compute attention. Set ``explain=True`` to print the four-stage trace."""
        t = self.trace(Q, K, V)
        return t.render(level) if explain else t.result

    @classmethod
    def demo(cls, seed: int = 0) -> Trace:
        """A tiny, reproducible 3-token / dₖ=4 example for docs and the CLI."""
        rng = np.random.default_rng(seed)
        Q = rng.normal(size=(3, 4)).round(2)
        K = rng.normal(size=(3, 4)).round(2)
        V = rng.normal(size=(3, 4)).round(2)
        return cls(d_k=4).trace(Q, K, V)
