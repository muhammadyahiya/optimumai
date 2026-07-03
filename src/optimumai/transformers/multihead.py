"""Multi-head self-attention — running attention in parallel subspaces.

One attention head can only express a single "mixing pattern" per token. Multi-
head attention splits the model dimension into ``n_heads`` slices and runs a
separate scaled dot-product self-attention in each, so different heads can
specialise on different relations (syntax, coreference, position, ...) at once:

    head_h = softmax( Xₕ·Xₕᵀ / √d_head ) · Xₕ
    MHA(X) = concat(head_1, ..., head_h)

With a *causal* mask the strictly-upper-triangular scores (future positions
``j > i``) are set to ``-∞`` before softmax, so a token can never attend to
tokens that come after it. That masked, autoregressive form is exactly the GPT
decoder.
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


class MultiHeadAttention(BaseOp):
    """Multi-head scaled dot-product *self*-attention.

    Args:
        n_heads: Number of parallel attention heads.
        d_model: Model dimension; must be divisible by ``n_heads``.
    """

    name = "multihead_attention"

    def __init__(self, n_heads: int, d_model: int):
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
            )
        self.n_heads = n_heads
        self.d_model = d_model
        self.head_dim = d_model // n_heads

    def trace(self, X, causal: bool = False) -> Trace:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D (tokens × d_model), got shape {X.shape}")
        if X.shape[1] != self.d_model:
            raise ValueError(
                f"X feature dim {X.shape[1]} does not match d_model {self.d_model}"
            )

        seq_len = X.shape[0]
        scale = np.sqrt(self.head_dim)
        t = Trace(
            op="multihead_attention",
            formula="MHA(X) = concat_h softmax(Xₕ·Xₕᵀ / √d_head) · Xₕ",
            complexity="O(h · n² · d_head)",
            why_ai=[
                "Each head attends to a different subspace/relation in parallel",
                "Splitting d_model keeps total cost the same as one full-width head",
                "Causal masking enforces autoregression (no peeking ahead) — the GPT decoder",
            ],
            meta={
                "n_heads": self.n_heads,
                "d_model": self.d_model,
                "head_dim": self.head_dim,
                "seq_len": seq_len,
                "causal": causal,
            },
        )

        if causal:
            # Strictly-upper-triangular entries (future positions j > i) → -inf.
            mask = np.triu(np.full((seq_len, seq_len), -np.inf), k=1)
            marks = np.triu(np.ones((seq_len, seq_len)), k=1)
            t.add(
                "Causal mask",
                f"add -∞ to future positions (j > i) before softmax\n{arr(marks)}",
                mask,
                detail="1s mark masked cells; softmax then zeroes them (no peeking ahead).",
            )
        else:
            mask = np.zeros((seq_len, seq_len))

        heads = np.split(X, self.n_heads, axis=1)
        outputs = []
        for h, head in enumerate(heads):
            scores = head @ head.T / scale + mask
            weights = _softmax_rows(scores)
            out_head = weights @ head
            outputs.append(out_head)
            lo, hi = h * self.head_dim, (h + 1) * self.head_dim
            t.add(
                f"Head {h}: softmax(XₕXₕᵀ/√{self.head_dim})",
                f"attention weights for columns [{lo}:{hi}]\n{arr(weights)}",
                weights,
                detail=f"Row sums = {arr(np.sum(weights, axis=-1))}; scale = 1/{num(scale)}.",
            )

        output = np.concatenate(outputs, axis=1)
        t.add(
            "Concatenate heads",
            f"stack {self.n_heads} head outputs back to d_model = {self.d_model}\n{arr(output)}",
            output,
            detail="Each head contributed head_dim columns; concatenation restores the full width.",
        )
        t.result = output
        return t

    def forward(self, X, causal: bool = False, explain: bool = False, level="engineer"):
        """Compute self-attention. Set ``explain=True`` to print the trace."""
        t = self.trace(X, causal=causal)
        return t.render(level) if explain else t.result

    @classmethod
    def demo(cls, seed: int = 0) -> Trace:
        """A tiny, reproducible 4-token / d_model=8 / 2-head causal example."""
        rng = np.random.default_rng(seed)
        X = rng.normal(size=(4, 8)).round(2)
        return cls(n_heads=2, d_model=8).trace(X, causal=True)
