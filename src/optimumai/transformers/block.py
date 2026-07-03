"""A single transformer block — the unit that stacks into GPT.

This is the pre-norm decoder block used by nanoGPT: normalise, attend, add;
normalise, feed-forward, add.

    x = x + MHA(LayerNorm(x), causal=True)
    x = x + FFN(LayerNorm(x))

Two ideas do the heavy lifting. **Residual connections** (`x + sublayer(x)`) give
gradients a highway straight back to the input, so very deep stacks stay
trainable. **Pre-norm** (LayerNorm *before* each sublayer rather than after)
keeps activations well-scaled and is what makes 100-layer models converge. The
feed-forward network (Linear → GELU → Linear, widening to ``d_ff = 4·d_model``)
is where most of the parameters — and most of the model's stored "knowledge" —
live.
"""

from __future__ import annotations

import numpy as np

from optimumai.core._fmt import arr
from optimumai.core.base_op import BaseOp
from optimumai.core.trace import Trace
from optimumai.transformers.multihead import MultiHeadAttention


def _layer_norm(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """LayerNorm over the feature axis with gamma=1, beta=0 for simplicity."""
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)


def _gelu(x: np.ndarray) -> np.ndarray:
    """The tanh approximation of GELU used throughout GPT-family models."""
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))


class TransformerBlock(BaseOp):
    """One pre-norm transformer decoder block (attention + feed-forward).

    Args:
        d_model: Model / residual-stream dimension.
        n_heads: Number of attention heads (must divide ``d_model``).
        d_ff: Hidden width of the feed-forward network (default ``4·d_model``).
    """

    name = "transformer_block"

    def __init__(self, d_model: int, n_heads: int, d_ff: int | None = None):
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
            )
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff if d_ff is not None else 4 * d_model
        self.mha = MultiHeadAttention(n_heads=n_heads, d_model=d_model)

        # Fixed, seeded FFN weights so a block is deterministic without training.
        rng = np.random.default_rng(0)
        self.W1 = rng.normal(size=(d_model, self.d_ff)) * (1.0 / np.sqrt(d_model))
        self.b1 = np.zeros(self.d_ff)
        self.W2 = rng.normal(size=(self.d_ff, d_model)) * (1.0 / np.sqrt(self.d_ff))
        self.b2 = np.zeros(d_model)

    def trace(self, X) -> Trace:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D (tokens × d_model), got shape {X.shape}")
        if X.shape[1] != self.d_model:
            raise ValueError(
                f"X feature dim {X.shape[1]} does not match d_model {self.d_model}"
            )

        t = Trace(
            op="transformer_block",
            formula="x = x + MHA(LN(x), causal); x = x + FFN(LN(x))",
            complexity="O(n²·d + n·d·d_ff) per block",
            why_ai=[
                "Residual connections give gradients a highway back through deep stacks",
                "Pre-norm (LayerNorm before each sublayer) stabilises very deep models",
                "The feed-forward network is where most parameters and 'knowledge' live",
            ],
            meta={
                "d_model": self.d_model,
                "n_heads": self.n_heads,
                "d_ff": self.d_ff,
                "seq_len": X.shape[0],
            },
        )

        # --- Attention sublayer ---
        normed1 = _layer_norm(X)
        t.add(
            "LayerNorm (pre-attention)",
            f"normalise each token over its {self.d_model} features\n{arr(normed1)}",
            normed1,
            detail="(x − mean) / √(var + eps); each row now has ~0 mean and unit variance.",
        )

        attn = self.mha.run(normed1, causal=True)
        t.add(
            "Multi-head self-attention (causal)",
            f"{self.n_heads} heads mix past tokens only\n{arr(attn)}",
            attn,
            detail="Causal mask keeps position i from attending to any j > i.",
        )

        x = X + attn
        t.add(
            "Residual add",
            f"x + attention\n{arr(x)}",
            x,
            detail="The attention output is added back to the untouched input stream.",
        )

        # --- Feed-forward sublayer ---
        normed2 = _layer_norm(x)
        t.add(
            "LayerNorm (pre-FFN)",
            f"normalise the updated stream again\n{arr(normed2)}",
            normed2,
        )

        ffn = _gelu(normed2 @ self.W1 + self.b1) @ self.W2 + self.b2
        t.add(
            "Feed-forward (GELU)",
            f"Linear → GELU → Linear, widen to d_ff = {self.d_ff}\n{arr(ffn)}",
            ffn,
            detail="Applied independently per token; GELU is the smooth activation used by GPT.",
        )

        out = x + ffn
        t.add(
            "Residual add",
            f"x + feed-forward\n{arr(out)}",
            out,
            detail="Second residual completes the block; output shape matches the input.",
        )
        t.result = out
        return t

    def forward(self, X, explain: bool = False, level="engineer"):
        """Run one transformer block. Set ``explain=True`` to print the trace."""
        t = self.trace(X)
        return t.render(level) if explain else t.result

    @classmethod
    def demo(cls, seed: int = 0) -> Trace:
        """A tiny, reproducible 4-token / d_model=8 / 2-head example."""
        rng = np.random.default_rng(seed)
        X = rng.normal(size=(4, 8)).round(2)
        return cls(d_model=8, n_heads=2).trace(X)
