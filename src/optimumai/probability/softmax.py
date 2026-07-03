"""Softmax — the function that turns raw scores into a distribution.

It is the last layer of a classifier, the thing that makes attention weights sum
to one, and the knob (via temperature) behind "creative" vs "focused" sampling.
This implementation uses the standard max-subtraction trick for numerical
stability and shows *why* that trick is safe.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from optimumai.core._fmt import arr, num
from optimumai.core.explain import ExplainLevel
from optimumai.core.trace import Trace


def softmax_trace(x: Iterable[float], temperature: float = 1.0) -> Trace:
    """Build the full trace of ``softmax(x)`` with an optional temperature."""
    vec = np.asarray(list(x), dtype=float)
    if vec.ndim != 1:
        raise ValueError(f"softmax_trace expects a 1-D input, got shape {vec.shape}")
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")

    t = Trace(
        op="softmax",
        formula="softmax(xᵢ) = e^(xᵢ) / Σⱼ e^(xⱼ)",
        complexity="O(n)",
        why_ai=[
            "Converts logits into a probability distribution that sums to 1",
            "Produces attention weights from scaled QKᵀ scores",
            "The final layer for classification and next-token prediction",
        ],
        meta={"temperature": temperature},
    )

    if temperature != 1.0:
        vec = vec / temperature
        t.add(
            "Apply temperature",
            f"xᵢ / T  with T = {num(temperature)}  →  {arr(vec)}",
            vec.copy(),
            detail="T < 1 sharpens the distribution; T > 1 flattens it.",
        )

    x_max = float(np.max(vec))
    shifted = vec - x_max
    t.add(
        "Subtract the max (stability)",
        f"xᵢ − {num(x_max)}  →  {arr(shifted)}",
        shifted,
        detail="Shifting by a constant leaves softmax unchanged but avoids overflow in e^x.",
    )

    exps = np.exp(shifted)
    t.add("Exponentiate", f"e^(xᵢ − max)  →  {arr(exps)}", exps)

    denom = float(np.sum(exps))
    t.add("Sum the exponentials", f"Σ = {num(denom)}", denom)

    probs = exps / denom
    t.add(
        "Normalize",
        f"e^(xᵢ) / Σ  →  {arr(probs)}",
        probs,
        detail=f"Check: the outputs sum to {num(float(np.sum(probs)))}.",
    )
    t.result = probs
    return t


def softmax(
    x: Iterable[float],
    temperature: float = 1.0,
    explain: bool = False,
    level: str | ExplainLevel = ExplainLevel.INTERMEDIATE,
) -> np.ndarray:
    """Return the softmax of ``x``. Set ``explain=True`` to print the trace."""
    t = softmax_trace(x, temperature=temperature)
    return t.render(level) if explain else t.result
