"""The math under AI, made concrete: tensors and numerical integration.

Two ideas that quietly power every model. **Tensors** are the data structure —
scalars, vectors, matrices and their higher-rank cousins — and *shape*, *rank*
and *broadcasting* are what almost every deep-learning bug is really about.
**Integration** is how continuous math sneaks in: an expectation ``E[f] = ∫ f·p``
is an integral, so likelihoods, the ELBO, and losses averaged over a distribution
all reduce to computing (or estimating) one. Run either trace with
``explain=True`` to watch the steps.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from optimumai.core._fmt import arr, num, shape_of
from optimumai.core.explain import ExplainLevel
from optimumai.core.trace import Trace


def tensor_intro_trace() -> Trace:
    """Build a trace introducing tensors as n-dimensional arrays."""
    t = Trace(
        op="tensor_intro",
        formula="tensor = n-dimensional array; rank = number of axes; shape = size per axis",
        complexity="storage O(∏ shape) — product of all axis sizes",
        why_ai=[
            "Tensors are THE data structure of deep learning — every input, "
            "weight, and activation is one",
            "Shape, rank, and broadcasting are what almost every model bug is about "
            "(a mismatched axis, a silent broadcast)",
            "A batch of token embeddings is a rank-3 tensor: (batch, seq, dim)",
        ],
        meta={},
    )

    scalar = np.asarray(3.0)
    t.add(
        "Rank 0 — scalar",
        f"value = {num(scalar)}, shape = {shape_of(scalar)}, rank = {scalar.ndim}",
        scalar,
        detail="A single number: a learning rate, a loss value, one logit.",
    )

    vector = np.array([1.0, 2.0, 3.0])
    t.add(
        "Rank 1 — vector",
        f"{arr(vector)}, shape = {shape_of(vector)}, rank = {vector.ndim}",
        vector,
        detail="A 1-D array: one token embedding, or one row of weights.",
    )

    matrix = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    t.add(
        "Rank 2 — matrix",
        f"shape = {shape_of(matrix)}, rank = {matrix.ndim}\n{arr(matrix)}",
        matrix,
        detail="A 2-D array: a weight matrix, or a (seq, dim) sequence of embeddings.",
    )

    tensor3 = np.arange(2 * 2 * 3, dtype=float).reshape(2, 2, 3)
    t.add(
        "Rank 3 — 3-tensor",
        f"shape = {shape_of(tensor3)}, rank = {tensor3.ndim}\n{arr(tensor3)}",
        tensor3,
        detail="A 3-D array: a batch of token embeddings, (batch, seq, dim).",
    )

    col = np.array([[10.0], [20.0], [30.0]])  # (3, 1)
    row = np.array([[1.0, 2.0, 3.0, 4.0]])  # (1, 4)
    broadcast = col + row  # (3, 4)
    t.add(
        "Broadcasting: (3,1) + (1,4) → (3,4)",
        f"col {shape_of(col)} + row {shape_of(row)} → {shape_of(broadcast)}\n{arr(broadcast)}",
        broadcast,
        detail=(
            "The rule: align shapes from the right; an axis of size 1 is stretched "
            "to match the other. No data is copied — it is a view-level trick that "
            "lets a (batch, 1, dim) bias add to a (batch, seq, dim) activation."
        ),
    )

    t.result = broadcast
    return t


def integrate_trace(
    f: Callable[[np.ndarray], np.ndarray],
    a: float,
    b: float,
    method: str = "trapezoid",
    n: int = 100,
) -> Trace:
    """Estimate ``∫ f(x) dx`` from ``a`` to ``b`` and trace the steps.

    Args:
        f: A vectorized function accepting a NumPy array and returning one.
        a: Lower bound of integration.
        b: Upper bound of integration.
        method: ``"trapezoid"`` (grid of trapezoids) or ``"monte_carlo"``
            (average of ``f`` at seeded random points, times the width).
        n: Number of grid intervals (trapezoid) or samples (Monte Carlo).
    """
    if method not in {"trapezoid", "monte_carlo"}:
        raise ValueError(f"method must be 'trapezoid' or 'monte_carlo', got {method!r}")
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")

    t = Trace(
        op="integrate",
        formula="∫ₐᵇ f(x) dx ≈ Σ (trapezoids)  |  ≈ (b−a)·mean(f(X)), X ~ U[a,b]",
        complexity="O(n) — n grid points or n samples",
        why_ai=[
            "Expectations ARE integrals: E[f] = ∫ f(x)·p(x) dx — the workhorse of "
            "probabilistic ML",
            "Likelihoods, the ELBO, and any loss averaged over a distribution reduce "
            "to computing one of these integrals",
            "Trapezoid: deterministic and accurate on a grid, but the grid size "
            "explodes with dimension",
            "Monte Carlo: error shrinks as 1/√n regardless of dimension, so it beats "
            "grid methods in high dimensions (the curse of dimensionality)",
        ],
        meta={"method": method, "a": a, "b": b, "n": n},
    )

    if method == "trapezoid":
        xs = np.linspace(a, b, n + 1)
        ys = np.asarray(f(xs), dtype=float)
        h = (b - a) / n
        t.add(
            "Build the grid",
            f"{n + 1} points from {num(a)} to {num(b)}, step h = {num(h)}",
            xs,
            detail="Trapezoid rule approximates the area under each interval by a trapezoid.",
        )
        t.add(
            "Evaluate f at the grid points",
            f"f(x): first few = {arr(ys[: min(5, ys.size)])} ...",
            ys,
        )
        # Trapezoid weights: endpoints count once, interior points twice.
        estimate = float(h * (ys[0] / 2 + ys[-1] / 2 + np.sum(ys[1:-1])))
        t.add(
            "Sum the trapezoids",
            f"h · (f₀/2 + f_n/2 + Σ interior)  →  {num(estimate)}",
            estimate,
            detail="Each trapezoid area is h·(fᵢ + fᵢ₊₁)/2; summing telescopes to this.",
        )
    else:  # monte_carlo
        rng = np.random.default_rng(0)
        xs = rng.uniform(a, b, size=n)
        ys = np.asarray(f(xs), dtype=float)
        t.add(
            "Draw random samples",
            f"{n} points X ~ Uniform[{num(a)}, {num(b)}] (seeded rng)",
            xs,
            detail="Monte Carlo replaces the deterministic grid with random draws.",
        )
        mean_y = float(np.mean(ys))
        t.add(
            "Average f over the samples",
            f"mean(f(X)) = {num(mean_y)}",
            mean_y,
        )
        estimate = float((b - a) * mean_y)
        t.add(
            "Scale by the interval width",
            f"(b − a) · mean(f(X)) = {num(b - a)} · {num(mean_y)}  →  {num(estimate)}",
            estimate,
            detail="Estimate = width × average height; error shrinks like 1/√n.",
        )

    t.result = estimate
    return t


def integrate(
    f: Callable[[np.ndarray], np.ndarray],
    a: float,
    b: float,
    method: str = "trapezoid",
    n: int = 100,
    explain: bool = False,
    level: str | ExplainLevel = ExplainLevel.INTERMEDIATE,
) -> float:
    """Return ``∫ f`` from ``a`` to ``b``. Set ``explain=True`` to print the trace."""
    t = integrate_trace(f, a, b, method=method, n=n)
    return t.render(level) if explain else t.result


def integrate_demo() -> Trace:
    """Trace ``∫₀¹ x² dx`` (true value = 1/3) via the trapezoid rule."""
    return integrate_trace(lambda x: x**2, 0.0, 1.0)


def demo() -> Trace:
    """The headline demo for this module: the tensor introduction."""
    return tensor_intro_trace()
