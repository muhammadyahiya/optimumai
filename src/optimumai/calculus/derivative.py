"""Derivatives — the slope that tells a network which way to move.

Two complementary views live here:

* **Numeric** — the finite-difference definition, ``f'(x) ≈ (f(x+h) − f(x−h)) / 2h``.
  Slow and approximate, but it's the *definition* and a perfect gradient check.
* **Exact** — the same slope from the autograd engine (:class:`~optimumai.autograd.value.Value`),
  which is what real frameworks use. The chain-rule demo shows the two agree.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from optimumai.autograd.value import Value
from optimumai.core._fmt import num
from optimumai.core.explain import ExplainLevel
from optimumai.core.trace import Trace


def derivative_trace(
    f: Callable[[float], float], x: float, h: float = 1e-5, label: str = "f"
) -> Trace:
    """Central-difference derivative of a scalar function at ``x``."""
    fx_plus = f(x + h)
    fx_minus = f(x - h)
    slope = (fx_plus - fx_minus) / (2 * h)
    t = Trace(
        op="derivative",
        formula="f'(x) = limₕ→₀ (f(x+h) − f(x−h)) / 2h",
        complexity="O(1) per evaluation",
        why_ai=[
            "The slope is how sensitive the output is to a tiny nudge in the input",
            "Gradients (vectors of derivatives) tell every optimizer which way is downhill",
            "Finite differences are the classic sanity check for a hand-written backward pass",
        ],
        meta={"x": x, "h": h},
    )
    t.add(f"Evaluate {label}(x+h)", f"{label}({num(x)}+{num(h)}) = {num(fx_plus)}", fx_plus)
    t.add(f"Evaluate {label}(x−h)", f"{label}({num(x)}−{num(h)}) = {num(fx_minus)}", fx_minus)
    t.add(
        "Central difference",
        f"({num(fx_plus)} − {num(fx_minus)}) / (2·{num(h)}) = {num(slope)}",
        slope,
        detail="Central differences cancel the first-order error, so they beat a one-sided slope.",
    )
    t.result = slope
    return t


def derivative(
    f: Callable[[float], float],
    x: float,
    h: float = 1e-5,
    explain: bool = False,
    level: str | ExplainLevel = ExplainLevel.INTERMEDIATE,
) -> float:
    """Numeric derivative of ``f`` at ``x`` (central difference)."""
    t = derivative_trace(f, x, h=h)
    return t.render(level) if explain else t.result


def gradient_trace(
    f: Callable[[Sequence[float]], float], point: Sequence[float], h: float = 1e-5
) -> Trace:
    """Numeric gradient (vector of partial derivatives) of ``f`` at ``point``."""
    point = list(point)
    t = Trace(
        op="gradient",
        formula="∇f = [∂f/∂x₀, ∂f/∂x₁, …]",
        complexity="O(n) function evaluations",
        why_ai=[
            "The gradient points in the direction of steepest ascent; we step the other way",
            "Every weight in a network gets one partial derivative — that's the gradient",
            "Backprop computes this whole vector in a single reverse pass",
        ],
        meta={"point": point},
    )
    grad = []
    for i in range(len(point)):
        bumped_up = list(point)
        bumped_down = list(point)
        bumped_up[i] += h
        bumped_down[i] -= h
        partial = (f(bumped_up) - f(bumped_down)) / (2 * h)
        grad.append(partial)
        t.add(f"∂f/∂x{i}", f"nudge x{i} by ±{num(h)} → {num(partial)}", partial)
    t.result = grad
    return t


def gradient(
    f: Callable[[Sequence[float]], float],
    point: Sequence[float],
    h: float = 1e-5,
    explain: bool = False,
    level: str | ExplainLevel = ExplainLevel.INTERMEDIATE,
) -> list[float]:
    """Numeric gradient of ``f`` at ``point``."""
    t = gradient_trace(f, point, h=h)
    return t.render(level) if explain else t.result


def chain_rule_trace(x: float = 1.5) -> Trace:
    """Demonstrate the chain rule on ``f(x) = tanh(x²)`` — exact vs numeric.

    dy/dx = tanh'(x²) · d(x²)/dx = (1 − tanh²(x²)) · 2x.
    """
    xv = Value(x, label="x")
    g = xv * xv          # inner: x²
    g.label = "g=x²"
    y = g.tanh()         # outer: tanh(g)
    y.label = "y"
    y.backward()

    numeric = derivative(lambda v: __import__("math").tanh(v * v), x)

    t = Trace(
        op="chain_rule",
        formula="dy/dx = (dy/dg)·(dg/dx)   for   y = tanh(g), g = x²",
        complexity="O(depth) — one local derivative per link in the chain",
        why_ai=[
            "The chain rule is the whole of backpropagation — nothing more",
            "Each layer contributes one local derivative; they multiply along the path",
            "Autograd just automates this bookkeeping over millions of links",
        ],
        meta={"x": x},
    )
    t.add("Inner function", f"g = x² = {num(g.data)}", g.data)
    t.add("Outer function", f"y = tanh(g) = {num(y.data)}", y.data)
    t.add("Local derivative dg/dx", f"d(x²)/dx = 2x = {num(2 * x)}", 2 * x)
    t.add(
        "Local derivative dy/dg",
        f"d(tanh)/dg = 1 − tanh²(g) = {num(1 - y.data**2)}",
        1 - y.data**2,
    )
    t.add(
        "Multiply along the chain (autograd)",
        f"dy/dx = {num(1 - y.data**2)} · {num(2 * x)} = {num(xv.grad)}",
        xv.grad,
        detail="This is exactly what x.grad holds after y.backward().",
    )
    t.add(
        "Cross-check (numeric)",
        f"finite-difference dy/dx ≈ {num(numeric)}",
        numeric,
        detail=f"Matches the exact autograd value {num(xv.grad)} to ~5 digits.",
    )
    t.result = xv.grad
    return t
