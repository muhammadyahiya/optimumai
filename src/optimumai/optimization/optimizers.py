"""Optimizers — how the gradient actually moves the weights.

Gradients say *which way* is downhill; an optimizer decides *how far* to step.
``SGD`` takes the raw step; ``Adam`` adapts the step per-parameter using running
estimates of the gradient's mean (momentum) and variance. Both operate on
:class:`~optimumai.autograd.value.Value` parameters, and :func:`minimize` records
the loss trajectory so you can watch convergence.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

from optimumai.autograd.value import Value
from optimumai.core._fmt import num
from optimumai.core.explain import ExplainLevel
from optimumai.core.trace import Trace

_SPARK = "▁▂▃▄▅▆▇█"


def _sparkline(values: Sequence[float]) -> str:
    """A one-line unicode chart of a sequence (great for a loss curve)."""
    lo, hi = min(values), max(values)
    span = hi - lo or 1.0
    n = len(_SPARK) - 1
    return "".join(_SPARK[min(n, int((v - lo) / span * n))] for v in values)


class SGD:
    """Vanilla stochastic gradient descent: ``w ← w − lr · ∂L/∂w``."""

    def __init__(self, params: Sequence[Value], lr: float = 0.1):
        self.params = list(params)
        self.lr = lr

    @property
    def name(self) -> str:
        return f"SGD(lr={num(self.lr)})"

    def step(self) -> None:
        for p in self.params:
            p.data -= self.lr * p.grad


class Adam:
    """Adam: per-parameter adaptive steps from 1st/2nd gradient moments.

        m ← β₁m + (1−β₁)g,   v ← β₂v + (1−β₂)g²
        m̂ = m/(1−β₁ᵗ),      v̂ = v/(1−β₂ᵗ)
        w ← w − lr · m̂ / (√v̂ + ε)
    """

    def __init__(
        self,
        params: Sequence[Value],
        lr: float = 0.1,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ):
        self.params = list(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = [0.0] * len(self.params)
        self.v = [0.0] * len(self.params)
        self.t = 0

    @property
    def name(self) -> str:
        return f"Adam(lr={num(self.lr)})"

    def step(self) -> None:
        self.t += 1
        for i, p in enumerate(self.params):
            g = p.grad
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * g * g
            m_hat = self.m[i] / (1 - self.beta1**self.t)
            v_hat = self.v[i] / (1 - self.beta2**self.t)
            p.data -= self.lr * m_hat / (math.sqrt(v_hat) + self.eps)


def minimize_trace(
    loss_fn: Callable[[], Value],
    params: Sequence[Value],
    optimizer: SGD | Adam,
    steps: int = 50,
) -> Trace:
    """Run ``steps`` optimization iterations, recording the loss each time.

    ``loss_fn`` must rebuild a scalar-:class:`Value` loss from the (persistent)
    ``params`` on every call. Each iteration recomputes the graph, backprops, and
    lets the optimizer nudge the parameters.
    """
    params = list(params)
    t = Trace(
        op="optimize",
        formula="repeat:  L = loss(θ);  θ.grad = ∂L/∂θ (backprop);  θ ← optimizer.step(θ)",
        complexity=f"{steps} steps × one backward pass",
        why_ai=[
            "This loop *is* training — every model you've used ran some version of it",
            "SGD follows the raw gradient; Adam adapts each step to the gradient's history",
            "Watching the loss fall is how you tell learning is actually happening",
        ],
        meta={"optimizer": optimizer.name, "steps": steps},
    )

    losses: list[float] = []
    log_every = max(1, steps // 10)
    for i in range(steps):
        loss = loss_fn()
        loss.backward()
        losses.append(loss.data)
        if i % log_every == 0 or i == steps - 1:
            theta = (
                f"   θ = [{', '.join(num(p.data) for p in params)}]"
                if len(params) <= 6
                else f"   ({len(params)} params)"
            )
            t.add(f"step {i:>3}", f"loss = {num(loss.data)}{theta}", loss.data)
        optimizer.step()

    final_loss = loss_fn().data
    t.add(
        "converged",
        f"loss {num(losses[0])} → {num(final_loss)}   {_sparkline(losses)}",
        final_loss,
        detail=f"{optimizer.name}: {num(losses[0])} → {num(final_loss)} "
        f"({(1 - final_loss / losses[0]) * 100:.1f}% reduction) over {steps} steps.",
    )
    t.result = [p.data for p in params]
    return t


def minimize(
    loss_fn: Callable[[], Value],
    params: Sequence[Value],
    optimizer: SGD | Adam,
    steps: int = 50,
    explain: bool = False,
    level: str | ExplainLevel = ExplainLevel.INTERMEDIATE,
) -> list[float]:
    """Minimize ``loss_fn`` over ``params`` and return the final parameter values."""
    t = minimize_trace(loss_fn, params, optimizer, steps=steps)
    return t.render(level) if explain else t.result


def descent_demo(optimizer: str = "adam", steps: int = 50) -> Trace:
    """Minimize the bowl ``f(x, y) = (x−3)² + (y+1)²`` from the origin.

    The global minimum is at ``(3, −1)`` with loss 0 — an easy target to watch
    the optimizer walk toward.
    """
    x = Value(0.0, label="x")
    y = Value(0.0, label="y")
    params = [x, y]
    opt: SGD | Adam = Adam(params, lr=0.3) if optimizer.lower() == "adam" else SGD(params, lr=0.1)

    def loss_fn() -> Value:
        return (x - 3.0) ** 2 + (y + 1.0) ** 2

    return minimize_trace(loss_fn, params, opt, steps=steps)
