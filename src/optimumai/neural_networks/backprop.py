"""Forward pass, loss, and full backprop through an MLP — end to end.

This is the payoff of the autograd engine: a real (if tiny) network that learns
a nonlinear decision boundary purely by ``loss.backward()`` and gradient steps.
"""

from __future__ import annotations

from collections.abc import Sequence

from optimumai.autograd.value import Value
from optimumai.core._fmt import num
from optimumai.core.explain import ExplainLevel
from optimumai.core.trace import Trace
from optimumai.neural_networks.mlp import MLP
from optimumai.optimization.optimizers import SGD, minimize_trace

# A classic tiny binary-classification set (targets in {-1, +1}).
_XS: list[list[float]] = [[2.0, 3.0, -1.0], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, -1.0]]
_YS: list[float] = [1.0, -1.0, -1.0, 1.0]


def forward_backward_trace(mlp: MLP, x: Sequence[float], target: float) -> Trace:
    """One example: forward to a prediction, squared-error loss, then backprop."""
    pred = mlp(x)
    if not isinstance(pred, Value):
        raise ValueError("this trace expects a single-output MLP")
    loss = (pred - target) ** 2
    loss.backward()

    params = mlp.parameters()
    grad_norm = sum(p.grad**2 for p in params) ** 0.5

    t = Trace(
        op="forward_backward",
        formula="ŷ = MLP(x);   L = (ŷ − y)²;   ∂L/∂θ via backprop",
        complexity=f"{len(params)} parameters, one forward + one backward pass",
        why_ai=[
            "Forward pass = predict; backward pass = assign blame to every weight",
            "One backward call fills in ∂L/∂w for all parameters at once",
            "The gradient's size tells you how strongly each weight wants to change",
        ],
        meta={"params": len(params)},
    )
    t.add("Forward pass", f"ŷ = MLP({list(x)}) = {num(pred.data)}", pred.data)
    t.add("Target", f"y = {num(target)}", target)
    t.add("Loss", f"(ŷ − y)² = ({num(pred.data)} − {num(target)})² = {num(loss.data)}", loss.data)
    t.add(
        "Backward pass",
        f"‖∂L/∂θ‖ = {num(grad_norm)} across {len(params)} parameters",
        grad_norm,
        detail="Each parameter now holds its own gradient, ready for an optimizer step.",
    )
    sample = params[:3]
    t.add(
        "Sample gradients (layer 0)",
        ", ".join(f"{p.label}.grad={num(p.grad)}" for p in sample),
        None,
    )
    t.result = loss.data
    return t


def train_demo(steps: int = 100, lr: float = 0.05, seed: int = 0) -> Trace:
    """Train a 3→4→4→1 MLP on the toy set and watch the loss fall."""
    mlp = MLP(3, [4, 4, 1], activation="tanh", seed=seed)
    params = mlp.parameters()

    def loss_fn() -> Value:
        total = Value(0.0)
        for x, y in zip(_XS, _YS, strict=True):
            pred = mlp(x)
            total = total + (pred - y) ** 2
        return total

    trace = minimize_trace(loss_fn, params, SGD(params, lr=lr), steps=steps)
    trace.op = "train_mlp"
    trace.meta["network"] = repr(mlp)
    trace.why_ai = [
        "A full training loop: predict → measure error → backprop → step, repeated",
        "The same loop scales from this 41-parameter net to trillion-parameter LLMs",
        "Karpathy's micrograd shows the whole thing fits in a few hundred lines",
    ]
    return trace


def train(
    steps: int = 100,
    lr: float = 0.05,
    seed: int = 0,
    explain: bool = False,
    level: str | ExplainLevel = ExplainLevel.INTERMEDIATE,
) -> list[float]:
    """Run the training demo; returns the final parameter vector."""
    t = train_demo(steps=steps, lr=lr, seed=seed)
    return t.render(level) if explain else t.result
