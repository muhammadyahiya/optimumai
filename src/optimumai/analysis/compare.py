"""Side-by-side comparison of operations on *your* input.

Pick two activations (``relu`` vs ``gelu``) and watch them diverge element by
element; or sweep a knob (softmax ``temperature``) and see the distribution
sharpen or flatten. The point is intuition: the shape of an activation is the
shape of its gradient, and gradient shape is what makes a network trainable.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from optimumai.core._fmt import arr, num
from optimumai.core.explain import ExplainLevel
from optimumai.core.trace import Trace

_SQRT_2_OVER_PI = np.sqrt(2.0 / np.pi)


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def _relu_deriv(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(float)


def _gelu(x: np.ndarray) -> np.ndarray:
    """GELU with the tanh approximation used by GPT-family models."""
    inner = _SQRT_2_OVER_PI * (x + 0.044715 * x**3)
    return 0.5 * x * (1.0 + np.tanh(inner))


def _gelu_deriv(x: np.ndarray) -> np.ndarray:
    inner = _SQRT_2_OVER_PI * (x + 0.044715 * x**3)
    tanh_inner = np.tanh(inner)
    d_inner = _SQRT_2_OVER_PI * (1.0 + 3 * 0.044715 * x**2)
    return 0.5 * (1.0 + tanh_inner) + 0.5 * x * (1.0 - tanh_inner**2) * d_inner


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _sigmoid_deriv(x: np.ndarray) -> np.ndarray:
    s = _sigmoid(x)
    return s * (1.0 - s)


def _tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


def _tanh_deriv(x: np.ndarray) -> np.ndarray:
    return 1.0 - np.tanh(x) ** 2


def _softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x)
    exps = np.exp(shifted)
    return exps / np.sum(exps)


_OPS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "relu": _relu,
    "gelu": _gelu,
    "sigmoid": _sigmoid,
    "tanh": _tanh,
    "softmax": _softmax,
}

_DERIVS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "relu": _relu_deriv,
    "gelu": _gelu_deriv,
    "sigmoid": _sigmoid_deriv,
    "tanh": _tanh_deriv,
}

_SATURATION_NOTE: dict[str, str] = {
    "relu": "ReLU is flat for x < 0: its derivative is exactly 0 there (dead neurons).",
    "gelu": "GELU is smooth everywhere; its gradient never fully vanishes, only tapers.",
    "sigmoid": "Sigmoid saturates for |x| ≫ 0, where its derivative → 0 (vanishing gradients).",
    "tanh": "Tanh saturates at ±1 for large |x|, where its derivative → 0.",
    "softmax": "Softmax gradients shrink as one component dominates (near one-hot).",
}


def _validate(name: str) -> Callable[[np.ndarray], np.ndarray]:
    if name not in _OPS:
        valid = ", ".join(sorted(_OPS))
        raise ValueError(f"unknown op {name!r}. Choose one of: {valid}.")
    return _OPS[name]


def compare_trace(op_a: str, op_b: str, inputs: list[float]) -> Trace:
    """Apply two ops to the same input vector and line them up step by step."""
    fn_a = _validate(op_a)
    fn_b = _validate(op_b)
    vec = np.asarray(inputs, dtype=float)
    if vec.ndim != 1:
        raise ValueError(f"compare_trace expects a 1-D input, got shape {vec.shape}")

    out_a = fn_a(vec)
    out_b = fn_b(vec)
    diff = out_a - out_b

    t = Trace(
        op="compare",
        formula=f"{op_a}(x)  vs  {op_b}(x)",
        complexity="O(n) per op",
        why_ai=[
            "Activation choice shapes gradients: ReLU is sparse and non-saturating",
            "GELU is smooth, giving well-behaved gradients everywhere",
            "Sigmoid/tanh saturate and kill gradients — why modern nets use ReLU/GELU",
        ],
        meta={"op_a": op_a, "op_b": op_b, "inputs": list(inputs)},
    )

    t.add("Input vector", f"x = {arr(vec)}", vec.copy())
    t.add(f"Apply {op_a}", f"{op_a}(x) = {arr(out_a)}", out_a)
    t.add(f"Apply {op_b}", f"{op_b}(x) = {arr(out_b)}", out_b)
    t.add(
        "Elementwise difference",
        f"{op_a}(x) − {op_b}(x) = {arr(diff)}",
        diff,
        detail="Where the two curves agree the difference is ~0; gaps show their character.",
    )

    for name in (op_a, op_b):
        if name in _DERIVS:
            d = _DERIVS[name](vec)
            t.add(
                f"Gradient of {name}",
                f"{name}'(x) = {arr(d)}",
                d,
                detail=_SATURATION_NOTE.get(name, ""),
            )
        elif name in _SATURATION_NOTE:
            t.add(
                f"Note on {name}",
                _SATURATION_NOTE[name],
                detail="Saturation zones are where learning stalls.",
            )

    t.result = np.vstack([out_a, out_b])
    return t


def compare(
    op_a: str,
    op_b: str,
    input: list[float],
    explain: bool = False,
    level: str | ExplainLevel = ExplainLevel.ENGINEER,
) -> Any:
    """Compare ``op_a`` and ``op_b`` on ``input``. Set ``explain=True`` to print."""
    t = compare_trace(op_a, op_b, input)
    return t.render(level) if explain else t.result


def sweep_trace(
    op: str,
    param: str,
    values: list[float],
    base_input: list[float] | None = None,
) -> Trace:
    """Sweep a parameter across ``values`` and show how the output evolves.

    Currently supports ``op="softmax"`` with ``param="temperature"``.
    """
    if op != "softmax":
        raise ValueError(f"sweep_trace currently supports op='softmax', got {op!r}")
    if param != "temperature":
        raise ValueError(
            f"sweep_trace supports param='temperature' for softmax, got {param!r}"
        )
    if not values:
        raise ValueError("values must be a non-empty list of temperatures")
    if any(v <= 0 for v in values):
        raise ValueError("softmax temperatures must be > 0")

    logits = np.asarray(
        base_input if base_input is not None else [2.0, 1.0, 0.1], dtype=float
    )
    if logits.ndim != 1:
        raise ValueError(f"base_input must be 1-D, got shape {logits.shape}")

    t = Trace(
        op="sweep",
        formula="softmax(x / T)   as T varies",
        complexity="O(len(values) · n)",
        why_ai=[
            "Temperature controls exploration vs determinism in sampling",
            "T → 0 collapses to argmax (greedy); T → ∞ flattens toward uniform",
            "It is the single knob behind 'focused' vs 'creative' generation",
        ],
        meta={"op": op, "param": param, "values": list(values), "logits": logits.tolist()},
    )

    t.add("Base logits", f"x = {arr(logits)}", logits.copy())

    rows = []
    for temp in values:
        dist = _softmax(logits / temp)
        rows.append(dist)
        peak = float(np.max(dist))
        if temp < 1:
            shape = "sharper (→ one-hot)"
        elif temp > 1:
            shape = "flatter (→ uniform)"
        else:
            shape = "reference T=1"
        t.add(
            f"{param} = {num(temp)}",
            f"softmax(x / {num(temp)}) = {arr(dist)}",
            dist,
            detail=f"peak probability {num(peak)} — {shape}",
        )

    t.result = np.vstack(rows)
    return t


def sweep(
    op: str,
    param: str,
    values: list[float],
    base_input: list[float] | None = None,
    explain: bool = False,
    level: str | ExplainLevel = ExplainLevel.ENGINEER,
) -> Any:
    """Sweep ``param`` of ``op`` across ``values``. Set ``explain=True`` to print."""
    t = sweep_trace(op, param, values, base_input=base_input)
    return t.render(level) if explain else t.result


def demo() -> Trace:
    """Compare ReLU and GELU on ``[-2, -1, 0, 1, 2]``."""
    return compare_trace("relu", "gelu", [-2, -1, 0, 1, 2])
