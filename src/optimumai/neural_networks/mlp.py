"""A multi-layer perceptron built on the autograd engine — Karpathy's ``nn.py``.

Every weight and bias is a :class:`~optimumai.autograd.value.Value`, so a forward
pass builds a computation graph and ``loss.backward()`` fills in every gradient
automatically. There is no special "neural network" math here — it is just the
scalar autograd engine composed a few thousand times.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

from optimumai.autograd.value import Value

_ACTIVATIONS = ("tanh", "relu", "linear")


class Neuron:
    """A single neuron: ``activation(Σ wᵢxᵢ + b)``."""

    def __init__(self, n_in: int, activation: str = "tanh", seed: int | None = None):
        if activation not in _ACTIVATIONS:
            raise ValueError(f"activation must be one of {_ACTIVATIONS}, got {activation!r}")
        rng = random.Random(seed)
        self.w = [Value(rng.uniform(-1, 1), label=f"w{i}") for i in range(n_in)]
        self.b = Value(0.0, label="b")
        self.activation = activation

    def __call__(self, x: Sequence[Value | float]) -> Value:
        act = self.b
        for wi, xi in zip(self.w, x, strict=True):
            act = act + wi * xi
        if self.activation == "tanh":
            return act.tanh()
        if self.activation == "relu":
            return act.relu()
        return act

    def parameters(self) -> list[Value]:
        return [*self.w, self.b]


class Layer:
    """A dense layer: ``n_out`` neurons, each seeing all inputs."""

    def __init__(self, n_in: int, n_out: int, activation: str = "tanh", seed: int | None = None):
        self.neurons = [
            Neuron(n_in, activation, seed=None if seed is None else seed * 1_000 + i)
            for i in range(n_out)
        ]

    def __call__(self, x: Sequence[Value | float]) -> Value | list[Value]:
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self) -> list[Value]:
        return [p for neuron in self.neurons for p in neuron.parameters()]


class MLP:
    """A stack of dense layers. Hidden layers use ``activation``; output is linear.

    Args:
        n_in: Number of input features.
        n_outs: Sizes of each layer, e.g. ``[4, 4, 1]``.
        activation: Hidden-layer nonlinearity (``"tanh"`` or ``"relu"``).
        seed: Seed for reproducible weight initialization.
    """

    def __init__(self, n_in: int, n_outs: Sequence[int], activation: str = "tanh", seed: int = 0):
        sizes = [n_in, *n_outs]
        self.layers = [
            Layer(
                sizes[i],
                sizes[i + 1],
                activation=activation if i < len(n_outs) - 1 else "linear",
                seed=seed * 100 + i,
            )
            for i in range(len(n_outs))
        ]

    def __call__(self, x: Sequence[Value | float]) -> Value | list[Value]:
        out: Sequence[Value | float] = x
        for layer in self.layers:
            out = layer(out)
            if isinstance(out, Value):
                out = [out]
        return out[0] if len(out) == 1 else list(out)

    def parameters(self) -> list[Value]:
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self) -> str:
        shape = " → ".join(str(len(layer.neurons)) for layer in self.layers)
        return f"MLP(params={len(self.parameters())}, layers={shape})"
