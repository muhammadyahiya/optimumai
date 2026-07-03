"""Neural networks built on the scalar autograd engine."""

from optimumai.neural_networks.backprop import (
    forward_backward_trace,
    train,
    train_demo,
)
from optimumai.neural_networks.mlp import MLP, Layer, Neuron

__all__ = [
    "MLP",
    "Layer",
    "Neuron",
    "forward_backward_trace",
    "train",
    "train_demo",
]
