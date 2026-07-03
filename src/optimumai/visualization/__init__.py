"""Visualization: terminal traces, plus matplotlib graphs (the ``[viz]`` extra).

The plotting functions import matplotlib lazily, so importing this package never
requires matplotlib — you only need it when you actually draw.
"""

from optimumai.visualization.landscape import landscape_demo, plot_loss_landscape
from optimumai.visualization.plots import (
    plot_activation,
    plot_attention,
    plot_embeddings,
    plot_heatmap,
    plot_softmax_temperature,
    plot_training_curve,
)
from optimumai.visualization.terminal import render_trace

__all__ = [
    "landscape_demo",
    "plot_activation",
    "plot_attention",
    "plot_embeddings",
    "plot_heatmap",
    "plot_loss_landscape",
    "plot_softmax_temperature",
    "plot_training_curve",
    "render_trace",
]
