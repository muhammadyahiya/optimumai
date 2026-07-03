"""Matplotlib plots — turning the math into pictures you can actually see.

Every earlier module produces a *trace* of numbers; this one turns the same
numbers into figures: activation curves, temperature-swept softmax bars, attention
heatmaps, a 2-D embedding projection, and a training loss curve. matplotlib is an
optional dependency (the ``optimumai[viz]`` extra) so it is imported lazily — the
base package still imports fine without it.

Each public function takes an optional ``out`` path: pass one to save a PNG and
get the path back, or leave it ``None`` to get the live :class:`Figure` object.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _mpl() -> Any:
    """Import matplotlib lazily in headless (Agg) mode, or explain how to get it."""
    try:
        import matplotlib

        matplotlib.use("Agg")  # headless
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError('plotting needs matplotlib: pip install "optimumai[viz]"') from exc
    return plt


def _finish(plt: Any, fig: Any, out: str | None) -> Any:
    """Either save ``fig`` to ``out`` (returning the path) or return the figure."""
    if out is not None:
        fig.savefig(out, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return out
    return fig


def _softmax_rows(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over the last axis."""
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exps = np.exp(shifted)
    return exps / np.sum(exps, axis=-1, keepdims=True)


def _activation(name: str, x: np.ndarray) -> np.ndarray:
    """Evaluate one of the supported activations on ``x``."""
    if name == "relu":
        return np.maximum(0.0, x)
    if name == "gelu":
        return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))
    if name == "sigmoid":
        return 1.0 / (1.0 + np.exp(-x))
    if name == "tanh":
        return np.tanh(x)
    raise ValueError(f"unknown activation {name!r}; expected one of relu, gelu, sigmoid, tanh")


def plot_activation(
    name: str = "gelu",
    xlim: tuple[float, float] = (-6.0, 6.0),
    out: str | None = None,
) -> Any:
    """Plot an activation and its (numeric) derivative on a single axes."""
    plt = _mpl()
    x = np.linspace(xlim[0], xlim[1], 400)
    y = _activation(name, x)
    dy = np.gradient(y, x)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(x, y, label=f"{name}(x)", color="tab:blue", linewidth=2)
    ax.plot(x, dy, label=f"{name}'(x)", color="tab:orange", linewidth=2, linestyle="--")
    ax.axhline(0.0, color="grey", linewidth=0.8)
    ax.axvline(0.0, color="grey", linewidth=0.8)
    ax.set_title(f"Activation {name!r} and its derivative")
    ax.set_xlabel("x")
    ax.set_ylabel("value")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return _finish(plt, fig, out)


def plot_softmax_temperature(
    logits: tuple[float, ...] = (2.0, 1.0, 0.1),
    temperatures: tuple[float, ...] = (0.5, 1.0, 2.0, 5.0),
    out: str | None = None,
) -> Any:
    """Grouped bars of softmax(logits / T) — sharp at low T, flat at high T."""
    plt = _mpl()
    logit_arr = np.asarray(logits, dtype=float)
    n_classes = logit_arr.size
    n_temps = len(temperatures)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    positions = np.arange(n_classes)
    width = 0.8 / n_temps
    for i, temp in enumerate(temperatures):
        probs = _softmax_rows(logit_arr / temp)
        offset = (i - (n_temps - 1) / 2) * width
        ax.bar(positions + offset, probs, width=width, label=f"T = {temp:g}")

    ax.set_title("Softmax distribution vs temperature")
    ax.set_xlabel("class index")
    ax.set_ylabel("probability")
    ax.set_xticks(positions)
    ax.set_xticklabels([str(i) for i in range(n_classes)])
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(title="temperature")
    return _finish(plt, fig, out)


def plot_heatmap(
    matrix: Any,
    title: str = "",
    labels: list[str] | None = None,
    out: str | None = None,
) -> Any:
    """imshow a 2-D array with a colorbar; annotate cells when the matrix is small."""
    plt = _mpl()
    mat = np.asarray(matrix, dtype=float)
    if mat.ndim != 2:
        raise ValueError(f"plot_heatmap expects a 2-D array, got shape {mat.shape}")

    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(mat, cmap="viridis", aspect="auto")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title or "Heatmap")

    if labels is not None:
        ax.set_xticks(range(mat.shape[1]))
        ax.set_yticks(range(mat.shape[0]))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    if mat.shape[0] <= 8 and mat.shape[1] <= 8:
        threshold = (mat.max() + mat.min()) / 2.0
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                colour = "white" if mat[i, j] < threshold else "black"
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", color=colour)
    return _finish(plt, fig, out)


def plot_attention(text: str | None = None, out: str | None = None) -> Any:
    """Heatmap of scaled dot-product self-attention weights with token labels."""
    if text is not None:
        tokens = text.split()
        if not tokens:
            raise ValueError("plot_attention got empty text after tokenizing")
        rng = np.random.default_rng(0)
        d = 8
        emb = rng.normal(size=(len(tokens), d))
        scores = emb @ emb.T / np.sqrt(d)
        weights = _softmax_rows(scores)
        labels = tokens
        title = "Self-attention weights"
    else:
        rng = np.random.default_rng(0)
        d = 4
        emb = rng.normal(size=(4, d))
        scores = emb @ emb.T / np.sqrt(d)
        weights = _softmax_rows(scores)
        labels = [f"t{i}" for i in range(4)]
        title = "Self-attention weights (seeded 4x4 example)"

    return plot_heatmap(weights, title=title, labels=labels, out=out)


def _pca_2d(matrix: np.ndarray) -> np.ndarray:
    """Project rows of ``matrix`` onto their top-2 principal components via SVD."""
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:2].T
    return centered @ components


def plot_embeddings(
    words: list[str] | None = None,
    dim: int = 16,
    out: str | None = None,
) -> Any:
    """Scatter seeded word embeddings after a numpy-SVD PCA down to 2-D."""
    plt = _mpl()
    if words is None:
        words = ["king", "queen", "man", "woman", "apple", "orange", "cat", "dog"]
    if not words:
        raise ValueError("plot_embeddings needs at least one word")

    rng = np.random.default_rng(0)
    emb = rng.normal(size=(len(words), dim))
    coords = _pca_2d(emb)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(coords[:, 0], coords[:, 1], color="tab:purple", s=60)
    for word, (px, py) in zip(words, coords, strict=True):
        ax.annotate(word, (px, py), textcoords="offset points", xytext=(6, 4))
    ax.set_title("Word embeddings projected to 2-D (random embeddings, PCA via SVD)")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.grid(True, alpha=0.3)
    return _finish(plt, fig, out)


def plot_training_curve(losses: list[float] | None = None, out: str | None = None) -> Any:
    """Plot training loss vs iteration on a log-y axis (runs the demo if needed)."""
    plt = _mpl()
    if losses is None:
        from optimumai.neural_networks.backprop import train_demo

        trace = train_demo(steps=120)
        losses = [
            float(step.value)
            for step in trace.steps
            if step.title.startswith("step") and isinstance(step.value, int | float)
        ]
    if not losses:
        raise ValueError("plot_training_curve got no loss values to plot")

    iterations = np.arange(len(losses))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(iterations, losses, color="tab:red", linewidth=2, marker="o", markersize=3)
    ax.set_yscale("log")
    ax.set_title("Training loss over iterations")
    ax.set_xlabel("iteration")
    ax.set_ylabel("loss (log scale)")
    ax.grid(True, which="both", alpha=0.3)
    return _finish(plt, fig, out)
