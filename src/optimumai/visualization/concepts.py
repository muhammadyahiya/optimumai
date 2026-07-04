"""One registry to visualize any concept — as a PNG, and as a GIF where motion helps.

    from optimumai.visualization.concepts import render_concept, list_concepts
    render_concept("attention", fmt="png", out="attn.png")
    render_concept("gradient_descent", fmt="gif", out="gd.gif")

PNG is available for every concept; GIF for the ones where animation adds insight.
All matplotlib imports are lazy (needs the ``[viz]`` extra).
"""

from __future__ import annotations

import numpy as np


def _mpl():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise ImportError('visualization needs matplotlib: pip install "optimumai[viz]"') from exc
    return plt


def _save(fig, out):
    fig.savefig(out, dpi=120, bbox_inches="tight")
    _mpl().close(fig)
    return out


# --- concept-specific PNGs not already covered by plots.py --------------------
def _png_matmul(out):
    plt = _mpl()
    rng = np.random.default_rng(0)
    A, B = rng.normal(size=(4, 4)), rng.normal(size=(4, 4))
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    for ax, m, title in zip(axes, [A, B, A @ B], ["A", "B", "A·B"], strict=True):
        im = ax.imshow(m, cmap="coolwarm")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046)
    return _save(fig, out)


def _png_positional(out):
    plt = _mpl()
    from optimumai.transformers.positional import positional_encoding

    pe = positional_encoding(32, 32)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(pe, cmap="RdBu", aspect="auto")
    ax.set_title("sinusoidal positional encoding")
    ax.set_xlabel("dimension")
    ax.set_ylabel("position")
    fig.colorbar(im, ax=ax)
    return _save(fig, out)


def _png_kv_cache(out):
    plt = _mpl()
    from optimumai.foundations.kv_cache import kv_cache_size

    seqs = np.array([512, 1024, 2048, 4096, 8192, 16384])
    gb = [kv_cache_size(32, 32, 128, int(s)) / 1024**3 for s in seqs]
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(seqs, gb, "o-", color="#c0392b")
    ax.set_title("KV cache grows linearly with context")
    ax.set_xlabel("sequence length")
    ax.set_ylabel("KV cache (GB)")
    ax.grid(alpha=0.3)
    return _save(fig, out)


def _png_vram(out):
    plt = _mpl()
    comps = ["weights", "gradients", "optimizer", "activations"]
    gb = [14, 14, 56, 6]
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(comps, gb, color=["#2563eb", "#ea7317", "#16a34a", "#9333ea"])
    ax.set_title("VRAM to train a 7B model (fp16, Adam)")
    ax.set_ylabel("GB")
    return _save(fig, out)


def _png_backprop(out):
    plt = _mpl()
    from optimumai.autograd import Value

    a, b, c = Value(2.0, label="a"), Value(-3.0, label="b"), Value(10.0, label="c")
    loss = (a * b + c).tanh()
    loss.backward()
    leaves = {"a": a.grad, "b": b.grad, "c": c.grad}
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(list(leaves), list(leaves.values()), color="#ea7317")
    ax.axhline(0, color="#333", lw=0.8)
    ax.set_title("backprop: ∂L/∂· at each leaf")
    ax.set_ylabel("gradient")
    return _save(fig, out)


def _png_dotcos(out):
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(4, 4))
    for vec, color, name in [((3, 1), "#2563eb", "a"), ((1, 2), "#ea7317", "b")]:
        ax.annotate("", xy=vec, xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=color, lw=2))
        ax.text(vec[0], vec[1], f" {name}", color=color)
    ax.set_xlim(-1, 4)
    ax.set_ylim(-1, 4)
    ax.grid(alpha=0.3)
    ax.set_title("vectors: dot = |a||b|cosθ")
    ax.set_aspect("equal")
    return _save(fig, out)


def _lazy_plots():
    from optimumai.visualization import animate, gallery, landscape, plots

    return plots, landscape, animate, gallery


def _registry():
    plots, landscape, animate, gallery = _lazy_plots()
    return {
        "activation": {"png": lambda o: plots.plot_activation(out=o), "gif": None},
        "softmax": {"png": lambda o: plots.plot_softmax_temperature(out=o),
                    "gif": lambda o: animate.animate_softmax_temperature(out=o)},
        "attention": {"png": lambda o: plots.plot_attention(out=o), "gif": None},
        "embeddings": {"png": lambda o: plots.plot_embeddings(out=o), "gif": None},
        "training": {"png": lambda o: plots.plot_training_curve(out=o), "gif": None},
        "gradient_descent": {"png": lambda o: landscape.plot_loss_landscape("rosenbrock", out=o),
                             "gif": lambda o: animate.animate_gradient_descent(out=o)},
        "loss_landscape": {"png": lambda o: landscape.plot_loss_landscape("saddle", out=o),
                           "gif": lambda o: animate.animate_gradient_descent("bowl", out=o)},
        "diffusion": {"png": lambda o: _png_matmul(o), "gif": lambda o: animate.animate_diffusion(out=o)},
        "matmul": {"png": _png_matmul, "gif": None},
        "positional": {"png": _png_positional, "gif": None},
        "kv_cache": {"png": _png_kv_cache, "gif": None},
        "vram": {"png": _png_vram, "gif": None},
        "backprop": {"png": _png_backprop, "gif": None},
        "dot": {"png": _png_dotcos, "gif": None},
        # v1.2 — classical-AI / ML gallery
        "kmeans": {"png": lambda o: gallery.plot_kmeans(out=o),
                   "gif": lambda o: gallery.animate_kmeans(out=o)},
        "decision_boundary": {"png": lambda o: gallery.plot_decision_boundary(out=o), "gif": None},
        "astar": {"png": lambda o: gallery.plot_astar_grid(out=o),
                  "gif": lambda o: gallery.animate_astar(out=o)},
        "value_iteration": {"png": lambda o: gallery.plot_value_function(out=o),
                            "gif": lambda o: gallery.animate_value_iteration(out=o)},
        "conv2d": {"png": lambda o: gallery.plot_conv_feature_map(out=o), "gif": None},
        "calibration": {"png": lambda o: gallery.plot_calibration(out=o), "gif": None},
        "ppo_clip": {"png": lambda o: gallery.plot_ppo_clip(out=o), "gif": None},
    }


def list_concepts() -> list[str]:
    """All visualizable concept names."""
    return list(_registry())


def concept_formats(name: str) -> list[str]:
    """Which formats ("png", "gif") a concept supports."""
    reg = _registry()
    if name not in reg:
        raise ValueError(f"unknown concept {name!r}; choose from {list(reg)}")
    return [fmt for fmt in ("png", "gif") if reg[name][fmt] is not None]


def render_concept(name: str, fmt: str = "png", out: str | None = None) -> str:
    """Render ``name`` as ``fmt`` ("png" or "gif") to ``out`` (or a default path)."""
    reg = _registry()
    if name not in reg:
        raise ValueError(f"unknown concept {name!r}; choose from {list(reg)}")
    maker = reg[name].get(fmt)
    if maker is None:
        raise ValueError(f"concept {name!r} has no {fmt}; available: {concept_formats(name)}")
    return maker(out or f"{name}.{fmt}")
