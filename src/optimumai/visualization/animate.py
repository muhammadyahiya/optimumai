"""Downloadable animated GIFs of models and algorithms (matplotlib, headless).

Turn a process into a shareable ``.gif``: an optimizer rolling downhill, a signal
being noised by diffusion, a softmax sharpening as temperature drops. Rendered
with matplotlib's Agg backend + Pillow writer, so it works with no display. Needs
the ``optimumai[viz]`` extra.
"""

from __future__ import annotations

import numpy as np


def _mpl():
    """Import matplotlib lazily with a headless backend, or raise a friendly error."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation, PillowWriter
    except ImportError as exc:  # pragma: no cover - env dependent
        raise ImportError('animation needs matplotlib: pip install "optimumai[viz]"') from exc
    return plt, FuncAnimation, PillowWriter


_PRESETS = {
    "bowl": lambda x, y: x**2 + y**2,
    "saddle": lambda x, y: x**2 - y**2,
    "rosenbrock": lambda x, y: (1 - x) ** 2 + 100 * (y - x**2) ** 2,
}


def animate_gradient_descent(
    func: str = "rosenbrock", out: str = "descent.gif",
    lr: float = 0.002, steps: int = 60, start: tuple[float, float] = (-1.5, 1.5), fps: int = 15,
) -> str:
    """Animate a point walking downhill on a loss surface, leaving a trail."""
    if func not in _PRESETS:
        raise ValueError(f"unknown preset {func!r}; choose from {list(_PRESETS)}")
    plt, FuncAnimation, PillowWriter = _mpl()
    f = _PRESETS[func]
    gx = np.linspace(-2, 2, 200)
    gy = np.linspace(-1, 3, 200)
    GX, GY = np.meshgrid(gx, gy)
    GZ = f(GX, GY)

    def grad(x, y, h=1e-4):
        dx = (f(x + h, y) - f(x - h, y)) / (2 * h)
        dy = (f(x, y + h) - f(x, y - h)) / (2 * h)
        return dx, dy

    xs, ys = [start[0]], [start[1]]
    for _ in range(steps):
        dx, dy = grad(xs[-1], ys[-1])
        xs.append(xs[-1] - lr * dx)
        ys.append(ys[-1] - lr * dy)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.contour(GX, GY, GZ, levels=30, cmap="viridis", linewidths=0.6)
    ax.set_title(f"gradient descent on {func}")
    (path,) = ax.plot([], [], "-", color="#ea7317", lw=1.5)
    (dot,) = ax.plot([], [], "o", color="#c0392b", ms=7)

    def update(i):
        path.set_data(xs[: i + 1], ys[: i + 1])
        dot.set_data([xs[i]], [ys[i]])
        return path, dot

    anim = FuncAnimation(fig, update, frames=len(xs), blit=True)
    anim.save(out, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return out


def animate_diffusion(
    out: str = "diffusion.gif", steps: int = 40, fps: int = 15, seed: int = 0
) -> str:
    """Animate a clean signal being progressively noised (forward diffusion)."""
    plt, FuncAnimation, PillowWriter = _mpl()
    rng = np.random.default_rng(seed)
    n = 64
    x0 = np.sin(np.linspace(0, 4 * np.pi, n))
    noise = rng.normal(size=n)
    betas = np.linspace(1e-4, 0.25, steps)
    abar = np.cumprod(1 - betas)

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.set_ylim(-2.5, 2.5)
    (line,) = ax.plot(x0, color="#2563eb")
    title = ax.set_title("t = 0  (clean signal)")

    def update(i):
        xt = np.sqrt(abar[i]) * x0 + np.sqrt(1 - abar[i]) * noise
        line.set_ydata(xt)
        title.set_text(f"t = {i + 1}/{steps}   noise level ≈ {1 - abar[i]:.2f}")
        return line, title

    anim = FuncAnimation(fig, update, frames=steps, blit=False)
    anim.save(out, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return out


def animate_softmax_temperature(
    logits: tuple[float, ...] = (2.0, 1.0, 0.1, -0.5), out: str = "softmax_temp.gif", fps: int = 12,
) -> str:
    """Animate a softmax distribution sharpening as temperature drops high → low."""
    plt, FuncAnimation, PillowWriter = _mpl()
    z = np.asarray(logits, dtype=float)
    temps = np.concatenate([np.linspace(3.0, 0.3, 30), np.linspace(0.3, 3.0, 15)])

    def softmax(t):
        s = z / t
        s = s - s.max()
        e = np.exp(s)
        return e / e.sum()

    fig, ax = plt.subplots(figsize=(5, 3))
    bars = ax.bar(range(len(z)), softmax(temps[0]), color="#2563eb")
    ax.set_ylim(0, 1)
    ax.set_xlabel("class")
    title = ax.set_title("")

    def update(i):
        probs = softmax(temps[i])
        for b, p in zip(bars, probs, strict=True):
            b.set_height(p)
        title.set_text(f"temperature = {temps[i]:.2f}")
        return list(bars) + [title]

    anim = FuncAnimation(fig, update, frames=len(temps), blit=False)
    anim.save(out, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return out


def demo(out: str = "descent.gif") -> str:
    """A representative GIF (gradient descent on Rosenbrock)."""
    return animate_gradient_descent(out=out)
