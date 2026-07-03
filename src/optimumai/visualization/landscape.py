"""Loss landscapes — LeCun's energy surface, with gradient descent walking it.

This is the headline picture of the whole SDK: a 2-variable function drawn as a
contour map and a 3-D surface, with a numeric gradient-descent trajectory tumbling
down toward a minimum. Presets cover the classic teaching shapes (a convex bowl, a
saddle, and the notorious Rosenbrock banana), and you can pass your own expression
in ``x`` and ``y`` evaluated in a locked-down numpy namespace.

matplotlib is an optional dependency (the ``optimumai[viz]`` extra) so it is
imported lazily; the base package imports fine without it.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

# 3-D projection registration is a matplotlib import side effect; it is triggered
# lazily inside the plotting functions, never at module import time.

_PRESETS: dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
    "bowl": lambda x, y: x**2 + y**2,
    "saddle": lambda x, y: x**2 - y**2,
    "rosenbrock": lambda x, y: (1 - x) ** 2 + 100 * (y - x**2) ** 2,
}

_SAFE_NAMES = {
    "np": np,
    "sin": np.sin,
    "cos": np.cos,
    "exp": np.exp,
    "sqrt": np.sqrt,
    "pi": np.pi,
}


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


def _resolve_func(func: str) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Turn a preset name or a custom x/y expression into a callable f(x, y)."""
    if func in _PRESETS:
        return _PRESETS[func]

    try:
        code = compile(func, "<landscape-expr>", "eval")
    except SyntaxError as exc:
        raise ValueError(f"invalid landscape expression {func!r}: {exc}") from exc

    allowed = set(_SAFE_NAMES) | {"x", "y"}
    used = set(code.co_names)
    illegal = used - allowed
    if illegal:
        raise ValueError(
            f"expression uses names outside the allowed set {sorted(allowed)}: {sorted(illegal)}"
        )

    def evaluated(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        namespace = {**_SAFE_NAMES, "x": x, "y": y}
        return eval(code, {"__builtins__": {}}, namespace)  # noqa: S307 - locked namespace

    return evaluated


def _numeric_gradient(
    f: Callable[[np.ndarray, np.ndarray], np.ndarray],
    x: float,
    y: float,
    eps: float = 1e-4,
) -> tuple[float, float]:
    """Central-difference gradient of ``f`` at ``(x, y)``."""
    dx = (f(x + eps, y) - f(x - eps, y)) / (2 * eps)
    dy = (f(x, y + eps) - f(x, y - eps)) / (2 * eps)
    return float(dx), float(dy)


def _descend(
    f: Callable[[np.ndarray, np.ndarray], np.ndarray],
    start: tuple[float, float],
    lr: float,
    steps: int,
) -> np.ndarray:
    """Run gradient descent from ``start`` and return the ``(steps + 1) x 3`` path."""
    x, y = float(start[0]), float(start[1])
    path = [(x, y, float(f(x, y)))]
    for _ in range(steps):
        gx, gy = _numeric_gradient(f, x, y)
        x -= lr * gx
        y -= lr * gy
        path.append((x, y, float(f(x, y))))
    return np.asarray(path, dtype=float)


def _grid(
    f: Callable[[np.ndarray, np.ndarray], np.ndarray],
    path: np.ndarray,
    pad: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build an evaluation grid that comfortably contains the descent path."""
    xs, ys = path[:, 0], path[:, 1]
    x_lo, x_hi = float(xs.min()) - pad, float(xs.max()) + pad
    y_lo, y_hi = float(ys.min()) - pad, float(ys.max()) + pad
    gx = np.linspace(x_lo, x_hi, 120)
    gy = np.linspace(y_lo, y_hi, 120)
    mesh_x, mesh_y = np.meshgrid(gx, gy)
    mesh_z = f(mesh_x, mesh_y)
    return mesh_x, mesh_y, mesh_z


def _draw_contour(ax: Any, mesh: tuple[np.ndarray, ...], path: np.ndarray, func: str) -> None:
    """Draw a contour map with the descent path overlaid."""
    mesh_x, mesh_y, mesh_z = mesh
    contour = ax.contourf(mesh_x, mesh_y, mesh_z, levels=30, cmap="viridis")
    ax.figure.colorbar(contour, ax=ax, fraction=0.046, pad=0.04)
    ax.plot(path[:, 0], path[:, 1], color="white", linewidth=1.5, marker="o", markersize=3)
    ax.annotate("start", (path[0, 0], path[0, 1]), color="white", fontweight="bold")
    ax.annotate("end", (path[-1, 0], path[-1, 1]), color="red", fontweight="bold")
    ax.set_title(f"Contour of {func!r} with descent path")
    ax.set_xlabel("x")
    ax.set_ylabel("y")


def _draw_surface(ax: Any, mesh: tuple[np.ndarray, ...], path: np.ndarray, func: str) -> None:
    """Draw a 3-D surface with the descent trajectory on it."""
    mesh_x, mesh_y, mesh_z = mesh
    ax.plot_surface(mesh_x, mesh_y, mesh_z, cmap="viridis", alpha=0.7, linewidth=0)
    ax.plot3D(
        path[:, 0], path[:, 1], path[:, 2], color="red", linewidth=2, marker="o", markersize=3
    )
    ax.set_title(f"Surface of {func!r} with trajectory")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("loss")


def plot_loss_landscape(
    func: str = "bowl",
    out: str | None = None,
    start: tuple[float, float] = (-1.8, 1.8),
    lr: float = 0.02,
    steps: int = 40,
    kind: str = "both",
) -> Any:
    """Draw a loss landscape (contour, surface, or both) with a descent path."""
    plt = _mpl()
    import mpl_toolkits.mplot3d  # noqa: F401 - registers the 3-D projection

    if kind not in {"contour", "surface", "both"}:
        raise ValueError(f"kind must be one of contour, surface, both; got {kind!r}")

    f = _resolve_func(func)
    path = _descend(f, start, lr, steps)
    mesh = _grid(f, path)

    if kind == "contour":
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        _draw_contour(ax, mesh, path, func)
    elif kind == "surface":
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection="3d")
        _draw_surface(ax, mesh, path, func)
    else:
        fig = plt.figure(figsize=(13, 5.5))
        ax_contour = fig.add_subplot(1, 2, 1)
        _draw_contour(ax_contour, mesh, path, func)
        ax_surface = fig.add_subplot(1, 2, 2, projection="3d")
        _draw_surface(ax_surface, mesh, path, func)

    return _finish(plt, fig, out)


def landscape_demo(out: str | None = None) -> Any:
    """The headline demo: gradient descent tumbling down the Rosenbrock banana."""
    return plot_loss_landscape("rosenbrock", out=out)
