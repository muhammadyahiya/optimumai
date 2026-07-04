"""A gallery of per-concept plots and GIFs for the v1.1 modules.

Where :mod:`optimumai.visualization.plots` and
:mod:`optimumai.visualization.animate` cover the v1.0 foundations (activations,
attention, embeddings, gradient descent, ...), this module does the same job
for v1.1: classical ML (:mod:`optimumai.ml`), search (:mod:`optimumai.search`),
reinforcement learning (:mod:`optimumai.rl`), vision (:mod:`optimumai.vision`),
and evaluation (:mod:`optimumai.evaluation`).

Every function computes real data by calling the actual v1.1 module (no
placeholder numbers), then renders it with matplotlib. matplotlib is an
optional dependency (the ``optimumai[viz]`` extra) so it is imported lazily
inside each function — the base package still imports fine without it. Static
plots save a PNG; animations save a GIF via matplotlib's ``PillowWriter``.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _mpl() -> Any:
    """Import matplotlib lazily in headless (Agg) mode, or explain how to get it."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError('plotting needs matplotlib: pip install "optimumai[viz]"') from exc
    return plt


def _mpl_anim() -> Any:
    """Import matplotlib + its animation writer lazily in headless mode."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation, PillowWriter
    except ImportError as exc:
        raise ImportError('animation needs matplotlib: pip install "optimumai[viz]"') from exc
    return plt, FuncAnimation, PillowWriter


def _save(plt: Any, fig: Any, out: str) -> str:
    """Save ``fig`` to ``out``, close it, and return the path."""
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


# --- ml: k-means -------------------------------------------------------------


def _kmeans_blobs(seed: int = 0) -> np.ndarray:
    """Two well-separated 2-D Gaussian blobs, small enough for a fast, tidy plot."""
    rng = np.random.default_rng(seed)
    blob_a = rng.normal(loc=(-3.0, -3.0), scale=0.6, size=(15, 2))
    blob_b = rng.normal(loc=(3.0, 3.0), scale=0.6, size=(15, 2))
    return np.vstack([blob_a, blob_b])


def plot_kmeans(out: str = "kmeans.png") -> str:
    """Plot 2-D points colored by their fitted k-means cluster, plus centroids.

    Fits :class:`optimumai.ml.KMeans` on two synthetic blobs and scatters every
    point in its assigned cluster's color, with an ``x`` marker at each final
    centroid.
    """
    from optimumai.ml import KMeans

    plt = _mpl()
    X = _kmeans_blobs()
    model = KMeans(k=2, max_iters=10).fit(X)

    fig, ax = plt.subplots(figsize=(6, 5))
    colors = ["tab:blue", "tab:orange"]
    for cluster in range(model.k):
        pts = X[model.labels_ == cluster]
        ax.scatter(pts[:, 0], pts[:, 1], color=colors[cluster], label=f"cluster {cluster}", s=40)
    ax.scatter(
        model.centroids[:, 0],
        model.centroids[:, 1],
        color="black",
        marker="x",
        s=150,
        linewidths=3,
        label="centroids",
    )
    ax.set_title(f"k-means (k={model.k}, inertia={model.inertia_:.2f})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return _save(plt, fig, out)


def animate_kmeans(out: str = "kmeans.gif", fps: int = 2) -> str:
    """Animate Lloyd's algorithm: points recolor and centroids move each iteration.

    Re-runs the assign/update steps of :func:`optimumai.ml.kmeans.kmeans_trace`
    by hand (small ``k`` and few points) so each frame is a real intermediate
    state of the algorithm, not an interpolation.
    """
    from optimumai.ml.kmeans import _inertia, _pairwise_distances

    plt, FuncAnimation, PillowWriter = _mpl_anim()
    X = _kmeans_blobs()
    k, max_iters = 2, 6
    centroids = X[:k].copy()

    frames_labels = []
    frames_centroids = []
    labels = np.full(X.shape[0], -1)
    for _ in range(max_iters):
        distances = _pairwise_distances(X, centroids)
        new_labels = np.argmin(distances, axis=1)
        frames_labels.append(new_labels.copy())
        frames_centroids.append(centroids.copy())
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        centroids = np.array(
            [
                X[labels == j].mean(axis=0) if np.any(labels == j) else centroids[j]
                for j in range(k)
            ]
        )

    colors = np.array(["tab:blue", "tab:orange"])
    fig, ax = plt.subplots(figsize=(5, 4))
    scatter = ax.scatter(X[:, 0], X[:, 1], c=colors[frames_labels[0]], s=40)
    centroid_scatter = ax.scatter(
        frames_centroids[0][:, 0], frames_centroids[0][:, 1], c="black", marker="x", s=150
    )
    title = ax.set_title("iteration 0")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    def update(i: int):
        scatter.set_color(colors[frames_labels[i]])
        centroid_scatter.set_offsets(frames_centroids[i])
        inertia = _inertia(X, frames_centroids[i], frames_labels[i])
        title.set_text(f"iteration {i}  (inertia={inertia:.2f})")
        return scatter, centroid_scatter, title

    anim = FuncAnimation(fig, update, frames=len(frames_labels), blit=False)
    anim.save(out, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return out


# --- ml: decision boundary ----------------------------------------------------


def plot_decision_boundary(out: str = "decision_boundary.png") -> str:
    """Plot a classifier's decision regions over a 2-D toy dataset.

    Trains :class:`optimumai.ml.LogisticRegression` on two separable blobs,
    scores a mesh grid covering the data, and shades the predicted region with
    ``contourf`` behind the scattered training points.
    """
    from optimumai.ml import LogisticRegression

    plt = _mpl()
    rng = np.random.default_rng(0)
    blob_a = rng.normal(loc=(-2.0, -2.0), scale=0.8, size=(20, 2))
    blob_b = rng.normal(loc=(2.0, 2.0), scale=0.8, size=(20, 2))
    X = np.vstack([blob_a, blob_b])
    y = np.array([0] * len(blob_a) + [1] * len(blob_b))

    model = LogisticRegression(lr=0.5, steps=300).fit(X, y)

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 150), np.linspace(y_min, y_max, 150))
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    preds = model.predict(grid).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.contourf(xx, yy, preds, levels=[-0.5, 0.5, 1.5], colors=["#a6c8ff", "#ffcf9e"], alpha=0.6)
    ax.scatter(X[y == 0, 0], X[y == 0, 1], color="tab:blue", label="class 0", edgecolor="k")
    ax.scatter(X[y == 1, 0], X[y == 1, 1], color="tab:orange", label="class 1", edgecolor="k")
    ax.set_title("Logistic regression decision boundary")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.legend()
    return _save(plt, fig, out)


# --- search: A* ----------------------------------------------------------------


def _astar_demo_grid() -> Any:
    """A 6x6 grid with a wall forcing a detour, matching the informed-search demo style."""
    from optimumai.search.problem import GridWorld

    walls = {(1, 0), (1, 1), (1, 2), (1, 3), (3, 2), (3, 3), (3, 4), (3, 5)}
    return GridWorld(width=6, height=6, walls=walls)


class _GridAdapter:
    """Binds ``GridWorld.heuristic`` to the 2-arg call ``informed.py`` expects."""

    def __init__(self, grid: Any) -> None:
        self._grid = grid

    def neighbors(self, state: Any) -> dict:
        return self._grid.neighbors(state)

    def cost(self, a: Any, b: Any) -> float:
        return self._grid.cost(a, b)

    def heuristic(self, state: Any, goal: Any) -> float:
        return self._grid.heuristic(state, goal, kind="manhattan")


def plot_astar_grid(out: str = "astar_grid.png") -> str:
    """Plot an A* search: walls, start/goal, explored cells, and the final path.

    Runs :func:`optimumai.search.informed.astar_trace` on a
    :class:`optimumai.search.problem.GridWorld` with a wall forcing a detour,
    then shades every expanded cell and overlays the reconstructed path.
    """
    from optimumai.search.informed import astar_trace

    plt = _mpl()
    grid = _astar_demo_grid()
    start, goal = (0, 0), (5, 5)
    trace = astar_trace(_GridAdapter(grid), start, goal)
    expansion_order = trace.meta["expansion_order"]
    path = trace.meta["path"]

    grid_img = np.zeros((grid.height, grid.width))
    for r, c in grid.walls:
        grid_img[r, c] = 1.0
    for r, c in expansion_order:
        if grid_img[r, c] == 0.0:
            grid_img[r, c] = 0.5

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.imshow(grid_img, cmap="Greys", vmin=0, vmax=1, origin="upper")
    path_rows = [p[0] for p in path]
    path_cols = [p[1] for p in path]
    ax.plot(path_cols, path_rows, color="tab:red", linewidth=3, marker="o", markersize=4)
    ax.scatter(*start[::-1], color="tab:green", s=200, marker="s", label="start", zorder=5)
    ax.scatter(*goal[::-1], color="tab:blue", s=200, marker="*", label="goal", zorder=5)
    ax.set_xticks(range(grid.width))
    ax.set_yticks(range(grid.height))
    ax.set_title(f"A* search (path cost={trace.meta['path_cost']:.0f}, "
                 f"expanded={trace.meta['nodes_expanded']})")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=3)
    return _save(plt, fig, out)


def animate_astar(out: str = "astar.gif", fps: int = 3) -> str:
    """Animate A*'s frontier expanding cell by cell, then reveal the final path.

    Uses the same :func:`optimumai.search.informed.astar_trace` expansion
    order as :func:`plot_astar_grid`, revealing one more expanded cell per
    frame and adding the reconstructed path in the last few frames.
    """
    from optimumai.search.informed import astar_trace

    plt, FuncAnimation, PillowWriter = _mpl_anim()
    grid = _astar_demo_grid()
    start, goal = (0, 0), (5, 5)
    trace = astar_trace(_GridAdapter(grid), start, goal)
    expansion_order = trace.meta["expansion_order"]
    path = trace.meta["path"]

    base = np.zeros((grid.height, grid.width))
    for r, c in grid.walls:
        base[r, c] = 1.0

    n_expand_frames = len(expansion_order)
    n_path_frames = len(path)
    total_frames = n_expand_frames + n_path_frames

    fig, ax = plt.subplots(figsize=(5, 5))
    image = ax.imshow(base, cmap="Greys", vmin=0, vmax=1, origin="upper")
    (path_line,) = ax.plot([], [], color="tab:red", linewidth=3, marker="o", markersize=4)
    ax.scatter(*start[::-1], color="tab:green", s=150, marker="s", zorder=5)
    ax.scatter(*goal[::-1], color="tab:blue", s=150, marker="*", zorder=5)
    ax.set_xticks(range(grid.width))
    ax.set_yticks(range(grid.height))
    title = ax.set_title("expanding frontier")

    def update(i: int):
        grid_img = base.copy()
        n_expanded = min(i + 1, n_expand_frames)
        for r, c in expansion_order[:n_expanded]:
            if grid_img[r, c] == 0.0:
                grid_img[r, c] = 0.5
        image.set_data(grid_img)
        if i >= n_expand_frames:
            n_path = i - n_expand_frames + 1
            path_line.set_data([p[1] for p in path[:n_path]], [p[0] for p in path[:n_path]])
            title.set_text("path found")
        else:
            title.set_text(f"expanded {n_expanded}/{n_expand_frames} cells")
        return image, path_line, title

    anim = FuncAnimation(fig, update, frames=total_frames, blit=False)
    anim.save(out, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return out


# --- rl: value iteration on a small gridworld MDP -----------------------------


def _gridworld_mdp(height: int = 4, width: int = 4, goal: tuple[int, int] = (3, 3)) -> Any:
    """Build a deterministic gridworld :class:`optimumai.rl.MDP`.

    4-connected moves (stepping off the grid leaves you in place), a single
    absorbing goal cell with reward +10, and -1 living cost everywhere else —
    small enough that value iteration converges in a handful of sweeps.
    """
    from optimumai.rl import MDP

    coords = [(r, c) for r in range(height) for c in range(width)]
    states = [f"{r},{c}" for r, c in coords]
    index = {rc: i for i, rc in enumerate(coords)}
    actions = ["up", "down", "left", "right"]
    moves = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
    n_s, n_a = len(states), len(actions)
    goal_idx = index[goal]

    p = np.zeros((n_s, n_a, n_s))
    r = np.zeros((n_s, n_a, n_s))
    for rc in coords:
        s = index[rc]
        if rc == goal:
            for a in range(n_a):
                p[s, a, s] = 1.0
            continue
        for a, name in enumerate(actions):
            dr, dc = moves[name]
            nxt = (rc[0] + dr, rc[1] + dc)
            if not (0 <= nxt[0] < height and 0 <= nxt[1] < width):
                nxt = rc
            s2 = index[nxt]
            p[s, a, s2] = 1.0
            r[s, a, s2] = 10.0 if nxt == goal else -1.0

    return MDP(
        states=states,
        actions=actions,
        transition=p,
        reward=r,
        gamma=0.9,
        terminal=frozenset({goal_idx}),
    ), height, width, coords, index


_ARROW = {"up": (0, -0.3), "down": (0, 0.3), "left": (-0.3, 0), "right": (0.3, 0)}


def plot_value_function(out: str = "value_function.png") -> str:
    """Plot a gridworld state-value heatmap with greedy-policy arrows.

    Runs :func:`optimumai.rl.mdp.value_iteration_trace` on a small
    deterministic gridworld MDP, then draws ``V*`` as a heatmap and overlays
    an arrow at every non-terminal cell pointing in its greedy action.
    """
    from optimumai.rl.mdp import value_iteration_trace

    plt = _mpl()
    mdp, height, width, coords, index = _gridworld_mdp()
    trace = value_iteration_trace(mdp)
    values = trace.result
    policy = trace.meta["policy"]

    grid_values = np.array(values).reshape(height, width)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    image = ax.imshow(grid_values, cmap="viridis", origin="upper")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="V*(s)")

    for rc in coords:
        r, c = rc
        s = f"{r},{c}"
        ax.text(c, r, f"{grid_values[r, c]:.1f}", ha="center", va="center", color="white",
                fontsize=8)
        if s in policy:
            dx, dy = _ARROW[policy[s]]
            ax.annotate(
                "", xy=(c + dx, r + dy), xytext=(c, r),
                arrowprops=dict(arrowstyle="->", color="red", lw=1.5),
            )
    ax.set_xticks(range(width))
    ax.set_yticks(range(height))
    ax.set_title(f"Value iteration: V* and greedy policy ({trace.meta['iterations']} sweeps)")
    return _save(plt, fig, out)


def animate_value_iteration(out: str = "value_iteration.gif", fps: int = 2) -> str:
    """Animate the value heatmap converging sweep by sweep.

    Re-runs the same Bellman backup as
    :func:`optimumai.rl.mdp.value_iteration_trace` by hand, capturing ``V``
    after every sweep so each frame is a real intermediate value function.
    """
    plt, FuncAnimation, PillowWriter = _mpl_anim()
    mdp, height, width, coords, index = _gridworld_mdp()

    n_s, n_a = mdp.n_states, mdp.n_actions
    values = np.zeros(n_s)
    frames = [values.copy()]
    for _ in range(20):
        new_values = values.copy()
        for s in range(n_s):
            if s in mdp.terminal:
                new_values[s] = 0.0
                continue
            q_sa = np.array([mdp.expected_backup(values, s, a) for a in range(n_a)])
            new_values[s] = float(np.max(q_sa))
        delta = float(np.max(np.abs(new_values - values)))
        values = new_values
        frames.append(values.copy())
        if delta < 1e-6:
            break

    grids = [f.reshape(height, width) for f in frames]
    vmin, vmax = min(g.min() for g in grids), max(g.max() for g in grids)

    fig, ax = plt.subplots(figsize=(5, 4.5))
    image = ax.imshow(grids[0], cmap="viridis", vmin=vmin, vmax=vmax, origin="upper")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="V(s)")
    texts = [
        [
            ax.text(c, r, "", ha="center", va="center", color="white", fontsize=8)
            for c in range(width)
        ]
        for r in range(height)
    ]
    title = ax.set_title("sweep 0")
    ax.set_xticks(range(width))
    ax.set_yticks(range(height))

    def update(i: int):
        image.set_data(grids[i])
        for r in range(height):
            for c in range(width):
                texts[r][c].set_text(f"{grids[i][r, c]:.1f}")
        title.set_text(f"sweep {i}")
        return [image, title, *[t for row in texts for t in row]]

    anim = FuncAnimation(fig, update, frames=len(grids), blit=False)
    anim.save(out, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return out


# --- vision: convolution -------------------------------------------------------


def plot_conv_feature_map(out: str = "conv_feature_map.png") -> str:
    """Plot an input image, a kernel, and the resulting feature map side by side.

    Uses :func:`optimumai.vision.conv2d` with a vertical-edge kernel on a
    small synthetic image (the same "half dark, half light" pattern used in
    the module's own demo).
    """
    from optimumai.vision import conv2d

    plt = _mpl()
    x = np.array(
        [
            [0.0, 0.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0, 1.0],
        ]
    )
    kernel = np.array([[1.0, 0.0, -1.0], [1.0, 0.0, -1.0], [1.0, 0.0, -1.0]])
    feature_map = conv2d(x, kernel, stride=1, padding=0)

    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    for ax, mat, title, cmap in zip(
        axes, [x, kernel, feature_map],
        ["input image", "kernel (vertical edge)", "feature map"],
        ["gray", "coolwarm", "coolwarm"],
        strict=True,
    ):
        im = ax.imshow(mat, cmap=cmap)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("2-D convolution: input ⊛ kernel = feature map")
    return _save(plt, fig, out)


# --- evaluation: calibration ---------------------------------------------------


def _calibration_demo_data() -> tuple[list[float], list[bool]]:
    """A deterministic, moderately overconfident set of predictions for the plot."""
    confidences = [
        0.95, 0.9, 0.92, 0.88, 0.6, 0.55, 0.62, 0.58, 0.3, 0.25,
        0.35, 0.2, 0.75, 0.8, 0.7, 0.78, 0.45, 0.5, 0.4, 0.48,
    ]
    correct = [
        True, True, False, True, True, False, True, False, False, True,
        False, True, True, True, False, True, False, True, False, False,
    ]
    return confidences, correct


def plot_calibration(out: str = "calibration.png") -> str:
    """Plot a reliability diagram: per-bin confidence vs accuracy, with the diagonal.

    Runs :func:`optimumai.evaluation.calibration.ece_trace` on a fixed,
    moderately overconfident set of predictions and draws a bar per bin
    (accuracy) next to the perfectly-calibrated diagonal.
    """
    from optimumai.evaluation.calibration import ece_trace

    plt = _mpl()
    confidences, correct = _calibration_demo_data()
    trace = ece_trace(confidences, correct, n_bins=5)
    bins = trace.meta["bins"]

    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="perfect calibration")
    for b in bins:
        lo, hi = b["range"]
        center = (lo + hi) / 2
        width = hi - lo
        ax.bar(center, b["accuracy"], width=width * 0.9, color="tab:blue", alpha=0.7,
               edgecolor="black")
        ax.scatter(center, b["mean_confidence"], color="tab:red", zorder=5, s=40)
    ax.scatter([], [], color="tab:red", label="mean confidence")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("confidence bin")
    ax.set_ylabel("accuracy")
    ax.set_title(f"Reliability diagram (ECE = {trace.result:.3f})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return _save(plt, fig, out)


# --- rl: PPO clipped objective --------------------------------------------------


def plot_ppo_clip(out: str = "ppo_clip.png", epsilon: float = 0.2) -> str:
    """Plot the PPO clipped surrogate objective vs. the probability ratio r.

    Sweeps ``r`` over a range and evaluates ``min(r·A, clip(r, 1-ε, 1+ε)·A)``
    directly (the same formula used inside
    :func:`optimumai.rl.ppo.ppo_clip_trace`) for both ``A=+1`` and ``A=-1`` —
    the classic clip diagram.
    """
    plt = _mpl()
    ratios = np.linspace(0.0, 2.0, 400)

    def clipped_objective(r: np.ndarray, advantage: float) -> np.ndarray:
        unclipped = r * advantage
        clipped = np.clip(r, 1 - epsilon, 1 + epsilon) * advantage
        return np.minimum(unclipped, clipped)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=False)
    for ax, advantage, title in zip(axes, [1.0, -1.0], ["A > 0", "A < 0"], strict=True):
        objective = clipped_objective(ratios, advantage)
        ax.plot(ratios, objective, color="tab:blue", linewidth=2, label="L^CLIP")
        ax.plot(ratios, ratios * advantage, color="grey", linestyle="--", linewidth=1,
                label="unclipped r·A")
        ax.axvline(1 - epsilon, color="tab:red", linestyle=":", linewidth=1)
        ax.axvline(1 + epsilon, color="tab:red", linestyle=":", linewidth=1)
        ax.axvline(1.0, color="black", linewidth=0.5)
        ax.set_title(f"PPO clipped objective, {title}")
        ax.set_xlabel("probability ratio r")
        ax.set_ylabel("L^CLIP(r)")
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"PPO clip diagram (ε={epsilon})")
    return _save(plt, fig, out)
