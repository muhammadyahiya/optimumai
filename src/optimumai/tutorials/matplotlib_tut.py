"""The matplotlib tutorial: Figure/Axes, the core chart types, and styling.

Runs headless (the ``Agg`` backend) so it works in CI and over SSH with no
display — every plotting cell saves a PNG to a temp file instead of calling
``plt.show()``. Requires the ``[viz]`` extra (``requires=("matplotlib",)``);
without it, the code and prose are still shown, just not executed.
"""

from __future__ import annotations

from optimumai.tutorials.core import Tutorial

_MPL = ("matplotlib",)


def build() -> Tutorial:
    t = Tutorial(
        name="matplotlib",
        title="matplotlib: the Figure/Axes mental model",
        summary="Line, bar, scatter, hist, subplots, styling, and saving figures.",
    )

    # ------------------------------------------------------------------
    # 1. headless setup + the Figure/Axes model
    # ------------------------------------------------------------------
    t.md(
        "## The mental model: Figure and Axes\n\n"
        "A matplotlib **Figure** is the whole canvas — the window or the saved "
        "image file. An **Axes** is one set of x/y (or x/y/z) coordinates drawn "
        "inside that figure — the thing that actually has data, ticks, labels, "
        "and a title. A figure can hold many axes (subplots); most simple plots "
        "have exactly one.\n\n"
        "The examples below use the *object-oriented* API — `fig, ax = "
        "plt.subplots()`, then calling methods on `ax` — because it scales "
        "cleanly to multi-panel figures, unlike the implicit `plt.plot(...)` "
        "state-machine style.\n\n"
        "This tutorial runs headless: no display is available, so we select the "
        "non-interactive `Agg` backend up front and save every figure to a file "
        "with `fig.savefig(...)` instead of calling `plt.show()` (which would "
        "warn or hang without a display)."
    )
    t.code(
        """import matplotlib

matplotlib.use("Agg")  # headless backend: no display needed
import tempfile

import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(6, 4))
print("fig is the whole canvas:", type(fig).__name__)
print("ax is one set of axes living inside it:", type(ax).__name__)
plt.close(fig)""",
        note="select Agg before importing pyplot; never call plt.show() headless",
        requires=_MPL,
    )

    # ------------------------------------------------------------------
    # 2. line plot
    # ------------------------------------------------------------------
    t.md(
        "## Line plots\n\n"
        "`ax.plot(x, y)` is the default: connect points with a line. Good for "
        "anything ordered along one axis — time series, loss curves, functions."
    )
    t.code(
        """x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x)

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(x, y, color="tab:blue")
ax.set_title("sin(x)")

path = tempfile.mktemp(suffix="_line.png")
fig.savefig(path)
plt.close(fig)
print("saved line plot to:", path)""",
        note="fig.savefig(...) writes the file; no plt.show() in a headless tutorial",
        requires=_MPL,
    )

    # ------------------------------------------------------------------
    # 3. bar chart
    # ------------------------------------------------------------------
    t.md(
        "## Bar charts\n\n"
        "`ax.bar(positions, heights)` compares discrete categories. Positions are "
        "usually just `np.arange(n)`, with the category names supplied separately "
        "via `set_xticks`/`set_xticklabels`."
    )
    t.code(
        """categories = ["a", "b", "c", "d"]
values = np.array([3, 7, 4, 9])

fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(np.arange(len(categories)), values, color="tab:orange")
ax.set_xticks(np.arange(len(categories)))
ax.set_xticklabels(categories)
ax.set_title("bar chart")

path = tempfile.mktemp(suffix="_bar.png")
fig.savefig(path)
plt.close(fig)
print("saved bar chart to:", path)""",
        note="bar positions are integers; labels are applied separately",
        requires=_MPL,
    )

    # ------------------------------------------------------------------
    # 4. scatter plot
    # ------------------------------------------------------------------
    t.md(
        "## Scatter plots\n\n"
        "`ax.scatter(x, y)` shows unconnected points — the right choice when you "
        "care about the *relationship* between two variables rather than a "
        "trend over an ordered axis."
    )
    t.code(
        """rng = np.random.default_rng(seed=0)
x = rng.normal(size=80)
y = 2.0 * x + rng.normal(scale=0.5, size=80)

fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(x, y, color="tab:green", alpha=0.7)
ax.set_title("scatter: y roughly tracks 2x")

path = tempfile.mktemp(suffix="_scatter.png")
fig.savefig(path)
plt.close(fig)
print("saved scatter plot to:", path)""",
        note="alpha < 1 helps when points overlap",
        requires=_MPL,
    )

    # ------------------------------------------------------------------
    # 5. histogram
    # ------------------------------------------------------------------
    t.md(
        "## Histograms\n\n"
        "`ax.hist(values, bins=...)` shows the *distribution* of a batch of "
        "numbers by bucketing them into ranges and counting how many fall in "
        "each. Increasing `bins` trades smoothness for resolution."
    )
    t.code(
        """rng = np.random.default_rng(seed=1)
samples = rng.normal(loc=0.0, scale=1.0, size=2000)

fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(samples, bins=30, color="tab:purple", edgecolor="white")
ax.set_title("histogram of 2000 standard-normal samples")

path = tempfile.mktemp(suffix="_hist.png")
fig.savefig(path)
plt.close(fig)
print("saved histogram to:", path)""",
        note="bins controls the resolution of the distribution shape",
        requires=_MPL,
    )

    # ------------------------------------------------------------------
    # 6. subplots
    # ------------------------------------------------------------------
    t.md(
        "## Subplots: multiple Axes, one Figure\n\n"
        "`plt.subplots(nrows, ncols)` returns one Figure and a grid of Axes so "
        "related charts can live side by side. `ax` is now a NumPy array of "
        "Axes objects that you index like any other array."
    )
    t.code(
        """x = np.linspace(-3, 3, 100)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(x, x**2, color="tab:blue")
axes[0].set_title("x^2")
axes[1].plot(x, np.sin(x), color="tab:red")
axes[1].set_title("sin(x)")
fig.suptitle("two axes, one figure")

path = tempfile.mktemp(suffix="_subplots.png")
fig.savefig(path)
plt.close(fig)
print("saved subplots figure to:", path)
print("axes grid shape:", axes.shape)""",
        note="axes is a NumPy array of Axes when nrows*ncols > 1",
        requires=_MPL,
    )

    # ------------------------------------------------------------------
    # 7. labels, legend, title, limits
    # ------------------------------------------------------------------
    t.md(
        "## Labels, legend, title, and axis limits\n\n"
        "A readable chart names its axes (`set_xlabel`/`set_ylabel`), names "
        "itself (`set_title`), and — when there's more than one series — "
        "explains which line is which (`label=...` on each call, then "
        "`ax.legend()`). `set_xlim`/`set_ylim` control what range is visible."
    )
    t.code(
        """x = np.linspace(0, 10, 200)

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(x, np.sin(x), label="sin(x)", color="tab:blue")
ax.plot(x, np.cos(x), label="cos(x)", color="tab:orange")
ax.set_xlabel("x")
ax.set_ylabel("f(x)")
ax.set_title("sin and cos, labeled")
ax.set_xlim(0, 10)
ax.set_ylim(-1.5, 1.5)
ax.legend()

path = tempfile.mktemp(suffix="_labeled.png")
fig.savefig(path)
plt.close(fig)
print("saved labeled figure to:", path)
print("legend entries:", [line.get_label() for line in ax.get_lines()])""",
        note="label= on each plot call + ax.legend() documents multi-series charts",
        requires=_MPL,
    )

    # ------------------------------------------------------------------
    # 8. styling & colors
    # ------------------------------------------------------------------
    t.md(
        "## Styling and colors\n\n"
        "Line style, marker, width, and color are all keyword arguments. "
        "matplotlib's `tab:*` palette (`tab:blue`, `tab:orange`, `tab:green`, "
        "...) is a good default: readable, colorblind-considerate, and "
        "consistent across the ecosystem. `plt.style.use(...)` swaps the whole "
        "visual theme in one line."
    )
    t.code(
        """x = np.linspace(0, 5, 30)

with plt.style.context("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in
                        plt.style.available else "default"):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, x, linestyle="--", marker="o", color="tab:blue", label="linear")
    ax.plot(x, x**1.5, linestyle="-", marker="s", color="tab:red", label="x^1.5")
    ax.legend()
    ax.set_title("custom line style, marker, and color")

path = tempfile.mktemp(suffix="_styled.png")
fig.savefig(path)
plt.close(fig)
print("saved styled figure to:", path)
print("available styles include seaborn-like themes:", len(plt.style.available) > 0)""",
        note="plt.style.context(...) applies a theme without leaking global state",
        requires=_MPL,
    )

    # ------------------------------------------------------------------
    # 9. saving a figure (formats, dpi)
    # ------------------------------------------------------------------
    t.md(
        "## Saving figures for real\n\n"
        "`fig.savefig(path, dpi=..., bbox_inches=\"tight\")` is what you'll use "
        "outside a headless demo, too: it writes PNG/PDF/SVG based on the file "
        "extension, `dpi` controls resolution, and `bbox_inches=\"tight\"` trims "
        "excess whitespace around the plot."
    )
    t.code(
        """import os

fig, ax = plt.subplots(figsize=(5, 3))
ax.bar(["x", "y", "z"], [3, 5, 2], color="tab:cyan")
ax.set_title("saved at 150 dpi, tight bounding box")

path = tempfile.mktemp(suffix="_saved.png")
fig.savefig(path, dpi=150, bbox_inches="tight")
plt.close(fig)
print("saved figure to:", path)
print("file exists on disk:", os.path.exists(path))
print("file size in bytes:", os.path.getsize(path))""",
        note="dpi + bbox_inches='tight' are the two options worth remembering",
        requires=_MPL,
    )

    # ------------------------------------------------------------------
    # 10. where to go next
    # ------------------------------------------------------------------
    t.md(
        "## Where to go next: Plot Studio\n\n"
        "For an interactive companion to everything above, use OptimumAI's own "
        "**Plot Studio**: `optimumai.visualization.plotstudio`. Its "
        "`plot_studio_playground()` writes a self-contained, offline HTML page "
        "where typing in a box of numbers updates a live chart, live numpy "
        "summary statistics, and the exact matplotlib+numpy source that "
        "reproduces the chart — all at once, no server required.\n\n"
        "From the CLI: `optimumai plot-studio`. From Python:\n\n"
        "```python\n"
        "from optimumai.visualization.plotstudio import plot_studio_playground\n"
        "plot_studio_playground('studio.html')\n"
        "```\n\n"
        "`plotstudio.plot_code(data, kind=\"bar\")` is also worth knowing: it "
        "returns the exact matplotlib+numpy source for a chart as a string, "
        "without ever importing matplotlib — handy for generating copy-paste "
        "snippets from data alone."
    )
    return t
