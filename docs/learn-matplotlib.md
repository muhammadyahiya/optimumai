# Learn matplotlib

matplotlib is the plotting library nearly everything else in Python's data
stack is built on top of (or renders through). Once you understand its
**Figure/Axes model**, every other plotting call — pandas' `.plot()`, seaborn,
OptimumAI's own visualization helpers — is just a shortcut on top of the same
two objects.

!!! tip "Run this interactively"
    ```bash
    optimumai tutorial matplotlib
    ```

    or from Python:

    ```python
    from optimumai.tutorials import get_tutorial
    get_tutorial("matplotlib").run()
    ```

    (Needs the `optimumai[viz]` extra installed for the interactive tutorial's
    plotting cells to actually render.)

All snippets on this page were executed against matplotlib 3.11 while writing
these docs, using the headless **Agg** backend (see [below](#the-agg-backend-and-headless-environments)).
Examples use `savefig`, never `show`, because `show()` requires a GUI/display
that a docs build, CI job, or SSH session usually doesn't have.

```python
import matplotlib
matplotlib.use("Agg")   # do this before importing pyplot, on headless machines
import matplotlib.pyplot as plt
import numpy as np
```

---

## 1. The Figure/Axes model

**Intuition.** Think of a **`Figure`** as the whole window or page, and an
**`Axes`** as one plot living inside it. A figure can hold many axes (a grid of
subplots); each axes has its own x/y limits, labels, and the actual drawn data
(lines, bars, points...). Almost every confusing matplotlib question — "why
didn't my title show up," "why are my two plots overlapping" — resolves once
you ask "which `Figure` and which `Axes` am I actually talking to?"

```python
fig, ax = plt.subplots()       # one Figure containing one Axes
ax.plot([1, 2, 3], [4, 1, 5])  # draw onto that Axes
fig.savefig("plot.png")        # save the whole Figure
```

`fig.axes` is the list of all Axes objects living in that figure — useful when
you build a grid and want to loop over every subplot.

---

## 2. pyplot vs. the object-oriented API

matplotlib has two ways to say the same thing.

**The `pyplot` (stateful) API** — `plt.plot(...)`, `plt.title(...)` — always
acts on "the current figure and current axes," tracked implicitly by
matplotlib. It reads like MATLAB and is fast for one-off, throwaway plots:

```python
x = np.linspace(0, 2 * np.pi, 100)
plt.plot(x, np.sin(x))
plt.title("sin(x)")
plt.savefig("pyplot_line.png")
plt.close()   # release the implicit current figure
```

**The object-oriented (OO) API** — get an explicit `fig, ax` and call methods
on `ax` directly. This is the recommended style for anything beyond a quick
sketch: multiple subplots, reusable plotting functions, or any code that lives
longer than one notebook cell, because there's no ambiguity about *which*
figure or axes you're modifying.

```python
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(x, np.sin(x), label="sin")
ax.plot(x, np.cos(x), label="cos")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Trig functions")
ax.legend()
fig.savefig("oo_line.png", dpi=150, bbox_inches="tight")
plt.close(fig)
```

!!! note "The mapping between the two APIs"
    Every `plt.xxx()` call is really "do `xxx` on the current axes" —
    `plt.plot(...)` → `ax.plot(...)`, `plt.title(...)` → `ax.set_title(...)`,
    `plt.xlabel(...)` → `ax.set_xlabel(...)`. Once you notice the pattern
    (`set_` prefix on the OO side for anything that's a property), switching
    between the two APIs, or reading code that mixes them, gets much easier.

This page uses the OO API from here on, since it's the one you want once
you're writing anything reusable.

---

## 3. Chart types

### Line

```python
fig, ax = plt.subplots()
ax.plot(x, np.sin(x))
fig.savefig("line.png")
plt.close(fig)
```

### Bar

```python
fig, ax = plt.subplots()
categories = ["A", "B", "C"]
values = [3, 7, 5]
ax.bar(categories, values, color="steelblue")
ax.set_title("Bar chart")
fig.savefig("bar.png")
plt.close(fig)
```

### Scatter

```python
rng = np.random.default_rng(0)
xs = rng.normal(size=200)
ys = rng.normal(size=200)

fig, ax = plt.subplots()
sc = ax.scatter(xs, ys, c=xs + ys, cmap="viridis", alpha=0.7)
fig.colorbar(sc, ax=ax, label="x+y")
fig.savefig("scatter.png")
plt.close(fig)
```

`c=` colors each point by a third variable (here `x + y`); `cmap=` picks the
colormap that maps those values to colors (more on colormaps
[below](#colormaps)).

### Histogram

```python
fig, ax = plt.subplots()
data = rng.normal(loc=0, scale=1, size=1000)
ax.hist(data, bins=30, color="salmon", edgecolor="black")
fig.savefig("hist.png")
plt.close(fig)
```

`bins` controls the resolution: too few and you smear over real structure, too
many and you're plotting noise. 20-50 bins is a reasonable default for a few
thousand points; scale roughly with `sqrt(n)` if you want a rule of thumb.

### Box plot

```python
fig, ax = plt.subplots()
groups = [rng.normal(size=100), rng.normal(loc=1, size=100), rng.normal(loc=-1, size=100)]
ax.boxplot(groups, tick_labels=["g1", "g2", "g3"])
fig.savefig("box.png")
plt.close(fig)
```

A box plot summarizes a distribution's median, quartiles, and outliers in one
compact shape — reach for it when you want to compare the *spread* of several
groups side by side, not just their means.

### Subplots

```python
fig, axes = plt.subplots(1, 2, figsize=(8, 3))
axes[0].plot(x, np.sin(x))
axes[0].set_title("sin")
axes[1].plot(x, np.cos(x))
axes[1].set_title("cos")
fig.tight_layout()   # fix overlapping labels/titles between subplots
fig.savefig("subplots.png")
plt.close(fig)
```

`plt.subplots(nrows, ncols)` returns `axes` as a NumPy array of `Axes` objects
when `nrows * ncols > 1` — index it like any array (`axes[0]`, or `axes[0, 1]`
for a 2-D grid).

---

## 4. Labels, legend, titles, limits

```python
fig, ax = plt.subplots()
ax.plot(x, np.sin(x), label="sin(x)")
ax.set_xlabel("x")
ax.set_ylabel("amplitude")
ax.set_title("A sine wave, zoomed")
ax.set_xlim(0, np.pi)
ax.set_ylim(-1.5, 1.5)
ax.legend()
fig.savefig("limits.png")
plt.close(fig)
```

- `set_xlabel` / `set_ylabel` / `set_title` — text annotations.
- `legend()` — draws a legend from every artist's `label=`; skip artists you
  don't want in it by leaving `label` unset.
- `set_xlim` / `set_ylim` — zoom the view without changing the underlying
  data; useful for focusing on a region of interest.

---

## 5. Styling, colors, and colormaps

### Colors

Pass `color=` (a name like `"steelblue"`, a hex string `"#1f77b4"`, or an RGB(A)
tuple) to any plotting call. matplotlib also has a default color cycle, so if
you don't specify colors, successive `plot()` calls on the same axes
automatically get different colors.

### Colormaps

A colormap turns a *scalar* value into a color — used whenever `c=` or
`cmap=` shows up (`scatter`, `imshow`, `pcolormesh`, contour plots). Pick the
family that matches your data:

- **Sequential** (`"viridis"`, `"plasma"`, `"cividis"`) — for data that goes
  from low to high with no meaningful zero crossing (counts, magnitudes).
  `"viridis"` is the modern matplotlib default and is perceptually uniform
  (equal steps in data look like equal steps in color) and colorblind-safe —
  prefer it over the old `"jet"` rainbow map, which is neither.
- **Diverging** (`"coolwarm"`, `"RdBu"`) — for data with a meaningful midpoint
  (e.g. positive/negative correlation, gains/losses around zero).
- **Qualitative** (`"tab10"`, `"Set2"`) — for categorical data with no
  inherent order.

```python
fig, ax = plt.subplots()
Z = rng.normal(size=(10, 10))
im = ax.imshow(Z, cmap="coolwarm")
fig.colorbar(im, ax=ax)
fig.savefig("heatmap.png")
plt.close(fig)
```

### Style sheets

```python
with plt.style.context("seaborn-v0_8-darkgrid"):
    fig, ax = plt.subplots()
    ax.plot(x, np.sin(x))
    fig.savefig("styled.png")
    plt.close(fig)
```

`plt.style.available` lists every built-in style name (`"ggplot"`,
`"fivethirtyeight"`, `"dark_background"`, ...). `plt.style.context(...)` scopes
the style change to the `with` block only — prefer it over the global
`plt.style.use(...)` unless you genuinely want every subsequent plot in the
process restyled.

---

## 6. Saving figures

```python
fig.savefig(
    "figure.png",
    dpi=150,           # resolution — higher for print/zoom, 100-150 is fine for screens
    bbox_inches="tight",  # trim excess whitespace around the figure
)
```

- **`dpi`** (dots per inch) controls resolution. The default is 100; use
  150-300 for anything you'll zoom into or print. Higher `dpi` means a larger
  file and slower save, so don't reach for 600 by default.
- **`bbox_inches="tight"`** re-crops the saved image to fit the actual content
  (labels, titles, legends) instead of clipping them or leaving dead margin.
  It's cheap and almost always what you want.
- The output format is inferred from the extension — `.png`, `.pdf`, `.svg`,
  `.jpg` are all supported by `savefig` directly. Vector formats (`.pdf`,
  `.svg`) scale losslessly and are the right choice for anything going into a
  paper or slide deck.

!!! warning "`savefig` before `show`/`close`, not after"
    Once you call `plt.show()` in an interactive session, or `plt.close()`,
    the figure may be torn down — call `savefig` *before* either, not after.
    In scripts (no `show()` at all), just call `savefig` then `plt.close(fig)`
    to free the memory once you're done with that figure.

### The Agg backend and headless environments

matplotlib's default backend tries to open an interactive window, which fails
(or silently does nothing useful) on a server, in CI, over SSH without X
forwarding, or inside most container images. **Agg** ("Anti-Grain Geometry")
is a backend that only rasterizes to image files — no display required:

```python
import matplotlib
matplotlib.use("Agg")   # must happen before `import matplotlib.pyplot`
```

Every snippet on this page ran with Agg. If you only ever call `savefig` and
never `show`, you can also just set the environment variable
`MPLBACKEND=Agg` before running your script, or rely on matplotlib
auto-selecting a non-interactive backend when no display is detected — but
explicitly calling `matplotlib.use("Agg")` first is the most reliable, portable
choice for scripts, notebooks-as-scripts, and docs builds alike.

---

## 7. When to reach for `optimumai plot-studio`

Once you're comfortable with the API above, `optimumai plot-studio` is a
teaching shortcut for the "I know what chart I want but forget the exact
matplotlib incantation" moment. Feed it numbers; it renders the chart **and**
prints the exact, copy-paste-runnable matplotlib + NumPy source that produced
it:

```bash
optimumai plot-studio "[3,1,4,1,5,9,2,6]" --kind hist
```

```python
from optimumai.visualization.plotstudio import plot_code, describe

# describe() gives you numpy summary stats without needing matplotlib at all
describe([1, 2, 3, 4, 5])
# {'n': 5, 'mean': 3.0, 'std': 1.41..., 'min': 1.0, 'q1': 2.0,
#  'median': 3.0, 'q3': 4.0, 'max': 5.0}

# plot_code() returns the source string — never executes it, always runnable
print(plot_code([1, 2, 3, 4, 5], kind="bar"))
```

which prints:

```python
import numpy as np
import matplotlib.pyplot as plt

data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

plt.figure(figsize=(7, 4.5))
plt.bar(np.arange(len(data)), data, color="tab:blue")
plt.xlabel("index")
plt.title("bar chart")
plt.tight_layout()
plt.show()
```

`plot_code` never imports matplotlib itself — it just builds and returns that
string, so it works even without the `[viz]` extra installed. `plot_data(...)`
is the sibling function that actually renders and saves the chart (needs
`optimumai[viz]`). Supported `kind` values: `bar`, `hist`, `scatter`, `box`,
`line`, `pie`, `violin`. Reach for Plot Studio when you want a quick chart from
raw numbers *and* a working code template to build on — it's a bridge from
"I have some numbers" to "here's idiomatic matplotlib code," not a replacement
for knowing the API in the sections above.

!!! note "Where to go next"
    matplotlib almost always plots NumPy arrays — if broadcasting, shapes, or
    `axis=` felt unfamiliar above, back up to [Learn NumPy](learn-numpy.md).
    From here, [Learn PyTorch](learn-pytorch.md) picks up where NumPy arrays
    leave off: tensors that also track gradients.
