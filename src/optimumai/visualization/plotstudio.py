"""Plot Studio — give it numbers, get back the chart *and* the code that made it.

Most plotting tools stop at the picture. Plot Studio is a teaching tool: every
chart it draws comes with the exact, copy-paste-runnable ``matplotlib`` +
``numpy`` source that reproduces it, so "how do I make this chart myself" is
answered by the tool that just made it for you.

Three ways in:

* :func:`describe` — numpy summary statistics for a batch of numbers.
* :func:`plot_code` — the source string, never executed, always runnable.
* :func:`plot_data` — actually render the chart (needs the ``[viz]`` extra).
* :func:`plot_studio_trace` — the same computation as a :class:`Trace`, so it
  renders with the rest of the SDK's ``.render()`` story.
* :func:`plot_studio_playground` — a self-contained, offline HTML page where
  typing numbers updates a live chart, live numpy stats, and live
  matplotlib+numpy code all at once — no server, no CDN.

matplotlib is imported lazily (inside functions only): importing this module,
calling :func:`describe`, or calling :func:`plot_code` never requires it.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from optimumai.core.trace import Trace

_KINDS = ("bar", "hist", "scatter", "box", "line", "pie", "violin")


def _mpl() -> Any:
    """Import matplotlib lazily in headless (Agg) mode, or explain how to get it."""
    try:
        import matplotlib

        matplotlib.use("Agg")  # headless
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError('plot studio needs matplotlib: pip install "optimumai[viz]"') from exc
    return plt


# --------------------------------------------------------------------------
# parsing: turn loose user input into clean numpy arrays
# --------------------------------------------------------------------------


def _is_pair(item: Any) -> bool:
    """Whether ``item`` looks like an ``(x, y)`` pair rather than a scalar."""
    return isinstance(item, list | tuple) and len(item) == 2 and not isinstance(item, str)


def _parse_values(data: Any) -> np.ndarray:
    """Parse ``data`` into a flat 1-D float array; raise on empty/bad input."""
    arr = np.asarray(data, dtype=float).ravel()
    if arr.size == 0:
        raise ValueError("plot studio got empty data")
    return arr


def _parse_xy(data: Any) -> tuple[np.ndarray, np.ndarray]:
    """Parse ``data`` for scatter plots.

    Accepts either a sequence of ``(x, y)`` pairs, or a two-item sequence of
    equal-length ``x`` and ``y`` sequences (``[xs, ys]``).
    """
    if data is None:
        raise ValueError("scatter needs data shaped as [(x, y), ...] or [xs, ys]")
    items = list(data)
    if not items:
        raise ValueError("plot studio got empty data")

    if all(_is_pair(item) for item in items):
        xs = np.asarray([float(item[0]) for item in items], dtype=float)
        ys = np.asarray([float(item[1]) for item in items], dtype=float)
        return xs, ys

    if len(items) == 2 and all(isinstance(item, list | tuple | np.ndarray) for item in items):
        xs = np.asarray(items[0], dtype=float)
        ys = np.asarray(items[1], dtype=float)
        if xs.shape != ys.shape:
            raise ValueError(
                f"scatter x/y length mismatch: {xs.shape[0]} vs {ys.shape[0]}"
            )
        return xs, ys

    raise ValueError(
        "scatter needs data shaped as [(x, y), ...] pairs or two equal-length "
        "sequences [xs, ys]"
    )


# --------------------------------------------------------------------------
# 1. describe — numpy summary statistics
# --------------------------------------------------------------------------


def describe(data: Any) -> dict[str, float | int]:
    """Numpy summary statistics for a batch of numbers.

    Returns ``n``, ``mean``, ``std``, ``min``, ``q1``, ``median``, ``q3``,
    ``max``. Raises :class:`ValueError` on empty data.
    """
    values = _parse_values(data)
    q1, median, q3 = np.percentile(values, [25, 50, 75])
    return {
        "n": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "q1": float(q1),
        "median": float(median),
        "q3": float(q3),
        "max": float(np.max(values)),
    }


# --------------------------------------------------------------------------
# 2. plot_code — the exact, runnable source
# --------------------------------------------------------------------------

_KIND_CALL: dict[str, str] = {
    "bar": 'plt.bar(np.arange(len(data)), data, color="tab:blue")\nplt.xlabel("index")',
    "hist": 'plt.hist(data, bins=bins, color="tab:blue", edgecolor="white")\nplt.xlabel("value")',
    "scatter": 'plt.scatter(x, y, color="tab:blue")\nplt.xlabel("x")\nplt.ylabel("y")',
    "box": 'plt.boxplot(data, vert=True)\nplt.ylabel("value")',
    "line": (
        'plt.plot(np.arange(len(data)), data, color="tab:blue", marker="o")\n'
        'plt.xlabel("index")'
    ),
    "pie": "plt.pie(data, labels=labels, autopct=\"%1.1f%%\")",
    "violin": 'plt.violinplot(data, vert=True, showmeans=True)\nplt.ylabel("value")',
}


def _validate_kind(kind: str) -> None:
    if kind not in _KINDS:
        raise ValueError(f"unknown kind {kind!r}; choose from {_KINDS}")


def plot_code(data: Any, kind: str = "bar", **opts: Any) -> str:
    """Return the exact, copy-paste-runnable ``matplotlib``+``numpy`` source.

    The string always starts with ``import numpy as np`` and
    ``import matplotlib.pyplot as plt``, builds the same array Plot Studio
    used, calls the matching plotting function, labels the axes/title, and
    ends with ``plt.show()``.

    ``kind`` is one of ``"bar"``, ``"hist"``, ``"scatter"``, ``"box"``,
    ``"line"``, ``"pie"``, ``"violin"``. For ``"scatter"``, ``data`` may be a
    list of ``(x, y)`` pairs or two equal-length sequences ``[xs, ys]``.
    ``opts`` accepts ``title`` (all kinds) and ``bins`` (``"hist"`` only,
    default 10).

    Raises :class:`ValueError` on empty data or an unknown ``kind``.
    """
    _validate_kind(kind)
    title = str(opts.get("title", f"{kind} chart"))
    lines = ["import numpy as np", "import matplotlib.pyplot as plt", ""]

    if kind == "scatter":
        xs, ys = _parse_xy(data)
        lines.append(f"x = np.array({xs.tolist()})")
        lines.append(f"y = np.array({ys.tolist()})")
    else:
        values = _parse_values(data)
        lines.append(f"data = np.array({values.tolist()})")
        if kind == "hist":
            bins = int(opts.get("bins", 10))
            lines.append(f"bins = {bins}")
        if kind == "pie":
            labels = list(opts.get("labels") or [str(i) for i in range(values.size)])
            lines.append(f"labels = {labels!r}")

    lines.append("")
    lines.append("plt.figure(figsize=(7, 4.5))")
    lines.append(_KIND_CALL[kind])
    lines.append(f'plt.title("{title}")')
    lines.append("plt.tight_layout()")
    lines.append("plt.show()")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# 3. plot_data — actually render with matplotlib
# --------------------------------------------------------------------------


def plot_data(data: Any, kind: str = "bar", *, out: str = "plot.png", **opts: Any) -> str:
    """Render ``data`` as ``kind`` with matplotlib and save it to ``out``.

    Returns the path to the saved PNG. See :func:`plot_code` for the
    supported ``kind`` values, the accepted ``data`` shapes, and ``opts``.
    """
    _validate_kind(kind)
    plt = _mpl()
    title = str(opts.get("title", f"{kind} chart"))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    if kind == "scatter":
        xs, ys = _parse_xy(data)
        ax.scatter(xs, ys, color="tab:blue")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    else:
        values = _parse_values(data)
        if kind == "bar":
            ax.bar(np.arange(values.size), values, color="tab:blue")
            ax.set_xlabel("index")
        elif kind == "hist":
            bins = int(opts.get("bins", 10))
            ax.hist(values, bins=bins, color="tab:blue", edgecolor="white")
            ax.set_xlabel("value")
        elif kind == "box":
            ax.boxplot(values, vert=True)
            ax.set_ylabel("value")
        elif kind == "line":
            ax.plot(np.arange(values.size), values, color="tab:blue", marker="o")
            ax.set_xlabel("index")
        elif kind == "pie":
            labels = list(opts.get("labels") or [str(i) for i in range(values.size)])
            ax.pie(values, labels=labels, autopct="%1.1f%%")
        elif kind == "violin":
            ax.violinplot(values, vert=True, showmeans=True)
            ax.set_ylabel("value")

    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


# --------------------------------------------------------------------------
# 4. plot_studio_trace — the SDK's step-by-step story
# --------------------------------------------------------------------------


def plot_studio_trace(data: Any, kind: str = "bar") -> Trace:
    """Build a :class:`Trace` narrating parse -> numpy stats -> chart -> code.

    ``trace.result`` is the :func:`plot_code` source string, so
    ``plot_studio_trace(...).render()`` prints the stats and ends by showing
    exactly the matplotlib+numpy code a learner would copy out.
    """
    _validate_kind(kind)
    trace = Trace(
        op="plot_studio",
        formula="x̄ = (1/n)Σxᵢ,  s = sqrt((1/n)Σ(xᵢ - x̄)²),  quartiles via percentile",
        why_ai=[
            "Every model-eval report starts as a Plot Studio-style loop: describe "
            "the numbers, then chart them.",
            "Histograms of activations/gradients catch dead ReLUs and exploding "
            "gradients before training blows up.",
            "Loss curves (line), calibration (scatter), and per-class accuracy "
            "(bar) are the three charts you'll draw the most as an ML engineer.",
        ],
    )

    if kind == "scatter":
        xs, ys = _parse_xy(data)
        trace.add(
            "Parse numbers",
            f"{len(xs)} (x, y) pairs -> two numpy arrays",
            value=list(zip(xs.tolist(), ys.tolist(), strict=True)),
        )
        stats = describe(np.concatenate([xs, ys]))
    else:
        values = _parse_values(data)
        trace.add(
            "Parse numbers",
            f"{values.size} values -> np.array(...)",
            value=values.tolist(),
        )
        stats = describe(values)

    trace.add(
        "Compute numpy stats",
        "np.mean, np.std, np.percentile([25, 50, 75]), np.min, np.max",
        value=stats,
        detail=(
            f"n={stats['n']}  mean={stats['mean']:.4g}  std={stats['std']:.4g}  "
            f"median={stats['median']:.4g}"
        ),
    )

    trace.add(
        f"Build the {kind} chart",
        f"matplotlib.pyplot.{_chart_fn_name(kind)}(...)",
        value=kind,
        detail="Rendered with optimumai.visualization.plotstudio.plot_data(...).",
    )

    code = plot_code(data, kind=kind)
    trace.add(
        "Emit reproducible code",
        "the exact matplotlib+numpy source that draws this chart",
        value=code,
    )
    trace.result = code
    return trace


def _chart_fn_name(kind: str) -> str:
    """The matplotlib.pyplot function name used for ``kind`` (for narration)."""
    return {
        "bar": "bar",
        "hist": "hist",
        "scatter": "scatter",
        "box": "boxplot",
        "line": "plot",
        "pie": "pie",
        "violin": "violinplot",
    }[kind]


# --------------------------------------------------------------------------
# 5. plot_studio_playground — self-contained HTML, no server, no CDN
# --------------------------------------------------------------------------

_EXAMPLE_DATA = "4, 8, 15, 16, 23, 42, 8, 15, 16, 4, 9, 30"
_EXAMPLE_SCATTER = "1,2\n2,4\n3,5\n4,8\n5,7\n6,10\n7,9\n8,13"

_PLAYGROUND_CSS = """
 body{font-family:system-ui,sans-serif;margin:0;padding:20px;color:#111;
      background:#fafafa}
 h2.sr-only{position:absolute;left:-9999px}
 h3{margin:0 0 6px}
 .sub{color:#555;margin:0 0 14px;max-width:760px}
 .row{display:flex;align-items:center;gap:10px;margin:8px 0;flex-wrap:wrap}
 textarea{font-family:monospace;font-size:13px;width:320px;height:140px;
          padding:8px;border-radius:6px;border:1px solid #ccc}
 select{font-family:inherit;font-size:14px;padding:5px 8px}
 button{font-family:inherit;font-size:13px;padding:6px 14px;border-radius:6px;
        border:1px solid #ccc;background:#fff;cursor:pointer}
 button:hover{background:#f0f0f0}
 button.primary{background:#2563eb;color:#fff;border-color:#2563eb}
 button.primary:hover{background:#1d4ed8}
 .grid{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:16px;
       align-items:start}
 .panel{border:1px solid #ddd;border-radius:8px;padding:12px;background:#fff}
 .panel h4{margin:0 0 8px;font-size:13px;color:#555;text-transform:uppercase;
           letter-spacing:.04em}
 canvas{border:1px solid #eee;border-radius:6px;background:#fff;max-width:100%}
 table.stats{border-collapse:collapse;font-family:monospace;font-size:13px}
 table.stats td{padding:3px 10px 3px 0}
 table.stats td.k{color:#555}
 pre#code{font-family:monospace;font-size:12px;background:#0f172a;color:#e2e8f0;
          padding:12px;border-radius:6px;overflow-x:auto;margin:0;max-height:360px}
 #err{color:#c0392b;font-family:monospace;font-size:13px;min-height:18px}
"""

_PLAYGROUND_SCRIPT = r"""
const KINDS = __KINDS__;
const dataBox = document.getElementById('dataBox');
const kindSelect = document.getElementById('kindSelect');
const canvas = document.getElementById('chart');
const ctx = canvas.getContext('2d');
const statsBox = document.getElementById('stats');
const codeBox = document.getElementById('code');
const errBox = document.getElementById('err');

function parseValues(text){
  const nums = text.split(/[\s,]+/).map(s => s.trim()).filter(s => s.length)
    .map(Number).filter(v => !Number.isNaN(v));
  if (nums.length === 0) throw new Error('enter at least one number');
  return nums;
}

function parseXY(text){
  const lines = text.split('\n').map(l => l.trim()).filter(l => l.length);
  const xs = [], ys = [];
  for (const line of lines){
    const parts = line.split(/[\s,]+/).map(Number);
    if (parts.length < 2 || parts.some(Number.isNaN)) continue;
    xs.push(parts[0]); ys.push(parts[1]);
  }
  if (xs.length === 0) throw new Error('enter "x,y" pairs, one per line');
  return {xs, ys};
}

function mean(a){ return a.reduce((s,v)=>s+v,0) / a.length; }

function percentile(sorted, p){
  const idx = (p / 100) * (sorted.length - 1);
  const lo = Math.floor(idx), hi = Math.ceil(idx);
  if (lo === hi) return sorted[lo];
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (idx - lo);
}

function describe(values){
  const sorted = [...values].sort((a,b) => a-b);
  const m = mean(values);
  const variance = mean(values.map(v => (v - m) ** 2));
  return {
    n: values.length,
    mean: m,
    std: Math.sqrt(variance),
    min: sorted[0],
    q1: percentile(sorted, 25),
    median: percentile(sorted, 50),
    q3: percentile(sorted, 75),
    max: sorted[sorted.length - 1],
  };
}

function fmt(v){
  return Number.isInteger(v) ? String(v) : v.toFixed(4).replace(/0+$/, '').replace(/\.$/, '');
}

function renderStats(values){
  const stats = describe(values);
  statsBox.innerHTML = Object.entries(stats).map(
    ([k, v]) => `<tr><td class="k">${k}</td><td>${fmt(v)}</td></tr>`
  ).join('');
  return stats;
}

function repr(arr){
  return '[' + arr.map(v => Number.isInteger(v) ? String(v) : v.toString()).join(', ') + ']';
}

const KIND_CALL = {
  bar: 'plt.bar(np.arange(len(data)), data, color="tab:blue")\nplt.xlabel("index")',
  hist: 'plt.hist(data, bins=bins, color="tab:blue", edgecolor="white")\nplt.xlabel("value")',
  scatter: 'plt.scatter(x, y, color="tab:blue")\nplt.xlabel("x")\nplt.ylabel("y")',
  box: 'plt.boxplot(data, vert=True)\nplt.ylabel("value")',
  line: 'plt.plot(np.arange(len(data)), data, color="tab:blue", marker="o")\nplt.xlabel("index")',
  pie: 'plt.pie(data, labels=labels, autopct="%1.1f%%")',
  violin: 'plt.violinplot(data, vert=True, showmeans=True)\nplt.ylabel("value")',
};

function buildCode(kind, values, xy){
  const lines = ['import numpy as np', 'import matplotlib.pyplot as plt', ''];
  if (kind === 'scatter'){
    lines.push(`x = np.array(${repr(xy.xs)})`);
    lines.push(`y = np.array(${repr(xy.ys)})`);
  } else {
    lines.push(`data = np.array(${repr(values)})`);
    if (kind === 'hist') lines.push('bins = 10');
    if (kind === 'pie') lines.push(`labels = ${JSON.stringify(values.map((_, i) => String(i)))}`);
  }
  lines.push('');
  lines.push('plt.figure(figsize=(7, 4.5))');
  lines.push(KIND_CALL[kind]);
  lines.push(`plt.title("${kind} chart")`);
  lines.push('plt.tight_layout()');
  lines.push('plt.show()');
  return lines.join('\n') + '\n';
}

function clear(){
  ctx.clearRect(0, 0, canvas.width, canvas.height);
}

function axes(padL, padB, w, h){
  ctx.strokeStyle = '#999';
  ctx.beginPath();
  ctx.moveTo(padL, 10); ctx.lineTo(padL, h - padB); ctx.lineTo(w - 10, h - padB);
  ctx.stroke();
}

function drawBar(values){
  const w = canvas.width, h = canvas.height, padL = 40, padB = 30;
  axes(padL, padB, w, h);
  const maxV = Math.max(...values, 0), minV = Math.min(...values, 0);
  const range = (maxV - minV) || 1;
  const barW = (w - padL - 20) / values.length;
  const zeroY = h - padB - ((0 - minV) / range) * (h - padB - 20);
  values.forEach((v, i) => {
    const barH = (v / range) * (h - padB - 20);
    const x = padL + i * barW + 2;
    ctx.fillStyle = '#2563eb';
    ctx.fillRect(x, Math.min(zeroY, zeroY - barH), Math.max(barW - 4, 1), Math.abs(barH));
  });
}

function drawHist(values){
  const bins = 10;
  const w = canvas.width, h = canvas.height, padL = 40, padB = 30;
  axes(padL, padB, w, h);
  const lo = Math.min(...values), hi = Math.max(...values);
  const span = (hi - lo) || 1;
  const counts = new Array(bins).fill(0);
  values.forEach(v => {
    let idx = Math.floor(((v - lo) / span) * bins);
    if (idx >= bins) idx = bins - 1;
    if (idx < 0) idx = 0;
    counts[idx] += 1;
  });
  const maxC = Math.max(...counts, 1);
  const binW = (w - padL - 20) / bins;
  counts.forEach((c, i) => {
    const barH = (c / maxC) * (h - padB - 20);
    ctx.fillStyle = '#2563eb';
    ctx.fillRect(padL + i * binW + 1, h - padB - barH, Math.max(binW - 2, 1), barH);
  });
}

function scale(values, lo, hi, outLo, outHi){
  const span = (hi - lo) || 1;
  return values.map(v => outLo + ((v - lo) / span) * (outHi - outLo));
}

function drawScatter(xy){
  const w = canvas.width, h = canvas.height, padL = 40, padB = 30;
  axes(padL, padB, w, h);
  const xs = scale(xy.xs, Math.min(...xy.xs), Math.max(...xy.xs), padL + 10, w - 20);
  const ys = scale(xy.ys, Math.min(...xy.ys), Math.max(...xy.ys), h - padB - 10, 20);
  ctx.fillStyle = '#2563eb';
  xs.forEach((x, i) => {
    ctx.beginPath();
    ctx.arc(x, ys[i], 4, 0, Math.PI * 2);
    ctx.fill();
  });
}

function drawLine(values){
  const w = canvas.width, h = canvas.height, padL = 40, padB = 30;
  axes(padL, padB, w, h);
  const lo = Math.min(...values), hi = Math.max(...values);
  const ys = scale(values, lo, hi, h - padB - 10, 20);
  const stepX = (w - padL - 30) / Math.max(values.length - 1, 1);
  ctx.strokeStyle = '#2563eb';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ys.forEach((y, i) => {
    const x = padL + 10 + i * stepX;
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();
  ctx.fillStyle = '#1d4ed8';
  ys.forEach((y, i) => {
    ctx.beginPath();
    ctx.arc(padL + 10 + i * stepX, y, 3, 0, Math.PI * 2);
    ctx.fill();
  });
}

function drawBox(values){
  const w = canvas.width, h = canvas.height, padL = 40, padB = 30;
  axes(padL, padB, w, h);
  const stats = describe(values);
  const lo = stats.min, hi = stats.max;
  const [y0, y1, yq1, ym, yq3] = scale(
    [lo, hi, stats.q1, stats.median, stats.q3], lo, hi, h - padB - 10, 20
  );
  const cx = w / 2, boxW = 80;
  ctx.strokeStyle = '#2563eb';
  ctx.lineWidth = 2;
  ctx.beginPath(); ctx.moveTo(cx, y0); ctx.lineTo(cx, yq1); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(cx, yq3); ctx.lineTo(cx, y1); ctx.stroke();
  ctx.strokeRect(cx - boxW / 2, Math.min(yq1, yq3), boxW, Math.abs(yq3 - yq1));
  ctx.beginPath(); ctx.moveTo(cx - boxW / 2, ym); ctx.lineTo(cx + boxW / 2, ym); ctx.stroke();
}

function drawViolin(values){
  // Approximate as a mirrored, smoothed histogram silhouette.
  const bins = 12;
  const w = canvas.width, h = canvas.height, padL = 40, padB = 30;
  axes(padL, padB, w, h);
  const lo = Math.min(...values), hi = Math.max(...values);
  const span = (hi - lo) || 1;
  const counts = new Array(bins).fill(0);
  values.forEach(v => {
    let idx = Math.floor(((v - lo) / span) * bins);
    if (idx >= bins) idx = bins - 1;
    if (idx < 0) idx = 0;
    counts[idx] += 1;
  });
  const maxC = Math.max(...counts, 1);
  const cx = w / 2, maxHalfW = 70;
  const stepY = (h - padB - 30) / (bins - 1 || 1);
  ctx.fillStyle = 'rgba(37,99,235,0.5)';
  ctx.strokeStyle = '#2563eb';
  ctx.beginPath();
  for (let i = 0; i < bins; i++){
    const y = h - padB - i * stepY;
    const halfW = (counts[i] / maxC) * maxHalfW;
    if (i === 0) ctx.moveTo(cx - halfW, y); else ctx.lineTo(cx - halfW, y);
  }
  for (let i = bins - 1; i >= 0; i--){
    const y = h - padB - i * stepY;
    const halfW = (counts[i] / maxC) * maxHalfW;
    ctx.lineTo(cx + halfW, y);
  }
  ctx.closePath();
  ctx.fill();
  ctx.stroke();
}

function drawPie(values){
  const w = canvas.width, h = canvas.height;
  const cx = w / 2, cy = h / 2, r = Math.min(w, h) / 2 - 20;
  const total = values.reduce((s, v) => s + Math.abs(v), 0) || 1;
  const colors = ['#2563eb', '#ea7317', '#16a34a', '#9333ea', '#c0392b',
                  '#0891b2', '#d97706', '#4338ca'];
  let start = -Math.PI / 2;
  values.forEach((v, i) => {
    const slice = (Math.abs(v) / total) * Math.PI * 2;
    ctx.fillStyle = colors[i % colors.length];
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.arc(cx, cy, r, start, start + slice);
    ctx.closePath();
    ctx.fill();
    start += slice;
  });
}

function update(){
  errBox.textContent = '';
  const kind = kindSelect.value;
  clear();
  try {
    if (kind === 'scatter'){
      const xy = parseXY(dataBox.value);
      drawScatter(xy);
      renderStats(xy.xs.concat(xy.ys));
      codeBox.textContent = buildCode(kind, null, xy);
      return;
    }
    const values = parseValues(dataBox.value);
    renderStats(values);
    codeBox.textContent = buildCode(kind, values, null);
    if (kind === 'bar') drawBar(values);
    else if (kind === 'hist') drawHist(values);
    else if (kind === 'line') drawLine(values);
    else if (kind === 'box') drawBox(values);
    else if (kind === 'violin') drawViolin(values);
    else if (kind === 'pie') drawPie(values);
  } catch (e) {
    errBox.textContent = e.message;
  }
}

const SEED_VALUES = [__EXAMPLE_DATA__.trim(), __EXAMPLE_SCATTER__.trim()];

function onKindChange(){
  // Only auto-swap the seed example when the box still holds *a* seed value
  // (never overwrite text the user typed themselves).
  const isSeeded = SEED_VALUES.includes(dataBox.value.trim());
  if (isSeeded) {
    dataBox.value = kindSelect.value === 'scatter' ? __EXAMPLE_SCATTER__ : __EXAMPLE_DATA__;
  }
  update();
}

kindSelect.addEventListener('change', onKindChange);
dataBox.addEventListener('input', update);

document.getElementById('copyBtn').addEventListener('click', () => {
  navigator.clipboard.writeText(codeBox.textContent).then(() => {
    const btn = document.getElementById('copyBtn');
    const old = btn.textContent;
    btn.textContent = 'Copied!';
    setTimeout(() => { btn.textContent = old; }, 1200);
  });
});

update();
"""


def plot_studio_playground(out: str | None = None) -> str:
    """Build the Plot Studio HTML playground: type numbers, get chart + code.

    A textarea accepts numbers separated by commas, whitespace, or newlines
    (for ``scatter``, one ``x,y`` pair per line). A dropdown picks one of the
    seven supported chart kinds. Three panels update live as you type: the
    rendered chart (drawn on a ``<canvas>`` by hand-rolled JavaScript), a
    numpy-style summary-statistics table, and the exact matplotlib+numpy
    source that reproduces the current chart, with a Copy button.

    Fully self-contained: inline CSS and vanilla JavaScript, no CDN, no
    server, works offline. Seeded with example data so it renders on open.
    """
    body = f"""
<h3>Plot Studio</h3>
<p class="sub">Type numbers (comma, space, or newline separated). For
scatter, enter one <code>x,y</code> pair per line. The chart, the numpy
stats, and the matplotlib+numpy code below all update live as you type.</p>
<div class="row">
  <textarea id="dataBox">{_EXAMPLE_DATA}</textarea>
  <label>chart kind:
    <select id="kindSelect">
      <option value="bar" selected>bar</option>
      <option value="hist">hist</option>
      <option value="scatter">scatter</option>
      <option value="box">box</option>
      <option value="line">line</option>
      <option value="pie">pie</option>
      <option value="violin">violin</option>
    </select>
  </label>
</div>
<div id="err"></div>
<div class="grid">
  <div class="panel">
    <h4>Chart</h4>
    <canvas id="chart" width="460" height="320"></canvas>
  </div>
  <div class="panel">
    <h4>numpy summary stats</h4>
    <table class="stats"><tbody id="stats"></tbody></table>
  </div>
  <div class="panel" style="grid-column:1 / -1">
    <h4>matplotlib + numpy code
      <button id="copyBtn" class="primary" style="float:right">Copy</button>
    </h4>
    <pre id="code"></pre>
  </div>
</div>
"""
    script = (
        _PLAYGROUND_SCRIPT.replace("__KINDS__", json.dumps(list(_KINDS)))
        .replace("__EXAMPLE_DATA__", json.dumps(_EXAMPLE_DATA))
        .replace("__EXAMPLE_SCATTER__", json.dumps(_EXAMPLE_SCATTER))
    )
    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>OptimumAI — Plot Studio</title>
<style>{_PLAYGROUND_CSS}</style></head><body>
<h2 class="sr-only">Plot Studio: type numbers to see a live chart, live
numpy summary statistics, and the exact matplotlib and numpy code that
reproduces the chart, with a button to copy the code.</h2>
{body}
<script>
{script}
</script></body></html>
"""
    target = out if out is not None else "plot_studio_playground.html"
    with open(target, "w", encoding="utf-8") as fh:
        fh.write(html)
    return target
