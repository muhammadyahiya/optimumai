"""Shared page chrome for the ``flows`` subpackage.

Every ``*_flow`` builder in this subpackage follows the same self-contained
pattern established in ``optimumai.visualization.playgrounds`` and
``optimumai.circuit.interactive``: compute real numbers once in Python
(seeded, deterministic), embed them as JSON via :func:`json.dumps`, and hand
the browser a single vanilla-JS renderer that draws inline SVG. No CDN, no
server, no build step, no runtime Python dependency — open the file offline.

What makes a *flow* different from a plain playground is the interaction
model: every flow is a pipeline of named stages (tokenize -> embed -> ... ->
output), rendered as a left-to-right (or top-to-bottom) row of SVG boxes
joined by arrows. A "Step" control advances a single active stage at a time,
dimming everything not yet reached and highlighting the current one, while a
plain-language caption underneath explains what just happened. Hovering a
matrix/vector cell anywhere pops a tooltip with the exact value and what it
means — the same hover-to-inspect affordance transformer-explainer made
famous.

This module only owns shared CSS/JS/HTML scaffolding; each flow file supplies
its own stage list, data, and captions.
"""

from __future__ import annotations

import json
from typing import Any

# --------------------------------------------------------------------------
# shared CSS
# --------------------------------------------------------------------------

FLOW_CSS = """
 body{font-family:system-ui,sans-serif;margin:0;padding:20px;color:#111;
      background:#fafafa}
 h2.sr-only{position:absolute;left:-9999px}
 h1{font-size:20px;margin:0 0 4px}
 h3{margin:0 0 6px}
 .sub{color:#555;margin:0 0 14px;max-width:760px;line-height:1.5}
 .row{display:flex;align-items:center;gap:10px;margin:8px 0;flex-wrap:wrap}
 button{font-family:inherit;font-size:14px;padding:6px 14px;border-radius:6px;
        border:1px solid #ccc;background:#fff;cursor:pointer}
 button:hover{background:#f0f0f0}
 button:disabled{opacity:.4;cursor:default}
 button.primary{background:#2563eb;color:#fff;border-color:#2563eb}
 button.primary:hover{background:#1d4ed8}
 .stat{font-family:monospace;color:#2563eb;font-weight:700}
 .legend{color:#555;font-size:13px;max-width:760px;margin-top:10px}
 #stage-caption{background:#eef2ff;border:1px solid #c7d2fe;border-radius:8px;
        padding:12px 16px;margin:14px 0;max-width:760px;min-height:24px;
        line-height:1.5}
 #stage-caption b{color:#1d4ed8}
 #stage-title{font-family:monospace;font-weight:700;color:#1d4ed8;margin:0 0 4px}
 #flow-wrap{width:100%;overflow-x:auto;border:1px solid #e5e7eb;border-radius:8px;
        background:#fff;padding:6px}
 #tooltip{position:fixed;pointer-events:none;background:#111;color:#fff;
        font-family:monospace;font-size:12px;padding:6px 9px;border-radius:6px;
        max-width:320px;line-height:1.4;opacity:0;transition:opacity .08s;
        z-index:10}
 .flow-stage-box{fill:#fff;stroke:#c7d2fe;stroke-width:1.5}
 .flow-stage-box.active{stroke:#2563eb;stroke-width:2.5}
 .flow-stage-box.done{stroke:#93c5fd}
 .flow-stage-label{font-family:monospace;font-size:12px;fill:#111}
 .flow-stage-label.dim{fill:#aaa}
 .flow-arrow{stroke:#94a3b8;stroke-width:2;marker-end:url(#arrowhead)}
 .flow-arrow.active{stroke:#2563eb;stroke-width:2.5}
 .cell{cursor:pointer}
 .cell:hover{stroke:#111;stroke-width:2}
 .cell-text{font-family:monospace;pointer-events:none;text-anchor:middle;
        dominant-baseline:middle}
"""

# --------------------------------------------------------------------------
# shared JS: stage stepper + tooltip
# --------------------------------------------------------------------------

FLOW_RUNTIME_JS = r"""
const STAGES = __STAGES__;             // [{id, title, caption}, ...]
let stageIdx = 0;

function tip(el, text){
  const box = document.getElementById('tooltip');
  el.addEventListener('mousemove', (e) => {
    box.style.left = (e.clientX + 14) + 'px';
    box.style.top = (e.clientY + 14) + 'px';
    box.style.opacity = '1';
    box.textContent = text;
  });
  el.addEventListener('mouseleave', () => { box.style.opacity = '0'; });
}

function setStage(i){
  stageIdx = Math.max(0, Math.min(STAGES.length - 1, i));
  STAGES.forEach((s, k) => {
    const grp = document.getElementById('stage-' + s.id);
    if (!grp) return;
    grp.classList.toggle('reached', k <= stageIdx);
    grp.classList.toggle('active-stage', k === stageIdx);
    grp.style.opacity = (k <= stageIdx) ? '1' : '0.28';
    const box = grp.querySelector('.flow-stage-box');
    if (box){
      box.classList.toggle('active', k === stageIdx);
      box.classList.toggle('done', k < stageIdx);
    }
    grp.querySelectorAll('.flow-stage-label').forEach((lbl) => {
      lbl.classList.toggle('dim', k > stageIdx);
    });
  });
  document.querySelectorAll('.flow-arrow').forEach((arrow) => {
    const upto = parseInt(arrow.dataset.after, 10);
    arrow.classList.toggle('active', upto <= stageIdx);
  });
  document.getElementById('stage-title').textContent =
    `Stage ${stageIdx + 1} / ${STAGES.length}: ${STAGES[stageIdx].title}`;
  document.getElementById('stage-caption').innerHTML = STAGES[stageIdx].caption;
  document.getElementById('prevBtn').disabled = stageIdx === 0;
  document.getElementById('nextBtn').disabled = stageIdx === STAGES.length - 1;
  document.getElementById('progress').textContent =
    `${stageIdx + 1} / ${STAGES.length}`;
}

document.getElementById('prevBtn').addEventListener('click', () => setStage(stageIdx - 1));
document.getElementById('nextBtn').addEventListener('click', () => setStage(stageIdx + 1));
document.getElementById('resetBtn').addEventListener('click', () => setStage(0));
document.getElementById('playBtn').addEventListener('click', () => {
  if (stageIdx >= STAGES.length - 1){ setStage(0); }
  const id = setInterval(() => {
    if (stageIdx >= STAGES.length - 1){ clearInterval(id); return; }
    setStage(stageIdx + 1);
  }, 1100);
});

setStage(0);
"""


def flow_controls_html() -> str:
    """The Step ▶ control row + stage title/caption placeholders, shared by every flow."""
    return """
<div class="row">
  <button id="resetBtn">&#8634; Reset</button>
  <button id="prevBtn">&larr; Prev</button>
  <button class="primary" id="nextBtn">Next &rarr;</button>
  <button id="playBtn">&#9654; Play</button>
  <span class="stat" id="progress">1 / 1</span>
</div>
<div id="stage-caption">
  <div id="stage-title"></div>
  <div id="stage-caption-text"></div>
</div>
"""


def page(
    title: str,
    heading_sr: str,
    body: str,
    script: str,
    css_extra: str = "",
) -> str:
    """Wrap ``body``/``script`` in the standard OptimumAI flow page shell."""
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>{FLOW_CSS}
{css_extra}
</style></head><body>
<h2 class="sr-only">{heading_sr}</h2>
{body}
<div id="tooltip"></div>
<script>
{script}
</script></body></html>
"""


def stages_json(stages: list[dict[str, Any]]) -> str:
    """Serialize a list of ``{"id", "title", "caption"}`` stage dicts to JSON."""
    return json.dumps(stages)


def runtime_script(extra_js: str, stages: list[dict[str, Any]]) -> str:
    """Compose a flow's own drawing JS with the shared stepper/tooltip runtime.

    ``extra_js`` should define the SVG-drawing logic and any data constants; the
    shared runtime (stage stepping, tooltips, play/pause) is appended after it so
    it can rely on the DOM the drawing code just built.
    """
    runtime = FLOW_RUNTIME_JS.replace("__STAGES__", stages_json(stages))
    return f"{extra_js}\n{runtime}"


def write(html: str, out: str | None, default_name: str) -> str:
    """Write ``html`` to ``out`` (defaulting to ``default_name``) and return the path."""
    target = out if out is not None else default_name
    with open(target, "w", encoding="utf-8") as fh:
        fh.write(html)
    return target


def svg_open(width: int, height: int, extra_defs: str = "") -> str:
    """Open an SVG tag with the standard arrowhead marker defined."""
    return f"""<svg id="flow-svg" viewBox="0 0 {width} {height}"
     xmlns="http://www.w3.org/2000/svg" font-family="monospace">
<defs>
<marker id="arrowhead" markerWidth="8" markerHeight="8" refX="7" refY="4"
        orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="#94a3b8"/></marker>
{extra_defs}
</defs>
"""


def heat_color(value: float, lo: float = 0.0, hi: float = 1.0) -> str:
    """Map ``value`` in ``[lo, hi]`` to a white->blue heatmap RGB string (Python side)."""
    span = (hi - lo) or 1.0
    p = max(0.0, min(1.0, (value - lo) / span))
    r = round(255 - p * (255 - 37))
    g = round(255 - p * (255 - 99))
    b = round(255 - p * (255 - 235))
    return f"rgb({r},{g},{b})"


def text_color_for(value: float, lo: float = 0.0, hi: float = 1.0) -> str:
    """Pick black/white text so it stays legible against :func:`heat_color`."""
    span = (hi - lo) or 1.0
    p = max(0.0, min(1.0, (value - lo) / span))
    return "#fff" if p > 0.6 else "#111"


def stage_group_open(stage_id: str, x: float, y: float) -> str:
    """Open a ``<g>`` for one pipeline stage, positioned at ``(x, y)``."""
    return f'<g id="stage-{stage_id}" class="flow-stage" transform="translate({x},{y})">'


def stage_group_close() -> str:
    return "</g>"


def stage_box(width: float, height: float, label: str, sublabel: str = "") -> str:
    """A titled rounded-rect box for one stage header (drawn inside a stage group)."""
    parts = [
        f'<rect class="flow-stage-box" width="{width}" height="{height}" rx="10"/>',
        f'<text class="flow-stage-label" x="{width / 2}" y="18" '
        f'text-anchor="middle" font-weight="700">{label}</text>',
    ]
    if sublabel:
        parts.append(
            f'<text class="flow-stage-label dim" x="{width / 2}" y="33" '
            f'text-anchor="middle" font-size="10">{sublabel}</text>'
        )
    return "\n".join(parts)


def arrow(x1: float, y1: float, x2: float, y2: float, after_stage_idx: int) -> str:
    """A straight connector between two stages, lit up once ``after_stage_idx`` is reached."""
    return (
        f'<line class="flow-arrow" data-after="{after_stage_idx}" '
        f'x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}"/>'
    )


def matrix_grid(
    values: list[list[float]],
    x: float,
    y: float,
    cell: float = 30,
    row_labels: list[str] | None = None,
    col_labels: list[str] | None = None,
    lo: float | None = None,
    hi: float | None = None,
    tooltip_fn: Any = None,
    id_prefix: str = "cell",
    decimals: int = 2,
) -> str:
    """Render a numeric matrix as a heatmap grid of hoverable SVG cells.

    ``tooltip_fn(i, j, value) -> str`` builds the tooltip text for cell (i, j);
    if omitted a generic "row i, col j = value" tooltip is used. Returns an SVG
    fragment (no wrapping ``<g>``); callers should already be inside a group
    translated to ``(x, y)`` or pass the absolute offset here.
    """
    flat = [v for row in values for v in row]
    lo = min(flat) if lo is None else lo
    hi = max(flat) if hi is None else hi
    if hi <= lo:
        hi = lo + 1.0
    label_w = 26 if row_labels else 0
    label_h = 16 if col_labels else 0
    out: list[str] = []
    if col_labels:
        for j, cl in enumerate(col_labels):
            cx = x + label_w + j * cell + cell / 2
            out.append(
                f'<text class="cell-text" x="{cx}" y="{y + label_h - 4}" '
                f'font-size="10" fill="#555">{cl}</text>'
            )
    for i, row in enumerate(values):
        if row_labels:
            ry = y + label_h + i * cell + cell / 2
            out.append(
                f'<text class="cell-text" x="{x + label_w - 6}" y="{ry}" '
                f'font-size="10" fill="#555" text-anchor="end">{row_labels[i]}</text>'
            )
        for j, val in enumerate(row):
            cx = x + label_w + j * cell
            cy = y + label_h + i * cell
            color = heat_color(val, lo, hi)
            fg = text_color_for(val, lo, hi)
            tip_text = tooltip_fn(i, j, val) if tooltip_fn else f"[{i},{j}] = {val:.4f}"
            out.append(
                f'<g class="cell" data-tip="{_esc(tip_text)}" '
                f'id="{id_prefix}-{i}-{j}">'
                f'<rect x="{cx}" y="{cy}" width="{cell}" height="{cell}" '
                f'fill="{color}" stroke="#eee"/>'
                f'<text class="cell-text" x="{cx + cell / 2}" y="{cy + cell / 2}" '
                f'font-size="9.5" fill="{fg}">{val:.{decimals}f}</text>'
                f"</g>"
            )
    return "\n".join(out)


def vector_chip(
    values: list[float],
    x: float,
    y: float,
    cell: float = 26,
    vertical: bool = False,
    tooltip_fn: Any = None,
    id_prefix: str = "vchip",
    lo: float | None = None,
    hi: float | None = None,
    decimals: int = 2,
) -> str:
    """Render a 1-D vector as a strip of hoverable colored cells."""
    lo = min(values) if lo is None else lo
    hi = max(values) if hi is None else hi
    if hi <= lo:
        hi = lo + 1.0
    out: list[str] = []
    for i, val in enumerate(values):
        cx = x if vertical else x + i * cell
        cy = y + i * cell if vertical else y
        color = heat_color(val, lo, hi)
        fg = text_color_for(val, lo, hi)
        tip_text = tooltip_fn(i, val) if tooltip_fn else f"[{i}] = {val:.4f}"
        out.append(
            f'<g class="cell" data-tip="{_esc(tip_text)}" id="{id_prefix}-{i}">'
            f'<rect x="{cx}" y="{cy}" width="{cell}" height="{cell}" '
            f'fill="{color}" stroke="#eee"/>'
            f'<text class="cell-text" x="{cx + cell / 2}" y="{cy + cell / 2}" '
            f'font-size="9.5" fill="{fg}">{val:.{decimals}f}</text>'
            f"</g>"
        )
    return "\n".join(out)


def _esc(text: str) -> str:
    """Escape a string for safe embedding inside an SVG ``data-tip`` attribute."""
    return (
        text.replace("&", "&amp;")
        .replace('"', "&quot;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


HOVER_TOOLTIP_JS = r"""
document.querySelectorAll('.cell[data-tip]').forEach((el) => {
  const box = document.getElementById('tooltip');
  el.addEventListener('mousemove', (e) => {
    box.style.left = (e.clientX + 14) + 'px';
    box.style.top = (e.clientY + 14) + 'px';
    box.style.opacity = '1';
    box.textContent = el.getAttribute('data-tip');
  });
  el.addEventListener('mouseleave', () => { box.style.opacity = '0'; });
});
"""
