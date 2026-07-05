"""Concept-agnostic D3 + KaTeX renderer for :class:`FlowTrace` objects.

This module never imports anything RAG-specific.  It receives a serialised
:class:`~optimumai.core.flow_trace.FlowTrace` dict plus a layout mapping and
produces a single self-contained HTML file that:

* draws the node/edge graph once with D3
* reveals edges progressively as the step index advances
* renders LaTeX formulas via KaTeX
* shows metrics as a plain table
* accepts keyboard arrow keys and progress-dot clicks

Swap the trace JSON for any other concept (value_iteration, quantization …)
and this file renders it unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from optimumai.core.flow_trace import FlowTrace

# ---------------------------------------------------------------------------
# Default hand-placed layout for the standard RAG graph topology
# (ingestion column → embedding column → index → retrieval → generation).
# For other concepts supply your own layout dict.
# ---------------------------------------------------------------------------

RAG_LAYOUT: dict[str, dict[str, float]] = {
    "doc":          {"x": 30,  "y": 270, "w": 130, "h": 44},
    "chunk_0":      {"x": 210, "y": 100, "w": 140, "h": 40},
    "chunk_1":      {"x": 210, "y": 270, "w": 140, "h": 40},
    "chunk_2":      {"x": 210, "y": 440, "w": 140, "h": 40},
    "chunk_vec_0":  {"x": 400, "y": 100, "w": 130, "h": 40},
    "chunk_vec_1":  {"x": 400, "y": 270, "w": 130, "h": 40},
    "chunk_vec_2":  {"x": 400, "y": 440, "w": 130, "h": 40},
    "query":        {"x": 30,  "y": 555, "w": 130, "h": 40},
    "query_vec":    {"x": 210, "y": 555, "w": 130, "h": 40},
    "index":        {"x": 580, "y": 270, "w": 120, "h": 50},
    "retrieved":    {"x": 750, "y": 185, "w": 140, "h": 40},
    "reranked":     {"x": 750, "y": 350, "w": 140, "h": 40},
    "context":      {"x": 920, "y": 210, "w": 130, "h": 40},
    "llm":          {"x": 920, "y": 335, "w": 130, "h": 44},
    "answer":       {"x": 920, "y": 455, "w": 130, "h": 40},
}

_VB_W = 1090
_VB_H = 630

# ---------------------------------------------------------------------------
# HTML template — renderer knows nothing about RAG
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<title>OptimumAI — {title}</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.9.0/d3.min.js"></script>
<link rel="stylesheet"
      href="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.js">
</script>
<style>
:root{{
  --bg:#0f1117; --panel:#171a23; --line:#2a2f3d; --text:#e7e9ee; --dim:#7d8497;
  --accent:#6ea8fe; --accent2:#ffb86b; --good:#5fd4a0;
  --ingestion:#6ea8fe; --query:#c792ea; --retrieval:#ffb86b; --generation:#5fd4a0;
}}
*{{ box-sizing:border-box; }}
body{{ margin:0; font-family:-apple-system,Segoe UI,Roboto,sans-serif;
       background:var(--bg); color:var(--text); }}
#app{{ display:flex; height:100vh; }}
#graph-pane{{ flex:1 1 63%; position:relative; border-right:1px solid var(--line); }}
#side-pane{{ flex:1 1 37%; padding:22px 24px; overflow-y:auto; }}
h1{{ font-size:13px; font-weight:600; letter-spacing:.06em; color:var(--dim);
     text-transform:uppercase; margin:0 0 4px; }}
h2{{ font-size:20px; margin:0 0 6px; }}
#desc{{ color:var(--dim); font-size:13px; margin-bottom:18px; }}
svg{{ width:100%; height:100%; display:block; }}
.node-box{{ fill:var(--panel); stroke:var(--line); stroke-width:1.4px; }}
.node-box.hl{{ stroke-width:2.4px; filter:drop-shadow(0 0 4px currentColor); }}
.node-label{{ fill:var(--text); font-size:11.5px; font-weight:500;
               pointer-events:none; }}
.node-sub{{ fill:var(--dim); font-size:9.5px; pointer-events:none; }}
.edge{{ fill:none; stroke:var(--line); stroke-width:1.6px; opacity:.18; }}
.edge.active{{ stroke:var(--accent); stroke-width:2.5px; opacity:.95; }}
.edge.past{{ stroke:#3d4356; stroke-width:1.8px; opacity:.55; }}
#controls{{ position:absolute; bottom:14px; left:14px; right:14px;
             display:flex; align-items:center; gap:10px; }}
button{{ background:var(--panel); border:1px solid var(--line); color:var(--text);
         padding:8px 16px; border-radius:8px; cursor:pointer; font-size:13px; }}
button:hover{{ border-color:var(--accent); color:var(--accent); }}
button:disabled{{ opacity:.3; cursor:default; }}
#progress{{ flex:1; display:flex; gap:5px; }}
.dot{{ flex:1; height:5px; border-radius:3px; background:var(--line);
       cursor:pointer; transition:background .15s; }}
.dot.done{{ background:var(--accent); }}
.dot.current{{ background:var(--accent2); }}
#step-count{{ color:var(--dim); font-size:12px; white-space:nowrap; }}
.stage-tag{{ display:inline-block; font-size:10.5px; text-transform:uppercase;
              letter-spacing:.06em; padding:3px 9px; border-radius:5px;
              margin-bottom:10px; }}
.op-tag{{ color:var(--dim); font-size:11px; margin-left:8px; }}
#step-title{{ font-size:17px; font-weight:600; margin:0 0 6px; }}
#narration{{ font-size:14.5px; line-height:1.6; margin:12px 0; color:#cdd0d9; }}
#formula{{ background:var(--panel); border:1px solid var(--line); border-radius:8px;
            padding:14px 18px; margin:14px 0; font-size:15px; overflow-x:auto;
            text-align:center; }}
.metrics-title,.io-title{{ color:var(--dim); font-size:11px; text-transform:uppercase;
                            letter-spacing:.04em; margin-bottom:6px; }}
table.metrics{{ width:100%; border-collapse:collapse; margin:6px 0 18px; font-size:13px; }}
table.metrics td{{ padding:6px 8px; border-bottom:1px solid var(--line); }}
table.metrics td:last-child{{ text-align:right; color:var(--accent);
                               font-variant-numeric:tabular-nums; font-weight:600; }}
.io-block{{ margin:14px 0; }}
.io-item{{ background:var(--panel); border:1px solid var(--line); border-radius:6px;
            padding:9px 11px; margin-bottom:6px; font-size:12.5px; line-height:1.5; }}
.io-item .k{{ color:var(--dim); }}
</style>
</head>
<body>
<div id="app">
  <div id="graph-pane">
    <svg viewBox="0 0 {vb_w} {vb_h}"></svg>
    <div id="controls">
      <button id="prev">&larr; Prev</button>
      <div id="progress"></div>
      <button id="next">Next &rarr;</button>
      <span id="step-count"></span>
    </div>
  </div>
  <div id="side-pane">
    <h1 id="concept-label"></h1>
    <h2 id="trace-title"></h2>
    <div id="desc"></div>
    <div id="stage-badge"></div>
    <div id="step-title"></div>
    <div id="narration"></div>
    <div id="formula" style="display:none;"></div>
    <div id="metrics-block" style="display:none;">
      <div class="metrics-title">Metrics</div>
      <table class="metrics"><tbody id="metrics-body"></tbody></table>
    </div>
    <div class="io-block" id="inputs-block" style="display:none;">
      <div class="io-title">Inputs</div>
      <div id="inputs-list"></div>
    </div>
    <div class="io-block" id="outputs-block" style="display:none;">
      <div class="io-title">Outputs</div>
      <div id="outputs-list"></div>
    </div>
  </div>
</div>

<script>
/* ============================================================
   TRACE — produced by optimumai.core.flow_trace.FlowTrace.to_dict()
   The renderer never reads the word "RAG" or any concept name.
   Swap this object for any other FlowTrace JSON and it works unchanged.
   ============================================================ */
const TRACE = {trace_json};

/* ---- layout: x, y, w, h in viewBox coordinates ---- */
const LAYOUT = {layout_json};

const GROUP_COLOR = {{
  ingestion:"var(--ingestion)", query:"var(--query)",
  retrieval:"var(--retrieval)", generation:"var(--generation)"
}};

const nodeById  = Object.fromEntries(TRACE.nodes.map(n => [n.id, n]));
const stepOrder = TRACE.steps.map(s => s.id);
let current = 0;

const svg       = d3.select("svg");
const edgeLayer = svg.append("g");
const nodeLayer = svg.append("g");

function edgePath(e){{
  const s = LAYOUT[e.source], t = LAYOUT[e.target];
  if (!s || !t) return "";
  const x1=s.x+s.w, y1=s.y+s.h/2, x2=t.x, y2=t.y+t.h/2, mx=(x1+x2)/2;
  return `M${{x1}},${{y1}} C${{mx}},${{y1}} ${{mx}},${{y2}} ${{x2}},${{y2}}`;
}}

/* draw edges once */
const edgeSel = edgeLayer.selectAll("path.edge")
  .data(TRACE.edges).join("path")
  .attr("class","edge")
  .attr("id", d => "edge-"+d.id)
  .attr("d", edgePath);

/* draw nodes once */
const nodeG = nodeLayer.selectAll("g.node")
  .data(TRACE.nodes).join("g")
  .attr("class","node")
  .attr("transform", d => {{
    const l = LAYOUT[d.id];
    return l ? `translate(${{l.x}},${{l.y}})` : "translate(0,0)";
  }});

nodeG.append("rect")
  .attr("class","node-box")
  .attr("id", d => "nodebox-"+d.id)
  .attr("rx", 8)
  .attr("width",  d => (LAYOUT[d.id]||{{w:0}}).w)
  .attr("height", d => (LAYOUT[d.id]||{{h:0}}).h)
  .attr("stroke", d => GROUP_COLOR[d.group] || "var(--line)");

nodeG.append("text").attr("class","node-label")
  .attr("x",10).attr("y", d => (LAYOUT[d.id]||{{h:24}}).h/2 - 2)
  .text(d => d.label);

nodeG.append("text").attr("class","node-sub")
  .attr("x",10).attr("y", d => (LAYOUT[d.id]||{{h:24}}).h/2 + 13)
  .text(d => d.kind);

/* ---- render loop ---- */
function activeStepSet(){{ return new Set(stepOrder.slice(0, current+1)); }}

function render(){{
  const step = TRACE.steps[current];
  const active = activeStepSet();
  const hlN = new Set(step.highlight_nodes);
  const hlE = new Set(step.highlight_edges);

  edgeSel
    .classed("active", d => hlE.has(d.id))
    .classed("past",   d => !hlE.has(d.id) && active.has(d.active_from_step))
    .attr("opacity",   d => active.has(d.active_from_step) ? null : 0.1);

  nodeG.select("rect")
    .classed("hl", d => hlN.has(d.id))
    .attr("opacity", d => {{
      const touched = TRACE.edges.some(
        e => (e.source===d.id||e.target===d.id) && active.has(e.active_from_step)
      );
      return (touched || hlN.has(d.id)) ? 1 : 0.22;
    }})
    .attr("stroke", d => {{
      if (hlN.has(d.id)) return GROUP_COLOR[d.group] || "var(--accent)";
      return GROUP_COLOR[d.group] || "var(--line)";
    }});

  /* side panel */
  const firstNode = nodeById[step.highlight_nodes[0]];
  const gc = firstNode ? (GROUP_COLOR[firstNode.group]||"var(--dim)") : "var(--dim)";

  d3.select("#concept-label").text(TRACE.concept);
  d3.select("#trace-title").text(TRACE.title);
  d3.select("#desc").text(TRACE.description);
  d3.select("#stage-badge").html(
    `<span class="stage-tag"
           style="background:${{gc}}22;color:${{gc}}">${{step.stage}}</span>`+
    `<span class="op-tag">op: ${{step.op}}</span>`
  );
  d3.select("#step-title").text(`${{step.index}}. ${{step.title}}`);
  d3.select("#narration").text(step.narration);

  const fEl = d3.select("#formula");
  if (step.formula){{
    fEl.style("display","block");
    katex.render(step.formula, fEl.node(), {{throwOnError:false, displayMode:true}});
  }} else {{
    fEl.style("display","none");
  }}

  const mEntries = Object.entries(step.metrics||{{}});
  const mBlock = d3.select("#metrics-block");
  d3.select("#metrics-body").selectAll("tr").remove();
  if (mEntries.length){{
    mBlock.style("display","block");
    mEntries.forEach(([k,v]) => {{
      const tr = d3.select("#metrics-body").append("tr");
      tr.append("td").text(k.replace(/_/g," "));
      tr.append("td").text(typeof v==="number" ? v.toFixed(4) : v);
    }});
  }} else {{ mBlock.style("display","none"); }}

  renderIO("#inputs-block","#inputs-list",step.inputs);
  renderIO("#outputs-block","#outputs-list",step.outputs);

  d3.select("#step-count").text(`Step ${{current+1}} / ${{TRACE.steps.length}}`);
  d3.select("#prev").property("disabled", current===0);
  d3.select("#next").property("disabled", current===TRACE.steps.length-1);
  renderDots();
}}

function renderIO(bSel, lSel, items){{
  const block = d3.select(bSel), list = d3.select(lSel);
  list.selectAll("*").remove();
  if (!items||!items.length){{ block.style("display","none"); return; }}
  block.style("display","block");
  items.forEach(it => {{
    list.append("div").attr("class","io-item")
      .html(`<span class="k">${{it.label}}</span> · ${{it.kind}}<br>${{it.preview}}`);
  }});
}}

function renderDots(){{
  const wrap = d3.select("#progress");
  wrap.selectAll("*").remove();
  TRACE.steps.forEach((_,i) => {{
    wrap.append("div")
      .attr("class","dot"+(i<current?" done":"")+(i===current?" current":""))
      .on("click", () => {{ current=i; render(); }});
  }});
}}

d3.select("#prev").on("click", () => {{ if(current>0){{ current--; render(); }} }});
d3.select("#next").on("click", () => {{
  if(current<TRACE.steps.length-1){{ current++; render(); }}
}});
document.addEventListener("keydown", e => {{
  if (e.key==="ArrowRight") d3.select("#next").dispatch("click");
  if (e.key==="ArrowLeft")  d3.select("#prev").dispatch("click");
}});

render();
</script>
</body>
</html>
"""


def render_flow_trace_html(
    trace: FlowTrace,
    layout: dict[str, dict[str, float]],
    out: str | None = None,
    vb_w: int = _VB_W,
    vb_h: int = _VB_H,
) -> str:
    """Render a :class:`~optimumai.core.flow_trace.FlowTrace` to a self-contained
    D3 + KaTeX HTML file.

    Parameters
    ----------
    trace:
        The flow trace to render.  Any concept works — the renderer only reads
        ``nodes``, ``edges``, ``steps``, ``title``, and ``concept``.
    layout:
        Mapping from ``node.id`` → ``{x, y, w, h}`` in SVG viewBox coordinates.
        Use :data:`RAG_LAYOUT` for the standard RAG topology.
    out:
        If given, write the HTML to this path and return the path string.
        Otherwise return the HTML string directly.
    vb_w, vb_h:
        SVG viewBox width and height (default 1090 × 630).

    Returns
    -------
    str
        The output file path (if ``out`` was given) or the raw HTML.

    Notes
    -----
    The rendered file requires an internet connection to load D3 v7 and KaTeX
    from their respective CDNs.  For fully offline use, host the JS/CSS locally
    and replace the CDN ``<script>``/``<link>`` tags.
    """
    trace_json = json.dumps(trace.to_dict(), indent=2, ensure_ascii=False)
    layout_json = json.dumps(layout, indent=2)

    html = _HTML_TEMPLATE.format(
        title=trace.title,
        vb_w=vb_w,
        vb_h=vb_h,
        trace_json=trace_json,
        layout_json=layout_json,
    )

    if out:
        Path(out).write_text(html, encoding="utf-8")
        return out
    return html
