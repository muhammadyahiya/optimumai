"""DAG-style concept explainers — formula + Python code, side by side.

Each concept in :data:`CONCEPTS` is a small computation graph (nodes + edges)
walked one step at a time. Every step carries a plain-language narration, a
KaTeX-rendered formula, and a real, runnable ``optimumai`` code snippet for
that exact operation — so the graph, the math, and the code stay in sync as
you step through.

    >>> from optimumai.visualization.explain import explain
    >>> explain("attention")               # doctest: +SKIP
    'explain_attention.html'

The page is a single self-contained HTML file (D3 + dagre + KaTeX from a
CDN; no server, no build step) — open it in any browser online. Use
:func:`explore_concepts` to build a searchable landing page linking to every
concept, and :func:`list_explain_concepts` to see what's available.
"""

from __future__ import annotations

import json
import webbrowser
from pathlib import Path

# --------------------------------------------------------------------------
# shared page chrome (dark DAG theme, matching optimumai.circuit.interactive)
# --------------------------------------------------------------------------

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<title>OptimumAI — __TITLE__</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.9.0/d3.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/dagre/0.8.5/dagre.min.js"></script>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.js"></script>
<style>
  :root{
    --bg:#0f1117; --panel:#171a23; --line:#2a2f3d; --text:#e7e9ee; --dim:#7d8497;
    --accent:#6ea8fe; --accent2:#ffb86b;
    --params:#6ea8fe; --grad:#ff8b8b; --optim:#5fd4a0; --branch:#c792ea;
    --ingestion:#6ea8fe; --query:#c792ea; --retrieval:#ffb86b; --generation:#5fd4a0;
  }
  * { box-sizing:border-box; }
  body{ margin:0; font-family:-apple-system,Segoe UI,Roboto,sans-serif; background:var(--bg); color:var(--text); }
  #app{ display:flex; height:100vh; }
  #graph-pane{ flex:1 1 60%; position:relative; border-right:1px solid var(--line); overflow:hidden; }
  #side-pane{ flex:1 1 40%; padding:20px 22px; overflow-y:auto; }
  h1{ font-size:14px; font-weight:600; letter-spacing:.02em; color:var(--dim); text-transform:uppercase; margin:0 0 4px; }
  h2{ font-size:19px; margin:0 0 6px; }
  #desc{ color:var(--dim); font-size:13px; margin-bottom:16px; }
  svg{ width:100%; height:100%; display:block; }
  .node-box{ fill:var(--panel); stroke:var(--line); stroke-width:1.3px; rx:8; }
  .node-box.active{ stroke-width:2.2px; }
  .node-label{ fill:var(--text); font-size:11.5px; font-weight:500; pointer-events:none; }
  .node-sub{ fill:var(--dim); font-size:9px; pointer-events:none; }
  .edge{ fill:none; stroke:var(--line); stroke-width:1.6px; opacity:.35; }
  .edge.active{ stroke:var(--accent); stroke-width:2.4px; opacity:.95; }
  .edge.past{ stroke:#3d4356; opacity:.6; }
  #controls{ position:absolute; bottom:16px; left:16px; right:16px; display:flex; align-items:center; gap:10px; }
  button{ background:var(--panel); border:1px solid var(--line); color:var(--text); padding:8px 14px; border-radius:8px; cursor:pointer; font-size:13px; }
  button:hover{ border-color:var(--accent); }
  button:disabled{ opacity:.35; cursor:default; }
  #progress{ flex:1; display:flex; gap:5px; }
  .dot{ flex:1; height:5px; border-radius:3px; background:var(--line); cursor:pointer; }
  .dot.done{ background:var(--accent); }
  .dot.current{ background:var(--accent2); }
  .stage-tag{ display:inline-block; font-size:10.5px; text-transform:uppercase; letter-spacing:.04em; padding:3px 8px; border-radius:5px; margin-bottom:10px; }
  .op-tag{ color:var(--dim); font-size:11px; margin-left:8px; }
  #narration{ font-size:14px; line-height:1.55; margin:12px 0; }
  #formula{ background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:12px 14px; margin:12px 0; font-size:15px; overflow-x:auto; }
  #code-block{ background:#0d1117; border:1px solid var(--line); border-radius:8px; padding:12px 14px; margin:12px 0; }
  .code-title{ color:var(--dim); font-size:10px; text-transform:uppercase; letter-spacing:.04em; margin-bottom:8px; }
  #code-pre{ margin:0; white-space:pre-wrap; word-break:break-word; font-family:'Cascadia Code','Fira Code',monospace; font-size:12px; color:#e7e9ee; line-height:1.55; overflow-x:auto; }
  table.metrics{ width:100%; border-collapse:collapse; margin:8px 0 16px; font-size:13px; }
  table.metrics td{ padding:5px 8px; border-bottom:1px solid var(--line); }
  table.metrics td:last-child{ text-align:right; color:var(--accent); font-variant-numeric:tabular-nums; }
  .io-block{ margin:12px 0; }
  .io-title{ color:var(--dim); font-size:11px; text-transform:uppercase; letter-spacing:.03em; margin-bottom:6px; }
  #step-count{ color:var(--dim); font-size:12px; white-space:nowrap; }
  #zoomhint{ position:absolute; top:12px; right:16px; color:var(--dim); font-size:11px; }
</style>
</head>
<body>
<div id="app">
  <div id="graph-pane">
    <svg></svg>
    <div id="zoomhint">scroll to zoom &middot; drag to pan</div>
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
    <h2 id="step-title" style="font-size:17px;"></h2>
    <div id="narration"></div>
    <div id="formula" style="display:none;"></div>
    <div id="code-block" style="display:none;">
      <div class="code-title">Python &middot; optimumai</div>
      <pre id="code-pre"></pre>
    </div>
    <div id="metrics-block" style="display:none;">
      <div class="io-title">Metrics</div>
      <table class="metrics"><tbody id="metrics-body"></tbody></table>
    </div>
  </div>
</div>

<script>
const TRACE = __TRACE_JSON__;
const GROUP_COLOR = {
  ingestion:"var(--ingestion)", query:"var(--query)", retrieval:"var(--retrieval)",
  generation:"var(--generation)", params:"var(--params)", grad:"var(--grad)",
  optim:"var(--optim)", branch:"var(--branch)"
};
const nodeById = Object.fromEntries(TRACE.nodes.map(n => [n.id, n]));
const stepIndexOrder = TRACE.steps.map(s => s.id);

const g = new dagre.graphlib.Graph();
g.setGraph({ rankdir: "LR", nodesep: 34, ranksep: 90, marginx: 30, marginy: 30 });
g.setDefaultEdgeLabel(() => ({}));
function estWidth(label){ return Math.max(110, label.length * 7 + 24); }
TRACE.nodes.forEach(n => g.setNode(n.id, { width: estWidth(n.label), height: 44, label: n.label }));
TRACE.edges.forEach(e => g.setEdge(e.source, e.target));
dagre.layout(g);

const GW = g.graph().width, GH = g.graph().height;
const svg = d3.select("svg").attr("viewBox", `0 0 ${GW} ${GH}`);
const zoomLayer = svg.append("g");
svg.call(d3.zoom().scaleExtent([0.4, 2.5]).on("zoom", (ev) => zoomLayer.attr("transform", ev.transform)));

const edgeLayer = zoomLayer.append("g").attr("class","edge-layer");
const nodeLayer = zoomLayer.append("g").attr("class","node-layer");
const line = d3.line().x(d => d.x).y(d => d.y).curve(d3.curveBasis);

const edgeSel = edgeLayer.selectAll("path.edge")
  .data(TRACE.edges).join("path").attr("class","edge")
  .attr("id", d => "edge-"+d.id)
  .attr("d", d => line(g.edge(d.source, d.target).points));

const nodeG = nodeLayer.selectAll("g.node")
  .data(TRACE.nodes).join("g").attr("class","node")
  .attr("transform", d => { const gn = g.node(d.id); return `translate(${gn.x - gn.width/2},${gn.y - gn.height/2})`; });

nodeG.append("rect").attr("class","node-box").attr("id", d => "nodebox-"+d.id)
  .attr("width", d => g.node(d.id).width).attr("height", d => g.node(d.id).height)
  .attr("stroke", d => GROUP_COLOR[d.group] || "var(--line)");
nodeG.append("text").attr("class","node-label").attr("x", 10).attr("y", 18).text(d => d.label);
nodeG.append("text").attr("class","node-sub").attr("x", 10).attr("y", 32).text(d => d.kind);

let current = 0;
function stepsUpToCurrent(){ return new Set(stepIndexOrder.slice(0, current+1)); }

function render(){
  const step = TRACE.steps[current];
  const activeSteps = stepsUpToCurrent();
  const hlNodes = new Set(step.highlight_nodes || []);
  const hlEdges = new Set(step.highlight_edges || []);

  edgeSel.classed("active", d => hlEdges.has(d.id))
    .classed("past", d => !hlEdges.has(d.id) && activeSteps.has(d.active_from_step))
    .attr("opacity", d => activeSteps.has(d.active_from_step) ? null : 0.12);

  nodeG.select("rect").classed("active", d => hlNodes.has(d.id))
    .attr("opacity", d => {
      const everTouched = TRACE.edges.some(e => (e.source===d.id||e.target===d.id) && activeSteps.has(e.active_from_step));
      return (everTouched || hlNodes.has(d.id)) ? 1 : 0.25;
    });

  d3.select("#concept-label").text(TRACE.concept);
  d3.select("#trace-title").text(TRACE.title);
  d3.select("#desc").text(TRACE.description);
  const firstHl = step.highlight_nodes && step.highlight_nodes[0];
  const grp = firstHl && nodeById[firstHl] ? nodeById[firstHl].group : null;
  d3.select("#stage-badge").html(
    `<span class="stage-tag" style="background:${GROUP_COLOR[grp]||'var(--panel)'}22;color:${GROUP_COLOR[grp]||'var(--dim)'}">${step.stage}</span><span class="op-tag">op: ${step.op}</span>`
  );
  d3.select("#step-title").text(`${step.index}. ${step.title}`);
  d3.select("#narration").text(step.narration);

  const formulaEl = d3.select("#formula");
  if (step.formula){
    formulaEl.style("display","block");
    katex.render(step.formula, formulaEl.node(), {throwOnError:false, displayMode:true});
  } else { formulaEl.style("display","none"); }

  const codeBlock = d3.select("#code-block");
  const codePre = d3.select("#code-pre");
  if (step.code){ codeBlock.style("display","block"); codePre.text(step.code); }
  else { codeBlock.style("display","none"); }

  const metricsBlock = d3.select("#metrics-block");
  const metricsBody = d3.select("#metrics-body");
  metricsBody.selectAll("tr").remove();
  const metricEntries = Object.entries(step.metrics || {});
  if (metricEntries.length){
    metricsBlock.style("display","block");
    metricEntries.forEach(([k,v]) => {
      const tr = metricsBody.append("tr");
      tr.append("td").text(k);
      tr.append("td").text(typeof v === "number" ? v.toFixed(4) : v);
    });
  } else { metricsBlock.style("display","none"); }

  d3.select("#step-count").text(`Step ${current+1} / ${TRACE.steps.length}`);
  d3.select("#prev").property("disabled", current === 0);
  d3.select("#next").property("disabled", current === TRACE.steps.length-1);
  renderProgress();
}

function renderProgress(){
  const wrap = d3.select("#progress"); wrap.selectAll("*").remove();
  TRACE.steps.forEach((s,i) => {
    wrap.append("div").attr("class", "dot" + (i < current ? " done" : "") + (i === current ? " current" : ""))
      .on("click", () => { current = i; render(); });
  });
}

d3.select("#prev").on("click", () => { if (current>0){ current--; render(); }});
d3.select("#next").on("click", () => { if (current<TRACE.steps.length-1){ current++; render(); }});
document.addEventListener("keydown", (e) => {
  if (e.key === "ArrowRight") d3.select("#next").dispatch("click");
  if (e.key === "ArrowLeft") d3.select("#prev").dispatch("click");
});
render();
</script>
</body>
</html>
"""

EXPLORE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>OptimumAI — Concept Explorer</title>
<style>
  :root{ --bg:#0f1117; --panel:#171a23; --line:#2a2f3d; --text:#e7e9ee; --dim:#7d8497; --accent:#6ea8fe; }
  * { box-sizing:border-box; }
  body{ margin:0; font-family:-apple-system,Segoe UI,Roboto,sans-serif; background:var(--bg); color:var(--text); padding:32px; }
  h1{ font-size:24px; margin:0 0 6px; }
  .subtitle{ color:var(--dim); margin-bottom:24px; }
  #search{ background:var(--panel); border:1px solid var(--line); color:var(--text); padding:10px 14px; border-radius:8px; width:100%; max-width:400px; font-size:14px; margin-bottom:24px; }
  #search:focus{ outline:none; border-color:var(--accent); }
  #grid{ display:grid; grid-template-columns:repeat(auto-fill, minmax(280px, 1fr)); gap:16px; }
  .card{ background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:20px; }
  .card-title{ font-size:15px; font-weight:600; margin:0 0 8px; color:var(--accent); }
  .card-desc{ color:var(--dim); font-size:13px; line-height:1.45; margin:0 0 16px; }
  .card-btn{ display:inline-block; background:var(--accent); color:#0f1117; padding:7px 16px; border-radius:7px; font-size:13px; font-weight:600; text-decoration:none; }
  .card-btn:hover{ opacity:.85; }
  .card.hidden{ display:none; }
  .count{ color:var(--dim); font-size:12px; margin-bottom:12px; }
</style>
</head>
<body>
<h1>OptimumAI — Concept Explorer</h1>
<p class="subtitle">__COUNT__ interactive AI/ML concept explainers — formula + code + interactive DAG graph. Click a card to launch it (generates the file alongside this page the first time via <code>optimumai explain &lt;concept&gt;</code>, or open directly if already built).</p>
<input id="search" placeholder="Search concepts..." autofocus />
<div class="count" id="count"></div>
<div id="grid"></div>
<script>
const CONCEPTS = __CONCEPTS_JSON__;
const grid = document.getElementById('grid');
CONCEPTS.forEach(c => {
  const card = document.createElement('div');
  card.className = 'card';
  card.dataset.title = (c.title + ' ' + c.key + ' ' + c.description).toLowerCase();
  card.innerHTML = `<div class="card-title">${c.title}</div><div class="card-desc">${c.description}</div><a class="card-btn" href="explain_${c.key}.html">Launch &rarr;</a>`;
  grid.appendChild(card);
});
function updateCount(){
  const visible = document.querySelectorAll('.card:not(.hidden)').length;
  document.getElementById('count').textContent = `${visible} / ${CONCEPTS.length} concepts`;
}
document.getElementById('search').addEventListener('input', (e) => {
  const q = e.target.value.toLowerCase();
  document.querySelectorAll('.card').forEach(card => {
    card.classList.toggle('hidden', !card.dataset.title.includes(q));
  });
  updateCount();
});
updateCount();
</script>
</body>
</html>
"""


# --------------------------------------------------------------------------
# concept registry — 30 foundational AI/ML concepts
# --------------------------------------------------------------------------


def _steps(*entries: dict) -> list[dict]:
    """Fill in id/index/highlight defaults for a concept's step list."""
    out = []
    for i, e in enumerate(entries, 1):
        step = {
            "id": f"s{i}",
            "index": i,
            "stage": e.get("stage", e["title"].split()[0].lower()),
            "title": e["title"],
            "op": e.get("op", "step"),
            "narration": e["narration"],
            "formula": e.get("formula"),
            "code": e.get("code"),
            "inputs": [],
            "outputs": [],
            "highlight_nodes": e.get("highlight_nodes", []),
            "highlight_edges": e.get("highlight_edges", []),
            "metrics": e.get("metrics", {}),
            "duration_hint_ms": 900,
            "chart": None,
        }
        out.append(step)
    return out


def _nodes(*entries: tuple) -> list[dict]:
    """Build node dicts from ``(id, label, kind, group)`` tuples."""
    return [
        {"id": nid, "label": label, "kind": kind, "group": group, "meta": {}}
        for nid, label, kind, group in entries
    ]


def _edges(*entries: tuple) -> list[dict]:
    """Build edge dicts from ``(id, source, target, active_from_step)`` tuples."""
    return [
        {"id": eid, "source": src, "target": tgt, "active_from_step": step, "label": None, "weight": None}
        for eid, src, tgt, step in entries
    ]


CONCEPTS: dict[str, dict] = {}

# --------------------------------------------------------------- 1. attention
CONCEPTS["attention"] = {
    "title": "Scaled Dot-Product Attention",
    "description": "Q, K, V → QKᵀ → scale → softmax → weighted sum of V. The core op inside every transformer.",
    "nodes": _nodes(
        ("q", "Query vector", "vector", "params"),
        ("k", "Key vectors (3)", "matrix", "params"),
        ("v", "Value vectors (3)", "matrix", "params"),
        ("scores", "QK^T scores", "vector", "branch"),
        ("scaled", "Scaled scores", "vector", "branch"),
        ("weights", "Softmax weights", "vector", "optim"),
        ("output", "Attention output", "vector", "generation"),
    ),
    "edges": _edges(
        ("e_q_scores", "q", "scores", "s1"),
        ("e_k_scores", "k", "scores", "s1"),
        ("e_scores_scaled", "scores", "scaled", "s2"),
        ("e_scaled_weights", "scaled", "weights", "s3"),
        ("e_weights_out", "weights", "output", "s4"),
        ("e_v_out", "v", "output", "s4"),
    ),
    "steps": _steps(
        {
            "title": "Compute QK^T similarity scores", "op": "matmul",
            "narration": "Each key is compared against the query via dot product — a higher score means that key/value pair is more relevant to this query.",
            "formula": r"\text{scores} = QK^T",
            "code": "from optimumai import Attention\nattn = Attention.demo(seed=0)\nscores = attn.scores  # shape (T, T)",
            "highlight_nodes": ["q", "k", "scores"], "highlight_edges": ["e_q_scores", "e_k_scores"],
        },
        {
            "title": "Scale by sqrt(d_k)", "op": "scale",
            "narration": "Dividing by sqrt(dimension) prevents dot products from growing too large as dimensionality increases, which would otherwise push softmax into a near-one-hot, gradient-killing regime.",
            "formula": r"\text{scaled} = \frac{QK^T}{\sqrt{d_k}}",
            "code": "import numpy as np\nd_k = 8\nscaled = scores / np.sqrt(d_k)",
            "highlight_nodes": ["scaled"], "highlight_edges": ["e_scores_scaled"],
        },
        {
            "title": "Softmax turns scores into weights", "op": "softmax",
            "narration": "Softmax converts the scaled scores into a probability distribution over the keys — these become the attention weights.",
            "formula": r"\alpha_i = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)_i",
            "code": "from optimumai import softmax\nweights = softmax(scaled.tolist())",
            "highlight_nodes": ["weights"], "highlight_edges": ["e_scaled_weights"],
        },
        {
            "title": "Weighted sum of values", "op": "weighted_sum",
            "narration": "The final attention output is a blend of all value vectors, weighted by relevance — the query 'attends' most to whichever key it matched best.",
            "formula": r"\text{out} = \sum_i \alpha_i V_i",
            "code": "output = weights @ v  # blends V rows by attention weight",
            "highlight_nodes": ["v", "output"], "highlight_edges": ["e_weights_out", "e_v_out"],
        },
    ),
}

# --------------------------------------------------------- 2. backpropagation
CONCEPTS["backpropagation"] = {
    "title": "Backpropagation",
    "description": "Forward pass computes output; backward pass applies chain rule in reverse to get every weight's gradient.",
    "nodes": _nodes(
        ("input", "Input x", "scalar", "params"),
        ("w1", "w1", "scalar", "params"),
        ("hidden", "Hidden h = ReLU(w1 x)", "scalar", "branch"),
        ("w2", "w2", "scalar", "params"),
        ("output", "Output y_hat = w2 h", "scalar", "branch"),
        ("loss", "Loss L", "scalar", "generation"),
        ("dL_dout", "dL/dy_hat", "scalar", "grad"),
        ("dL_dw2", "dL/dw2", "scalar", "grad"),
        ("dL_dh", "dL/dh", "scalar", "grad"),
        ("dL_dw1", "dL/dw1", "scalar", "grad"),
    ),
    "edges": _edges(
        ("e_in_h", "input", "hidden", "s1"),
        ("e_w1_h", "w1", "hidden", "s1"),
        ("e_h_out", "hidden", "output", "s2"),
        ("e_w2_out", "w2", "output", "s2"),
        ("e_out_loss", "output", "loss", "s3"),
        ("e_loss_dout", "loss", "dL_dout", "s4"),
        ("e_dout_dw2", "dL_dout", "dL_dw2", "s4"),
        ("e_dout_dh", "dL_dout", "dL_dh", "s4"),
        ("e_dh_dw1", "dL_dh", "dL_dw1", "s5"),
    ),
    "steps": _steps(
        {
            "title": "Forward: compute hidden h = ReLU(w1 x)", "op": "forward",
            "narration": "The forward pass starts by combining the input with the first weight and applying a non-linearity.",
            "formula": r"h = \text{ReLU}(w_1 x)",
            "code": "from optimumai import Value\nx = Value(1.5); w1 = Value(0.4)\nh = (x * w1).relu()",
            "highlight_nodes": ["input", "w1", "hidden"], "highlight_edges": ["e_in_h", "e_w1_h"],
        },
        {
            "title": "Forward: output y_hat = w2 h", "op": "forward",
            "narration": "The hidden activation is scaled by the second weight to produce the model's prediction.",
            "formula": r"\hat{y} = w_2 h",
            "code": "w2 = Value(0.9)\ny_hat = w2 * h",
            "highlight_nodes": ["w2", "output"], "highlight_edges": ["e_h_out", "e_w2_out"],
        },
        {
            "title": "Loss: MSE between prediction and target", "op": "loss",
            "narration": "The loss measures how far the prediction is from the true value — this is what we differentiate.",
            "formula": r"L = (\hat{y} - y)^2",
            "code": "y = Value(1.0)\nloss = (y_hat - y) ** 2",
            "highlight_nodes": ["loss"], "highlight_edges": ["e_out_loss"],
        },
        {
            "title": "Backward: propagate dL/dy_hat", "op": "backward",
            "narration": "The backward pass starts at the loss and walks the graph in reverse, applying the chain rule at every node.",
            "formula": r"\frac{\partial L}{\partial \hat{y}} = 2(\hat{y} - y)",
            "code": "loss.backward()  # computes all gradients in one call",
            "highlight_nodes": ["dL_dout", "dL_dw2", "dL_dh"], "highlight_edges": ["e_loss_dout", "e_dout_dw2", "e_dout_dh"],
        },
        {
            "title": "Gradient at w1 via the chain rule", "op": "backward",
            "narration": "The gradient at w1 requires chaining through every operation between it and the loss — this is exactly what autograd automates.",
            "formula": r"\frac{\partial L}{\partial w_1} = \frac{\partial L}{\partial \hat{y}}\cdot w_2 \cdot \mathbf{1}[h>0]\cdot x",
            "code": "print(w1.grad)  # dL/dw1, computed via the chain rule",
            "highlight_nodes": ["dL_dw1"], "highlight_edges": ["e_dh_dw1"],
        },
    ),
}

# ------------------------------------------------------------------ 3. gradient
CONCEPTS["gradient"] = {
    "title": "Gradient",
    "description": "A gradient is the vector of partial derivatives of a scalar function with respect to each parameter — it points toward steepest ascent.",
    "nodes": _nodes(
        ("f", "f(x, y)", "scalar", "branch"),
        ("x", "x", "scalar", "params"),
        ("y", "y", "scalar", "params"),
        ("partial_x", "∂f/∂x", "scalar", "grad"),
        ("partial_y", "∂f/∂y", "scalar", "grad"),
        ("gradient_vec", "∇f", "vector", "grad"),
    ),
    "edges": _edges(
        ("e_x_f", "x", "f", "s1"),
        ("e_y_f", "y", "f", "s1"),
        ("e_f_px", "f", "partial_x", "s2"),
        ("e_f_py", "f", "partial_y", "s3"),
        ("e_px_grad", "partial_x", "gradient_vec", "s4"),
        ("e_py_grad", "partial_y", "gradient_vec", "s4"),
    ),
    "steps": _steps(
        {
            "title": "Define the scalar function f(x, y)", "op": "define",
            "narration": "We start with a scalar-valued function of two variables — gradients are always taken of a scalar output.",
            "formula": r"f(x, y) = x^2 + 2y^2",
            "code": "from optimumai import Value\nx = Value(3.0); y = Value(1.0)\nf = x**2 + 2*y**2",
            "highlight_nodes": ["x", "y", "f"], "highlight_edges": ["e_x_f", "e_y_f"],
        },
        {
            "title": "Partial derivative with respect to x", "op": "diff",
            "narration": "Holding y fixed, differentiate f with respect to x.",
            "formula": r"\frac{\partial f}{\partial x} = 2x",
            "code": "f.backward()\nprint(x.grad)  # = 6.0",
            "highlight_nodes": ["partial_x"], "highlight_edges": ["e_f_px"], "metrics": {"df_dx": 6.0},
        },
        {
            "title": "Partial derivative with respect to y", "op": "diff",
            "narration": "Holding x fixed, differentiate f with respect to y.",
            "formula": r"\frac{\partial f}{\partial y} = 4y",
            "code": "print(y.grad)  # = 4.0",
            "highlight_nodes": ["partial_y"], "highlight_edges": ["e_f_py"], "metrics": {"df_dy": 4.0},
        },
        {
            "title": "Assemble the gradient vector", "op": "assemble",
            "narration": "Stacking every partial derivative into a vector gives the gradient — it points in the direction of steepest ascent of f.",
            "formula": r"\nabla f = \begin{pmatrix} 2x \\ 4y \end{pmatrix} = \begin{pmatrix} 6 \\ 4 \end{pmatrix}",
            "code": "from optimumai import gradient\ngrad = gradient(lambda v: v[0]**2 + 2*v[1]**2, [3.0, 1.0])\nprint(grad)  # [6.0, 4.0]",
            "highlight_nodes": ["gradient_vec"], "highlight_edges": ["e_px_grad", "e_py_grad"],
        },
    ),
}

# --------------------------------------------------------- 4. gradient_descent
CONCEPTS["gradient_descent"] = {
    "title": "Gradient Descent",
    "description": "Iteratively move parameters opposite to the gradient to minimize a loss function. The learning rate controls the step size.",
    "nodes": _nodes(
        ("params", "Parameters θ", "vector", "params"),
        ("loss", "Loss L(θ)", "scalar", "branch"),
        ("gradient", "Gradient ∇L", "vector", "grad"),
        ("update", "Update rule", "vector", "optim"),
        ("new_params", "New θ", "vector", "optim"),
    ),
    "edges": _edges(
        ("e_params_loss", "params", "loss", "s1"),
        ("e_loss_grad", "loss", "gradient", "s2"),
        ("e_grad_update", "gradient", "update", "s3"),
        ("e_update_new", "update", "new_params", "s3"),
        ("e_new_loop", "new_params", "loss", "s4"),
    ),
    "steps": _steps(
        {
            "title": "Compute loss L(θ)", "op": "loss",
            "narration": "Every gradient descent step starts by evaluating how wrong the current parameters are.",
            "formula": r"L(\theta) = \frac{1}{n}\sum_i (y_i - \hat{y}_i)^2",
            "code": "from optimumai import LinearRegression\nmodel = LinearRegression()\nloss = model.mse(X_train, y_train)",
            "highlight_nodes": ["params", "loss"], "highlight_edges": ["e_params_loss"],
        },
        {
            "title": "Compute the gradient ∇L", "op": "diff",
            "narration": "The gradient tells us how the loss changes with each parameter — the direction to move away from.",
            "formula": r"\nabla_\theta L = \frac{2}{n} X^T(X\theta - y)",
            "code": "from optimumai import gradient\ngrad = gradient(model.loss_fn, model.params)",
            "highlight_nodes": ["gradient"], "highlight_edges": ["e_loss_grad"],
        },
        {
            "title": "Apply the update rule", "op": "update",
            "narration": "Step opposite to the gradient, scaled by the learning rate η.",
            "formula": r"\theta \leftarrow \theta - \eta \nabla_\theta L",
            "code": "from optimumai import SGD\noptimizer = SGD(lr=0.01)\nparams = optimizer.step(params, grad)",
            "highlight_nodes": ["update", "new_params"], "highlight_edges": ["e_grad_update", "e_update_new"],
        },
        {
            "title": "Repeat until convergence", "op": "loop",
            "narration": "Looping this process drives the loss down step by step until it stops improving meaningfully.",
            "formula": r"L^{(t+1)} < L^{(t)}",
            "code": "for step in range(100):\n    loss = model.loss_fn(params)\n    grad = gradient(model.loss_fn, params)\n    params = optimizer.step(params, grad)",
            "highlight_nodes": ["new_params", "loss"], "highlight_edges": ["e_new_loop"],
        },
        {
            "title": "Final parameters minimize the loss", "op": "converge",
            "narration": "After enough steps, the parameters settle near a minimum of the loss surface.",
            "formula": r"\hat{\theta} = \arg\min_\theta L(\theta)",
            "code": 'print(f"Final loss: {loss:.4f}")\nprint(f"Params: {params}")',
            "highlight_nodes": ["new_params"], "highlight_edges": [],
        },
    ),
}

# ---------------------------------------------------- 5. activation_functions
CONCEPTS["activation_functions"] = {
    "title": "Activation Functions",
    "description": "Non-linear functions applied element-wise after a linear layer — they're what let neural networks approximate arbitrary functions.",
    "nodes": _nodes(
        ("linear_out", "Pre-activation z", "scalar", "params"),
        ("relu_node", "ReLU(z)", "scalar", "branch"),
        ("gelu_node", "GELU(z)", "scalar", "branch"),
        ("sigmoid_node", "Sigmoid(z)", "scalar", "branch"),
        ("tanh_node", "Tanh(z)", "scalar", "branch"),
        ("act_out", "Activated output", "scalar", "generation"),
    ),
    "edges": _edges(
        ("e_lin_relu", "linear_out", "relu_node", "s2"),
        ("e_lin_gelu", "linear_out", "gelu_node", "s3"),
        ("e_lin_sig", "linear_out", "sigmoid_node", "s4"),
        ("e_lin_tanh", "linear_out", "tanh_node", "s5"),
        ("e_relu_out", "relu_node", "act_out", "s5"),
    ),
    "steps": _steps(
        {
            "title": "Why activations? The problem with pure linear layers", "op": "motivate",
            "narration": "Stacking linear layers without a non-linearity collapses into a single linear transform — no matter how deep, the network could never learn curves.",
            "formula": r"\text{Linear}^n(x) = Wx + b \quad (\text{still linear})",
            "code": "# Without activation, stacking layers = one linear transform\nfrom optimumai import Vector\nz = Vector([2.0, -1.0, 0.5])  # pre-activation",
            "highlight_nodes": ["linear_out"], "highlight_edges": [],
        },
        {
            "title": "ReLU: simple and powerful", "op": "relu",
            "narration": "ReLU zeroes out negative values and passes positives through unchanged — cheap to compute, and the default for hidden layers.",
            "formula": r"\text{ReLU}(z) = \max(0, z)",
            "code": "from optimumai import Value\nz = Value(-0.5)\nout = z.relu()  # = 0.0\nz2 = Value(1.3)\nout2 = z2.relu()  # = 1.3",
            "highlight_nodes": ["relu_node"], "highlight_edges": ["e_lin_relu"],
        },
        {
            "title": "GELU: a smoother ReLU (used in GPT)", "op": "gelu",
            "narration": "GELU weights inputs by their value under a Gaussian CDF, giving a smooth curve instead of ReLU's hard kink — standard in transformer feed-forward layers.",
            "formula": r"\text{GELU}(z) \approx z \cdot \sigma(1.702 z)",
            "code": "import numpy as np\ndef gelu(z):\n    return z * (1 / (1 + np.exp(-1.702 * z)))\nprint(gelu(1.0))  # ≈ 0.841",
            "highlight_nodes": ["gelu_node"], "highlight_edges": ["e_lin_gelu"],
        },
        {
            "title": "Sigmoid: squashes to (0, 1)", "op": "sigmoid",
            "narration": "Sigmoid maps any real number into a probability-like range — used for binary classification outputs and gates.",
            "formula": r"\sigma(z) = \frac{1}{1+e^{-z}}",
            "code": "from optimumai import Value\nz = Value(1.5)\nout = z.sigmoid()  # ≈ 0.818",
            "highlight_nodes": ["sigmoid_node"], "highlight_edges": ["e_lin_sig"],
        },
        {
            "title": "Tanh: squashes to (-1, 1)", "op": "tanh",
            "narration": "Tanh is a zero-centered sigmoid variant — often preferred in RNN hidden states because it keeps outputs balanced around zero.",
            "formula": r"\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}",
            "code": "import numpy as np\nz = np.array([0.0, 1.0, -1.0, 2.0])\nprint(np.tanh(z))  # [0, 0.76, -0.76, 0.96]",
            "highlight_nodes": ["tanh_node", "act_out"], "highlight_edges": ["e_lin_tanh", "e_relu_out"],
        },
    ),
}

# ---------------------------------------------------------- 6. adam_optimizer
CONCEPTS["adam_optimizer"] = {
    "title": "Adam Optimizer",
    "description": "Adaptive Moment Estimation: tracks a moving average of gradients (m) and squared gradients (v), with bias correction, for stable per-parameter learning rates.",
    "nodes": _nodes(
        ("grad", "Gradient g_t", "scalar", "grad"),
        ("m_t", "1st moment m_t", "scalar", "optim"),
        ("v_t", "2nd moment v_t", "scalar", "optim"),
        ("m_hat", "Bias-corrected m̂", "scalar", "optim"),
        ("v_hat", "Bias-corrected v̂", "scalar", "optim"),
        ("theta", "Parameter θ", "scalar", "params"),
    ),
    "edges": _edges(
        ("e_grad_m", "grad", "m_t", "s1"),
        ("e_grad_v", "grad", "v_t", "s2"),
        ("e_m_mhat", "m_t", "m_hat", "s3"),
        ("e_v_vhat", "v_t", "v_hat", "s3"),
        ("e_mhat_theta", "m_hat", "theta", "s4"),
        ("e_vhat_theta", "v_hat", "theta", "s4"),
    ),
    "steps": _steps(
        {
            "title": "First moment: gradient momentum", "op": "moment1",
            "narration": "m tracks an exponential moving average of the gradient — this smooths out noisy step-to-step gradients.",
            "formula": r"m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t",
            "code": "beta1, beta2, eps = 0.9, 0.999, 1e-8\nm = 0.0; v = 0.0\ng = 0.3  # gradient at step t\nm = beta1 * m + (1 - beta1) * g",
            "highlight_nodes": ["grad", "m_t"], "highlight_edges": ["e_grad_m"],
        },
        {
            "title": "Second moment: squared gradient", "op": "moment2",
            "narration": "v tracks an exponential moving average of the squared gradient — this estimates each parameter's gradient variance.",
            "formula": r"v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2",
            "code": "v = beta2 * v + (1 - beta2) * g**2",
            "highlight_nodes": ["v_t"], "highlight_edges": ["e_grad_v"],
        },
        {
            "title": "Bias correction (early steps are biased toward 0)", "op": "bias_correct",
            "narration": "Because m and v start at zero, early estimates are biased low — dividing by (1 - β^t) corrects for this.",
            "formula": r"\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1-\beta_2^t}",
            "code": "t = 1\nm_hat = m / (1 - beta1**t)\nv_hat = v / (1 - beta2**t)",
            "highlight_nodes": ["m_hat", "v_hat"], "highlight_edges": ["e_m_mhat", "e_v_vhat"],
        },
        {
            "title": "Parameter update", "op": "update",
            "narration": "The update divides the momentum by the root of the second moment — giving each parameter its own adaptive step size.",
            "formula": r"\theta \leftarrow \theta - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \varepsilon}",
            "code": "lr = 0.001; theta = 1.0\ntheta -= lr * m_hat / (v_hat**0.5 + eps)\nprint(theta)  # ≈ 0.999",
            "highlight_nodes": ["theta"], "highlight_edges": ["e_mhat_theta", "e_vhat_theta"],
        },
        {
            "title": "Full Adam step via optimumai", "op": "apply",
            "narration": "optimumai's Adam optimizer handles m, v, and the step counter t internally — you just call step().",
            "formula": r"\eta_{\text{eff}} = \eta \frac{\sqrt{1-\beta_2^t}}{1-\beta_1^t}",
            "code": "from optimumai import Adam\nopt = Adam(lr=0.001)\nparams = opt.step(params, grads)  # handles m, v, t internally",
            "highlight_nodes": ["theta"], "highlight_edges": [],
        },
    ),
}

# --------------------------------------------------------- 7. adamw_optimizer
CONCEPTS["adamw_optimizer"] = {
    "title": "AdamW Optimizer",
    "description": "Adam + decoupled weight decay: weight decay is applied directly to the parameters (not through the adaptive gradient), fixing Adam's regularization behavior.",
    "nodes": _nodes(
        ("grad", "Gradient g_t", "scalar", "grad"),
        ("m_t", "Momentum m_t, v_t", "scalar", "optim"),
        ("wd_term", "Weight decay term", "scalar", "optim"),
        ("theta", "Parameter θ", "scalar", "params"),
    ),
    "edges": _edges(
        ("e_grad_m", "grad", "m_t", "s1"),
        ("e_m_theta", "m_t", "theta", "s2"),
        ("e_wd_theta", "wd_term", "theta", "s2"),
    ),
    "steps": _steps(
        {
            "title": "Adam update (same momentum machinery as Adam)", "op": "moment",
            "narration": "AdamW starts from the exact same first/second moment estimates as Adam.",
            "formula": r"\Delta\theta = -\eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\varepsilon}",
            "code": "# Same momentum steps as Adam\nbeta1, beta2, eps = 0.9, 0.999, 1e-8\nm = beta1*m + (1-beta1)*g\nv = beta2*v + (1-beta2)*g**2",
            "highlight_nodes": ["grad", "m_t"], "highlight_edges": ["e_grad_m"],
        },
        {
            "title": "Decoupled weight decay", "op": "decay",
            "narration": "Instead of folding weight decay into the gradient (which Adam's adaptive scaling would distort), AdamW shrinks the parameter directly.",
            "formula": r"\theta \leftarrow \theta(1 - \eta\lambda) - \eta\frac{\hat{m}}{\sqrt{\hat{v}}+\varepsilon}",
            "code": "lam = 0.01  # weight decay\ntheta = theta * (1 - lr * lam) - lr * m_hat / (v_hat**0.5 + eps)",
            "highlight_nodes": ["wd_term", "theta"], "highlight_edges": ["e_m_theta", "e_wd_theta"],
        },
        {
            "title": "Why decoupled? Adam's L2 regularization isn't weight decay", "op": "explain",
            "narration": "Adding λθ to the gradient (classic L2) gets rescaled by Adam's adaptive denominator — the actual shrinkage ends up parameter-dependent, which isn't what you want.",
            "formula": r"\text{L2 reg}: \nabla L + \lambda\theta \quad \text{vs.} \quad \text{WD}: \theta(1-\eta\lambda)",
            "code": "# L2 reg scales with gradient magnitude (wrong)\n# AdamW applies weight decay directly to params (correct)\n# Use AdamW for transformers by default",
            "highlight_nodes": ["theta"], "highlight_edges": [],
        },
        {
            "title": "AdamW via optimumai", "op": "apply",
            "narration": "Passing a weight_decay to optimumai's Adam applies the decoupled update automatically.",
            "formula": r"\theta^{(t+1)} = (1-\eta\lambda)\theta^{(t)} - \eta\hat{m}/(\sqrt{\hat{v}}+\varepsilon)",
            "code": "from optimumai import Adam\n# Adam with weight_decay kwarg applies decoupled WD\nopt = Adam(lr=1e-3, weight_decay=0.01)\nparams = opt.step(params, grads)",
            "highlight_nodes": ["theta"], "highlight_edges": [],
        },
    ),
}

# ------------------------------------------------------------------ 8. softmax
CONCEPTS["softmax"] = {
    "title": "Softmax",
    "description": "Converts a vector of raw scores (logits) into a probability distribution that sums to 1 — used in classification and attention.",
    "nodes": _nodes(
        ("logits", "Logits z", "vector", "params"),
        ("shifted", "Shifted z - max(z)", "vector", "branch"),
        ("exp_vals", "exp(z')", "vector", "branch"),
        ("sum_exp", "Σ exp(z')", "scalar", "branch"),
        ("probs", "Probabilities", "vector", "generation"),
    ),
    "edges": _edges(
        ("e_logits_shifted", "logits", "shifted", "s2"),
        ("e_shifted_exp", "shifted", "exp_vals", "s3"),
        ("e_exp_sum", "exp_vals", "sum_exp", "s4"),
        ("e_exp_probs", "exp_vals", "probs", "s4"),
        ("e_sum_probs", "sum_exp", "probs", "s4"),
    ),
    "steps": _steps(
        {
            "title": "Raw logits", "op": "input",
            "narration": "Softmax starts with raw, unnormalized scores — they can be any real number, positive or negative.",
            "formula": r"z = [z_1, z_2, \ldots, z_C]",
            "code": "from optimumai import Vector\nlogits = Vector([2.0, 1.0, 0.1])\nprint(logits)  # raw unnormalized scores",
            "highlight_nodes": ["logits"], "highlight_edges": [],
        },
        {
            "title": "Numerical stability: subtract the max", "op": "shift",
            "narration": "Subtracting the max before exponentiating avoids overflow, without changing the final result — softmax is shift-invariant.",
            "formula": r"z' = z - \max(z)",
            "code": "import numpy as np\nz = np.array([2.0, 1.0, 0.1])\nz_stable = z - z.max()  # [0, -1, -1.9]",
            "highlight_nodes": ["shifted"], "highlight_edges": ["e_logits_shifted"],
        },
        {
            "title": "Exponentiate", "op": "exp",
            "narration": "Exponentiating makes every value positive and amplifies differences between scores.",
            "formula": r"e^{z'} = [e^{z'_1}, \ldots, e^{z'_C}]",
            "code": "exp_z = np.exp(z_stable)  # [1.0, 0.368, 0.150]",
            "highlight_nodes": ["exp_vals"], "highlight_edges": ["e_shifted_exp"],
        },
        {
            "title": "Normalize into probabilities", "op": "normalize",
            "narration": "Dividing each exponentiated value by their sum yields a valid probability distribution that sums to exactly 1.",
            "formula": r"\text{softmax}(z)_i = \frac{e^{z_i}}{\sum_j e^{z_j}}",
            "code": "from optimumai import softmax\nprobs = softmax([2.0, 1.0, 0.1])\nprint(probs)  # [0.659, 0.242, 0.099], sums to 1",
            "highlight_nodes": ["sum_exp", "probs"], "highlight_edges": ["e_exp_sum", "e_exp_probs", "e_sum_probs"],
        },
    ),
}

# ---------------------------------------------------------- 9. cross_entropy_loss
CONCEPTS["cross_entropy_loss"] = {
    "title": "Cross-Entropy Loss",
    "description": "Measures how well predicted probabilities match one-hot true labels — the standard loss for classification tasks.",
    "nodes": _nodes(
        ("logits", "Logits", "vector", "params"),
        ("probs", "Predicted probs", "vector", "branch"),
        ("true_label", "True label (one-hot)", "vector", "params"),
        ("log_probs", "log(probs)", "vector", "branch"),
        ("loss", "Cross-entropy loss", "scalar", "generation"),
    ),
    "edges": _edges(
        ("e_logits_probs", "logits", "probs", "s1"),
        ("e_probs_log", "probs", "log_probs", "s3"),
        ("e_label_loss", "true_label", "loss", "s4"),
        ("e_log_loss", "log_probs", "loss", "s4"),
    ),
    "steps": _steps(
        {
            "title": "Predicted probabilities via softmax", "op": "softmax",
            "narration": "The model's raw logits are converted to a probability distribution over classes.",
            "formula": r"\hat{p} = \text{softmax}(z)",
            "code": "from optimumai import softmax\nlogits = [3.0, 1.0, 0.2]\np_hat = softmax(logits)  # [0.844, 0.114, 0.042]",
            "highlight_nodes": ["logits", "probs"], "highlight_edges": ["e_logits_probs"],
        },
        {
            "title": "True label as one-hot vector", "op": "label",
            "narration": "The ground truth is represented as a one-hot vector — 1 for the correct class, 0 elsewhere.",
            "formula": r"y = [0, 1, 0] \quad (\text{class 1 is correct})",
            "code": "import numpy as np\ny_true = np.array([0, 1, 0])  # class 1 is correct\n# The model put most mass on class 0 -- high loss expected",
            "highlight_nodes": ["true_label"], "highlight_edges": [],
        },
        {
            "title": "Log of the predicted probability", "op": "log",
            "narration": "We only care about the log-probability the model assigned to the true class — everything else is zeroed by the one-hot label.",
            "formula": r"-\log \hat{p}_{y}",
            "code": "log_p = np.log(p_hat)\nloss_per_class = -y_true * log_p\nprint(loss_per_class)  # [0, -log(0.114), 0]",
            "highlight_nodes": ["log_probs"], "highlight_edges": ["e_probs_log"],
        },
        {
            "title": "Sum into the cross-entropy loss", "op": "loss",
            "narration": "The final loss is just the negative log-probability of the true class — low when the model is confident and correct, high when it's confidently wrong.",
            "formula": r"L = -\sum_c y_c \log \hat{p}_c = -\log \hat{p}_{\text{true}}",
            "code": 'loss = -np.sum(y_true * np.log(p_hat))\nprint(f"CE loss: {loss:.4f}")  # ≈ 2.17',
            "highlight_nodes": ["loss"], "highlight_edges": ["e_label_loss", "e_log_loss"],
        },
    ),
}

# ------------------------------------------------------- 10. layer_normalization
CONCEPTS["layer_normalization"] = {
    "title": "Layer Normalization",
    "description": "Normalize activations across the feature dimension (not batch) — stabilizes training and is used in every transformer layer.",
    "nodes": _nodes(
        ("x", "Input activations x", "vector", "params"),
        ("mean_node", "Mean μ", "scalar", "branch"),
        ("std_node", "Std σ", "scalar", "branch"),
        ("normalized", "Normalized x̂", "vector", "branch"),
        ("scaled_shifted", "γ, β", "vector", "params"),
        ("output", "Output y", "vector", "generation"),
    ),
    "edges": _edges(
        ("e_x_mean", "x", "mean_node", "s2"),
        ("e_x_std", "x", "std_node", "s2"),
        ("e_mean_norm", "mean_node", "normalized", "s3"),
        ("e_std_norm", "std_node", "normalized", "s3"),
        ("e_norm_out", "normalized", "output", "s4"),
        ("e_scale_out", "scaled_shifted", "output", "s4"),
    ),
    "steps": _steps(
        {
            "title": "Input activations", "op": "input",
            "narration": "A single residual-stream vector, whose feature values may vary wildly in scale.",
            "formula": r"x = [x_1, \ldots, x_d]",
            "code": "import numpy as np\nx = np.array([0.2, 1.4, -0.5, 0.9, 2.1])  # one residual stream",
            "highlight_nodes": ["x"], "highlight_edges": [],
        },
        {
            "title": "Compute mean and std across features", "op": "stats",
            "narration": "Unlike batch norm, layer norm computes statistics across the feature dimension of a single example — no dependence on batch size.",
            "formula": r"\mu = \frac{1}{d}\sum_i x_i, \quad \sigma = \sqrt{\frac{1}{d}\sum_i (x_i - \mu)^2}",
            "code": "mu = x.mean()    # 0.82\nsigma = x.std()  # 0.87",
            "highlight_nodes": ["mean_node", "std_node"], "highlight_edges": ["e_x_mean", "e_x_std"],
        },
        {
            "title": "Normalize to zero mean, unit variance", "op": "normalize",
            "narration": "Subtracting the mean and dividing by the std rescales the vector so its features are comparable in magnitude.",
            "formula": r"\hat{x}_i = \frac{x_i - \mu}{\sigma + \varepsilon}",
            "code": "eps = 1e-5\nx_hat = (x - mu) / (sigma + eps)\nprint(x_hat)  # zero-mean, unit-variance",
            "highlight_nodes": ["normalized"], "highlight_edges": ["e_mean_norm", "e_std_norm"],
        },
        {
            "title": "Learnable scale γ and shift β", "op": "affine",
            "narration": "A learned affine transform lets the network recover any scale/shift it actually needs — normalization doesn't have to mean losing expressiveness.",
            "formula": r"y_i = \gamma \hat{x}_i + \beta",
            "code": "gamma = np.ones(5)   # learned scale\nbeta  = np.zeros(5)  # learned shift\ny = gamma * x_hat + beta  # output",
            "highlight_nodes": ["scaled_shifted", "output"], "highlight_edges": ["e_norm_out", "e_scale_out"],
        },
    ),
}

# ---------------------------------------------------- 11. multi_agentic_workflow
CONCEPTS["multi_agentic_workflow"] = {
    "title": "Multi-Agentic Workflow",
    "description": "Multiple AI agents collaborate: an orchestrator decomposes a task, sub-agents execute with tools, results are synthesized. Used in coding assistants, research, RAG pipelines.",
    "nodes": _nodes(
        ("user_task", "User task", "text", "ingestion"),
        ("orchestrator", "Orchestrator", "agent", "query"),
        ("plan", "Sub-task plan", "list", "branch"),
        ("agent_a", "Agent A", "agent", "retrieval"),
        ("agent_b", "Agent B", "agent", "retrieval"),
        ("tool_call", "Tool calls", "action", "branch"),
        ("results", "Sub-results", "list", "branch"),
        ("synthesizer", "Synthesizer", "agent", "query"),
        ("final_output", "Final output", "text", "generation"),
    ),
    "edges": _edges(
        ("e_task_orch", "user_task", "orchestrator", "s1"),
        ("e_orch_plan", "orchestrator", "plan", "s2"),
        ("e_plan_a", "plan", "agent_a", "s3"),
        ("e_plan_b", "plan", "agent_b", "s3"),
        ("e_a_tool", "agent_a", "tool_call", "s4"),
        ("e_b_tool", "agent_b", "tool_call", "s4"),
        ("e_a_results", "agent_a", "results", "s5"),
        ("e_b_results", "agent_b", "results", "s5"),
        ("e_results_synth", "results", "synthesizer", "s5"),
        ("e_synth_final", "synthesizer", "final_output", "s5"),
    ),
    "steps": _steps(
        {
            "title": "User task arrives at the orchestrator", "op": "route",
            "narration": "A single high-level request kicks off the workflow — the orchestrator's job is to figure out how to get it done.",
            "formula": r"\text{Task} \xrightarrow{\text{route}} \text{Orchestrator}",
            "code": 'from optimumai import generate\ntask = "Research and summarize transformer attention"\nplan = generate(f"Break this into sub-tasks: {task}")',
            "highlight_nodes": ["user_task", "orchestrator"], "highlight_edges": ["e_task_orch"],
        },
        {
            "title": "Orchestrator decomposes into sub-tasks", "op": "plan",
            "narration": "The orchestrator breaks the task into smaller, independently executable pieces.",
            "formula": r"P = \{t_1, t_2, \ldots, t_n\}",
            "code": 'sub_tasks = [\n    "Explain scaled dot-product attention",\n    "Give a code example with optimumai",\n    "List real-world use cases",\n]',
            "highlight_nodes": ["plan"], "highlight_edges": ["e_orch_plan"],
        },
        {
            "title": "Sub-agents execute in parallel", "op": "dispatch",
            "narration": "Each sub-task is handed to a dedicated agent, which can work concurrently with the others.",
            "formula": r"r_i = \text{Agent}_i(t_i, \text{tools})",
            "code": "from optimumai import attention_flow\nresult_a = attention_flow()  # Agent A: generate diagram\nresult_b = generate(\"Code example for attention\")",
            "highlight_nodes": ["agent_a", "agent_b"], "highlight_edges": ["e_plan_a", "e_plan_b"],
        },
        {
            "title": "Agents call tools to augment themselves", "op": "tool_use",
            "narration": "An agent alone is limited to its training data — tools (search, code execution, retrieval) let it act on live, external information.",
            "formula": r"\text{Agent} + \text{Tool} \rightarrow \text{Augmented Agent}",
            "code": 'tools = [\n    {"name": "search", "fn": web_search},\n    {"name": "code", "fn": run_code},\n]\n# Each agent calls the tools relevant to its sub-task',
            "highlight_nodes": ["tool_call"], "highlight_edges": ["e_a_tool", "e_b_tool"],
        },
        {
            "title": "Synthesizer combines all sub-results", "op": "synthesize",
            "narration": "A final synthesis step merges every agent's output into one coherent answer for the user.",
            "formula": r"\text{Final} = \text{Synthesize}(r_1, r_2, \ldots, r_n)",
            "code": 'final = generate(\n    "Synthesize: " + "\\n".join(results),\n    provider="anthropic"\n)',
            "highlight_nodes": ["results", "synthesizer", "final_output"],
            "highlight_edges": ["e_a_results", "e_b_results", "e_results_synth", "e_synth_final"],
        },
    ),
}

# ---------------------------------------------------------- 12. weights_bias_neuron
CONCEPTS["weights_bias_neuron"] = {
    "title": "Weights, Bias & the Weighted Sum",
    "description": "x → weighted sum (Σwx) → +bias → activation → output. The atomic unit behind linear regression and every neural network layer.",
    "nodes": _nodes(
        ("x1", "x1", "scalar", "params"),
        ("x2", "x2", "scalar", "params"),
        ("x3", "x3", "scalar", "params"),
        ("w1", "w1", "scalar", "params"),
        ("w2", "w2", "scalar", "params"),
        ("w3", "w3", "scalar", "params"),
        ("weighted_sum", "Σ w_i x_i", "scalar", "branch"),
        ("bias", "bias b", "scalar", "params"),
        ("pre_act", "z = Σwx + b", "scalar", "branch"),
        ("activation", "ReLU(z)", "scalar", "generation"),
        ("output", "output", "scalar", "generation"),
    ),
    "edges": _edges(
        ("e_x1_sum", "x1", "weighted_sum", "s1"),
        ("e_x2_sum", "x2", "weighted_sum", "s1"),
        ("e_x3_sum", "x3", "weighted_sum", "s1"),
        ("e_w1_sum", "w1", "weighted_sum", "s1"),
        ("e_w2_sum", "w2", "weighted_sum", "s1"),
        ("e_w3_sum", "w3", "weighted_sum", "s1"),
        ("e_sum_pre", "weighted_sum", "pre_act", "s2"),
        ("e_bias_pre", "bias", "pre_act", "s2"),
        ("e_pre_act", "pre_act", "activation", "s3"),
        ("e_act_out", "activation", "output", "s3"),
    ),
    "steps": _steps(
        {
            "title": "Inputs combine into a weighted sum", "op": "dot",
            "narration": "Each input is multiplied by its own learned weight, then all the products are summed — this is a dot product.",
            "formula": r"z = \sum_i w_i x_i",
            "code": "from optimumai import Vector\nx = Vector([1.0, 2.0, 3.0])\nw = Vector([0.5, -0.3, 0.8])\nz = x.dot(w)  # = 0.5 - 0.6 + 2.4 = 2.3",
            "highlight_nodes": ["x1", "x2", "x3", "w1", "w2", "w3", "weighted_sum"],
            "highlight_edges": ["e_x1_sum", "e_x2_sum", "e_x3_sum", "e_w1_sum", "e_w2_sum", "e_w3_sum"],
        },
        {
            "title": "Add the bias term", "op": "add",
            "narration": "A bias term shifts the weighted sum, letting the neuron fire even when all inputs are zero.",
            "formula": r"z' = z + b",
            "code": "b = 0.1\nz_prime = z + b  # = 2.4",
            "highlight_nodes": ["bias", "pre_act"], "highlight_edges": ["e_sum_pre", "e_bias_pre"],
        },
        {
            "title": "Apply the activation function", "op": "activate",
            "narration": "The non-linear activation decides how strongly this neuron fires, given its pre-activation value.",
            "formula": r"a = \text{ReLU}(z')",
            "code": "from optimumai import Value\nz_val = Value(2.4)\na = z_val.relu()  # = 2.4 (positive, so unchanged)",
            "highlight_nodes": ["activation", "output"], "highlight_edges": ["e_pre_act", "e_act_out"],
        },
        {
            "title": "Gradient with respect to each weight", "op": "backward",
            "narration": "During training, backprop computes how much each weight contributed to the final error.",
            "formula": r"\frac{\partial L}{\partial w_i} = \frac{\partial L}{\partial a} \cdot x_i",
            "code": "a.backward()  # backprop\nprint([w.grad for w in [w1, w2, w3]])  # dL/dw_i",
            "highlight_nodes": ["w1", "w2", "w3"], "highlight_edges": [],
        },
        {
            "title": "Update every weight", "op": "update",
            "narration": "Each weight is nudged opposite to its gradient, scaled by the learning rate.",
            "formula": r"w_i \leftarrow w_i - \eta \frac{\partial L}{\partial w_i}",
            "code": "lr = 0.01\nfor i, (wi, xi) in enumerate(zip(weights, x_vals)):\n    weights[i] -= lr * dl_da * xi",
            "highlight_nodes": ["w1", "w2", "w3"], "highlight_edges": [],
        },
    ),
}

# -------------------------------------------------------------- 13. linear_regression
CONCEPTS["linear_regression"] = {
    "title": "Linear Regression",
    "description": "Fit a line to data by minimizing the sum of squared residuals. Closed-form solution via the normal equation, or iterative via gradient descent.",
    "nodes": _nodes(
        ("x_data", "X data", "matrix", "params"),
        ("y_data", "y data", "vector", "params"),
        ("theta", "θ", "vector", "params"),
        ("y_hat", "ŷ = Xθ", "vector", "branch"),
        ("residuals", "residuals y - ŷ", "vector", "branch"),
        ("loss", "MSE loss", "scalar", "branch"),
        ("update", "θ̂ (fitted)", "vector", "generation"),
    ),
    "edges": _edges(
        ("e_x_yhat", "x_data", "y_hat", "s1"),
        ("e_theta_yhat", "theta", "y_hat", "s1"),
        ("e_y_resid", "y_data", "residuals", "s2"),
        ("e_yhat_resid", "y_hat", "residuals", "s2"),
        ("e_resid_loss", "residuals", "loss", "s2"),
        ("e_loss_update", "loss", "update", "s3"),
    ),
    "steps": _steps(
        {
            "title": "Model: predict ŷ = Xθ", "op": "predict",
            "narration": "Linear regression assumes the target is a linear combination of the input features.",
            "formula": r"\hat{y} = X\theta = \theta_0 + \theta_1 x",
            "code": "from optimumai import LinearRegression\nimport numpy as np\nX = np.array([[1,1],[1,2],[1,3]])\ny = np.array([2.0, 2.5, 3.5])\nmodel = LinearRegression()",
            "highlight_nodes": ["x_data", "theta", "y_hat"], "highlight_edges": ["e_x_yhat", "e_theta_yhat"],
        },
        {
            "title": "Mean squared error loss", "op": "loss",
            "narration": "The loss penalizes the squared distance between predictions and true targets, averaged over all examples.",
            "formula": r"L = \frac{1}{n}\|y - X\theta\|^2",
            "code": 'y_hat = X @ model.theta\nloss = np.mean((y - y_hat)**2)\nprint(f"MSE: {loss:.4f}")',
            "highlight_nodes": ["y_data", "residuals", "loss"], "highlight_edges": ["e_y_resid", "e_yhat_resid", "e_resid_loss"],
        },
        {
            "title": "Normal equation: closed-form solution", "op": "solve",
            "narration": "Because MSE is convex and differentiable, we can solve for the optimal θ directly — no iteration required.",
            "formula": r"\hat{\theta} = (X^TX)^{-1}X^Ty",
            "code": "theta_hat = np.linalg.solve(X.T @ X, X.T @ y)\nprint(theta_hat)  # [1.0, 0.75]",
            "highlight_nodes": ["update"], "highlight_edges": ["e_loss_update"],
        },
        {
            "title": "Fit via optimumai (explained)", "op": "fit",
            "narration": "optimumai's LinearRegression walks through the exact same normal-equation steps, with an explained trace.",
            "formula": r"R^2 = 1 - \frac{\text{SS}_{\text{res}}}{\text{SS}_{\text{tot}}}",
            "code": "model.fit(X[:, 1:], y, explain=True)\nprint(model.coef_, model.intercept_)",
            "highlight_nodes": ["update"], "highlight_edges": [],
        },
    ),
}

# ------------------------------------------------------------ 14. logistic_regression
CONCEPTS["logistic_regression"] = {
    "title": "Logistic Regression",
    "description": "Binary classification: apply sigmoid to a linear model to get a probability, then minimize cross-entropy loss via gradient descent.",
    "nodes": _nodes(
        ("x_data", "Input x", "vector", "params"),
        ("linear_out", "z = wᵀx + b", "scalar", "branch"),
        ("sigmoid_out", "σ(z)", "scalar", "branch"),
        ("predicted_prob", "ŷ (probability)", "scalar", "generation"),
        ("loss", "BCE loss", "scalar", "branch"),
        ("gradient", "∇_w L", "vector", "grad"),
        ("update", "updated w", "vector", "optim"),
    ),
    "edges": _edges(
        ("e_x_linear", "x_data", "linear_out", "s1"),
        ("e_linear_sig", "linear_out", "sigmoid_out", "s2"),
        ("e_sig_prob", "sigmoid_out", "predicted_prob", "s2"),
        ("e_prob_loss", "predicted_prob", "loss", "s3"),
        ("e_loss_grad", "loss", "gradient", "s4"),
        ("e_grad_update", "gradient", "update", "s4"),
    ),
    "steps": _steps(
        {
            "title": "Linear combination of inputs", "op": "linear",
            "narration": "Just like linear regression, we start with a weighted sum of the input features plus a bias.",
            "formula": r"z = w^T x + b",
            "code": "from optimumai import Vector\nx = Vector([2.5, 1.0])\nw = Vector([0.4, -0.6])\nb = 0.1\nz = x.dot(w) + b  # = 0.4",
            "highlight_nodes": ["x_data", "linear_out"], "highlight_edges": ["e_x_linear"],
        },
        {
            "title": "Sigmoid turns z into a probability", "op": "sigmoid",
            "narration": "Sigmoid squashes the linear score into (0, 1), which we interpret as the probability of the positive class.",
            "formula": r"\hat{p} = \sigma(z) = \frac{1}{1+e^{-z}}",
            "code": "from optimumai import Value\nz_val = Value(0.4)\np_hat = z_val.sigmoid()  # ≈ 0.599",
            "highlight_nodes": ["sigmoid_out", "predicted_prob"], "highlight_edges": ["e_linear_sig", "e_sig_prob"],
        },
        {
            "title": "Binary cross-entropy loss", "op": "loss",
            "narration": "The loss penalizes confident wrong predictions much more heavily than uncertain ones.",
            "formula": r"L = -[y\log\hat{p} + (1-y)\log(1-\hat{p})]",
            "code": 'import numpy as np\ny = 1  # true label\nloss = -(y*np.log(0.599) + (1-y)*np.log(0.401))\nprint(f"BCE: {loss:.4f}")  # ≈ 0.513',
            "highlight_nodes": ["loss"], "highlight_edges": ["e_prob_loss"],
        },
        {
            "title": "Gradient and weight update", "op": "update",
            "narration": "The gradient of the BCE loss with respect to the weights has a remarkably clean form: the prediction error times the input.",
            "formula": r"\nabla_w L = (\hat{p} - y)x, \quad w \leftarrow w - \eta\nabla_w L",
            "code": "from optimumai import LogisticRegression\nmodel = LogisticRegression()\nmodel.fit(X, y, explain=True)",
            "highlight_nodes": ["gradient", "update"], "highlight_edges": ["e_loss_grad", "e_grad_update"],
        },
    ),
}

# --------------------------------------------------------- 15. bias_variance_tradeoff
CONCEPTS["bias_variance_tradeoff"] = {
    "title": "Bias–Variance Tradeoff",
    "description": "Prediction error = Bias² + Variance + Noise. High bias means underfitting, high variance means overfitting. Regularization shifts the balance.",
    "nodes": _nodes(
        ("true_fn", "True function", "curve", "params"),
        ("model", "Fitted model", "curve", "branch"),
        ("bias_node", "Bias²", "scalar", "grad"),
        ("variance_node", "Variance", "scalar", "grad"),
        ("noise", "Irreducible noise σ²", "scalar", "params"),
        ("total_error", "Total error", "scalar", "generation"),
    ),
    "edges": _edges(
        ("e_true_bias", "true_fn", "bias_node", "s2"),
        ("e_model_bias", "model", "bias_node", "s2"),
        ("e_model_var", "model", "variance_node", "s3"),
        ("e_bias_total", "bias_node", "total_error", "s1"),
        ("e_var_total", "variance_node", "total_error", "s1"),
        ("e_noise_total", "noise", "total_error", "s1"),
    ),
    "steps": _steps(
        {
            "title": "Decompose expected error", "op": "decompose",
            "narration": "Any model's expected squared error can be split into three additive terms: how wrong it is on average (bias), how much it varies across training sets (variance), and noise it can never remove.",
            "formula": r"\mathbb{E}[(y - \hat{y})^2] = \text{Bias}^2 + \text{Var} + \sigma^2",
            "code": "# Total expected error = bias^2 + variance + irreducible noise\nbias_sq = (expected_pred - true_val)**2\nvariance = np.var(predictions_across_datasets)",
            "highlight_nodes": ["total_error"], "highlight_edges": ["e_bias_total", "e_var_total", "e_noise_total"],
        },
        {
            "title": "High bias: underfitting", "op": "bias",
            "narration": "A model too simple for the data (e.g. linear on non-linear structure) is systematically wrong — that's high bias.",
            "formula": r"\text{Bias} = \mathbb{E}[\hat{y}] - y",
            "code": "# Linear model on nonlinear data -> high bias\nfrom optimumai import LinearRegression\nmodel = LinearRegression()\nmodel.fit(X_nonlinear, y)  # systematically wrong",
            "highlight_nodes": ["true_fn", "model", "bias_node"], "highlight_edges": ["e_true_bias", "e_model_bias"],
        },
        {
            "title": "High variance: overfitting", "op": "variance",
            "narration": "A model that memorizes training data will fit wildly differently depending on which training set it saw — that's high variance.",
            "formula": r"\text{Var} = \mathbb{E}[(\hat{y} - \mathbb{E}[\hat{y}])^2]",
            "code": "from optimumai import DecisionTree\nmodel = DecisionTree(max_depth=None)  # memorizes training data\nmodel.fit(X, y)  # train=perfect, test=poor",
            "highlight_nodes": ["variance_node"], "highlight_edges": ["e_model_var"],
        },
        {
            "title": "Regularization controls the tradeoff", "op": "regularize",
            "narration": "Adding a penalty on model complexity trades a little bias for a lot less variance — usually a net win.",
            "formula": r"L_{\text{reg}} = L + \lambda\|\theta\|^2",
            "code": '# Ridge (L2) shrinks weights -> reduces variance\nfrom optimumai import LinearRegression\nmodel = LinearRegression(regularization="ridge", lam=0.1)\nmodel.fit(X, y, explain=True)',
            "highlight_nodes": ["total_error"], "highlight_edges": [],
        },
    ),
}

# ------------------------------------------------------------- 16. embedding_lookup
CONCEPTS["embedding_lookup"] = {
    "title": "Embedding Lookup",
    "description": "Map discrete token IDs to dense continuous vectors — a differentiable table lookup that's the first step in every language model.",
    "nodes": _nodes(
        ("token_id", "Token ID", "scalar", "ingestion"),
        ("embed_table", "Embedding table W_E", "matrix", "params"),
        ("embedding_vec", "Embedding vector", "vector", "branch"),
        ("position", "Position index", "scalar", "ingestion"),
        ("pos_enc", "Positional encoding", "vector", "branch"),
        ("combined", "Token + position", "vector", "generation"),
    ),
    "edges": _edges(
        ("e_id_table", "token_id", "embed_table", "s2"),
        ("e_table_vec", "embed_table", "embedding_vec", "s2"),
        ("e_pos_enc", "position", "pos_enc", "s3"),
        ("e_vec_combined", "embedding_vec", "combined", "s3"),
        ("e_pe_combined", "pos_enc", "combined", "s3"),
    ),
    "steps": _steps(
        {
            "title": "Token becomes an integer ID", "op": "tokenize",
            "narration": "Before a model can process text, every token needs a unique integer ID from the vocabulary.",
            "formula": r"\text{token} \xrightarrow{\text{vocab}} \text{id} \in \{0,\ldots,V-1\}",
            "code": 'from optimumai import BPETokenizer\ntok = BPETokenizer()\nids = tok.encode("the cat")  # e.g. [7, 342]',
            "highlight_nodes": ["token_id"], "highlight_edges": [],
        },
        {
            "title": "Embedding table lookup", "op": "lookup",
            "narration": "The embedding table is just a matrix, one learned row per vocabulary entry — lookup is a row index.",
            "formula": r"E[i] = W_E[\text{id}_i] \in \mathbb{R}^d",
            "code": "from optimumai import embedding_lookup\nW_E = np.random.randn(10000, 128)  # vocab x d\nvec = embedding_lookup(W_E, token_id=342)  # row 342",
            "highlight_nodes": ["embed_table", "embedding_vec"], "highlight_edges": ["e_id_table", "e_table_vec"],
        },
        {
            "title": "Add positional encoding", "op": "position",
            "narration": "Because attention has no inherent sense of order, a positional signal is added so the model knows where each token sits in the sequence.",
            "formula": r"\text{PE}_{(pos,2i)} = \sin\!\left(\frac{pos}{10000^{2i/d}}\right)",
            "code": "from optimumai import positional_encoding\npe = positional_encoding(seq_len=2, d_model=128)\nembeds = vec + pe  # token + position info",
            "highlight_nodes": ["position", "pos_enc", "combined"], "highlight_edges": ["e_pos_enc", "e_vec_combined", "e_pe_combined"],
        },
        {
            "title": "Gradients flow back into the table", "op": "backward",
            "narration": "During training, gradients update only the rows of the embedding table that were actually looked up — similar words end up nearby.",
            "formula": r"\frac{\partial L}{\partial W_E[i]} = \sum_t \delta_t \mathbf{1}[\text{id}_t = i]",
            "code": "from optimumai import nearest_neighbors\n# After training, similar tokens are nearby in embedding space\nnn = nearest_neighbors(W_E, vec, k=5)\nprint(nn.labels)",
            "highlight_nodes": ["combined"], "highlight_edges": [],
        },
    ),
}

# -------------------------------------------------------------- 17. kmeans_clustering
CONCEPTS["kmeans_clustering"] = {
    "title": "K-Means Clustering",
    "description": "Assign each point to its nearest centroid, then recompute centroids as cluster means. Repeat until convergence. Minimizes within-cluster variance.",
    "nodes": _nodes(
        ("data", "Data points", "matrix", "ingestion"),
        ("centroids", "Centroids μ_k", "matrix", "params"),
        ("assignments", "Cluster assignments", "vector", "branch"),
        ("new_centroids", "Updated centroids", "matrix", "branch"),
        ("inertia", "Inertia", "scalar", "generation"),
    ),
    "edges": _edges(
        ("e_data_assign", "data", "assignments", "s2"),
        ("e_cent_assign", "centroids", "assignments", "s2"),
        ("e_assign_new", "assignments", "new_centroids", "s3"),
        ("e_new_inertia", "new_centroids", "inertia", "s4"),
    ),
    "steps": _steps(
        {
            "title": "Initialize centroids randomly", "op": "init",
            "narration": "K-means starts by picking k initial centroid positions, often sampled from the data itself.",
            "formula": r"\mu_k \sim \mathcal{U}(\text{data})",
            "code": "from optimumai import KMeans\nimport numpy as np\nX = np.random.randn(100, 2)\nkmeans = KMeans(k=3, seed=0)\nkmeans.init_centroids(X)",
            "highlight_nodes": ["data", "centroids"], "highlight_edges": [],
        },
        {
            "title": "Assign each point to its nearest centroid", "op": "assign",
            "narration": "Every data point is labeled with whichever centroid it's closest to in Euclidean distance.",
            "formula": r"c_i = \arg\min_k \|x_i - \mu_k\|^2",
            "code": "assignments = kmeans.assign(X)\nprint(assignments[:10])  # [0, 2, 1, 0, ...]",
            "highlight_nodes": ["assignments"], "highlight_edges": ["e_data_assign", "e_cent_assign"],
        },
        {
            "title": "Recompute centroids as cluster means", "op": "update",
            "narration": "Each centroid moves to the average position of all points currently assigned to it.",
            "formula": r"\mu_k = \frac{1}{|C_k|}\sum_{i \in C_k} x_i",
            "code": "kmeans.update_centroids(X, assignments)\nprint(kmeans.centroids)  # new centroid positions",
            "highlight_nodes": ["new_centroids"], "highlight_edges": ["e_assign_new"],
        },
        {
            "title": "Convergence: inertia reaches a minimum", "op": "converge",
            "narration": "Repeating assign-then-update always decreases (or holds) inertia — the algorithm stops once it stabilizes.",
            "formula": r"\text{inertia} = \sum_i \|x_i - \mu_{c_i}\|^2",
            "code": 'kmeans.fit(X, explain=True)\nprint(f"Inertia: {kmeans.inertia_:.2f}")',
            "highlight_nodes": ["inertia"], "highlight_edges": ["e_new_inertia"],
        },
    ),
}

# ---------------------------------------------------------------------- 18. kv_cache
CONCEPTS["kv_cache"] = {
    "title": "KV Cache",
    "description": "Cache Key and Value matrices from previous tokens so decoding only computes attention for the new token — reduces attention cost from quadratic to linear per step.",
    "nodes": _nodes(
        ("past_tokens", "Past tokens", "list", "ingestion"),
        ("k_cache", "K cache", "matrix", "params"),
        ("v_cache", "V cache", "matrix", "params"),
        ("new_token", "New token", "scalar", "ingestion"),
        ("new_k", "new k_t", "vector", "branch"),
        ("new_v", "new v_t", "vector", "branch"),
        ("attn_out", "Attention output", "vector", "generation"),
    ),
    "edges": _edges(
        ("e_past_kcache", "past_tokens", "k_cache", "s2"),
        ("e_past_vcache", "past_tokens", "v_cache", "s2"),
        ("e_new_k", "new_token", "new_k", "s3"),
        ("e_new_v", "new_token", "new_v", "s3"),
        ("e_kcache_out", "k_cache", "attn_out", "s4"),
        ("e_vcache_out", "v_cache", "attn_out", "s4"),
        ("e_newk_out", "new_k", "attn_out", "s4"),
        ("e_newv_out", "new_v", "attn_out", "s4"),
    ),
    "steps": _steps(
        {
            "title": "Without a cache: recompute everything each step", "op": "naive",
            "narration": "Naively, generating each new token means re-running attention over the full sequence from scratch — wasteful and quadratic.",
            "formula": r"\text{Cost} \propto n^2 \quad (n = \text{sequence length})",
            "code": "# Naive: re-run full attention every token\ndef decode_naive(tokens):\n    for t in range(len(tokens)):\n        attn_out = full_attention(tokens[:t+1])",
            "highlight_nodes": ["past_tokens"], "highlight_edges": [],
        },
        {
            "title": "Build the KV cache from past tokens", "op": "build_cache",
            "narration": "Since the K and V projections of past tokens never change, we compute them once and store them.",
            "formula": r"K_{\text{cache}} = [k_1, k_2, \ldots, k_{t-1}]",
            "code": 'from optimumai import kv_cache_size\nmemory_mb = kv_cache_size(layers=32, heads=32,\n    head_dim=128, seq_len=4096)\nprint(f"{memory_mb:.1f} MB")',
            "highlight_nodes": ["k_cache", "v_cache"], "highlight_edges": ["e_past_kcache", "e_past_vcache"],
        },
        {
            "title": "New token: compute only its own k, v", "op": "extend",
            "narration": "Each decoding step only needs to project the single new token into a key and a value, then append it.",
            "formula": r"k_t = W_K x_t, \quad v_t = W_V x_t",
            "code": "# Append new k, v to cache -- O(d) not O(n*d)\nk_cache = np.vstack([k_cache, new_k])\nv_cache = np.vstack([v_cache, new_v])",
            "highlight_nodes": ["new_token", "new_k", "new_v"], "highlight_edges": ["e_new_k", "e_new_v"],
        },
        {
            "title": "Attend against the full cache", "op": "attend",
            "narration": "The new query attends over the entire cached history, but only one new row was ever computed — memory scales linearly with context.",
            "formula": r"\text{attn}_t = \text{softmax}\!\left(\frac{q_t K_{\text{cache}}^T}{\sqrt{d}}\right) V_{\text{cache}}",
            "code": 'from optimumai import kv_cache_size\n# Memory scales linearly with context, not quadratically\nfor seq in [1024, 4096, 16384]:\n    mb = kv_cache_size(layers=32, heads=32, head_dim=128, seq_len=seq)\n    print(f"seq={seq}: {mb:.0f} MB")',
            "highlight_nodes": ["attn_out"], "highlight_edges": ["e_kcache_out", "e_vcache_out", "e_newk_out", "e_newv_out"],
        },
    ),
}

# ------------------------------------------------------------------- 19. model_drift
CONCEPTS["model_drift"] = {
    "title": "Model Drift",
    "description": "Model performance degrades over time as the real-world data distribution shifts away from the training distribution — detect it with statistical tests.",
    "nodes": _nodes(
        ("train_dist", "Training distribution", "curve", "params"),
        ("prod_dist", "Production distribution", "curve", "branch"),
        ("feature_drift", "Feature drift test", "scalar", "branch"),
        ("label_drift", "Label / performance drift", "scalar", "grad"),
        ("perf_drop", "Performance drop", "scalar", "grad"),
        ("alert", "Retrain / adapt", "action", "generation"),
    ),
    "edges": _edges(
        ("e_train_feat", "train_dist", "feature_drift", "s2"),
        ("e_prod_feat", "prod_dist", "feature_drift", "s2"),
        ("e_feat_label", "feature_drift", "label_drift", "s3"),
        ("e_label_drop", "label_drift", "perf_drop", "s3"),
        ("e_drop_alert", "perf_drop", "alert", "s4"),
    ),
    "steps": _steps(
        {
            "title": "Training distribution vs. production", "op": "compare",
            "narration": "Over time, the data a deployed model sees can drift away from what it was trained on — new user behavior, seasonality, or upstream changes.",
            "formula": r"p_{\text{train}}(x) \neq p_{\text{prod}}(x)",
            "code": "import numpy as np\ntrain_dist = np.random.normal(loc=0, scale=1, size=1000)\nprod_dist  = np.random.normal(loc=0.8, scale=1.2, size=1000)\n# Means shifted, variance changed",
            "highlight_nodes": ["train_dist", "prod_dist"], "highlight_edges": [],
        },
        {
            "title": "Detect feature drift statistically", "op": "detect",
            "narration": "A statistical test (KS test, or KL divergence) quantifies how different the two distributions are.",
            "formula": r"D_{KL}(P\|Q) = \sum_x P(x)\log\frac{P(x)}{Q(x)}",
            "code": '# KS test: are these samples from the same distribution?\nfrom scipy import stats\nstat, p_val = stats.ks_2samp(train_dist, prod_dist)\nprint(f"p-value: {p_val:.4f}")  # small -> drift detected',
            "highlight_nodes": ["feature_drift"], "highlight_edges": ["e_train_feat", "e_prod_feat"],
        },
        {
            "title": "Performance metric degradation", "op": "measure",
            "narration": "Drift matters because it usually shows up as a real drop in downstream accuracy or business metrics.",
            "formula": r"\Delta \text{acc} = \text{acc}_{\text{train}} - \text{acc}_{\text{prod}}",
            "code": 'train_acc = 0.92\nprod_acc  = 0.74  # 18-point drop\ndrift_severity = train_acc - prod_acc\nif drift_severity > 0.1:\n    alert("Model drift detected -- consider retraining")',
            "highlight_nodes": ["label_drift", "perf_drop"], "highlight_edges": ["e_feat_label", "e_label_drop"],
        },
        {
            "title": "Mitigate: retrain or adapt", "op": "mitigate",
            "narration": "Once drift is confirmed, the usual fix is retraining on recent data or blending old and new distributions.",
            "formula": r"p_{\text{new}} \leftarrow \alpha\, p_{\text{train}} + (1-\alpha)\, p_{\text{recent}}",
            "code": "# Options: retrain on recent data, online adaptation,\n# or confidence-based rejection\nfrom optimumai import LinearRegression\nmodel_v2 = LinearRegression()\nmodel_v2.fit(X_recent, y_recent)",
            "highlight_nodes": ["alert"], "highlight_edges": ["e_drop_alert"],
        },
    ),
}

# -------------------------------------------------------------------------- 20. pca
CONCEPTS["pca"] = {
    "title": "Principal Component Analysis",
    "description": "Find the directions of maximum variance in data. Project data onto the top-k eigenvectors of the covariance matrix to reduce dimensionality while retaining structure.",
    "nodes": _nodes(
        ("x_data", "X data", "matrix", "ingestion"),
        ("centered", "Centered X̃", "matrix", "branch"),
        ("cov_matrix", "Covariance Σ", "matrix", "branch"),
        ("eigenvectors", "Eigenvectors V", "matrix", "branch"),
        ("eigenvalues", "Eigenvalues λ", "vector", "branch"),
        ("projection", "Projection Z", "matrix", "generation"),
        ("explained_var", "Explained variance ratio", "vector", "generation"),
    ),
    "edges": _edges(
        ("e_x_centered", "x_data", "centered", "s1"),
        ("e_centered_cov", "centered", "cov_matrix", "s2"),
        ("e_cov_eigvec", "cov_matrix", "eigenvectors", "s3"),
        ("e_cov_eigval", "cov_matrix", "eigenvalues", "s3"),
        ("e_eigvec_proj", "eigenvectors", "projection", "s4"),
        ("e_centered_proj", "centered", "projection", "s4"),
        ("e_eigval_var", "eigenvalues", "explained_var", "s5"),
    ),
    "steps": _steps(
        {
            "title": "Center the data", "op": "center",
            "narration": "PCA first removes each feature's mean so the data is centered at the origin — required before covariance makes sense.",
            "formula": r"\tilde{X} = X - \bar{X}",
            "code": "from optimumai import PCA\nimport numpy as np\nX = np.random.randn(100, 5)\nX_centered = X - X.mean(axis=0)",
            "highlight_nodes": ["x_data", "centered"], "highlight_edges": ["e_x_centered"],
        },
        {
            "title": "Compute the covariance matrix", "op": "covariance",
            "narration": "The covariance matrix captures how every pair of features varies together.",
            "formula": r"\Sigma = \frac{1}{n-1}\tilde{X}^T\tilde{X}",
            "code": "cov = np.cov(X_centered.T)  # shape (5, 5)\nprint(cov.shape)  # (5, 5)",
            "highlight_nodes": ["cov_matrix"], "highlight_edges": ["e_centered_cov"],
        },
        {
            "title": "Eigendecomposition of the covariance", "op": "eig",
            "narration": "The eigenvectors of the covariance matrix are the principal directions of variance; the eigenvalues tell us how much variance lies along each.",
            "formula": r"\Sigma = V\Lambda V^T, \quad \Lambda = \text{diag}(\lambda_1, \ldots, \lambda_d)",
            "code": "eigenvalues, eigenvectors = np.linalg.eigh(cov)\n# Sort by descending eigenvalue\nidx = np.argsort(eigenvalues)[::-1]\nPC = eigenvectors[:, idx]",
            "highlight_nodes": ["eigenvectors", "eigenvalues"], "highlight_edges": ["e_cov_eigvec", "e_cov_eigval"],
        },
        {
            "title": "Project onto the top-k eigenvectors", "op": "project",
            "narration": "Keeping only the eigenvectors with the largest eigenvalues gives a lower-dimensional projection that retains as much variance as possible.",
            "formula": r"Z = \tilde{X} V_k \quad (V_k = \text{top } k \text{ eigenvectors})",
            "code": "k = 2\nZ = X_centered @ PC[:, :k]  # shape (100, 2)",
            "highlight_nodes": ["projection"], "highlight_edges": ["e_eigvec_proj", "e_centered_proj"],
        },
        {
            "title": "How much variance did we keep?", "op": "explain_variance",
            "narration": "The explained variance ratio tells us what fraction of total variance survives the projection.",
            "formula": r"\text{EVR}_k = \frac{\sum_{i=1}^k \lambda_i}{\sum_i \lambda_i}",
            "code": "pca = PCA(n_components=2)\npca.fit(X, explain=True)\nprint(f\"Explained variance: {pca.explained_variance_ratio_}\")",
            "highlight_nodes": ["explained_var"], "highlight_edges": ["e_eigval_var"],
        },
    ),
}

# --------------------------------------------------------------------- 21. q_learning
CONCEPTS["q_learning"] = {
    "title": "Q-Learning",
    "description": "Learn an action-value function Q(s,a) via the Bellman equation and temporal difference updates — off-policy and model-free.",
    "nodes": _nodes(
        ("state", "State s", "scalar", "ingestion"),
        ("action", "Action a", "scalar", "query"),
        ("reward", "Reward r", "scalar", "branch"),
        ("next_state", "Next state s'", "scalar", "ingestion"),
        ("q_table", "Q-table", "matrix", "params"),
        ("td_error", "TD error δ", "scalar", "grad"),
        ("update", "Updated Q(s,a)", "scalar", "optim"),
    ),
    "edges": _edges(
        ("e_state_action", "state", "action", "s2"),
        ("e_qtable_action", "q_table", "action", "s2"),
        ("e_action_reward", "action", "reward", "s3"),
        ("e_action_next", "action", "next_state", "s3"),
        ("e_reward_td", "reward", "td_error", "s3"),
        ("e_next_td", "next_state", "td_error", "s3"),
        ("e_td_update", "td_error", "update", "s4"),
    ),
    "steps": _steps(
        {
            "title": "Q-table: value of every (state, action) pair", "op": "init",
            "narration": "Q-learning maintains a table estimating the long-run value of taking each action from each state.",
            "formula": r"Q(s, a) \in \mathbb{R}^{|S| \times |A|}",
            "code": "from optimumai import MDP, q_learning\nimport numpy as np\n# States=4, Actions=2\nQ = np.zeros((4, 2))  # Q-table initialized to 0",
            "highlight_nodes": ["q_table"], "highlight_edges": [],
        },
        {
            "title": "Choose an action (ε-greedy)", "op": "act",
            "narration": "The agent mostly exploits its current best-known action, but occasionally explores randomly to discover better ones.",
            "formula": r"a = \begin{cases} \arg\max_a Q(s,a) & \text{w.p. } 1-\varepsilon \\ \text{random} & \text{w.p. } \varepsilon \end{cases}",
            "code": "epsilon = 0.1\nif np.random.rand() < epsilon:\n    action = np.random.randint(2)  # explore\nelse:\n    action = np.argmax(Q[state])   # exploit",
            "highlight_nodes": ["state", "action"], "highlight_edges": ["e_state_action", "e_qtable_action"],
        },
        {
            "title": "Compute the TD error", "op": "td_error",
            "narration": "The temporal-difference error measures the gap between the current Q estimate and a better bootstrap estimate using the observed reward and the best next action.",
            "formula": r"\delta = r + \gamma \max_{a'} Q(s',a') - Q(s,a)",
            "code": "gamma = 0.9; lr = 0.1\nnext_val = np.max(Q[next_state])\ntd_error = reward + gamma * next_val - Q[state, action]",
            "highlight_nodes": ["reward", "next_state", "td_error"], "highlight_edges": ["e_action_reward", "e_action_next", "e_reward_td", "e_next_td"],
        },
        {
            "title": "Update the Q-value", "op": "update",
            "narration": "Q(s,a) is nudged toward the better bootstrap estimate, scaled by the learning rate.",
            "formula": r"Q(s,a) \leftarrow Q(s,a) + \alpha\delta",
            "code": "Q[state, action] += lr * td_error\n\n# Full training via optimumai:\nfrom optimumai import q_learning\nresult = q_learning(MDP.demo(), episodes=200, explain=True)",
            "highlight_nodes": ["update"], "highlight_edges": ["e_td_update"],
        },
    ),
}

# ----------------------------------------------- 22. reinforcement_learning_overview
CONCEPTS["reinforcement_learning_overview"] = {
    "title": "Reinforcement Learning",
    "description": "An agent learns by trial and error: take actions, receive rewards, update its policy to maximize cumulative discounted return G = Σ γᵗ r_t.",
    "nodes": _nodes(
        ("env", "Environment", "system", "ingestion"),
        ("agent", "Agent", "agent", "query"),
        ("state", "State s_t", "scalar", "ingestion"),
        ("action", "Action a_t", "scalar", "query"),
        ("reward", "Reward r_t", "scalar", "branch"),
        ("next_state", "State s_{t+1}", "scalar", "ingestion"),
        ("policy", "Policy π", "params", "params"),
        ("value_fn", "Value function V", "params", "generation"),
    ),
    "edges": _edges(
        ("e_state_agent", "state", "agent", "s1"),
        ("e_agent_action", "agent", "action", "s1"),
        ("e_action_env", "action", "env", "s1"),
        ("e_env_reward", "env", "reward", "s2"),
        ("e_env_next", "env", "next_state", "s2"),
        ("e_reward_value", "reward", "value_fn", "s3"),
        ("e_value_policy", "value_fn", "policy", "s4"),
    ),
    "steps": _steps(
        {
            "title": "Agent-environment interaction loop", "op": "loop",
            "narration": "The agent observes a state, picks an action via its policy, and the environment responds with a reward and a new state.",
            "formula": r"s_0 \xrightarrow{\pi} a_0 \xrightarrow{\text{env}} r_0, s_1 \xrightarrow{\pi} \ldots",
            "code": "from optimumai import MDP\nenv = MDP.demo()\nstate = env.reset()\nreward, next_state = env.step(action=0)",
            "highlight_nodes": ["state", "agent", "action", "env"], "highlight_edges": ["e_state_agent", "e_agent_action", "e_action_env"],
        },
        {
            "title": "Return: the discounted sum of future rewards", "op": "return",
            "narration": "The agent doesn't just care about the immediate reward — it optimizes the discounted sum of all future rewards.",
            "formula": r"G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k}",
            "code": "gamma = 0.9\nrewards = [1.0, 0.0, 1.0, 1.0, 0.0]\nG = sum(gamma**k * r for k, r in enumerate(rewards))\nprint(f\"Return: {G:.3f}\")",
            "highlight_nodes": ["reward", "next_state"], "highlight_edges": ["e_env_reward", "e_env_next"],
        },
        {
            "title": "Value function: expected return from a state", "op": "value",
            "narration": "The value function estimates how good it is to be in a given state, under the current policy.",
            "formula": r"V^\pi(s) = \mathbb{E}_\pi[G_t | s_t = s]",
            "code": "from optimumai import value_iteration, MDP\nmdp = MDP.demo()\nV = value_iteration(mdp, explain=True)\nprint(V)  # expected return from each state",
            "highlight_nodes": ["value_fn"], "highlight_edges": ["e_reward_value"],
        },
        {
            "title": "The Bellman equation: recursive structure", "op": "bellman",
            "narration": "Every value function satisfies a recursive relationship: the value of a state equals the expected immediate reward plus the discounted value of what comes next.",
            "formula": r"V^\pi(s) = \sum_a \pi(a|s)\sum_{s'} P(s'|s,a)[r + \gamma V^\pi(s')]",
            "code": "from optimumai import policy_iteration\npolicy = policy_iteration(mdp, explain=True)\nprint(policy)  # optimal action per state",
            "highlight_nodes": ["policy"], "highlight_edges": ["e_value_policy"],
        },
        {
            "title": "Policy gradient: learn directly from experience", "op": "policy_gradient",
            "narration": "Instead of learning values first, policy-gradient methods directly increase the probability of actions that led to high returns.",
            "formula": r"\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}[G_t \nabla_\theta \log\pi_\theta(a|s)]",
            "code": "from optimumai import reinforce\nresult = reinforce(mdp, episodes=1000, explain=True)",
            "highlight_nodes": ["policy"], "highlight_edges": [],
        },
    ),
}

# ----------------------------------------------------------- 23. sum_and_dot_product
CONCEPTS["sum_and_dot_product"] = {
    "title": "Sum & Dot Product",
    "description": "The summation and dot product are the most fundamental operations in AI: every weighted sum, attention score, and layer output traces back to these.",
    "nodes": _nodes(
        ("a_vec", "Vector a", "vector", "params"),
        ("b_vec", "Vector b", "vector", "params"),
        ("products", "Element-wise products", "vector", "branch"),
        ("sum_node", "Σ products", "scalar", "branch"),
        ("dot_result", "a · b", "scalar", "generation"),
    ),
    "edges": _edges(
        ("e_a_prod", "a_vec", "products", "s1"),
        ("e_b_prod", "b_vec", "products", "s1"),
        ("e_prod_sum", "products", "sum_node", "s2"),
        ("e_sum_dot", "sum_node", "dot_result", "s3"),
    ),
    "steps": _steps(
        {
            "title": "Multiply corresponding elements", "op": "multiply",
            "narration": "The first step of a dot product multiplies each pair of corresponding elements from the two vectors.",
            "formula": r"a_i \cdot b_i \quad \forall i",
            "code": "from optimumai import Vector\na = Vector([1.0, 2.0, 3.0])\nb = Vector([4.0, 5.0, 6.0])\nproducts = [ai * bi for ai, bi in zip(a, b)]  # [4, 10, 18]",
            "highlight_nodes": ["a_vec", "b_vec", "products"], "highlight_edges": ["e_a_prod", "e_b_prod"],
        },
        {
            "title": "Sum the products", "op": "sum",
            "narration": "Adding up every product collapses the vectors down to a single number.",
            "formula": r"\sum_i a_i b_i = 4 + 10 + 18 = 32",
            "code": "total = sum(products)  # 32",
            "highlight_nodes": ["sum_node"], "highlight_edges": ["e_prod_sum"],
        },
        {
            "title": "Dot product, all in one call", "op": "dot",
            "narration": "optimumai's Vector.dot does the multiply-and-sum in a single explained call.",
            "formula": r"a \cdot b = \|a\|\|b\|\cos\theta",
            "code": "result = a.dot(b, explain=True)\nprint(result)  # 32.0",
            "highlight_nodes": ["dot_result"], "highlight_edges": ["e_sum_dot"],
        },
        {
            "title": "Cosine similarity: a normalized dot product", "op": "cosine",
            "narration": "Dividing the dot product by both vectors' magnitudes gives a similarity score that ignores scale — the basis of embedding similarity search.",
            "formula": r"\cos\theta = \frac{a \cdot b}{\|a\|\|b\|}",
            "code": 'sim = a.cosine_similarity(b, explain=True)\nprint(f"Cosine similarity: {sim:.4f}")',
            "highlight_nodes": ["dot_result"], "highlight_edges": [],
        },
    ),
}

# --------------------------------------------------------------- 24. supervised_ml
CONCEPTS["supervised_ml"] = {
    "title": "Supervised Machine Learning",
    "description": "Learn a mapping f: X → y from labeled examples (input, label) pairs. Training minimizes a loss between predictions and true labels.",
    "nodes": _nodes(
        ("training_data", "Training data (X, y)", "matrix", "ingestion"),
        ("model", "Model f_θ", "params", "params"),
        ("prediction", "Prediction ŷ", "vector", "branch"),
        ("loss", "Loss", "scalar", "branch"),
        ("gradient", "Gradient", "vector", "grad"),
        ("updated_model", "Updated model", "params", "optim"),
        ("test_data", "Test data", "matrix", "ingestion"),
        ("evaluation", "Test accuracy", "scalar", "generation"),
    ),
    "edges": _edges(
        ("e_data_model", "training_data", "model", "s1"),
        ("e_model_pred", "model", "prediction", "s2"),
        ("e_pred_loss", "prediction", "loss", "s3"),
        ("e_loss_grad", "loss", "gradient", "s4"),
        ("e_grad_updated", "gradient", "updated_model", "s4"),
        ("e_updated_test", "updated_model", "test_data", "s5"),
        ("e_test_eval", "test_data", "evaluation", "s5"),
    ),
    "steps": _steps(
        {
            "title": "Labeled training data", "op": "data",
            "narration": "Supervised learning requires a dataset where every input has a known, correct output.",
            "formula": r"\mathcal{D} = \{(x_i, y_i)\}_{i=1}^n",
            "code": "import numpy as np\nX_train = np.random.randn(100, 4)  # 100 samples, 4 features\ny_train = (X_train[:, 0] + X_train[:, 1] > 0).astype(int)",
            "highlight_nodes": ["training_data"], "highlight_edges": [],
        },
        {
            "title": "The model produces a prediction", "op": "predict",
            "narration": "A parameterized model maps every input to a prediction — initially, a bad one.",
            "formula": r"\hat{y} = f_\theta(x)",
            "code": "from optimumai import LogisticRegression\nmodel = LogisticRegression()\ny_pred = model.predict_proba(X_train)  # before training",
            "highlight_nodes": ["model", "prediction"], "highlight_edges": ["e_data_model", "e_model_pred"],
        },
        {
            "title": "Compute the loss", "op": "loss",
            "narration": "The loss measures how far off the current predictions are from the true labels.",
            "formula": r"L(\theta) = \frac{1}{n}\sum_i \ell(y_i, f_\theta(x_i))",
            "code": '# For regression: MSE; for classification: cross-entropy\nloss = model.loss(X_train, y_train)\nprint(f"Loss: {loss:.4f}")',
            "highlight_nodes": ["loss"], "highlight_edges": ["e_pred_loss"],
        },
        {
            "title": "Minimize loss to update parameters", "op": "fit",
            "narration": "Gradient-based optimization repeatedly nudges the parameters to reduce the loss.",
            "formula": r"\theta^* = \arg\min_\theta L(\theta)",
            "code": "model.fit(X_train, y_train, explain=True)",
            "highlight_nodes": ["gradient", "updated_model"], "highlight_edges": ["e_loss_grad", "e_grad_updated"],
        },
        {
            "title": "Evaluate on held-out test data", "op": "evaluate",
            "narration": "The real test of a model is how well it generalizes to data it never trained on.",
            "formula": r"\text{acc} = \frac{1}{m}\sum_j \mathbf{1}[\hat{y}_j = y_j]",
            "code": 'from optimumai.ml.metrics import accuracy\ny_test_pred = model.predict(X_test)\nacc = accuracy(y_test, y_test_pred)\nprint(f"Test accuracy: {acc:.2%}")',
            "highlight_nodes": ["test_data", "evaluation"], "highlight_edges": ["e_updated_test", "e_test_eval"],
        },
    ),
}

# ------------------------------------------------------------- 25. unsupervised_ml
CONCEPTS["unsupervised_ml"] = {
    "title": "Unsupervised Machine Learning",
    "description": "Find structure in data without labels: cluster similar points, reduce dimensionality, or model the data distribution itself.",
    "nodes": _nodes(
        ("raw_data", "Raw data (no labels)", "matrix", "ingestion"),
        ("clustering", "Clustering", "branch", "branch"),
        ("pca_reduction", "Dimensionality reduction", "branch", "branch"),
        ("distribution_model", "Density / anomaly model", "branch", "branch"),
        ("patterns", "Discovered patterns", "generation", "generation"),
    ),
    "edges": _edges(
        ("e_data_cluster", "raw_data", "clustering", "s2"),
        ("e_data_pca", "raw_data", "pca_reduction", "s3"),
        ("e_data_dist", "raw_data", "distribution_model", "s4"),
        ("e_cluster_patterns", "clustering", "patterns", "s4"),
        ("e_pca_patterns", "pca_reduction", "patterns", "s4"),
        ("e_dist_patterns", "distribution_model", "patterns", "s4"),
    ),
    "steps": _steps(
        {
            "title": "Data without labels", "op": "data",
            "narration": "Unlike supervised learning, there's no target y — the goal is to discover structure in X alone.",
            "formula": r"\mathcal{D} = \{x_i\}_{i=1}^n \quad (\text{no } y_i)",
            "code": "import numpy as np\nX = np.random.randn(200, 10)  # 200 samples, 10 features\n# Goal: discover structure without labels",
            "highlight_nodes": ["raw_data"], "highlight_edges": [],
        },
        {
            "title": "Clustering: find natural groups", "op": "cluster",
            "narration": "Clustering algorithms partition data into groups such that points within a group are more similar to each other than to points in other groups.",
            "formula": r"\min_{\mu} \sum_i \min_k \|x_i - \mu_k\|^2",
            "code": "from optimumai import KMeans\nkmeans = KMeans(k=4)\nlabels = kmeans.fit_predict(X)\nprint(f\"4 clusters found\")",
            "highlight_nodes": ["clustering"], "highlight_edges": ["e_data_cluster"],
        },
        {
            "title": "Dimensionality reduction via PCA", "op": "reduce",
            "narration": "High-dimensional data can often be projected into a much smaller space while preserving most of its structure.",
            "formula": r"Z = X V_k \quad (k \ll d)",
            "code": "from optimumai import PCA\npca = PCA(n_components=2)\nZ = pca.fit_transform(X)  # 200x10 -> 200x2\nprint(f\"Kept {pca.explained_variance_ratio_.sum():.0%} of variance\")",
            "highlight_nodes": ["pca_reduction"], "highlight_edges": ["e_data_pca"],
        },
        {
            "title": "Anomaly detection: low-density regions", "op": "anomaly",
            "narration": "Points that don't fit any discovered structure well — far from every cluster, or in a low-probability region — are flagged as anomalies.",
            "formula": r"\text{anomaly} \Leftrightarrow p(x) < \tau",
            "code": "# Points far from any centroid are anomalous\ndists = kmeans.transform(X).min(axis=1)  # dist to nearest centroid\nthreshold = np.percentile(dists, 95)\nanomalies = X[dists > threshold]",
            "highlight_nodes": ["distribution_model", "patterns"], "highlight_edges": ["e_data_dist", "e_cluster_patterns", "e_pca_patterns", "e_dist_patterns"],
        },
    ),
}

# -------------------------------------------------------------------- 26. tokenizer
CONCEPTS["tokenizer"] = {
    "title": "Tokenizer & BPE",
    "description": "Split raw text into a vocabulary of sub-word pieces using Byte Pair Encoding (BPE). Used in GPT, BERT, and all modern LLMs.",
    "nodes": _nodes(
        ("raw_text", "Raw corpus", "text", "ingestion"),
        ("chars", "Character vocab", "list", "branch"),
        ("byte_pairs", "Most frequent pair", "scalar", "branch"),
        ("merges", "Merge rules", "list", "branch"),
        ("vocab", "Final vocabulary", "list", "generation"),
        ("token_ids", "Encoded token IDs", "vector", "generation"),
    ),
    "edges": _edges(
        ("e_text_chars", "raw_text", "chars", "s1"),
        ("e_chars_pairs", "chars", "byte_pairs", "s2"),
        ("e_pairs_merges", "byte_pairs", "merges", "s3"),
        ("e_merges_vocab", "merges", "vocab", "s4"),
        ("e_vocab_ids", "vocab", "token_ids", "s5"),
    ),
    "steps": _steps(
        {
            "title": "Start: character-level vocabulary", "op": "init",
            "narration": "BPE begins with the simplest possible vocabulary — every unique character in the corpus.",
            "formula": r"V_0 = \{\text{unique chars in corpus}\}",
            "code": 'from optimumai import BPETokenizer\ncorpus = ["low", "lower", "lowest", "newer", "widest"]\ntok = BPETokenizer()\nvocab = tok.init_vocab(corpus)\nprint(vocab)',
            "highlight_nodes": ["raw_text", "chars"], "highlight_edges": ["e_text_chars"],
        },
        {
            "title": "Find the most frequent adjacent pair", "op": "count",
            "narration": "BPE scans the corpus for the pair of adjacent symbols that occurs most often.",
            "formula": r"(a, b) = \arg\max_{(a,b)} \text{freq}(ab)",
            "code": "pair = tok.most_frequent_pair(corpus)\nprint(f\"Most frequent pair: {pair}\")  # e.g. ('e', 'r')",
            "highlight_nodes": ["byte_pairs"], "highlight_edges": ["e_chars_pairs"],
        },
        {
            "title": "Merge that pair into a new token", "op": "merge",
            "narration": "The most frequent pair is merged into a single new symbol, added to the vocabulary.",
            "formula": r"V_{i+1} = V_i \cup \{ab\} \setminus \{a, b\}",
            "code": "tok.merge(pair)\nprint(tok.vocab)  # 'e'+'r' merged into 'er'",
            "highlight_nodes": ["merges"], "highlight_edges": ["e_pairs_merges"],
        },
        {
            "title": "Repeat for n merges", "op": "iterate",
            "narration": "Repeating the count-and-merge step many times grows the vocabulary from individual characters up to common sub-words.",
            "formula": r"V_n = V_0 + n \text{ merges}",
            "code": 'tok.learn_merges(corpus, num_merges=8, explain=True)\nprint(f"Vocab size: {len(tok.vocab)}")',
            "highlight_nodes": ["vocab"], "highlight_edges": ["e_merges_vocab"],
        },
        {
            "title": "Encode new text with the learned vocabulary", "op": "encode",
            "narration": "Once training is done, any new text is greedily split using the learned merge rules.",
            "formula": r"\text{encode}(t) = [v_1, v_2, \ldots]",
            "code": 'ids = tok.encode("lower")\nprint(ids)  # e.g. [12, 5]  (sub-word IDs)',
            "highlight_nodes": ["token_ids"], "highlight_edges": ["e_vocab_ids"],
        },
    ),
}

# --------------------------------------------------------------- 27. transformer_block
CONCEPTS["transformer_block"] = {
    "title": "Transformer Block",
    "description": "The building block of GPT/BERT: multi-head attention (so each token can attend anywhere) + feed-forward layer + residual connections + layer norm.",
    "nodes": _nodes(
        ("residual_in", "Input x", "vector", "params"),
        ("layer_norm1", "LayerNorm 1", "vector", "branch"),
        ("mha", "Multi-head attention", "matrix", "branch"),
        ("residual1", "x + MHA(LN(x))", "vector", "branch"),
        ("layer_norm2", "LayerNorm 2", "vector", "branch"),
        ("ffn", "Feed-forward", "vector", "branch"),
        ("residual2", "x + FFN(LN(x))", "vector", "generation"),
        ("output", "Block output", "vector", "generation"),
    ),
    "edges": _edges(
        ("e_in_ln1", "residual_in", "layer_norm1", "s2"),
        ("e_ln1_mha", "layer_norm1", "mha", "s3"),
        ("e_mha_res1", "mha", "residual1", "s3"),
        ("e_in_res1", "residual_in", "residual1", "s3"),
        ("e_res1_ln2", "residual1", "layer_norm2", "s4"),
        ("e_ln2_ffn", "layer_norm2", "ffn", "s4"),
        ("e_ffn_res2", "ffn", "residual2", "s4"),
        ("e_res1_res2", "residual1", "residual2", "s4"),
        ("e_res2_out", "residual2", "output", "s5"),
    ),
    "steps": _steps(
        {
            "title": "Input residual stream", "op": "input",
            "narration": "A transformer block takes in one vector per token — the running 'residual stream' that every layer reads from and writes back to.",
            "formula": r"x \in \mathbb{R}^{T \times d}",
            "code": "from optimumai import TransformerBlock\nimport numpy as np\nx = np.random.randn(4, 64)  # 4 tokens, d=64\nblock = TransformerBlock(d_model=64, n_heads=4, d_ff=256)",
            "highlight_nodes": ["residual_in"], "highlight_edges": [],
        },
        {
            "title": "Layer norm before attention", "op": "norm",
            "narration": "Pre-norm transformers normalize the input before feeding it into attention, which stabilizes deep training.",
            "formula": r"x' = \text{LayerNorm}(x)",
            "code": "x_norm = block.ln1(x)  # normalize over d=64 dim",
            "highlight_nodes": ["layer_norm1"], "highlight_edges": ["e_in_ln1"],
        },
        {
            "title": "Multi-head self-attention + residual", "op": "attention",
            "narration": "Attention lets every token gather information from every other token; the result is added back to the original stream via a residual connection.",
            "formula": r"x \leftarrow x + \text{MHA}(\text{LN}(x))",
            "code": "from optimumai import MultiHeadAttention\nattn_out = block.attn(x_norm)\nx = x + attn_out  # residual connection",
            "highlight_nodes": ["mha", "residual1"], "highlight_edges": ["e_ln1_mha", "e_mha_res1", "e_in_res1"],
        },
        {
            "title": "Layer norm, feed-forward, and residual again", "op": "ffn",
            "narration": "A second normalize-transform-residual cycle applies a position-wise MLP, giving the model room to process each token independently.",
            "formula": r"x \leftarrow x + \text{FFN}(\text{LN}(x))",
            "code": "x_norm2 = block.ln2(x)\nffn_out = block.ffn(x_norm2)  # Linear -> ReLU/GELU -> Linear\nx = x + ffn_out",
            "highlight_nodes": ["layer_norm2", "ffn", "residual2"], "highlight_edges": ["e_res1_ln2", "e_ln2_ffn", "e_ffn_res2", "e_res1_res2"],
        },
        {
            "title": "Full block forward pass", "op": "forward",
            "narration": "Stacking many of these blocks, each refining the residual stream, is literally what GPT/BERT are made of.",
            "formula": r"y = \text{Block}(x) = x + \text{FFN}(\text{LN}(x + \text{MHA}(\text{LN}(x))))",
            "code": "y = block.forward(x)  # all steps together\nprint(y.shape)  # (4, 64) -- same shape, richer representation",
            "highlight_nodes": ["output"], "highlight_edges": ["e_res2_out"],
        },
    ),
}

# ------------------------------------------------------------------------ 28. variance
CONCEPTS["variance"] = {
    "title": "Variance & Standard Deviation",
    "description": "Variance measures how spread out a distribution is. It's the average squared deviation from the mean — fundamental to statistics and all of ML.",
    "nodes": _nodes(
        ("data", "Data x", "vector", "ingestion"),
        ("mean_node", "Mean x̄", "scalar", "branch"),
        ("deviations", "Deviations", "vector", "branch"),
        ("squared_devs", "Squared deviations", "vector", "branch"),
        ("variance_node", "Variance σ²", "scalar", "generation"),
        ("std_node", "Std σ", "scalar", "generation"),
    ),
    "edges": _edges(
        ("e_data_mean", "data", "mean_node", "s1"),
        ("e_data_dev", "data", "deviations", "s2"),
        ("e_mean_dev", "mean_node", "deviations", "s2"),
        ("e_dev_sq", "deviations", "squared_devs", "s3"),
        ("e_sq_var", "squared_devs", "variance_node", "s3"),
        ("e_var_std", "variance_node", "std_node", "s4"),
    ),
    "steps": _steps(
        {
            "title": "Data and its mean", "op": "mean",
            "narration": "Every measure of spread starts by locating the center of the data — the arithmetic mean.",
            "formula": r"\bar{x} = \frac{1}{n}\sum_i x_i",
            "code": "import numpy as np\nx = np.array([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])\nmean = x.mean()  # 5.0",
            "highlight_nodes": ["data", "mean_node"], "highlight_edges": ["e_data_mean"],
        },
        {
            "title": "Deviations from the mean", "op": "deviate",
            "narration": "Subtracting the mean from every point shows how far each one strays from the center.",
            "formula": r"d_i = x_i - \bar{x}",
            "code": "deviations = x - mean\nprint(deviations)  # [-3, -1, -1, -1, 0, 0, 2, 4]",
            "highlight_nodes": ["deviations"], "highlight_edges": ["e_data_dev", "e_mean_dev"],
        },
        {
            "title": "Variance: average squared deviation", "op": "variance",
            "narration": "Squaring the deviations before averaging keeps them positive and penalizes large deviations more heavily.",
            "formula": r"\sigma^2 = \frac{1}{n}\sum_i (x_i - \bar{x})^2",
            "code": "variance = np.mean(deviations**2)  # population var\nvar_sample = np.var(x, ddof=1)  # sample var (n-1)\nprint(f\"σ² = {variance:.2f}\")  # = 4.0",
            "highlight_nodes": ["squared_devs", "variance_node"], "highlight_edges": ["e_dev_sq", "e_sq_var"],
        },
        {
            "title": "Standard deviation: back to the original units", "op": "std",
            "narration": "Taking the square root of variance returns to the same units as the original data, making it directly interpretable.",
            "formula": r"\sigma = \sqrt{\sigma^2}",
            "code": "std = np.sqrt(variance)\nprint(f\"σ = {std:.2f}\")  # = 2.0\n# 68% of data within 1 sigma of the mean in a normal distribution",
            "highlight_nodes": ["std_node"], "highlight_edges": ["e_var_std"],
        },
    ),
}

# ------------------------------------------------------------------------ 29. dropout
CONCEPTS["dropout"] = {
    "title": "Dropout Regularization",
    "description": "During training, randomly zero out each neuron activation with probability p. Forces the network to learn redundant representations, reducing overfitting.",
    "nodes": _nodes(
        ("layer_input", "Pre-dropout activations", "vector", "params"),
        ("mask", "Bernoulli mask", "vector", "branch"),
        ("dropped_out", "Masked activations", "vector", "branch"),
        ("scaled_out", "Inverted-scaled output", "vector", "generation"),
        ("layer_output", "Layer output", "vector", "generation"),
    ),
    "edges": _edges(
        ("e_input_mask", "layer_input", "mask", "s2"),
        ("e_input_drop", "layer_input", "dropped_out", "s3"),
        ("e_mask_drop", "mask", "dropped_out", "s3"),
        ("e_drop_scaled", "dropped_out", "scaled_out", "s3"),
        ("e_scaled_out", "scaled_out", "layer_output", "s4"),
    ),
    "steps": _steps(
        {
            "title": "Pre-dropout activations", "op": "input",
            "narration": "Dropout is applied to a layer's activations, after the non-linearity.",
            "formula": r"h = \text{ReLU}(Wx + b)",
            "code": "import numpy as np\nh = np.array([0.5, 1.2, 0.0, 2.1, -0.3, 0.8])  # layer output",
            "highlight_nodes": ["layer_input"], "highlight_edges": [],
        },
        {
            "title": "Sample a dropout mask (p = 0.5)", "op": "sample",
            "narration": "A random binary mask is sampled each forward pass — each neuron independently survives with probability 1 - p.",
            "formula": r"m_i \sim \text{Bernoulli}(1 - p)",
            "code": "p = 0.5\nmask = np.random.binomial(1, 1-p, size=h.shape)\nprint(mask)  # e.g. [1, 0, 1, 0, 1, 1]",
            "highlight_nodes": ["mask"], "highlight_edges": ["e_input_mask"],
        },
        {
            "title": "Apply the mask (inverted dropout)", "op": "mask",
            "narration": "Masked-out neurons contribute nothing; surviving neurons are scaled up by 1/(1-p) so the expected output stays the same as without dropout.",
            "formula": r"\tilde{h}_i = \frac{h_i \cdot m_i}{1-p}",
            "code": "h_dropped = (h * mask) / (1 - p)\nprint(h_dropped)  # zeros some neurons, scales up the rest",
            "highlight_nodes": ["dropped_out", "scaled_out"], "highlight_edges": ["e_input_drop", "e_mask_drop", "e_drop_scaled"],
        },
        {
            "title": "At inference: no dropout applied", "op": "inference",
            "narration": "During evaluation, every neuron is kept active — the inverted scaling during training ensures this matches the expected training-time output.",
            "formula": r"\tilde{h}^{(\text{test})}_i = h_i \quad \forall i",
            "code": "# During eval, keep all neurons (no masking)\n# The 1/(1-p) scaling ensures the same expected output\nif training:\n    h = (h * np.random.binomial(1, 1-p, h.shape)) / (1-p)\nelse:\n    pass  # h unchanged",
            "highlight_nodes": ["layer_output"], "highlight_edges": ["e_scaled_out"],
        },
    ),
}

# --------------------------------------------------------------------------- 30. tfidf
CONCEPTS["tfidf"] = {
    "title": "TF-IDF",
    "description": "Term Frequency–Inverse Document Frequency weights words by how often they appear in a document (TF) but penalizes words common across all documents (IDF).",
    "nodes": _nodes(
        ("corpus", "Corpus (documents)", "text", "ingestion"),
        ("term_freq", "Term frequency (TF)", "matrix", "branch"),
        ("doc_freq", "Document frequency (DF)", "vector", "branch"),
        ("idf", "IDF", "vector", "branch"),
        ("tfidf_matrix", "TF-IDF matrix", "matrix", "generation"),
        ("query_vec", "Query vector", "vector", "query"),
    ),
    "edges": _edges(
        ("e_corpus_tf", "corpus", "term_freq", "s1"),
        ("e_corpus_df", "corpus", "doc_freq", "s2"),
        ("e_df_idf", "doc_freq", "idf", "s3"),
        ("e_tf_matrix", "term_freq", "tfidf_matrix", "s4"),
        ("e_idf_matrix", "idf", "tfidf_matrix", "s4"),
    ),
    "steps": _steps(
        {
            "title": "Term frequency: how often a word appears in a document", "op": "tf",
            "narration": "TF measures local importance — how often a word shows up within one particular document, normalized by document length.",
            "formula": r"\text{TF}(t, d) = \frac{f_{t,d}}{\sum_{t'} f_{t',d}}",
            "code": 'from optimumai import TfidfVectorizer\ndocs = ["the cat sat on the mat", "the cat is a cat"]\nvec = TfidfVectorizer()\nvec.fit(docs)',
            "highlight_nodes": ["corpus", "term_freq"], "highlight_edges": ["e_corpus_tf"],
        },
        {
            "title": "Document frequency: how many documents contain the word", "op": "df",
            "narration": "DF counts, across the whole corpus, in how many documents each word appears at all.",
            "formula": r"\text{DF}(t) = |\{d : t \in d\}|",
            "code": "df = vec.document_freq_\nprint(df)  # {'the': 2, 'cat': 2, 'sat': 1, ...}",
            "highlight_nodes": ["doc_freq"], "highlight_edges": ["e_corpus_df"],
        },
        {
            "title": "IDF: rare words get a higher weight", "op": "idf",
            "narration": "The inverse document frequency downweights words that appear in almost every document — they carry little distinguishing signal.",
            "formula": r"\text{IDF}(t) = \log\frac{N}{\text{DF}(t)}",
            "code": "import numpy as np\nN = len(docs)  # 2\nidf = {t: np.log(N / df[t]) for t in df}\nprint(idf)  # 'sat': log(2/1)=0.69, 'the': log(2/2)=0",
            "highlight_nodes": ["idf"], "highlight_edges": ["e_df_idf"],
        },
        {
            "title": "TF-IDF: score every (document, word) pair", "op": "tfidf",
            "narration": "Multiplying TF by IDF rewards words that are frequent locally but rare globally — exactly the words that best characterize a document.",
            "formula": r"\text{TF-IDF}(t, d) = \text{TF}(t,d) \cdot \text{IDF}(t)",
            "code": "matrix = vec.transform(docs)  # explain=True shows steps\nprint(matrix)  # shape (2, vocab_size)\n# 'the' gets 0 (too common); 'sat' gets a high score",
            "highlight_nodes": ["tfidf_matrix"], "highlight_edges": ["e_tf_matrix", "e_idf_matrix"],
        },
    ),
}


# --------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------


def _build_html(concept: str) -> str:
    data = CONCEPTS[concept]
    trace = {
        "schema_version": "1.0",
        "concept": concept,
        "title": data["title"],
        "description": data["description"],
        "nodes": data["nodes"],
        "edges": data["edges"],
        "steps": data["steps"],
        "meta": {},
    }
    return HTML_TEMPLATE.replace("__TRACE_JSON__", json.dumps(trace)).replace(
        "__TITLE__", data["title"]
    )


def _write_html(path: Path, html: str) -> str:
    """Write HTML to disk, creating parent directories if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return str(path)


def list_explain_concepts() -> list[str]:
    """Return every concept key :func:`explain` accepts, sorted alphabetically."""
    return sorted(CONCEPTS)


def explain(concept: str, out: str | None = None, open_browser: bool = True) -> str:
    """Build an interactive DAG explainer (formula + code per step) for ``concept``.

    Args:
        concept: One of :func:`list_explain_concepts` (case/dash/space-insensitive).
        out: Output HTML path; defaults to ``explain_{concept}.html``.
        open_browser: Open the generated file in the default browser.

    Returns:
        The path the HTML was written to.
    """
    key = concept.lower().strip().replace("-", "_").replace(" ", "_")
    if key not in CONCEPTS:
        valid = ", ".join(list_explain_concepts())
        raise ValueError(f"unknown concept {concept!r}; choose from: {valid}")
    html = _build_html(key)
    target = Path(out) if out is not None else Path(f"explain_{key}.html")
    _write_html(target, html)
    if open_browser:
        webbrowser.open(f"file://{target.resolve()}")
    return str(target)


def explore_concepts(out: str | None = None, open_browser: bool = True) -> str:
    """Build a searchable landing page linking to every :func:`explain` concept.

    Each card links to ``explain_{key}.html`` alongside the generated page; run
    :func:`explain` for a concept (or ``optimumai explain <concept>``) to
    inspect one directly.
    """
    target = Path(out) if out is not None else Path("explore.html")
    concept_dir = target.parent
    cards = [
        {"key": key, "title": data["title"], "description": data["description"]}
        for key, data in sorted(CONCEPTS.items())
    ]
    for card in cards:
        explain_path = concept_dir / f"explain_{card['key']}.html"
        _write_html(explain_path, _build_html(card["key"]))
    html = EXPLORE_TEMPLATE.replace("__CONCEPTS_JSON__", json.dumps(cards)).replace(
        "__COUNT__", str(len(cards))
    )
    _write_html(target, html)
    if open_browser:
        webbrowser.open(f"file://{target.resolve()}")
    return str(target)
