"""The transformer forward pass, drawn as a circuit — the flagship flow.

``transformer_flow`` takes a short sentence and renders every stage of a toy
(but numerically real) transformer forward pass as an interactive, self-
contained HTML page:

    tokens -> embeddings -> + positional -> Q,K,V -> scores = QKᵀ/√d ->
    softmax -> attention matrix -> weighted sum (·V) -> FFN -> logits ->
    softmax over vocab

Every number on the page — the embedding table, Q/K/V, the attention matrix,
the FFN activations, the final next-token distribution — is computed once in
Python with a seeded RNG (so the page is fully deterministic), then embedded
as JSON and drawn as inline SVG. The browser owns only the *interaction*: a
Step control walks through the pipeline one stage at a time, dimming stages
not yet reached, and every matrix/vector cell responds to hover with the
exact value and a plain-language explanation of what it represents.

This intentionally mirrors ``optimumai.transformers.attention`` and
``optimumai.transformers.text_pipeline`` (same formulas, same scaling), but
recomputes a single, simplified head directly so the diagram can show every
intermediate (Q, K, V separately, not just the fused block output).
"""

from __future__ import annotations

import re

import numpy as np

from optimumai.flows._shared import (
    HOVER_TOOLTIP_JS,
    arrow,
    flow_controls_html,
    matrix_grid,
    page,
    runtime_script,
    stage_box,
    stage_group_close,
    stage_group_open,
    svg_open,
    vector_chip,
    write,
)
from optimumai.transformers.positional import positional_encoding

_WORD_RE = re.compile(r"[a-z0-9]+")

_CSS_EXTRA = """
 #flow-svg{min-width:1500px}
 .flow-stage{transition:opacity .25s}
"""


def _tokenize(text: str) -> list[str]:
    tokens = _WORD_RE.findall(text.lower())
    if not tokens:
        raise ValueError("text must contain at least one word character")
    return tokens


def transformer_flow(
    text: str = "the cat sat on the mat",
    d_model: int = 8,
    seed: int = 0,
    temperature: float = 1.0,
    top_k: int = 5,
    top_p: float = 0.9,
    out: str | None = None,
) -> str:
    """Build an interactive transformer explainer for ``text`` as self-contained HTML.

    Args:
        text: A prompt to analyze.
        d_model: Embedding / model dimension for the toy transformer.
        seed: Seed for the (untrained) Q/K/V and feed-forward weight matrices.
        temperature: Initial sampling temperature.
        top_k: Initial top-k filter for the next-token head.
        top_p: Initial nucleus filter for the next-token head.
        out: Path to write the HTML to (defaults to ``"transformer_flow.html"``).

    Returns:
        The path the HTML was written to.
    """
    import json

    def _tokenize_live(text_: str) -> list[str]:
        tokens = re.findall(r"[A-Za-z0-9']+|[^\sA-Za-z0-9']", text_.strip().lower())
        return tokens[:12] or ["<empty>"]

    def _rand_rows(rng: np.random.Generator, rows: int, cols: int, scale: float) -> list[list[float]]:
        return (rng.normal(size=(rows, cols)) * scale).round(4).tolist()

    def _build_stage_svg() -> str:
        stage_defs = [
            ("tokenize", "1. Tokenize"),
            ("embed", "2. Embed + position"),
            ("qkv", "3. Q, K, V"),
            ("scores", "4. Scores"),
            ("attention", "5. Attention"),
            ("ffn", "6. Feed-forward"),
            ("output", "7. Output"),
        ]
        box_w, box_h, gap = 172, 42, 44
        svg_w = 40 + len(stage_defs) * (box_w + gap)
        svg_h = 122
        row_y = 30
        parts = [svg_open(svg_w, svg_h)]
        for idx, (sid, title) in enumerate(stage_defs):
            gx = 20 + idx * (box_w + gap)
            parts.append(stage_group_open(sid, gx, row_y))
            parts.append(stage_box(box_w, box_h, title))
            parts.append(stage_group_close())
            if idx < len(stage_defs) - 1:
                ay = row_y + box_h / 2
                parts.append(arrow(gx + box_w, ay, gx + box_w + gap, ay, idx))
        parts.append("</svg>")
        return "\n".join(parts)

    rng = np.random.default_rng(seed)
    model = {
        "seed": int(seed),
        "d_model": int(d_model),
        "w_q": _rand_rows(rng, d_model, d_model, 0.5),
        "w_k": _rand_rows(rng, d_model, d_model, 0.5),
        "w_v": _rand_rows(rng, d_model, d_model, 0.5),
        "w1": _rand_rows(rng, d_model, 4 * d_model, 1.0 / np.sqrt(d_model)),
        "w2": _rand_rows(rng, 4 * d_model, d_model, 1.0 / np.sqrt(4 * d_model)),
    }
    stages = [
        {"id": "tokenize", "title": "Tokenize", "caption": "<b>Tokenize.</b> Split the prompt into visible tokens and choose the one you want to inspect."},
        {"id": "embed", "title": "Embed + position", "caption": "<b>Embed + position.</b> The demo creates deterministic token vectors and adds a position signal."},
        {"id": "qkv", "title": "Q, K, V", "caption": "<b>Q, K, V.</b> Each token becomes a query, key, and value vector through learned projections."},
        {"id": "scores", "title": "Scores", "caption": "<b>Scores.</b> Q·Kᵀ/√d produces a causal relevance matrix before softmax."},
        {"id": "attention", "title": "Attention", "caption": "<b>Attention.</b> Row-wise softmax turns scores into weights; click tokens to inspect any row."},
        {"id": "ffn", "title": "Feed-forward", "caption": "<b>Feed-forward.</b> The context vector is refined by a small MLP."},
        {"id": "output", "title": "Output", "caption": "<b>Output.</b> The final vector is scored against a small candidate vocabulary and sampled."},
    ]
    common_candidates = [
        "the", "a", "an", "to", "of", "and", "in", "for", "with", "on", "by", "from",
        "that", "this", "is", "are", "can", "will", "model", "token", "attention",
        "learn", "output", "next",
    ]

    body = f"""
<h1>Transformer forward pass — interactive explainer</h1>
<p class="sub">Edit the prompt, tune temperature, top-k, and top-p, then step through
tokenization, embeddings, attention, and next-token sampling. The numbers are toy
values, but the mechanics mirror a real transformer flow.</p>
<div class="lab-shell">
  <div class="lab-panel">
    <div class="field">
      <label for="prompt">Prompt</label>
      <textarea id="prompt" rows="3">{text}</textarea>
    </div>
    <div class="row">
      <button class="primary" id="analyzeBtn">Analyze prompt</button>
      <button id="demoBtn">Use demo prompt</button>
      <span class="stat" id="analysis-summary"></span>
    </div>
    <div class="sl" title="Higher values flatten the softmax distribution.">
      <label>Temperature</label>
      <input id="temperature" type="range" min="0.2" max="2.5" step="0.05" value="{temperature}">
      <span class="stat" id="temperatureValue"></span>
    </div>
    <div class="sl" title="Keep only the top-k candidate tokens before sampling.">
      <label>Top-k</label>
      <input id="topk" type="range" min="1" max="12" step="1" value="{top_k}">
      <span class="stat" id="topkValue"></span>
    </div>
    <div class="sl" title="Keep the smallest set whose cumulative probability exceeds p.">
      <label>Top-p</label>
      <input id="topp" type="range" min="0.1" max="1.0" step="0.05" value="{top_p}">
      <span class="stat" id="toppValue"></span>
    </div>
    <div class="sl">
      <label>Seed</label>
      <input id="seed" type="number" min="0" step="1" value="{seed}">
    </div>
    <div id="token-strip" class="token-strip"></div>
  </div>
</div>
{flow_controls_html()}
<div id="flow-wrap">
{_build_stage_svg()}
</div>
<div id="detail-stage" class="detail-stage"></div>
<p class="legend">The attention heatmap is row-wise causal softmax(QKᵀ/√d). The output
panel scores a tiny candidate vocabulary, filters it with top-k/top-p, and samples
the next token from the remaining probabilities.</p>
"""

    script = r"""
const MODEL = __MODEL_JSON__;
const COMMON = __COMMON_JSON__;
const DEFAULT_PROMPT = __PROMPT_JSON__;
const MAX_TOKENS = 12;

const state = {
  prompt: DEFAULT_PROMPT,
  temperature: __TEMPERATURE__,
  topK: __TOP_K__,
  topP: __TOP_P__,
  seed: __SEED__,
  activeToken: 0,
  stageIdx: 0,
  analysis: null,
};

function esc(text){
  return String(text)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function hashString(text){
  let h = 2166136261;
  for (let i = 0; i < text.length; i++){
    h ^= text.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

function rngFrom(seed){
  let x = seed >>> 0;
  return function(){
    x += 0x6D2B79F5;
    let t = Math.imul(x ^ (x >>> 15), 1 | x);
    t ^= t + Math.imul(t ^ (t >>> 7), 61 | t);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function tokenize(text){
  const raw = text.trim().toLowerCase().match(/[a-z0-9']+|[^\s]/g);
  const tokens = raw && raw.length ? raw.slice(0, MAX_TOKENS) : ["<empty>"];
  return tokens;
}

function tokenVector(tag, dim, seed){
  const next = rngFrom(hashString(`${tag}|${seed}`));
  const out = [];
  for (let i = 0; i < dim; i++) out.push(next() * 2 - 1);
  return out;
}

function positionalEncoding(pos, dim){
  const out = [];
  for (let i = 0; i < dim; i++){
    const denom = Math.pow(10000, (2 * Math.floor(i / 2)) / dim);
    out.push(i % 2 === 0 ? Math.sin(pos / denom) : Math.cos(pos / denom));
  }
  return out;
}

function addVec(a, b){ return a.map((v, i) => v + b[i]); }
function dot(a, b){ return a.reduce((sum, v, i) => sum + v * b[i], 0); }
function matVec(mat, vec){ return mat.map(row => dot(row, vec)); }
function relu(vec){ return vec.map(v => Math.max(0, v)); }
function weightedSum(weights, rows){
  const dim = rows[0].length;
  const out = Array(dim).fill(0);
  for (let i = 0; i < rows.length; i++){
    for (let j = 0; j < dim; j++) out[j] += weights[i] * rows[i][j];
  }
  return out;
}
function softmax(values, temperature){
  const scaled = values.map(v => v / Math.max(temperature, 0.05));
  const finite = scaled.filter(v => Number.isFinite(v));
  const max = finite.length ? Math.max(...finite) : 0;
  const exps = scaled.map(v => Number.isFinite(v) ? Math.exp(v - max) : 0);
  const sum = exps.reduce((a, b) => a + b, 0) || 1;
  return exps.map(v => v / sum);
}
function filterTopKTopP(logits, temperature, topK, topP){
  const probs = softmax(logits, temperature);
  const ranked = probs.map((p, i) => ({ p, i })).sort((a, b) => b.p - a.p);
  const topKSet = new Set(ranked.slice(0, Math.max(1, Math.min(topK, probs.length))).map(x => x.i));
  let cumulative = 0;
  const nucleus = new Set();
  for (const item of ranked){
    nucleus.add(item.i);
    cumulative += item.p;
    if (cumulative >= topP) break;
  }
  const keep = probs.map((_, i) => topKSet.has(i) && nucleus.has(i));
  const filtered = probs.map((p, i) => keep[i] ? p : 0);
  const total = filtered.reduce((a, b) => a + b, 0) || 1;
  return { probs: filtered.map(v => v / total), keep };
}
function sampleIndex(probs, seed){
  const rnd = rngFrom(seed);
  let r = rnd();
  for (let i = 0; i < probs.length; i++){
    r -= probs[i];
    if (r <= 0) return i;
  }
  return probs.length - 1;
}
function heatColor(v, lo, hi){
  const span = (hi - lo) || 1;
  const p = Math.max(0, Math.min(1, (v - lo) / span));
  const r = Math.round(255 - p * (255 - 37));
  const g = Math.round(255 - p * (255 - 99));
  const b = Math.round(255 - p * (255 - 235));
  return `rgb(${r},${g},${b})`;
}
function activeToken(){
  return Math.max(0, Math.min(state.activeToken, state.analysis.tokens.length - 1));
}
function buildAnalysis(){
  const tokens = tokenize(state.prompt);
  const n = tokens.length;
  const d = MODEL.d_model;
  const embed = tokens.map((tok, i) => addVec(
    tokenVector(`tok:${tok}`, d, state.seed + i * 17),
    positionalEncoding(i, d)
  ));
  const q = embed.map(v => matVec(MODEL.w_q, v));
  const k = embed.map(v => matVec(MODEL.w_k, v));
  const v = embed.map(v => matVec(MODEL.w_v, v));
  const scores = q.map((qi, i) => k.map((kj, j) => j <= i ? dot(qi, kj) / Math.sqrt(d) : Number.NEGATIVE_INFINITY));
  const attn = scores.map(row => softmax(row, state.temperature));
  const context = attn.map(row => weightedSum(row, v));
  const active = Math.max(0, Math.min(state.activeToken, n - 1));
  const hidden = relu(matVec(MODEL.w1, context[active]));
  const ffn = matVec(MODEL.w2, hidden);
  const candidateVocab = [...new Set([...COMMON, ...tokens.map(t => t.toLowerCase())])].slice(0, 16);
  const candidateVecs = candidateVocab.map(word => tokenVector(`cand:${word}`, d, state.seed + 701));
  const logits = candidateVecs.map(vec => dot(ffn, vec));
  const filtered = filterTopKTopP(logits, state.temperature, state.topK, state.topP);
  const sampledIndex = sampleIndex(filtered.probs, hashString(`${state.prompt}|${state.seed}|${state.temperature}|${state.topK}|${state.topP}`));
  return {
    tokens, embed, q, k, v, scores, attn, context, hidden, ffn,
    candidateVocab, logits, probs: filtered.probs, keep: filtered.keep, sampledIndex,
  };
}
function vectorCard(title, values, note = ""){
  return `
    <section class="viz-card">
      <div class="viz-title">${esc(title)}</div>
      ${note ? `<div class="viz-note">${esc(note)}</div>` : ""}
      <div class="vec-row">${values.map((v, i) => `<span class="vec-chip" title="${i}: ${v.toFixed(4)}">${v.toFixed(3)}</span>`).join("")}</div>
    </section>`;
}
function matrixTable(title, matrix, rowLabels, colLabels, activeRow){
  let lo = Infinity, hi = -Infinity;
  matrix.forEach(row => row.forEach(v => { if (Number.isFinite(v)) { lo = Math.min(lo, v); hi = Math.max(hi, v); } }));
  if (!Number.isFinite(lo)) { lo = 0; hi = 1; }
  const rows = matrix.map((row, i) => `
    <tr class="${i === activeRow ? "active-row" : ""}">
      <th class="row-label" data-row="${i}">${esc(rowLabels[i])}</th>
      ${row.map((v, j) => Number.isFinite(v)
        ? `<td style="background:${heatColor(v, lo, hi)}" title="${esc(rowLabels[i])} → ${esc(colLabels[j])}: ${v.toFixed(4)}">${v.toFixed(2)}</td>`
        : `<td class="masked" title="masked future token">·</td>`).join("")}
    </tr>`).join("");
  return `
    <section class="viz-card">
      <div class="viz-title">${esc(title)}</div>
      <table class="heat-table">
        <thead><tr><th></th>${colLabels.map(label => `<th>${esc(label)}</th>`).join("")}</tr></thead>
        <tbody>${rows}</tbody>
      </table>
    </section>`;
}
function barChart(title, names, probs, sampledIndex, keep){
  return `
    <section class="viz-card">
      <div class="viz-title">${esc(title)}</div>
      <div class="bars">
        ${names.map((name, i) => `
          <div class="bar-row ${i === sampledIndex ? "sampled" : ""} ${keep[i] ? "" : "dimmed"}">
            <div class="bar-name">${esc(name)}</div>
            <div class="bar-track"><div class="bar-fill" style="width:${(probs[i] * 100).toFixed(1)}%"></div></div>
            <div class="bar-val">${(probs[i] * 100).toFixed(1)}%</div>
          </div>`).join("")}
      </div>
    </section>`;
}
function renderTokenStrip(){
  const strip = document.getElementById("token-strip");
  strip.innerHTML = state.analysis.tokens.map((tok, i) => `
    <button class="tok ${i === activeToken() ? "active" : ""}" data-idx="${i}">${esc(tok)}</button>
  `).join("");
  strip.querySelectorAll("button").forEach(btn => {
    btn.addEventListener("click", () => {
      state.activeToken = Number(btn.dataset.idx);
      renderTokenStrip();
      renderDetail();
    });
  });
}
function renderSummary(){
  const a = state.analysis;
  const summary = document.getElementById("analysis-summary");
  const sample = a.candidateVocab[a.sampledIndex];
  summary.textContent = `${a.tokens.length} token(s) · active: ${a.tokens[activeToken()]} · sample: ${sample}`;
  document.getElementById("temperatureValue").textContent = state.temperature.toFixed(2);
  document.getElementById("topkValue").textContent = String(state.topK);
  document.getElementById("toppValue").textContent = state.topP.toFixed(2);
}
function renderDetail(){
  const a = state.analysis;
  const idx = activeToken();
  const stageIdx = state.stageIdx;
  const stage = STAGES[stageIdx];
  const root = document.getElementById("detail-stage");
  if (stageIdx === 0){
    root.innerHTML = `
      <section class="viz-card">
        <div class="viz-title">${esc(stage.title)} · ${a.tokens.length} token(s)</div>
        <div class="viz-note">Click a token chip to inspect its embedding, attention row, and output path.</div>
        <div class="token-grid">${a.tokens.map((tok, i) => `<span class="token-pill ${i === idx ? "active" : ""}">${esc(tok)}</span>`).join("")}</div>
        <div class="mini-meta">Visible vocabulary: ${a.tokens.join(", ")}</div>
      </section>
      ${vectorCard("Token IDs", a.tokens.map((_, i) => i), "A local index per visible token in this tiny demo.")}
    `;
  } else if (stageIdx === 1){
    root.innerHTML = `
      <div class="detail-grid">
        ${vectorCard(`Embedding for "${a.tokens[idx]}"`, addVec(tokenVector(`tok:${a.tokens[idx]}`, MODEL.d_model, state.seed + idx * 17), positionalEncoding(idx, MODEL.d_model)), "Token embedding + position encoding.")}
        ${vectorCard("Position encoding", positionalEncoding(idx, MODEL.d_model), "Sinusoidal position signal.")}
        ${vectorCard("Combined input", a.embed[idx], "What enters the Q/K/V projections.")}
      </div>`;
  } else if (stageIdx === 2){
    root.innerHTML = `
      <div class="detail-grid">
        ${vectorCard("Query Q", a.q[idx], "What this token is looking for.")}
        ${vectorCard("Key K", a.k[idx], "What this token offers to others.")}
        ${vectorCard("Value V", a.v[idx], "What gets passed on if selected.")}
      </div>`;
  } else if (stageIdx === 3){
    root.innerHTML = matrixTable("Causal score matrix QKᵀ/√d", a.scores, a.tokens, a.tokens, idx);
  } else if (stageIdx === 4){
    root.innerHTML = `
      <div class="detail-grid">
        ${matrixTable("Attention weights (row-wise softmax)", a.attn, a.tokens, a.tokens, idx)}
        ${vectorCard(`Attention row for "${a.tokens[idx]}"`, a.attn[idx], "How much this token attends to each visible token.")}
      </div>`;
  } else if (stageIdx === 5){
    root.innerHTML = `
      <div class="detail-grid">
        ${vectorCard("Context vector", a.context[idx], "Weighted sum of values.")}
        ${vectorCard("Hidden state", a.hidden, "After the feed-forward expansion + ReLU.")}
        ${vectorCard("FFN output", a.ffn, "Projected back to model dimension.")}
      </div>`;
  } else {
    root.innerHTML = `
      <div class="detail-grid">
        ${barChart("Candidate next tokens", a.candidateVocab, a.probs, a.sampledIndex, a.keep)}
        <section class="viz-card">
          <div class="viz-title">Sampling summary</div>
          <div class="viz-note">temperature=${state.temperature.toFixed(2)}, top-k=${state.topK}, top-p=${state.topP.toFixed(2)}</div>
          <div class="mini-meta">Sampled token: <b>${esc(a.candidateVocab[a.sampledIndex])}</b></div>
        </section>
      </div>`;
  }
  root.querySelectorAll("[data-row]").forEach(el => {
    el.addEventListener("click", () => {
      state.activeToken = Number(el.dataset.row);
      renderTokenStrip();
      renderDetail();
    });
  });
}
function recompute(){
  state.analysis = buildAnalysis();
  if (state.activeToken >= state.analysis.tokens.length) state.activeToken = 0;
  renderTokenStrip();
  renderSummary();
  renderDetail();
}

document.getElementById("prompt").addEventListener("input", (e) => { state.prompt = e.target.value; });
document.getElementById("analyzeBtn").addEventListener("click", () => {
  state.prompt = document.getElementById("prompt").value;
  state.activeToken = 0;
  recompute();
  if (window.setStage) window.setStage(0);
});
document.getElementById("demoBtn").addEventListener("click", () => {
  document.getElementById("prompt").value = DEFAULT_PROMPT;
  state.prompt = DEFAULT_PROMPT;
  state.activeToken = 0;
  recompute();
  if (window.setStage) window.setStage(0);
});
document.getElementById("temperature").addEventListener("input", (e) => {
  state.temperature = +e.target.value;
  recompute();
});
document.getElementById("topk").addEventListener("input", (e) => {
  state.topK = +e.target.value;
  recompute();
});
document.getElementById("topp").addEventListener("input", (e) => {
  state.topP = +e.target.value;
  recompute();
});
document.getElementById("seed").addEventListener("input", (e) => {
  state.seed = Math.max(0, +e.target.value || 0);
  recompute();
});
document.getElementById("prompt").addEventListener("keydown", (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
    document.getElementById("analyzeBtn").click();
  }
});
window.__FLOW_STAGE_HOOK__ = (stageIdx) => {
  state.stageIdx = stageIdx;
  renderDetail();
};

recompute();
"""

    script = (
        script
        .replace("__MODEL_JSON__", json.dumps(model))
        .replace("__COMMON_JSON__", json.dumps(common_candidates))
        .replace("__PROMPT_JSON__", json.dumps(text))
        .replace("__TEMPERATURE__", json.dumps(float(temperature)))
        .replace("__TOP_K__", json.dumps(int(top_k)))
        .replace("__TOP_P__", json.dumps(float(top_p)))
        .replace("__SEED__", json.dumps(int(seed)))
    )
    html = page(
        title="OptimumAI — transformer flow",
        heading_sr=(
            "Interactive transformer explainer with editable prompt, attention heatmaps, "
            "and next-token sampling controls."
        ),
        body=body,
        script=runtime_script(script, stages),
        css_extra="""
 #flow-svg{min-width:1200px}
 .lab-shell{display:block;margin-bottom:12px}
 .lab-panel{background:#fff;border:1px solid #e5e7eb;border-radius:8px;padding:14px 16px;max-width:1100px}
 .field{display:flex;flex-direction:column;gap:6px;margin-bottom:12px}
 .field label,.sl label{font-family:monospace;font-size:13px;font-weight:700}
 textarea{width:100%;font:inherit;font-family:monospace;font-size:14px;padding:10px;border:1px solid #cbd5e1;border-radius:6px;resize:vertical}
 .sl{display:flex;align-items:center;gap:10px;margin:10px 0;flex-wrap:wrap}
 .sl input[type=range]{flex:1;min-width:180px}
 .sl input[type=number]{width:90px;font:inherit;font-family:monospace;padding:5px 8px}
 .token-strip{display:flex;flex-wrap:wrap;gap:8px;margin-top:12px}
 .tok,.token-pill{border:1px solid #c7d2fe;background:#eef2ff;color:#1d4ed8;border-radius:999px;padding:6px 10px;font-family:monospace;font-size:12px;cursor:pointer}
 .tok.active,.token-pill.active{background:#2563eb;color:#fff;border-color:#1d4ed8}
 .detail-stage{margin-top:14px;max-width:1100px}
 .detail-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:12px}
 .viz-card{background:#fff;border:1px solid #e5e7eb;border-radius:8px;padding:12px 14px}
 .viz-title{font-family:monospace;font-size:13px;font-weight:700;color:#1d4ed8;margin-bottom:4px}
 .viz-note,.mini-meta{color:#475569;font-size:13px;line-height:1.45}
 .vec-row{display:flex;flex-wrap:wrap;gap:6px;margin-top:10px}
 .vec-chip{display:inline-block;background:#f8fafc;border:1px solid #e2e8f0;border-radius:6px;padding:4px 8px;font-family:monospace;font-size:12px}
 .heat-table{border-collapse:collapse;width:100%;overflow:auto;font-size:12px}
 .heat-table th,.heat-table td{border:1px solid #e5e7eb;padding:4px 6px;text-align:center;font-family:monospace}
 .heat-table .row-label{cursor:pointer;background:#f8fafc;text-align:left;color:#1d4ed8}
 .heat-table .active-row .row-label{font-weight:700}
 .heat-table .masked{color:#94a3b8;background:#f8fafc}
 .bars{display:flex;flex-direction:column;gap:8px;margin-top:10px}
 .bar-row{display:grid;grid-template-columns:110px 1fr 64px;gap:10px;align-items:center}
 .bar-row.sampled .bar-name{font-weight:700;color:#1d4ed8}
 .bar-row.dimmed{opacity:.5}
 .bar-name,.bar-val{font-family:monospace;font-size:12px}
 .bar-track{height:10px;background:#e2e8f0;border-radius:999px;overflow:hidden}
 .bar-fill{height:100%;background:#2563eb;border-radius:999px}
 .legend{max-width:1100px}
""",
    )
    return write(html, out, "transformer_flow.html")

    tokens = _tokenize(text)
    n = len(tokens)
    vocab = sorted(set(tokens))
    vocab_size = len(vocab)
    tok_to_id = {t: i for i, t in enumerate(vocab)}
    ids = [tok_to_id[t] for t in tokens]

    rng = np.random.default_rng(seed)
    scale_e = 1.0 / np.sqrt(d_model)
    embed_table = rng.normal(size=(vocab_size, d_model)) * scale_e
    embed = embed_table[ids]

    pe = positional_encoding(n, d_model)
    x = embed + pe

    w_q = rng.normal(size=(d_model, d_model)) * 0.5
    w_k = rng.normal(size=(d_model, d_model)) * 0.5
    w_v = rng.normal(size=(d_model, d_model)) * 0.5
    q = x @ w_q
    k = x @ w_k
    v = x @ w_v

    d_k = d_model
    scores = q @ k.T
    scaled = scores / np.sqrt(d_k)
    shifted = scaled - scaled.max(axis=-1, keepdims=True)
    exps = np.exp(shifted)
    attn = exps / exps.sum(axis=-1, keepdims=True)
    attn_out = attn @ v

    d_ff = 4 * d_model
    w1 = rng.normal(size=(d_model, d_ff)) * (1.0 / np.sqrt(d_model))
    w2 = rng.normal(size=(d_ff, d_model)) * (1.0 / np.sqrt(d_ff))
    hidden = np.maximum(attn_out @ w1, 0.0)  # ReLU FFN for a readable diagram
    ffn_out = hidden @ w2

    head = rng.normal(size=(d_model, vocab_size)) * (1.0 / np.sqrt(d_model))
    logits = ffn_out[-1] @ head
    l_shift = logits - logits.max()
    l_exp = np.exp(l_shift)
    probs = l_exp / l_exp.sum()
    top_idx = np.argsort(probs)[::-1][: min(3, vocab_size)]

    # ---------------------------------------------------------------- SVG
    box_w, box_h, gap = 150, 42, 60
    stage_defs = [
        ("tokens", "1. Tokens"),
        ("embed", "2. Embeddings"),
        ("posenc", "3. + Positional"),
        ("qkv", "4. Q, K, V"),
        ("scores", "5. Scores QKᵀ/√d"),
        ("softmax", "6. Softmax"),
        ("weighted", "7. Weighted sum ·V"),
        ("ffn", "8. Feed-forward"),
        ("logits", "9. Logits"),
        ("output", "10. Softmax → next token"),
    ]
    n_stages = len(stage_defs)
    content_h = 260
    svg_w = 40 + n_stages * (box_w + gap)
    svg_h = content_h + 80

    svg_parts = [svg_open(svg_w, svg_h)]
    row_y = 60
    for idx, (sid, title) in enumerate(stage_defs):
        gx = 20 + idx * (box_w + gap)
        svg_parts.append(stage_group_open(sid, gx, row_y))
        svg_parts.append(stage_box(box_w, box_h, title))
        svg_parts.append('<g transform="translate(0,52)">')

        if sid == "tokens":
            chips = "".join(
                f'<g class="cell" data-tip="{_esc(t)} → id {i}">'
                f'<rect x="{j * 34}" y="0" width="30" height="26" rx="4" '
                f'fill="#eef2ff" stroke="#c7d2fe"/>'
                f'<text class="cell-text" x="{j * 34 + 15}" y="13" font-size="9">{t}</text>'
                f"</g>"
                for j, (t, i) in enumerate(zip(tokens, ids, strict=True))
            )
            svg_parts.append(chips)
        elif sid == "embed":
            svg_parts.append(
                matrix_grid(
                    embed.round(3).tolist(),
                    0,
                    0,
                    cell=22,
                    row_labels=tokens,
                    id_prefix="embed",
                    tooltip_fn=lambda i, j, v: (
                        f"embed[{tokens[i]}][{j}] = {v:.4f} "
                        f"(row {i}'s lookup vector, dim {j})"
                    ),
                )
            )
        elif sid == "posenc":
            svg_parts.append(
                matrix_grid(
                    x.round(3).tolist(),
                    0,
                    0,
                    cell=22,
                    row_labels=tokens,
                    id_prefix="posenc",
                    tooltip_fn=lambda i, j, v: (
                        f"x[{tokens[i]}][{j}] = embed + PE = {v:.4f} "
                        f"(position {i} baked in)"
                    ),
                )
            )
        elif sid == "qkv":
            svg_parts.append('<text class="flow-stage-label" x="0" y="10" font-size="10">Q</text>')
            svg_parts.append(
                matrix_grid(
                    q.round(3).tolist(),
                    0,
                    14,
                    cell=16,
                    row_labels=tokens,
                    id_prefix="qmat",
                    tooltip_fn=lambda i, j, v: f"Q[{tokens[i]}][{j}] = {v:.4f} (what I seek)",
                )
            )
            k_y = 14 + n * 16 + 18
            svg_parts.append(
                f'<text class="flow-stage-label" x="0" y="{k_y - 4}" font-size="10">K</text>'
            )
            svg_parts.append(
                matrix_grid(
                    k.round(3).tolist(),
                    0,
                    k_y,
                    cell=16,
                    row_labels=tokens,
                    id_prefix="kmat",
                    tooltip_fn=lambda i, j, v: f"K[{tokens[i]}][{j}] = {v:.4f} (what I offer)",
                )
            )
        elif sid == "scores":
            svg_parts.append(
                matrix_grid(
                    scaled.round(3).tolist(),
                    0,
                    0,
                    cell=22,
                    row_labels=tokens,
                    col_labels=tokens,
                    id_prefix="scores",
                    tooltip_fn=lambda i, j, v: (
                        f"{tokens[i]} · {tokens[j]} / √d = {v:.4f} "
                        f"(raw relevance before softmax)"
                    ),
                )
            )
        elif sid == "softmax":
            svg_parts.append(
                matrix_grid(
                    attn.round(4).tolist(),
                    0,
                    0,
                    cell=22,
                    row_labels=tokens,
                    col_labels=tokens,
                    lo=0.0,
                    hi=1.0,
                    id_prefix="attn",
                    decimals=2,
                    tooltip_fn=lambda i, j, v: (
                        f"token {tokens[i]!r} attends to token {tokens[j]!r} = {v:.3f} "
                        f"({v * 100:.1f}% of {tokens[i]!r}'s attention)"
                    ),
                )
            )
        elif sid == "weighted":
            svg_parts.append(
                matrix_grid(
                    attn_out.round(3).tolist(),
                    0,
                    0,
                    cell=22,
                    row_labels=tokens,
                    id_prefix="attnout",
                    tooltip_fn=lambda i, j, v: (
                        f"context[{tokens[i]}][{j}] = {v:.4f} "
                        f"(blend of V rows, weighted by row {i} of the attention matrix)"
                    ),
                )
            )
        elif sid == "ffn":
            svg_parts.append(
                matrix_grid(
                    ffn_out.round(3).tolist(),
                    0,
                    0,
                    cell=22,
                    row_labels=tokens,
                    id_prefix="ffnout",
                    tooltip_fn=lambda i, j, v: (
                        f"ffn_out[{tokens[i]}][{j}] = {v:.4f} "
                        f"(ReLU(x·W1)·W2 — a per-token nonlinear transform)"
                    ),
                )
            )
        elif sid == "logits":
            svg_parts.append(
                vector_chip(
                    logits.round(3).tolist(),
                    0,
                    0,
                    cell=26,
                    vertical=True,
                    id_prefix="logit",
                    tooltip_fn=lambda i, v: f"logit[{vocab[i]!r}] = {v:.4f}",
                )
            )
        elif sid == "output":
            svg_parts.append(
                vector_chip(
                    probs.round(4).tolist(),
                    0,
                    0,
                    cell=26,
                    vertical=True,
                    lo=0.0,
                    hi=1.0,
                    id_prefix="prob",
                    tooltip_fn=lambda i, v: (
                        f"P(next={vocab[i]!r}) = {v:.4f}"
                        + (" <- top pick" if i == top_idx[0] else "")
                    ),
                )
            )

        svg_parts.append("</g>")
        svg_parts.append(stage_group_close())

        if idx < n_stages - 1:
            ax = gx + box_w
            ay = row_y + box_h / 2
            svg_parts.append(arrow(ax, ay, ax + gap, ay, idx))

    svg_parts.append("</svg>")
    svg = "\n".join(svg_parts)

    top_str = ", ".join(f"{vocab[i]!r} ({probs[i] * 100:.1f}%)" for i in top_idx)
    captions = [
        (
            f"<b>Tokenize.</b> {text!r} splits into {n} word tokens: {tokens}. "
            f"Vocabulary has {vocab_size} unique words, each getting a stable integer id."
        ),
        (
            "<b>Embed.</b> Every token id looks up one row of a "
            f"{vocab_size}×{d_model} table — a learned (here, seeded/random) vector "
            "that stands in for the word's meaning before any context is added."
        ),
        (
            "<b>Add positional encoding.</b> Self-attention has no notion of order — "
            "shuffle the tokens and the math doesn't change — so a sinusoidal signal "
            "unique to each position is added on top of the embedding."
        ),
        (
            "<b>Project to Q, K, V.</b> Three learned matrices turn each token's vector "
            "into a Query ('what am I looking for'), a Key ('what do I offer'), and a "
            "Value ('what do I pass on if picked')."
        ),
        (
            "<b>Score every pair.</b> scores = Q·Kᵀ / √d. Each cell [i,j] is how well "
            "query token i matches key token j, scaled down so softmax doesn't saturate."
        ),
        (
            "<b>Softmax, row by row.</b> Each row is turned into a probability "
            "distribution over which tokens to attend to — this IS the attention matrix; "
            "rows sum to 1."
        ),
        (
            "<b>Weighted sum.</b> attention · V blends every value vector by how much "
            "attention its token received — each output row is a convex combination of "
            "the other tokens' values."
        ),
        (
            "<b>Feed-forward network.</b> Each token's blended vector is pushed through "
            "an independent 2-layer MLP (widen, ReLU, narrow) — this is where most of a "
            "transformer's parameters (and 'knowledge') actually live."
        ),
        (
            "<b>Project to logits.</b> The final token's vector is matmul'd against an "
            "output head to produce one raw score per vocabulary word."
        ),
        (
            f"<b>Softmax over the vocabulary.</b> The logits become a probability "
            f"distribution over what comes next. Top picks here: {top_str}."
        ),
    ]
    stages = [
        {"id": sid, "title": title.split(". ", 1)[1], "caption": cap}
        for (sid, title), cap in zip(stage_defs, captions, strict=True)
    ]

    body = f"""
<h1>Transformer forward pass — {text!r}</h1>
<p class="sub">Every stage of a toy (but real) transformer forward pass, computed
once in Python with a seeded RNG and rendered live below. Step through the
pipeline, or hover any cell in a matrix to see its exact value.</p>
{flow_controls_html()}
<div id="flow-wrap">
{svg}
</div>
<p class="legend">Attention(Q,K,V) = softmax(Q·Kᵀ/√d)·V — the core operation of
every transformer. Q/K/V weights and the FFN are randomly initialized (untrained),
so treat the numbers as illustrative of the *mechanism*, not a trained model's
actual beliefs.</p>
"""
    script = runtime_script(HOVER_TOOLTIP_JS, stages)
    html = page(
        title="OptimumAI — transformer forward-pass flow",
        heading_sr=(
            "Interactive transformer forward pass: step through tokenize, embed, "
            "positional encoding, Q/K/V projection, attention scores, softmax, "
            "weighted sum, feed-forward, logits, and final softmax, hovering any "
            "matrix cell for its exact value."
        ),
        body=body,
        script=script,
        css_extra=_CSS_EXTRA,
    )
    return write(html, out, "transformer_flow.html")


def _esc(text: str) -> str:
    return text.replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;")
