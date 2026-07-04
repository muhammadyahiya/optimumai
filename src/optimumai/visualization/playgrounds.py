"""Interactive playgrounds — poke a small model and watch it react, live.

Inspired by `poloclub.github.io/transformer-explainer` and
`playground.tensorflow.org`: each function here computes a small, deterministic
example in Python (numpy, seeded), embeds the numbers as JSON inside a single
``.html`` file, and lets vanilla JavaScript own all of the interaction (canvas
drawing, sliders, click handlers). There is no server, no build step, and no
runtime Python dependency — a user opens the file in any browser, online or
off, and starts dragging things.

Three playgrounds ship today:

* :func:`transformer_attention_playground` — hover a token to see who it
  attends to; drag a temperature slider to watch the attention distribution
  sharpen (low temperature) or flatten (high temperature).
* :func:`kmeans_playground` — click to drop 2-D points, then step or run
  Lloyd's algorithm and watch centroids chase their clusters.
* :func:`astar_playground` — draw walls on a grid, then watch A* search
  expand its frontier before tracing the shortest path.

Use :func:`playground` to build any of them by name (handy for a CLI).
"""

from __future__ import annotations

import json

import numpy as np

# --------------------------------------------------------------------------
# shared page chrome
# --------------------------------------------------------------------------

_BASE_CSS = """
 body{font-family:system-ui,sans-serif;margin:0;padding:20px;color:#111;
      background:#fafafa}
 h2.sr-only{position:absolute;left:-9999px}
 h3{margin:0 0 6px}
 .sub{color:#555;margin:0 0 14px;max-width:640px}
 .row{display:flex;align-items:center;gap:10px;margin:8px 0;flex-wrap:wrap}
 button{font-family:inherit;font-size:14px;padding:6px 14px;border-radius:6px;
        border:1px solid #ccc;background:#fff;cursor:pointer}
 button:hover{background:#f0f0f0}
 button.primary{background:#2563eb;color:#fff;border-color:#2563eb}
 button.primary:hover{background:#1d4ed8}
 select,input[type=number]{font-family:inherit;font-size:14px;padding:4px 6px}
 .stat{font-family:monospace;color:#2563eb;font-weight:700}
 .legend{color:#555;font-size:13px;max-width:640px;margin-top:10px}
 canvas{border:1px solid #ccc;border-radius:6px;background:#fff}
"""


def _page(title: str, heading_sr: str, body: str, script: str, css_extra: str = "") -> str:
    """Wrap ``body``/``script`` in the standard OptimumAI playground shell."""
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>{_BASE_CSS}
{css_extra}
</style></head><body>
<h2 class="sr-only">{heading_sr}</h2>
{body}
<script>
{script}
</script></body></html>
"""


# ==========================================================================
# 1. Transformer attention playground
# ==========================================================================

_ATTENTION_CSS = """
 #tokens{display:flex;gap:6px;flex-wrap:wrap;margin:10px 0}
 .tok{padding:6px 10px;border-radius:6px;background:#eef2ff;border:1px solid
      #c7d2fe;font-family:monospace;cursor:pointer;user-select:none}
 .tok.active{background:#2563eb;color:#fff;border-color:#1d4ed8}
 #heat{border-collapse:collapse;margin-top:10px}
 #heat td{width:34px;height:28px;text-align:center;font-family:monospace;
          font-size:11px;border:1px solid #eee;cursor:pointer}
 #heat td.rowlabel,#heat th{font-family:monospace;font-size:12px;padding:0 6px;
          border:none;text-align:right}
 .sl{display:flex;align-items:center;gap:10px;margin:10px 0}
 .sl label{font-family:monospace;width:150px}
"""

_ATTENTION_SCRIPT = r"""
const TOKENS = __TOKENS__;
const SCORES = __SCORES__;      // T x T raw scores, S[i][j] = query i . key j
const T = TOKENS.length;
let active = 0;
let temperature = 1.0;

function softmaxRow(row, temp){
  const scaled = row.map(v => v / temp);
  const m = Math.max(...scaled);
  const ex = scaled.map(v => Math.exp(v - m));
  const s = ex.reduce((a,b)=>a+b, 0);
  return ex.map(v => v / s);
}

function colorFor(p){
  // 0 -> white, 1 -> deep blue
  const c = Math.round(255 - p * (255 - 37));
  const g = Math.round(255 - p * (255 - 99));
  const b = Math.round(255 - p * (255 - 235));
  return `rgb(${c},${g},${b})`;
}

const tokensDiv = document.getElementById('tokens');
TOKENS.forEach((tok, i) => {
  const span = document.createElement('span');
  span.className = 'tok' + (i === active ? ' active' : '');
  span.textContent = tok;
  span.dataset.i = i;
  span.addEventListener('mouseenter', () => setActive(i));
  span.addEventListener('click', () => setActive(i));
  tokensDiv.appendChild(span);
});

function setActive(i){
  active = i;
  [...tokensDiv.children].forEach((el, k) => {
    el.classList.toggle('active', k === i);
  });
  render();
}

function render(){
  const table = document.getElementById('heat');
  table.innerHTML = '';
  const head = document.createElement('tr');
  head.appendChild(document.createElement('th'));
  TOKENS.forEach(tok => {
    const th = document.createElement('th');
    th.textContent = tok;
    head.appendChild(th);
  });
  table.appendChild(head);

  const weights = SCORES.map(row => softmaxRow(row, temperature));
  for (let i = 0; i < T; i++){
    const tr = document.createElement('tr');
    const label = document.createElement('td');
    label.className = 'rowlabel';
    label.textContent = TOKENS[i];
    tr.appendChild(label);
    for (let j = 0; j < T; j++){
      const td = document.createElement('td');
      const p = weights[i][j];
      td.textContent = p.toFixed(2);
      const dim = (i === active) ? 1.0 : 0.25;
      td.style.background = colorFor(p * dim + (1 - dim) * 0.02);
      td.style.opacity = (i === active) ? '1' : '0.35';
      td.title = `${TOKENS[i]} -> ${TOKENS[j]}: ${p.toFixed(3)}`;
      td.addEventListener('click', () => setActive(i));
      tr.appendChild(td);
    }
    table.appendChild(tr);
  }

  document.getElementById('activeLabel').textContent = TOKENS[active];
  document.getElementById('tempVal').textContent = temperature.toFixed(2);
}

document.getElementById('temp').addEventListener('input', (e) => {
  temperature = +e.target.value;
  render();
});

render();
"""


def transformer_attention_playground(
    text: str = "the cat sat on the mat",
    out: str | None = None,
) -> str:
    """Transformer-Explainer-style self-attention playground.

    Splits ``text`` into whitespace tokens, builds small seeded random query
    and key projections, and computes the raw self-attention score matrix
    ``S = Q @ K.T / sqrt(d)``. Python computes all the numbers once (so the
    page is fully deterministic); the browser then owns the interaction:

    * hover or click a token to highlight the row of the score matrix that
      belongs to it — "each row is where one token looks";
    * drag the temperature slider to recompute ``softmax(S / temperature)``
      live in JavaScript. Low temperature sharpens the distribution onto a
      single key; high temperature flattens it toward uniform attention.
    """
    tokens = text.split()
    if not tokens:
        tokens = ["<empty>"]
    n_tokens = len(tokens)
    d_model = 8

    rng = np.random.default_rng(0)
    embed = rng.normal(size=(n_tokens, d_model))
    w_q = rng.normal(size=(d_model, d_model)) * 0.5
    w_k = rng.normal(size=(d_model, d_model)) * 0.5

    q = embed @ w_q
    k = embed @ w_k
    scores = (q @ k.T) / np.sqrt(d_model)

    body = """
<h3>Transformer attention playground</h3>
<p class="sub">Self-attention scores for a tiny 8-dim toy embedding of the
sentence below. Hover or click a token to see the row of the score matrix
that belongs to it: <b>each row is where one token looks</b>. Drag the
temperature slider to re-apply softmax(S / T) live.</p>
<div id="tokens"></div>
<div class="sl">
  <label>temperature T = <span id="tempVal">1.00</span></label>
  <input id="temp" type="range" min="0.1" max="5" step="0.05" value="1.0">
</div>
<p>Attention distribution for query token: <b id="activeLabel"></b></p>
<table id="heat"></table>
<p class="legend">Each cell (row i, column j) is softmax(S/T)[i, j]: how much
query token i attends to key token j after temperature-scaled softmax. Rows
always sum to 1. Low T &rarr; peaky (near one-hot); high T &rarr; near
uniform attention over all keys.</p>
"""
    script = (
        _ATTENTION_SCRIPT.replace("__TOKENS__", json.dumps(tokens))
        .replace("__SCORES__", json.dumps(scores.round(6).tolist()))
    )
    html = _page(
        title="OptimumAI — transformer attention playground",
        heading_sr=(
            "Interactive self-attention: hover a token to see its attention "
            "row, drag the temperature slider to sharpen or flatten the "
            "softmax distribution."
        ),
        body=body,
        script=script,
        css_extra=_ATTENTION_CSS,
    )
    return _write(html, out, "transformer_attention_playground.html")


# ==========================================================================
# 2. k-means playground
# ==========================================================================

_KMEANS_CSS = """
 #wrap canvas{cursor:crosshair}
 .palette{display:flex;gap:6px;align-items:center}
 .swatch{width:14px;height:14px;border-radius:50%;display:inline-block}
"""

_KMEANS_SCRIPT = r"""
const CANVAS_W = 560, CANVAS_H = 400;
const COLORS = ['#ef4444', '#2563eb', '#16a34a', '#d97706'];
const canvas = document.getElementById('kmeans-canvas');
const ctx = canvas.getContext('2d');

let points = __POINTS__.map(p => ({x: p[0], y: p[1], cluster: -1}));
let k = 3;
let centroids = [];
let iteration = 0;
let running = false;

function randomPoint(){
  const x = 20 + Math.random() * (CANVAS_W - 40);
  const y = 20 + Math.random() * (CANVAS_H - 40);
  return {x: x, y: y, cluster: -1};
}

function initCentroids(){
  centroids = [];
  const shuffled = [...points].sort(() => Math.random() - 0.5);
  for (let i = 0; i < k; i++){
    const src = shuffled[i % Math.max(shuffled.length, 1)] || randomPoint();
    centroids.push({x: src.x, y: src.y});
  }
  iteration = 0;
}

function assignClusters(){
  points.forEach(p => {
    let best = 0, bestDist = Infinity;
    centroids.forEach((c, ci) => {
      const d = (p.x - c.x) ** 2 + (p.y - c.y) ** 2;
      if (d < bestDist){ bestDist = d; best = ci; }
    });
    p.cluster = best;
  });
}

function updateCentroids(){
  let moved = 0;
  centroids.forEach((c, ci) => {
    const members = points.filter(p => p.cluster === ci);
    if (members.length === 0) return;
    const mx = members.reduce((a, p) => a + p.x, 0) / members.length;
    const my = members.reduce((a, p) => a + p.y, 0) / members.length;
    moved += Math.hypot(mx - c.x, my - c.y);
    c.x = mx; c.y = my;
  });
  return moved;
}

function inertia(){
  return points.reduce((sum, p) => {
    if (p.cluster < 0) return sum;
    const c = centroids[p.cluster];
    return sum + (p.x - c.x) ** 2 + (p.y - c.y) ** 2;
  }, 0);
}

function step(){
  if (centroids.length === 0) initCentroids();
  assignClusters();
  const moved = updateCentroids();
  iteration += 1;
  render();
  return moved;
}

function render(){
  ctx.clearRect(0, 0, CANVAS_W, CANVAS_H);
  points.forEach(p => {
    ctx.beginPath();
    ctx.arc(p.x, p.y, 5, 0, Math.PI * 2);
    ctx.fillStyle = p.cluster >= 0 ? COLORS[p.cluster % COLORS.length] : '#999';
    ctx.fill();
  });
  centroids.forEach((c, ci) => {
    ctx.beginPath();
    ctx.arc(c.x, c.y, 9, 0, Math.PI * 2);
    ctx.fillStyle = COLORS[ci % COLORS.length];
    ctx.strokeStyle = '#111';
    ctx.lineWidth = 2;
    ctx.fill();
    ctx.stroke();
  });
  document.getElementById('iterLabel').textContent = iteration;
  document.getElementById('inertiaLabel').textContent =
    points.some(p => p.cluster >= 0) ? inertia().toFixed(1) : '-';
}

canvas.addEventListener('click', (e) => {
  const rect = canvas.getBoundingClientRect();
  points.push({x: e.clientX - rect.left, y: e.clientY - rect.top, cluster: -1});
  render();
});

document.getElementById('kSelect').addEventListener('change', (e) => {
  k = +e.target.value;
  centroids = [];
  points.forEach(p => p.cluster = -1);
  render();
});

document.getElementById('addRandom').addEventListener('click', () => {
  for (let i = 0; i < 20; i++) points.push(randomPoint());
  render();
});

document.getElementById('stepBtn').addEventListener('click', () => step());

document.getElementById('runBtn').addEventListener('click', () => {
  if (running) return;
  running = true;
  let lastMoved = Infinity;
  const id = setInterval(() => {
    lastMoved = step();
    if (lastMoved < 0.5 || iteration > 100){
      clearInterval(id);
      running = false;
    }
  }, 300);
});

document.getElementById('resetBtn').addEventListener('click', () => {
  points = [];
  centroids = [];
  iteration = 0;
  render();
});

render();
"""


def kmeans_playground(out: str | None = None) -> str:
    """Lloyd's k-means playground: click to add points, then step or run.

    A blank canvas starts with a small seeded set of points (so the page is
    non-empty and deterministic on load). Buttons let you add random points,
    single-step Lloyd's algorithm, run it to convergence with an animation,
    or reset. All of the k-means math (assignment + centroid update +
    inertia) runs in JavaScript so it can animate; Python only supplies the
    seeded starting points.
    """
    rng = np.random.default_rng(0)
    blobs = []
    centers = [(140, 120), (420, 120), (280, 300)]
    for cx, cy in centers:
        for _ in range(8):
            blobs.append(
                (
                    float(np.clip(cx + rng.normal(scale=35), 15, 545)),
                    float(np.clip(cy + rng.normal(scale=35), 15, 385)),
                )
            )

    body = """
<h3>k-means playground</h3>
<p class="sub">Click anywhere on the canvas to drop a new point. Choose k,
then Step through Lloyd's algorithm one iteration at a time or Run it to
convergence and watch the centroids chase their clusters.</p>
<div class="row">
  <label>k = <select id="kSelect">
    <option value="2">2</option>
    <option value="3" selected>3</option>
    <option value="4">4</option>
  </select></label>
  <button id="addRandom">Add random</button>
  <button id="stepBtn">Step</button>
  <button class="primary" id="runBtn">Run</button>
  <button id="resetBtn">Reset</button>
</div>
<div class="row">
  iteration = <span class="stat" id="iterLabel">0</span>
  &nbsp;&nbsp; inertia = <span class="stat" id="inertiaLabel">-</span>
</div>
<div id="wrap"><canvas id="kmeans-canvas" width="560" height="400"></canvas></div>
<p class="legend">Lloyd's algorithm alternates two steps: (1) assign each
point to its nearest centroid, (2) move each centroid to the mean of its
assigned points. Inertia is the total squared distance from points to their
centroid — it should only ever decrease.</p>
"""
    points_json = json.dumps([[round(x, 2), round(y, 2)] for x, y in blobs])
    script = _KMEANS_SCRIPT.replace("__POINTS__", points_json)
    html = _page(
        title="OptimumAI — k-means playground",
        heading_sr=(
            "Interactive k-means clustering: click to add points, choose k, "
            "then step or run Lloyd's algorithm and watch centroids move."
        ),
        body=body,
        script=script,
        css_extra=_KMEANS_CSS,
    )
    return _write(html, out, "kmeans_playground.html")


# ==========================================================================
# 3. A* pathfinding playground
# ==========================================================================

_ASTAR_CSS = """
 #astar-canvas{cursor:pointer}
 .legend .swatch{display:inline-block;width:12px;height:12px;
   border-radius:2px;margin:0 4px -1px 10px}
"""

_ASTAR_SCRIPT = r"""
const COLS = __COLS__, ROWS = __ROWS__, CELL = __CELL__;
const canvas = document.getElementById('astar-canvas');
const ctx = canvas.getContext('2d');
canvas.width = COLS * CELL;
canvas.height = ROWS * CELL;

let walls = new Set();
let start = [1, Math.floor(ROWS / 2)];
let goal = [COLS - 2, Math.floor(ROWS / 2)];
let mode = 'wall';
let mouseDown = false;
let lastResult = null;

function key(x, y){ return x + ',' + y; }

function neighbors(x, y){
  const out = [];
  [[1,0],[-1,0],[0,1],[0,-1]].forEach(([dx,dy]) => {
    const nx = x + dx, ny = y + dy;
    if (nx >= 0 && nx < COLS && ny >= 0 && ny < ROWS && !walls.has(key(nx, ny))){
      out.push([nx, ny]);
    }
  });
  return out;
}

function heuristic(a, b){
  return Math.abs(a[0]-b[0]) + Math.abs(a[1]-b[1]);
}

async function runAstar(){
  const openSet = new Map();
  const cameFrom = new Map();
  const gScore = new Map();
  const fScore = new Map();
  const startKey = key(...start);
  gScore.set(startKey, 0);
  fScore.set(startKey, heuristic(start, goal));
  openSet.set(startKey, start);
  const visited = [];
  let expanded = 0;

  while (openSet.size > 0){
    let currentKey = null, current = null, bestF = Infinity;
    for (const [k2, pos] of openSet){
      const f = fScore.get(k2) ?? Infinity;
      if (f < bestF){ bestF = f; currentKey = k2; current = pos; }
    }
    openSet.delete(currentKey);
    expanded += 1;
    visited.push(current);

    if (current[0] === goal[0] && current[1] === goal[1]){
      const path = [current];
      let ck = currentKey;
      while (cameFrom.has(ck)){
        const prev = cameFrom.get(ck);
        path.push(prev.pos);
        ck = prev.key;
      }
      path.reverse();
      return {path, expanded, visited};
    }

    for (const nb of neighbors(...current)){
      const nbKey = key(...nb);
      const tentativeG = (gScore.get(currentKey) ?? Infinity) + 1;
      if (tentativeG < (gScore.get(nbKey) ?? Infinity)){
        cameFrom.set(nbKey, {pos: current, key: currentKey});
        gScore.set(nbKey, tentativeG);
        fScore.set(nbKey, tentativeG + heuristic(nb, goal));
        openSet.set(nbKey, nb);
      }
    }
  }
  return {path: null, expanded, visited};
}

function draw(frontierVisited, path){
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  for (let x = 0; x < COLS; x++){
    for (let y = 0; y < ROWS; y++){
      ctx.strokeStyle = '#eee';
      ctx.strokeRect(x * CELL, y * CELL, CELL, CELL);
    }
  }
  walls.forEach(k2 => {
    const [x, y] = k2.split(',').map(Number);
    ctx.fillStyle = '#333';
    ctx.fillRect(x * CELL, y * CELL, CELL, CELL);
  });
  if (frontierVisited){
    ctx.fillStyle = 'rgba(37,99,235,0.25)';
    frontierVisited.forEach(([x, y]) => ctx.fillRect(x * CELL, y * CELL, CELL, CELL));
  }
  if (path){
    ctx.fillStyle = 'rgba(217,119,6,0.85)';
    path.forEach(([x, y]) => ctx.fillRect(x * CELL, y * CELL, CELL, CELL));
  }
  ctx.fillStyle = '#16a34a';
  ctx.fillRect(start[0] * CELL, start[1] * CELL, CELL, CELL);
  ctx.fillStyle = '#ef4444';
  ctx.fillRect(goal[0] * CELL, goal[1] * CELL, CELL, CELL);
}

function cellFromEvent(e){
  const rect = canvas.getBoundingClientRect();
  const x = Math.floor((e.clientX - rect.left) / CELL);
  const y = Math.floor((e.clientY - rect.top) / CELL);
  return [x, y];
}

function toggleWall(x, y){
  if ((x === start[0] && y === start[1]) || (x === goal[0] && y === goal[1])) return;
  const k2 = key(x, y);
  if (mode === 'wall'){
    if (walls.has(k2)) walls.delete(k2); else walls.add(k2);
  } else if (mode === 'start'){
    start = [x, y];
  } else if (mode === 'goal'){
    goal = [x, y];
  }
  draw(null, null);
}

canvas.addEventListener('mousedown', (e) => {
  mouseDown = true;
  toggleWall(...cellFromEvent(e));
});
canvas.addEventListener('mousemove', (e) => {
  if (mouseDown && mode === 'wall') toggleWall(...cellFromEvent(e));
});
window.addEventListener('mouseup', () => { mouseDown = false; });

document.getElementById('modeSelect').addEventListener('change', (e) => {
  mode = e.target.value;
});

document.getElementById('runBtn').addEventListener('click', async () => {
  const result = await runAstar();
  lastResult = result;
  draw(result.visited, result.path);
  document.getElementById('expandedLabel').textContent = result.expanded;
  document.getElementById('pathLabel').textContent = result.path ? result.path.length : 'no path';
});

document.getElementById('resetBtn').addEventListener('click', () => {
  walls = new Set();
  lastResult = null;
  document.getElementById('expandedLabel').textContent = '0';
  document.getElementById('pathLabel').textContent = '-';
  draw(null, null);
});

draw(null, null);
"""


def astar_playground(out: str | None = None) -> str:
    """A* pathfinding playground on a grid with a Manhattan heuristic.

    Draws a grid canvas with a fixed start (green) and goal (red) cell.
    Click or drag on the grid to toggle walls; the mode selector lets you
    instead move the start or goal cell. Pressing "Run" executes A* (in
    JavaScript, so the whole search — including the visited-node overlay and
    the final path — can be drawn) with ``h(n) = |dx| + |dy|``, then reports
    the number of nodes expanded and the path length.
    """
    cols, rows, cell = 24, 16, 24
    body = f"""
<h3>A* pathfinding playground</h3>
<p class="sub">Click or drag on the grid to draw walls. Switch mode to move
the start or goal cell, then press Run to watch A* (Manhattan heuristic)
expand its frontier and trace the shortest path.</p>
<div class="row">
  <label>mode:
    <select id="modeSelect">
      <option value="wall" selected>draw walls</option>
      <option value="start">move start</option>
      <option value="goal">move goal</option>
    </select>
  </label>
  <button class="primary" id="runBtn">Run</button>
  <button id="resetBtn">Reset</button>
</div>
<div class="row">
  nodes expanded = <span class="stat" id="expandedLabel">0</span>
  &nbsp;&nbsp; path length = <span class="stat" id="pathLabel">-</span>
</div>
<canvas id="astar-canvas" width="{cols * cell}" height="{rows * cell}"></canvas>
<p class="legend">Green = start, red = goal, dark = wall, light blue = nodes
A* expanded while searching, orange = final shortest path. Heuristic:
h(n) = |dx| + |dy| (Manhattan distance to the goal).</p>
"""
    script = (
        _ASTAR_SCRIPT.replace("__COLS__", str(cols))
        .replace("__ROWS__", str(rows))
        .replace("__CELL__", str(cell))
    )
    html = _page(
        title="OptimumAI — A* pathfinding playground",
        heading_sr=(
            "Interactive A* search: draw walls on a grid, move the start or "
            "goal, then run A* with a Manhattan heuristic and watch the "
            "frontier expand before the shortest path is drawn."
        ),
        body=body,
        script=script,
        css_extra=_ASTAR_CSS,
    )
    return _write(html, out, "astar_playground.html")


# ==========================================================================
# ==========================================================================
# 4. Neural-net playground (powered by the OptiX TypeScript kit)
# ==========================================================================

_NN_CSS = """
 .cap{color:#475569;max-width:640px;line-height:1.5}
 .optix-nn-playground{margin-top:14px}
 .optix-nn-playground canvas{border:1px solid #e2e8f0;border-radius:8px;background:#fff}
 .optix-nn-playground .optix-nn-readout{font-family:monospace;margin-top:8px}
 .optix-nn-playground button{margin-right:6px;padding:5px 12px;border-radius:6px;
      border:1px solid #c7d2fe;background:#eef2ff;cursor:pointer}
 .optix-nn-playground label{font-family:monospace;margin-right:14px}
 .optix-nn-playground input[type=range]{vertical-align:middle}
"""


def nn_playground(out: str | None = None) -> str:
    """A TensorFlow-Playground-style neural-net playground, powered by OptiX.

    Pick a 2-D dataset (XOR / circle / spiral), set the learning rate and hidden
    width, and train a tiny MLP while its decision boundary forms live. The math
    — seeded MLP init, forward pass, and backprop — is **OptiX**, a typed,
    unit-tested TypeScript kit compiled into OptimumAI. Self-contained, offline.
    """
    from optimumai.visualization.assets import optix_js

    body = (
        '<h1>Neural-net playground</h1>'
        '<p class="cap">Pick a dataset, set the learning rate and hidden units, then '
        '<b>Train</b> — watch the decision boundary bend until it separates the two '
        'classes. The network (seeded init, forward pass, and backprop) runs in '
        '<b>OptiX</b>, a typed, unit-tested TypeScript kit compiled into OptimumAI.</p>'
        '<div id="optix-nn"></div>'
    )
    boot = 'OptiX.mount.nnPlayground("#optix-nn", {dataset:"xor", hidden:8, lr:0.5, seed:42});'
    html = _page("OptimumAI · Neural-net playground", "Neural network playground",
                 body, optix_js() + "\n" + boot, css_extra=_NN_CSS)
    return _write(html, out, "nn_playground.html")


# ==========================================================================
# dispatcher + shared file-writing helper
# ==========================================================================

_PLAYGROUNDS = {
    "attention": transformer_attention_playground,
    "kmeans": kmeans_playground,
    "astar": astar_playground,
    "nn": nn_playground,
}


def playground(name: str, out: str | None = None) -> str:
    """Build the named playground ("attention", "kmeans", "astar", or "nn")."""
    try:
        builder = _PLAYGROUNDS[name]
    except KeyError as exc:
        valid = ", ".join(sorted(_PLAYGROUNDS))
        raise ValueError(f"unknown playground {name!r}; choose from: {valid}") from exc
    return builder(out=out)


def _write(html: str, out: str | None, default_name: str) -> str:
    """Write ``html`` to ``out`` (defaulting to ``default_name``) and return the path."""
    target = out if out is not None else default_name
    with open(target, "w", encoding="utf-8") as fh:
        fh.write(html)
    return target
