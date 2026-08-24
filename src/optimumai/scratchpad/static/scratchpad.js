/**
 * optimumai scratchpad — generic board renderers.
 * ------------------------------------------------------------------
 * There is one renderer per board *kind*, not per concept. A concept is a
 * BoardSpec (see concepts.py); this file interprets it. Adding a concept means
 * adding a Python dict entry, not a function here.
 *
 * Note what this file does NOT contain:
 *   - the maths of any curve. Python differentiates with SymPy and hands us
 *     JavaScript via SymPy's jscode printer, so f and f' cannot disagree.
 *   - any colour literal. They arrive in BOARD.palette from optimumai.design.
 */

const BOARD = window.BOARD_SPEC;
const P = BOARD.palette;

/* Compiled from a SymPy-generated string that originates in our own
 * concepts.py -- there is no path from user input to here. */
function fnFromJs(jsSource) {
  return new Function("x", "return (" + jsSource + ");");
}

function katexInto(el, tex) {
  if (window.katex) katex.render(tex, el, { throwOnError: false });
  else el.textContent = tex;
}

// --- generic chrome -------------------------------------------------

function buildReadouts() {
  const wrap = document.getElementById("readouts");
  wrap.innerHTML = "";
  BOARD.readouts.forEach((r, i) => {
    const card = document.createElement("div");
    card.className = "card metric" + (i === BOARD.readouts.length - 1 ? " accent" : "");
    card.innerHTML =
      '<p class="card-label">' + r.label + '</p><p class="metric-value" id="ro-' + r.key + '">—</p>';
    wrap.appendChild(card);
  });
}
function setReadout(key, value) {
  const el = document.getElementById("ro-" + key);
  if (el) el.textContent = value;
}

function buildParams(onChange) {
  const wrap = document.getElementById("params");
  if (!BOARD.params.length) { wrap.style.display = "none"; return {}; }
  wrap.innerHTML = "";
  const state = {};
  BOARD.params.forEach((p) => {
    state[p.name] = p.default;
    const row = document.createElement("div");
    row.className = "param-row";
    row.innerHTML =
      '<label for="p-' + p.name + '">' + p.label + '</label>' +
      '<input type="range" id="p-' + p.name + '" min="' + p.min + '" max="' + p.max +
      '" step="' + p.step + '" value="' + p.default + '">' +
      '<span class="param-value" id="pv-' + p.name + '">' + p.default + '</span>';
    wrap.appendChild(row);
    row.querySelector("input").addEventListener("input", (e) => {
      state[p.name] = parseFloat(e.target.value);
      document.getElementById("pv-" + p.name).textContent = state[p.name];
      onChange(state);
    });
  });
  return state;
}
function syncParamInputs(state) {
  Object.entries(state).forEach(([k, v]) => {
    const input = document.getElementById("p-" + k);
    if (input) { input.value = v; document.getElementById("pv-" + k).textContent = v; }
  });
}

/** Snapshots are the author's claim about where something interesting happens. */
function buildSnapshots(apply) {
  const wrap = document.getElementById("snapshots");
  if (!BOARD.snapshots.length) { wrap.style.display = "none"; return; }
  wrap.innerHTML = '<p class="card-label">worth a look</p>';
  BOARD.snapshots.forEach((s) => {
    const chip = document.createElement("button");
    chip.className = "snapshot";
    chip.textContent = s.label;
    chip.title = s.note;
    chip.addEventListener("click", () => {
      apply(s.values);
      document.getElementById("snapshot-note").textContent = s.note;
    });
    wrap.appendChild(chip);
  });
}

function newBoard() {
  return JXG.JSXGraph.initBoard("jxgbox", {
    boundingbox: BOARD.bounding_box,
    axis: true, showCopyright: false, showNavigation: false,
  });
}

// --- kind: vectors --------------------------------------------------

function initVectors() {
  const board = newBoard();
  const origin = board.create("point", [0, 0], { visible: false, fixed: true });
  const pts = {};
  BOARD.points.forEach((p) => {
    pts[p.name] = board.create("point", [p.x, p.y], { name: p.name, color: p.color, size: 4 });
    board.create("arrow", [origin, pts[p.name]], { strokeColor: p.color, strokeWidth: 2 });
  });
  const a = pts.a, b = pts.b;

  function update() {
    const [a1, a2] = [a.X(), a.Y()], [b1, b2] = [b.X(), b.Y()];
    const dot = a1 * b1 + a2 * b2;
    const magA = Math.hypot(a1, a2), magB = Math.hypot(b1, b2);
    const cos = magA > 0 && magB > 0 ? dot / (magA * magB) : 0;
    katexInto(document.getElementById("trace-eq"),
      `${a1.toFixed(1)} \\times ${b1.toFixed(1)} + ${a2.toFixed(1)} \\times ${b2.toFixed(1)} = ${dot.toFixed(2)}`);
    setReadout("magA", magA.toFixed(2));
    setReadout("magB", magB.toFixed(2));
    setReadout("cos", cos.toFixed(2));
  }
  a.on("drag", update); b.on("drag", update);
  buildSnapshots((v) => {
    a.moveTo([v.ax, v.ay]); b.moveTo([v.bx, v.by]); board.update(); update();
  });
  update();
}

// --- kind: function -------------------------------------------------

function initFunction() {
  const board = newBoard();
  const expr = BOARD.expression;
  const f = fnFromJs(expr.js), fp = fnFromJs(expr.derivative_js);

  const curve = board.create("functiongraph", [f], { strokeColor: P.grid, strokeWidth: 2 });
  const glider = board.create("glider", [1, f(1), curve],
    { name: "x", color: P.attention, size: 4 });
  board.create("tangent", [glider],
    { strokeColor: P.derivative, strokeWidth: 2, dash: 2 });

  document.getElementById("expr-label").innerHTML = "";
  katexInto(document.getElementById("expr-label"),
    `f(x) = ${expr.latex} \\qquad f'(x) = ${expr.derivative_latex}`);

  function update() {
    const x = glider.X();
    katexInto(document.getElementById("trace-eq"),
      `f'(${x.toFixed(2)}) = ${fp(x).toFixed(2)}`);
    setReadout("x", x.toFixed(2));
    setReadout("fx", f(x).toFixed(2));
    setReadout("slope", fp(x).toFixed(2));
  }
  glider.on("drag", update);
  buildSnapshots((v) => { glider.moveTo([v.x, f(v.x)]); board.update(); update(); });
  update();
}

// --- kind: matrix ---------------------------------------------------

function initMatrix() {
  const board = newBoard();
  const [xmin, ymax, xmax, ymin] = BOARD.bounding_box;
  const R = Math.max(Math.abs(xmin), Math.abs(xmax), Math.abs(ymin), Math.abs(ymax));

  // faded reference grid: where space *was*, so you can see what moved
  for (let k = -R; k <= R; k++) {
    board.create("segment", [[k, -R], [k, R]],
      { strokeColor: P.grid_faded, strokeWidth: 1, opacity: 0.35, fixed: true, highlight: false });
    board.create("segment", [[-R, k], [R, k]],
      { strokeColor: P.grid_faded, strokeWidth: 1, opacity: 0.35, fixed: true, highlight: false });
  }

  const origin = board.create("point", [0, 0], { visible: false, fixed: true });
  const spec = {}; BOARD.points.forEach((p) => (spec[p.name] = p));
  const i = board.create("point", [spec.i.x, spec.i.y],
    { name: "i", color: spec.i.color, size: 4, strokeWidth: 3 });
  const j = board.create("point", [spec.j.x, spec.j.y],
    { name: "j", color: spec.j.color, size: 4, strokeWidth: 3 });

  // transformed grid: the image of the line x=k is k*i + t*j
  for (let k = -R; k <= R; k++) {
    board.create("segment", [
      () => [k * i.X() - R * j.X(), k * i.Y() - R * j.Y()],
      () => [k * i.X() + R * j.X(), k * i.Y() + R * j.Y()],
    ], { strokeColor: P.grid, strokeWidth: k === 0 ? 2 : 1, opacity: k === 0 ? 1 : 0.55 });
    board.create("segment", [
      () => [k * j.X() - R * i.X(), k * j.Y() - R * i.Y()],
      () => [k * j.X() + R * i.X(), k * j.Y() + R * i.Y()],
    ], { strokeColor: P.grid, strokeWidth: k === 0 ? 2 : 1, opacity: k === 0 ? 1 : 0.55 });
  }

  // the unit square: its area IS the determinant
  board.create("polygon", [
    origin, i, () => [i.X() + j.X(), i.Y() + j.Y()], j,
  ], {
    fillColor: P.attention, fillOpacity: 0.3, highlight: false,
    borders: { strokeColor: P.attention, strokeWidth: 2 },
  });
  board.create("arrow", [origin, i], { strokeColor: spec.i.color, strokeWidth: 4 });
  board.create("arrow", [origin, j], { strokeColor: spec.j.color, strokeWidth: 4 });

  function update() {
    const det = i.X() * j.Y() - i.Y() * j.X();
    katexInto(document.getElementById("trace-eq"),
      `\\det\\begin{pmatrix}${i.X().toFixed(2)} & ${j.X().toFixed(2)}\\\\` +
      `${i.Y().toFixed(2)} & ${j.Y().toFixed(2)}\\end{pmatrix} = ${det.toFixed(2)}`);
    setReadout("det", det.toFixed(3));
    setReadout("area", Math.abs(det).toFixed(3) + (det < 0 ? " (flipped)" : ""));
    const note = document.getElementById("snapshot-note");
    if (Math.abs(det) < 0.02) {
      note.textContent = "Determinant ~0: space is squashed onto a line. Not invertible.";
    }
  }
  i.on("drag", update); j.on("drag", update);
  buildSnapshots((v) => {
    i.moveTo([v.ix, v.iy]); j.moveTo([v.jx, v.jy]); board.update(); update();
  });
  update();
}

// --- kind: descent --------------------------------------------------

function initDescent() {
  const board = newBoard();
  const expr = BOARD.expression;
  const f = fnFromJs(expr.js), fp = fnFromJs(expr.derivative_js);
  board.create("functiongraph", [f], { strokeColor: P.grid, strokeWidth: 2 });

  katexInto(document.getElementById("expr-label"),
    `L(x) = ${expr.latex} \\qquad L'(x) = ${expr.derivative_latex}`);

  let marks = [];
  function clearMarks() { marks.forEach((m) => board.removeObject(m)); marks = []; }

  function iterate(state) {
    const lr = state.lr, steps = Math.round(state.steps);
    let x = state.x0;
    const xs = [x];
    for (let k = 0; k < steps; k++) {
      x = x - lr * fp(x);
      if (!isFinite(x) || Math.abs(x) > 1e6) { xs.push(x); break; }
      xs.push(x);
    }
    return xs;
  }

  /* The iteration on a quadratic is x <- x(1 - lr*L''), so behaviour is decided
   * by whether |x| grows and whether the sign alternates -- not by whether |x|
   * has yet reached some arbitrary magnitude. Checking a large threshold misses
   * divergence that simply has not had enough steps to blow up. */
  function classify(xs) {
    const last = xs[xs.length - 1];
    if (!isFinite(last)) return ["diverging", P.counterexample];
    const tail = xs.slice(-Math.min(4, xs.length));
    const mags = tail.map(Math.abs);
    const growing = mags.length > 1 && mags[mags.length - 1] > mags[0] * (1 + 1e-9);
    const alternating = tail.some((v, k) => k > 0 && v * tail[k - 1] < 0);
    if (growing || Math.abs(last) > 1e3) return ["diverging", P.counterexample];
    if (Math.abs(last) < 0.05) return [alternating ? "converged (oscillating)" : "converged", P.result];
    if (alternating) return ["oscillating", P.attention];
    return ["still descending", P.given];
  }

  function render(state) {
    clearMarks();
    const xs = iterate(state);
    const [status, colour] = classify(xs);
    xs.forEach((x, k) => {
      if (!isFinite(x) || Math.abs(x) > 1e4) return;
      const isLast = k === xs.length - 1;
      marks.push(board.create("point", [x, f(x)], {
        name: "", size: isLast ? 4 : 2, fixed: true, highlight: false,
        color: isLast ? colour : P.attention,
        // a fading trail: older iterates recede
        opacity: 0.25 + 0.75 * (k / Math.max(1, xs.length - 1)),
      }));
      if (k > 0) {
        const prev = xs[k - 1];
        if (isFinite(prev) && Math.abs(prev) <= 1e4) {
          marks.push(board.create("segment", [[prev, f(prev)], [x, f(x)]], {
            strokeColor: colour, strokeWidth: 1.5, dash: 2, fixed: true, highlight: false,
          }));
        }
      }
    });
    const last = xs[xs.length - 1];
    katexInto(document.getElementById("trace-eq"),
      `x \\leftarrow x - ${state.lr.toFixed(2)}\\,L'(x)`);
    setReadout("final_x", isFinite(last) ? last.toFixed(3) : "diverged");
    setReadout("final_y", isFinite(last) && Math.abs(last) < 1e4 ? f(last).toFixed(3) : "∞");
    setReadout("status", status);
    document.getElementById("snapshot-note").textContent =
      status === "diverging"
        ? "The step overshoots further each time — this is why the learning rate matters."
        : "";
  }

  const state = buildParams(render);
  buildSnapshots((v) => { Object.assign(state, v); syncParamInputs(state); render(state); });
  render(state);
}

// --- quiz + boot ----------------------------------------------------

function wireQuiz() {
  const btn = document.getElementById("reveal-btn");
  const answer = document.getElementById("quiz-a");
  if (!btn) return;
  btn.addEventListener("click", () => { answer.classList.remove("hidden"); btn.disabled = true; });
}

function wireComplete() {
  const btn = document.getElementById("complete-btn");
  if (!btn) return;
  btn.addEventListener("click", async () => {
    btn.disabled = true;
    try {
      const r = await fetch("/api/complete/" + window.ACTIVE_CONCEPT, { method: "POST" });
      const d = await r.json();
      btn.textContent = d.completed ? "Marked complete ✓" : "Could not save";
    } catch (e) {
      btn.textContent = "Could not save";
    }
  });
}

const RENDERERS = {
  vectors: initVectors, function: initFunction, matrix: initMatrix, descent: initDescent,
};

function boot() {
  wireQuiz();
  wireComplete();
  buildReadouts();
  const render = RENDERERS[BOARD.kind];
  if (!render) { console.warn("no renderer for board kind:", BOARD.kind); return; }
  render();
}

document.addEventListener("DOMContentLoaded", boot);
