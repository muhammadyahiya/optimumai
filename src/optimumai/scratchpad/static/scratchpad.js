/**
 * optimumai scratchpad — Tier 1 boards
 * -------------------------------------
 * Each initXBoard() function owns one JSXGraph board and wires it to the
 * trace panel in index.html. All math happens client-side — dragging a
 * point never round-trips to the server, matching the "instant, no
 * server round-trip" pattern used by JSXGraph/Plotly across this feature.
 *
 * To add a new concept board: write initYourConcept(board), add a case to
 * the switch in boot(), and add a matching entry in concepts.py.
 */

function katexRender(el, tex) {
  katex.render(tex, el, { throwOnError: false });
}

// ---------------------------------------------------------------------
// Tier 1 — Vector algebra: dot product & cosine similarity
// ---------------------------------------------------------------------
function initDotProductBoard() {
  const board = JXG.JSXGraph.initBoard("jxgbox", {
    boundingbox: [-6, 6, 6, -6],
    axis: true,
    showCopyright: false,
    showNavigation: false,
  });

  const origin = board.create("point", [0, 0], { visible: false, fixed: true });
  const a = board.create("point", [3, 2], { name: "a", color: "#D85A30", size: 4 });
  const b = board.create("point", [2, -2], { name: "b", color: "#378ADD", size: 4 });

  board.create("arrow", [origin, a], { strokeColor: "#D85A30", strokeWidth: 2 });
  board.create("arrow", [origin, b], { strokeColor: "#378ADD", strokeWidth: 2 });

  document.getElementById("metricA-label").textContent = "|a|";
  document.getElementById("metricB-label").textContent = "|b|";
  document.getElementById("metricC-label").textContent = "cos similarity";

  function update() {
    const [a1, a2] = [a.X(), a.Y()];
    const [b1, b2] = [b.X(), b.Y()];
    const dot = a1 * b1 + a2 * b2;
    const magA = Math.sqrt(a1 * a1 + a2 * a2);
    const magB = Math.sqrt(b1 * b1 + b2 * b2);
    const cos = magA > 0 && magB > 0 ? dot / (magA * magB) : 0;

    katexRender(
      document.getElementById("trace-eq"),
      `${a1.toFixed(1)} \\times ${b1.toFixed(1)} + ${a2.toFixed(1)} \\times ${b2.toFixed(1)} = ${dot.toFixed(2)}`
    );
    document.getElementById("metricA").textContent = magA.toFixed(2);
    document.getElementById("metricB").textContent = magB.toFixed(2);
    document.getElementById("metricC").textContent = cos.toFixed(2);
  }

  a.on("drag", update);
  b.on("drag", update);
  update();
}

// ---------------------------------------------------------------------
// Tier 1 — Derivatives: tangent line as instantaneous slope
// ---------------------------------------------------------------------
function initTangentLineBoard() {
  const board = JXG.JSXGraph.initBoard("jxgbox", {
    boundingbox: [-5, 8, 5, -8],
    axis: true,
    showCopyright: false,
    showNavigation: false,
  });

  const f = (x) => 0.3 * x * x * x - 2 * x; // a curve with a visible min/max
  const fPrime = (x) => 0.9 * x * x - 2;

  const curve = board.create("functiongraph", [f], { strokeColor: "#534AB7", strokeWidth: 2 });
  const glider = board.create("glider", [1, f(1), curve], { name: "x", color: "#D85A30", size: 4 });

  const tangent = board.create("tangent", [glider], { strokeColor: "#D85A30", strokeWidth: 2, dash: 2 });

  document.getElementById("metricA-label").textContent = "x";
  document.getElementById("metricB-label").textContent = "f(x)";
  document.getElementById("metricC-label").textContent = "slope f'(x)";

  function update() {
    const x = glider.X();
    const y = f(x);
    const slope = fPrime(x);

    katexRender(
      document.getElementById("trace-eq"),
      `f'(${x.toFixed(2)}) = ${slope.toFixed(2)}`
    );
    document.getElementById("metricA").textContent = x.toFixed(2);
    document.getElementById("metricB").textContent = y.toFixed(2);
    document.getElementById("metricC").textContent = slope.toFixed(2);
  }

  glider.on("drag", update);
  update();
}

// ---------------------------------------------------------------------
// Quiz reveal + boot
// ---------------------------------------------------------------------
function wireQuiz() {
  const btn = document.getElementById("reveal-btn");
  const answer = document.getElementById("quiz-a");
  if (!btn) return;
  btn.addEventListener("click", () => {
    answer.classList.remove("hidden");
    btn.disabled = true;
  });
}

function boot() {
  wireQuiz();
  switch (window.ACTIVE_CONCEPT) {
    case "dot_product":
      initDotProductBoard();
      break;
    case "tangent_line":
      initTangentLineBoard();
      break;
    default:
      console.warn("No board wired for concept:", window.ACTIVE_CONCEPT);
  }
}

document.addEventListener("DOMContentLoaded", boot);
