// mount.ts — builds a TensorFlow-Playground-style neural-net widget into a
// host element: canvas decision boundary + scatter, dataset/lr/hidden-units
// controls, Train/Step/Reset buttons, and a live epoch+loss readout.
// Vanilla DOM/canvas only — no external dependencies.

import { decisionBoundary, scatter, DEFAULT_BOUNDS, type Bounds } from "./canvas.js";
import { makeMLP, trainStep, datasets, type MLP, type DatasetKind } from "./nn.js";

export interface NnPlaygroundOptions {
  dataset?: DatasetKind;
  hidden?: number;
  lr?: number;
  seed?: number;
  nPoints?: number;
  canvasSize?: number;
  bounds?: Bounds;
}

export interface NnPlaygroundHandle {
  root: HTMLElement;
  canvas: HTMLCanvasElement;
  train(): void;
  stop(): void;
  step(): void;
  reset(): void;
  destroy(): void;
}

const DEFAULTS: Required<Omit<NnPlaygroundOptions, "bounds">> & { bounds: Bounds } = {
  dataset: "xor",
  hidden: 8,
  lr: 0.5,
  seed: 42,
  nPoints: 200,
  canvasSize: 320,
  bounds: DEFAULT_BOUNDS,
};

function resolveRoot(root: HTMLElement | string): HTMLElement {
  if (typeof root === "string") {
    const el = document.querySelector(root);
    if (!el) throw new Error(`nnPlayground: root selector not found: ${root}`);
    return el as HTMLElement;
  }
  return root;
}

function el<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  className?: string,
): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  if (className) node.className = className;
  return node;
}

/**
 * Build a full neural-net playground UI (dataset + canvas + controls) inside
 * `root`, wiring buttons to nn.trainStep and redrawing the decision boundary
 * on every step. Returns a handle for programmatic control / cleanup.
 */
export function nnPlayground(
  root: HTMLElement | string,
  opts: NnPlaygroundOptions = {},
): NnPlaygroundHandle {
  const hostRoot = resolveRoot(root);
  const cfg = { ...DEFAULTS, ...opts };

  let dataset = cfg.dataset;
  let hidden = cfg.hidden;
  let lr = cfg.lr;
  let data = datasets(dataset, cfg.nPoints, cfg.seed);
  let mlp: MLP = makeMLP([2, hidden, 1], cfg.seed);
  let epoch = 0;
  let lastLoss = Number.NaN;
  let timer: ReturnType<typeof setInterval> | null = null;

  // --- structure ---------------------------------------------------------
  const container = el("div", "optix-nn-playground");

  const canvas = el("canvas", "optix-nn-canvas");
  canvas.width = cfg.canvasSize;
  canvas.height = cfg.canvasSize;
  container.appendChild(canvas);

  const controls = el("div", "optix-nn-controls");
  container.appendChild(controls);

  // dataset select
  const datasetLabel = el("label", "optix-nn-label");
  datasetLabel.textContent = "Dataset";
  const datasetSelect = el("select", "optix-nn-dataset");
  datasetSelect.setAttribute("aria-label", "Dataset");
  (["xor", "circle", "spiral"] as DatasetKind[]).forEach((kind) => {
    const option = el("option");
    option.value = kind;
    option.textContent = kind;
    if (kind === dataset) option.selected = true;
    datasetSelect.appendChild(option);
  });
  datasetLabel.appendChild(datasetSelect);
  controls.appendChild(datasetLabel);

  // learning-rate slider
  const lrLabel = el("label", "optix-nn-label");
  lrLabel.textContent = "Learning rate";
  const lrSlider = el("input", "optix-nn-lr");
  lrSlider.type = "range";
  lrSlider.min = "0.01";
  lrSlider.max = "2";
  lrSlider.step = "0.01";
  lrSlider.value = String(lr);
  lrSlider.setAttribute("aria-label", "Learning rate");
  lrLabel.appendChild(lrSlider);
  controls.appendChild(lrLabel);

  // hidden-units slider
  const hiddenLabel = el("label", "optix-nn-label");
  hiddenLabel.textContent = "Hidden units";
  const hiddenSlider = el("input", "optix-nn-hidden");
  hiddenSlider.type = "range";
  hiddenSlider.min = "1";
  hiddenSlider.max = "16";
  hiddenSlider.step = "1";
  hiddenSlider.value = String(hidden);
  hiddenSlider.setAttribute("aria-label", "Hidden units");
  hiddenLabel.appendChild(hiddenSlider);
  controls.appendChild(hiddenLabel);

  // buttons
  const buttonRow = el("div", "optix-nn-buttons");
  const trainButton = el("button", "optix-nn-train");
  trainButton.type = "button";
  trainButton.textContent = "Train";
  const stepButton = el("button", "optix-nn-step");
  stepButton.type = "button";
  stepButton.textContent = "Step";
  const resetButton = el("button", "optix-nn-reset");
  resetButton.type = "button";
  resetButton.textContent = "Reset";
  buttonRow.appendChild(trainButton);
  buttonRow.appendChild(stepButton);
  buttonRow.appendChild(resetButton);
  controls.appendChild(buttonRow);

  // readout
  const readout = el("div", "optix-nn-readout");
  readout.setAttribute("role", "status");
  container.appendChild(readout);

  hostRoot.appendChild(container);

  // --- behavior ------------------------------------------------------------
  function updateReadout(): void {
    const lossText = Number.isFinite(lastLoss) ? lastLoss.toFixed(4) : "-";
    readout.textContent = `Epoch: ${epoch}  Loss: ${lossText}`;
  }

  function redraw(): void {
    const ctx = canvas.getContext("2d");
    if (ctx) {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      decisionBoundary(ctx, mlp, cfg.bounds, 40);
      scatter(ctx, data.X, data.y, cfg.bounds);
    }
    updateReadout();
  }

  function doStep(): void {
    lastLoss = trainStep(mlp, data.X, data.y, lr);
    epoch += 1;
    redraw();
  }

  function stop(): void {
    if (timer !== null) {
      clearInterval(timer);
      timer = null;
    }
  }

  function train(): void {
    if (timer !== null) return;
    timer = setInterval(doStep, 16);
  }

  function reset(): void {
    stop();
    epoch = 0;
    lastLoss = Number.NaN;
    data = datasets(dataset, cfg.nPoints, cfg.seed);
    mlp = makeMLP([2, hidden, 1], cfg.seed);
    redraw();
  }

  trainButton.addEventListener("click", () => {
    if (timer === null) {
      train();
      trainButton.textContent = "Pause";
    } else {
      stop();
      trainButton.textContent = "Train";
    }
  });
  stepButton.addEventListener("click", () => {
    stop();
    trainButton.textContent = "Train";
    doStep();
  });
  resetButton.addEventListener("click", () => {
    trainButton.textContent = "Train";
    reset();
  });
  datasetSelect.addEventListener("change", () => {
    dataset = datasetSelect.value as DatasetKind;
    reset();
  });
  lrSlider.addEventListener("input", () => {
    lr = parseFloat(lrSlider.value);
  });
  hiddenSlider.addEventListener("change", () => {
    hidden = parseInt(hiddenSlider.value, 10);
    reset();
  });

  redraw();

  return {
    root: container,
    canvas,
    train,
    stop,
    step: doStep,
    reset,
    destroy(): void {
      stop();
      container.remove();
    },
  };
}
