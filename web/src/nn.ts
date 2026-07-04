// nn.ts — a tiny, dependency-free MLP used by the decision-boundary playground.
// Architecture is fixed to: input -> hidden (tanh) -> output (sigmoid),
// trained with full-batch gradient descent via manual backprop.

import { mulberry32, uniform, type PRNG } from "./rng.js";

export interface MLP {
  /** Layer sizes, e.g. [2, 8, 1] for 2 inputs, 8 hidden units, 1 output. */
  sizes: number[];
  /** Hidden-layer weights: hidden x inputSize. */
  W1: number[][];
  /** Hidden-layer biases: hidden. */
  b1: number[];
  /** Output-layer weights: 1 x hidden. */
  W2: number[][];
  /** Output-layer bias: length 1. */
  b2: number[];
}

function randMatrix(rows: number, cols: number, rng: PRNG, scale: number): number[][] {
  const m: number[][] = [];
  for (let i = 0; i < rows; i++) {
    const row: number[] = [];
    for (let j = 0; j < cols; j++) row.push(uniform(rng, -scale, scale));
    m.push(row);
  }
  return m;
}

/**
 * Build a 2-layer MLP: inputSize -> hidden (tanh) -> 1 (sigmoid).
 * `sizes` is [inputSize, hiddenSize, outputSize] (outputSize is expected to be 1).
 */
export function makeMLP(sizes: number[], seed: number): MLP {
  if (sizes.length !== 3) {
    throw new Error("makeMLP expects sizes = [inputSize, hiddenSize, outputSize]");
  }
  const [inputSize, hiddenSize, outputSize] = sizes;
  const rng = mulberry32(seed);
  const scale1 = Math.sqrt(1 / inputSize);
  const scale2 = Math.sqrt(1 / hiddenSize);
  return {
    sizes: [...sizes],
    W1: randMatrix(hiddenSize, inputSize, rng, scale1),
    b1: new Array(hiddenSize).fill(0),
    W2: randMatrix(outputSize, hiddenSize, rng, scale2),
    b2: new Array(outputSize).fill(0),
  };
}

function tanhScalar(x: number): number {
  return Math.tanh(x);
}
function sigmoidScalar(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

interface ForwardCache {
  z1: number[];
  a1: number[];
  z2: number;
  a2: number;
}

function forwardCached(mlp: MLP, x: number[]): ForwardCache {
  const z1 = mlp.W1.map((row, i) => {
    let s = mlp.b1[i];
    for (let j = 0; j < row.length; j++) s += row[j] * x[j];
    return s;
  });
  const a1 = z1.map(tanhScalar);
  let z2 = mlp.b2[0];
  for (let j = 0; j < mlp.W2[0].length; j++) z2 += mlp.W2[0][j] * a1[j];
  const a2 = sigmoidScalar(z2);
  return { z1, a1, z2, a2 };
}

/** Forward pass returning the scalar sigmoid output for a single input vector. */
export function forward(mlp: MLP, x: number[]): number {
  return forwardCached(mlp, x).a2;
}

/**
 * One full-batch gradient-descent step (manual backprop) over X/y using
 * binary-cross-entropy-free mean-squared-error loss. Mutates `mlp` in place
 * and returns the mean squared-error loss *before* the update.
 */
export function trainStep(mlp: MLP, X: number[][], y: number[], lr: number): number {
  const n = X.length;
  const hidden = mlp.sizes[1];
  const gW1 = Array.from({ length: hidden }, () => new Array(mlp.sizes[0]).fill(0));
  const gb1 = new Array(hidden).fill(0);
  const gW2 = [new Array(hidden).fill(0)];
  const gb2 = [0];

  let lossSum = 0;

  for (let i = 0; i < n; i++) {
    const x = X[i];
    const target = y[i];
    const { a1, a2 } = forwardCached(mlp, x);

    const err = a2 - target;
    lossSum += err * err;

    // dL/dz2 for MSE loss with sigmoid output: dL/da2 * da2/dz2
    const dLda2 = 2 * err;
    const da2dz2 = a2 * (1 - a2);
    const dz2 = dLda2 * da2dz2;

    for (let j = 0; j < hidden; j++) {
      gW2[0][j] += dz2 * a1[j];
    }
    gb2[0] += dz2;

    for (let j = 0; j < hidden; j++) {
      const da1 = dz2 * mlp.W2[0][j];
      const dz1 = da1 * (1 - a1[j] * a1[j]); // tanh'
      for (let k = 0; k < mlp.sizes[0]; k++) {
        gW1[j][k] += dz1 * x[k];
      }
      gb1[j] += dz1;
    }
  }

  const invN = 1 / n;
  for (let j = 0; j < hidden; j++) {
    for (let k = 0; k < mlp.sizes[0]; k++) {
      mlp.W1[j][k] -= lr * gW1[j][k] * invN;
    }
    mlp.b1[j] -= lr * gb1[j] * invN;
    mlp.W2[0][j] -= lr * gW2[0][j] * invN;
  }
  mlp.b2[0] -= lr * gb2[0] * invN;

  return lossSum / n;
}

export type DatasetKind = "xor" | "circle" | "spiral";

export interface Dataset {
  X: number[][];
  y: number[];
}

/** Generate a deterministic 2-D toy dataset for the playground. */
export function datasets(kind: DatasetKind, n: number, seed: number): Dataset {
  const rng = mulberry32(seed);
  const X: number[][] = [];
  const y: number[] = [];

  if (kind === "xor") {
    for (let i = 0; i < n; i++) {
      const x0 = uniform(rng, -1, 1);
      const x1 = uniform(rng, -1, 1);
      const label = x0 * x1 >= 0 ? 1 : 0;
      X.push([x0, x1]);
      y.push(label);
    }
  } else if (kind === "circle") {
    for (let i = 0; i < n; i++) {
      const x0 = uniform(rng, -1, 1);
      const x1 = uniform(rng, -1, 1);
      const r = Math.sqrt(x0 * x0 + x1 * x1);
      const label = r < 0.5 ? 1 : 0;
      X.push([x0, x1]);
      y.push(label);
    }
  } else if (kind === "spiral") {
    for (let i = 0; i < n; i++) {
      const label = i % 2;
      const idx = Math.floor(i / 2);
      const t = (idx / (n / 2)) * 4; // 0..4
      const angleOffset = label === 0 ? 0 : Math.PI;
      const radius = t / 4;
      const angle = t * Math.PI + angleOffset;
      const noise = uniform(rng, -0.05, 0.05);
      const x0 = radius * Math.sin(angle) + noise;
      const x1 = radius * Math.cos(angle) + noise;
      X.push([x0, x1]);
      y.push(label);
    }
  } else {
    throw new Error(`Unknown dataset kind: ${String(kind)}`);
  }

  return { X, y };
}
