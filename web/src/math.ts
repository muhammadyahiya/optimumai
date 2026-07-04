// math.ts — small numerically-stable math helpers shared by the OptiX widgets.

/** Numerically stable softmax. Always sums to 1 (for finite inputs). */
export function softmax(xs: number[], temp = 1): number[] {
  if (xs.length === 0) return [];
  const t = temp === 0 ? 1e-12 : temp;
  const scaled = xs.map((x) => x / t);
  const max = Math.max(...scaled);
  const exps = scaled.map((x) => Math.exp(x - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / sum);
}

/** Index of the maximum value. Returns -1 for an empty array. */
export function argmax(xs: number[]): number {
  if (xs.length === 0) return -1;
  let best = 0;
  for (let i = 1; i < xs.length; i++) {
    if (xs[i] > xs[best]) best = i;
  }
  return best;
}

/** Dot product of two equal-length vectors. */
export function dot(a: number[], b: number[]): number {
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += a[i] * b[i];
  return sum;
}

/** Matrix-vector product: W (rows x cols) times x (cols) -> (rows). */
export function matVec(W: number[][], x: number[]): number[] {
  return W.map((row) => dot(row, x));
}

/** Elementwise tanh. */
export function tanh(xs: number[]): number[] {
  return xs.map((x) => Math.tanh(x));
}

/** Elementwise sigmoid. */
export function sigmoid(xs: number[]): number[] {
  return xs.map((x) => 1 / (1 + Math.exp(-x)));
}

/** Elementwise ReLU. */
export function relu(xs: number[]): number[] {
  return xs.map((x) => Math.max(0, x));
}
