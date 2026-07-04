// canvas.ts — vanilla-canvas rendering helpers for the decision-boundary
// playground: a background heatmap of network output + a scatter overlay.

import { forward, type MLP } from "./nn.js";

export interface Bounds {
  xMin: number;
  xMax: number;
  yMin: number;
  yMax: number;
}

export const DEFAULT_BOUNDS: Bounds = { xMin: -1, xMax: 1, yMin: -1, yMax: 1 };

/** Map a network output in [0,1] to an RGB color (blue -> white -> orange). */
function colorFor(v: number): string {
  const t = Math.max(0, Math.min(1, v));
  // 0 -> orange-ish (class 0), 1 -> blue-ish (class 1), midpoint white.
  const orange: [number, number, number] = [245, 166, 35];
  const blue: [number, number, number] = [66, 133, 244];
  const mix = (a: number, b: number, f: number): number => Math.round(a + (b - a) * f);
  const [r, g, b] = [
    mix(orange[0], blue[0], t),
    mix(orange[1], blue[1], t),
    mix(orange[2], blue[2], t),
  ];
  return `rgb(${r}, ${g}, ${b})`;
}

/**
 * Fill the canvas with a coarse `res`x`res` grid of the network's output,
 * mapped from data-space `bounds` to canvas pixel-space.
 */
export function decisionBoundary(
  ctx: CanvasRenderingContext2D,
  mlp: MLP,
  bounds: Bounds = DEFAULT_BOUNDS,
  res = 40,
): void {
  const { canvas } = ctx;
  const w = canvas.width;
  const h = canvas.height;
  const cellW = w / res;
  const cellH = h / res;

  for (let gy = 0; gy < res; gy++) {
    const dataY = bounds.yMax - (gy / (res - 1)) * (bounds.yMax - bounds.yMin);
    for (let gx = 0; gx < res; gx++) {
      const dataX = bounds.xMin + (gx / (res - 1)) * (bounds.xMax - bounds.xMin);
      const out = forward(mlp, [dataX, dataY]);
      ctx.fillStyle = colorFor(out);
      ctx.fillRect(gx * cellW, gy * cellH, cellW + 1, cellH + 1);
    }
  }
}

/** Draw the dataset points on top of the decision boundary. */
export function scatter(
  ctx: CanvasRenderingContext2D,
  X: number[][],
  y: number[],
  bounds: Bounds = DEFAULT_BOUNDS,
): void {
  const { canvas } = ctx;
  const w = canvas.width;
  const h = canvas.height;
  const radius = 4;

  for (let i = 0; i < X.length; i++) {
    const [dataX, dataY] = X[i];
    const px = ((dataX - bounds.xMin) / (bounds.xMax - bounds.xMin)) * w;
    const py = h - ((dataY - bounds.yMin) / (bounds.yMax - bounds.yMin)) * h;

    ctx.beginPath();
    ctx.arc(px, py, radius, 0, Math.PI * 2);
    ctx.fillStyle = y[i] >= 0.5 ? "#1a56db" : "#c2410c";
    ctx.fill();
    ctx.lineWidth = 1;
    ctx.strokeStyle = "#ffffff";
    ctx.stroke();
  }
}
