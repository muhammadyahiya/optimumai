import { describe, it, expect } from "vitest";
import { makeMLP, forward, trainStep, datasets } from "../src/nn.js";

describe("forward", () => {
  it("is deterministic for a fixed seed", () => {
    const mlp1 = makeMLP([2, 4, 1], 7);
    const mlp2 = makeMLP([2, 4, 1], 7);
    const x = [0.3, -0.6];
    expect(forward(mlp1, x)).toBe(forward(mlp2, x));
  });

  it("produces different weights for different seeds", () => {
    const mlp1 = makeMLP([2, 4, 1], 1);
    const mlp2 = makeMLP([2, 4, 1], 2);
    expect(mlp1.W1).not.toEqual(mlp2.W1);
  });

  it("output is always within sigmoid range (0, 1)", () => {
    const mlp = makeMLP([2, 6, 1], 3);
    for (const x of [
      [0, 0],
      [1, 1],
      [-1, -1],
      [5, -5],
    ]) {
      const out = forward(mlp, x);
      expect(out).toBeGreaterThan(0);
      expect(out).toBeLessThan(1);
    }
  });
});

describe("trainStep on XOR", () => {
  it("reduces loss over ~500 steps of full-batch gradient descent", () => {
    const { X, y } = datasets("xor", 200, 42);
    const mlp = makeMLP([2, 8, 1], 42);

    const initialLoss = trainStep(mlp, X, y, 0.5);
    let loss = initialLoss;
    for (let i = 0; i < 499; i++) {
      loss = trainStep(mlp, X, y, 0.5);
    }

    expect(loss).toBeLessThan(initialLoss);
    expect(loss).toBeLessThan(0.2);
  });
});

describe("datasets", () => {
  it("returns matching shapes for xor", () => {
    const { X, y } = datasets("xor", 50, 1);
    expect(X.length).toBe(50);
    expect(y.length).toBe(50);
    for (const point of X) expect(point.length).toBe(2);
    for (const label of y) expect([0, 1]).toContain(label);
  });

  it("returns matching shapes for circle", () => {
    const { X, y } = datasets("circle", 30, 2);
    expect(X.length).toBe(30);
    expect(y.length).toBe(30);
    for (const point of X) expect(point.length).toBe(2);
    for (const label of y) expect([0, 1]).toContain(label);
  });

  it("returns matching shapes for spiral", () => {
    const { X, y } = datasets("spiral", 40, 3);
    expect(X.length).toBe(40);
    expect(y.length).toBe(40);
    for (const point of X) expect(point.length).toBe(2);
    for (const label of y) expect([0, 1]).toContain(label);
  });

  it("is deterministic for a fixed seed", () => {
    const a = datasets("circle", 20, 99);
    const b = datasets("circle", 20, 99);
    expect(a.X).toEqual(b.X);
    expect(a.y).toEqual(b.y);
  });
});
