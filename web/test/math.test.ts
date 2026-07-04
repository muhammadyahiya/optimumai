import { describe, it, expect } from "vitest";
import { softmax, argmax, dot, matVec, tanh, sigmoid, relu } from "../src/math.js";

describe("softmax", () => {
  it("sums to 1", () => {
    const out = softmax([1, 2, 3, 4]);
    const sum = out.reduce((a, b) => a + b, 0);
    expect(sum).toBeCloseTo(1, 10);
  });

  it("sums to 1 for negative / mixed inputs", () => {
    const out = softmax([-5, 0, 5, 100]);
    const sum = out.reduce((a, b) => a + b, 0);
    expect(sum).toBeCloseTo(1, 10);
  });

  it("returns empty array for empty input", () => {
    expect(softmax([])).toEqual([]);
  });

  it("lower temperature sharpens the distribution", () => {
    const xs = [1, 2, 3];
    const sharp = softmax(xs, 0.1);
    const normal = softmax(xs, 1);
    expect(Math.max(...sharp)).toBeGreaterThan(Math.max(...normal));
  });

  it("higher temperature flattens the distribution", () => {
    const xs = [1, 2, 3];
    const flat = softmax(xs, 10);
    const normal = softmax(xs, 1);
    expect(Math.max(...flat)).toBeLessThan(Math.max(...normal));
    // flattened distribution should approach uniform
    const spread = Math.max(...flat) - Math.min(...flat);
    expect(spread).toBeLessThan(0.1);
  });
});

describe("argmax", () => {
  it("finds the index of the maximum value", () => {
    expect(argmax([1, 5, 3])).toBe(1);
    expect(argmax([9, 1, 2])).toBe(0);
    expect(argmax([1, 2, 9])).toBe(2);
  });

  it("returns -1 for an empty array", () => {
    expect(argmax([])).toBe(-1);
  });

  it("returns first occurrence on ties", () => {
    expect(argmax([3, 3, 1])).toBe(0);
  });
});

describe("dot / matVec", () => {
  it("computes dot product", () => {
    expect(dot([1, 2, 3], [4, 5, 6])).toBe(32);
  });

  it("computes matrix-vector product", () => {
    const W = [
      [1, 0],
      [0, 1],
      [1, 1],
    ];
    expect(matVec(W, [3, 4])).toEqual([3, 4, 7]);
  });
});

describe("elementwise activations", () => {
  it("tanh output stays within (-1, 1)", () => {
    const out = tanh([-100, -1, 0, 1, 100]);
    for (const v of out) {
      expect(v).toBeGreaterThanOrEqual(-1);
      expect(v).toBeLessThanOrEqual(1);
    }
    expect(out[2]).toBeCloseTo(0, 10);
  });

  it("sigmoid output stays within [0, 1], strictly within for moderate inputs", () => {
    const moderate = sigmoid([-10, -1, 0, 1, 10]);
    for (const v of moderate) {
      expect(v).toBeGreaterThan(0);
      expect(v).toBeLessThan(1);
    }
    expect(moderate[2]).toBeCloseTo(0.5, 10);

    // Extreme inputs saturate to 0/1 at double precision, which is correct.
    const extreme = sigmoid([-100, 100]);
    expect(extreme[0]).toBeGreaterThanOrEqual(0);
    expect(extreme[1]).toBeLessThanOrEqual(1);
  });

  it("relu clamps negatives to zero and passes positives through", () => {
    expect(relu([-2, -1, 0, 1, 2])).toEqual([0, 0, 0, 1, 2]);
  });
});
