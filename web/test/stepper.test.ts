import { describe, it, expect } from "vitest";
import { createStepper } from "../src/stepper.js";

describe("createStepper", () => {
  it("starts at 0", () => {
    const s = createStepper(5);
    expect(s.current()).toBe(0);
  });

  it("next advances and clamps at nStages - 1", () => {
    const s = createStepper(3);
    s.next();
    expect(s.current()).toBe(1);
    s.next();
    expect(s.current()).toBe(2);
    s.next();
    expect(s.current()).toBe(2); // clamped
  });

  it("prev decreases and clamps at 0", () => {
    const s = createStepper(3);
    s.prev();
    expect(s.current()).toBe(0); // clamped, cannot go below 0
    s.next();
    s.next();
    s.prev();
    expect(s.current()).toBe(1);
  });

  it("atEnd reports correctly", () => {
    const s = createStepper(2);
    expect(s.atEnd()).toBe(false);
    s.next();
    expect(s.atEnd()).toBe(true);
    s.next();
    expect(s.atEnd()).toBe(true); // still clamped at end
  });

  it("reset returns to 0", () => {
    const s = createStepper(4);
    s.next();
    s.next();
    s.reset();
    expect(s.current()).toBe(0);
  });

  it("handles nStages = 1 (always at end, index 0)", () => {
    const s = createStepper(1);
    expect(s.current()).toBe(0);
    expect(s.atEnd()).toBe(true);
    s.next();
    expect(s.current()).toBe(0);
  });
});
