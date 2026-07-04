// @vitest-environment happy-dom
import { describe, it, expect, afterEach } from "vitest";
import { nnPlayground } from "../src/mount.js";

describe("nnPlayground", () => {
  afterEach(() => {
    document.body.innerHTML = "";
  });

  it("mounts a canvas and the expected controls without throwing", () => {
    expect(() => nnPlayground(document.body)).not.toThrow();

    const canvas = document.body.querySelector("canvas");
    expect(canvas).not.toBeNull();

    const select = document.body.querySelector("select.optix-nn-dataset");
    expect(select).not.toBeNull();
    const options = select ? Array.from(select.querySelectorAll("option")).map((o) => o.value) : [];
    expect(options).toEqual(["xor", "circle", "spiral"]);

    const lr = document.body.querySelector("input.optix-nn-lr");
    expect(lr).not.toBeNull();

    const hidden = document.body.querySelector("input.optix-nn-hidden");
    expect(hidden).not.toBeNull();

    const trainBtn = document.body.querySelector("button.optix-nn-train");
    const stepBtn = document.body.querySelector("button.optix-nn-step");
    const resetBtn = document.body.querySelector("button.optix-nn-reset");
    expect(trainBtn).not.toBeNull();
    expect(stepBtn).not.toBeNull();
    expect(resetBtn).not.toBeNull();

    const readout = document.body.querySelector(".optix-nn-readout");
    expect(readout).not.toBeNull();
  });

  it("accepts a selector string as root", () => {
    const container = document.createElement("div");
    container.id = "optix-root";
    document.body.appendChild(container);

    expect(() => nnPlayground("#optix-root")).not.toThrow();
    expect(container.querySelector("canvas")).not.toBeNull();
  });

  it("step() advances the epoch readout", () => {
    const handle = nnPlayground(document.body, { nPoints: 20 });
    const readoutBefore = document.body.querySelector(".optix-nn-readout")?.textContent;
    handle.step();
    const readoutAfter = document.body.querySelector(".optix-nn-readout")?.textContent;
    expect(readoutAfter).not.toBe(readoutBefore);
    handle.destroy();
  });

  it("reset() restores epoch 0", () => {
    const handle = nnPlayground(document.body, { nPoints: 20 });
    handle.step();
    handle.step();
    handle.reset();
    const readout = document.body.querySelector(".optix-nn-readout")?.textContent ?? "";
    expect(readout.startsWith("Epoch: 0")).toBe(true);
    handle.destroy();
  });

  it("destroy() removes the mounted container", () => {
    const handle = nnPlayground(document.body);
    expect(document.body.querySelector("canvas")).not.toBeNull();
    handle.destroy();
    expect(document.body.querySelector("canvas")).toBeNull();
  });
});
