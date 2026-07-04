// index.ts — public entry point for the OptiX widget kit.
// Bundled as an IIFE, this exposes `window.OptiX = { math, nn, stepper, mount }`.

import * as math from "./math.js";
import * as nn from "./nn.js";
import { createStepper } from "./stepper.js";
import { nnPlayground } from "./mount.js";

const stepper = { createStepper };
const mount = { nnPlayground };

export { math, nn, stepper, mount };
