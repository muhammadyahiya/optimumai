// build.mjs — bundles OptiX into a single self-contained IIFE that the
// OptimumAI Python package embeds inline into its generated offline HTML.
import { build } from "esbuild";
import { mkdirSync, statSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

const __dirname = dirname(fileURLToPath(import.meta.url));
const outdir = resolve(__dirname, "../src/optimumai/visualization/_static");
const outfile = resolve(outdir, "optix.js");

mkdirSync(outdir, { recursive: true });

await build({
  entryPoints: [resolve(__dirname, "src/index.ts")],
  outfile,
  bundle: true,
  format: "iife",
  globalName: "OptiX",
  minify: true,
  target: ["es2019"],
  banner: { js: "/*! OptiX — OptimumAI widget kit (generated from web/, do not edit) */" },
});

const { size } = statSync(outfile);
console.log(`OptiX bundle written: ${outfile}`);
console.log(`Bundle size: ${size} bytes (${(size / 1024).toFixed(2)} KiB)`);
