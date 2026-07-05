# API stability

From **v1.0.0**, OptimumAI follows [semantic versioning](https://semver.org/).

## What's stable

Everything exported from the top-level `optimumai` namespace — the names in
`optimumai.__all__` — is the **public API** and is stable:

- No breaking changes to these names without a **major** version bump.
- New features arrive in **minor** versions; bug fixes in **patch** versions.

The stable names include:

```
Vector, Matrix, Value, MLP, Attention, MultiHeadAttention, TransformerBlock,
TextPipeline, softmax, derivative, gradient, integrate, Adam, minimize,
JEPA, superposition, embedding_lookup, nearest_neighbors, RAGPipeline,
forward_diffusion, kv_cache_size, vram_estimate, generate,
Trace, ExplainLevel, COURSE, Course, Lesson, ProgressTracker,
Quiz, ReviewScheduler, Workbook, KernelWorkbench, GpuSim,
render_concept, editable_plot
```

and all other names in `__all__`.

## What may change

- Anything imported from a **submodule** rather than the top level.
- Any name prefixed with an underscore (`_`).
- The exact wording/formatting of rendered traces (they're for humans, not
  parsing) and the `meta` dict on a `Trace`.
- Optional-extra behavior that depends on third-party libraries.
- Internal CLI output formatting.

## Deprecation policy

When a public name must change, the old one keeps working for at least one
**minor** release and emits a `DeprecationWarning` pointing to the replacement.

## Versioning summary

| Bump | When |
|---|---|
| Patch `x.y.Z` | Bug fixes, documentation corrections |
| Minor `x.Y.0` | New features, new public names (backwards-compatible) |
| Major `X.0.0` | Breaking changes to public API |
