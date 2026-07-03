# API stability

From **v1.0.0**, OptimumAI follows [semantic versioning](https://semver.org/).

## What's stable

Everything exported from the top-level `optimumai` namespace — i.e. the names in
`optimumai.__all__` — is the **public API** and is stable:

- No breaking changes to these names without a **major** version bump.
- New features arrive in **minor** versions; fixes in **patch** versions.

That includes: `Vector`, `Matrix`, `Value`, `MLP`, `Attention`,
`MultiHeadAttention`, `TransformerBlock`, `softmax`, `Trace`, `ExplainLevel`,
`COURSE`, `Course`, `Lesson`, `ProgressTracker`, `Quiz`, `ReviewScheduler`,
`Workbook`, `KernelWorkbench`, `GpuSim`, `TextPipeline`, `Tutor`, `generate`,
`render_concept`, `editable_plot`, and the other names in `__all__`.

## What may change

- Anything imported from a **submodule** rather than the top level.
- Any name prefixed with an underscore (`_`).
- The exact wording/formatting of rendered traces (they're for humans, not
  parsing) and the contents of the `meta` dict on a `Trace`.
- Optional-extra behavior that depends on third-party libraries.

## Deprecation policy

When a public name must change, the old one keeps working for at least one minor
release and emits a `DeprecationWarning` pointing to the replacement.
