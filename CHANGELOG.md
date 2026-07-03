# Changelog

All notable changes to OptimumAI are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project adheres to
[Semantic Versioning](https://semver.org/).

## [0.2.0] — 2026-07-03

The fundamentals behind modern AI, each runnable with `explain=True`. Grounded in
Karpathy's micrograd/nanoGPT, Yann LeCun's world models, and Anthropic's
interpretability work — see [PHILOSOPHY.md](PHILOSOPHY.md).

### Added

- **autograd** — `Value`, a micrograd-style scalar autograd engine building a
  computation DAG, with `backward()` and a `backward_trace()` that renders the
  chain rule flowing backwards node by node.
- **calculus** — numeric `derivative`, `gradient`, and a `chain_rule` demo that
  cross-checks exact autograd against finite differences.
- **optimization** — `SGD` and `Adam` (with bias-corrected moments) operating on
  `Value` parameters, plus `minimize`/`minimize_trace` and a loss-curve sparkline.
- **neural_networks** — `Neuron`, `Layer`, `MLP` built entirely on `Value`, a
  `forward_backward_trace`, and a `train`/`train_demo` loop that learns a toy set.
- **transformers** — `MultiHeadAttention` (parallel heads + causal mask, nanoGPT
  style), sinusoidal `positional_encoding`, and a pre-norm `TransformerBlock`
  (LayerNorm → attention → FFN with residuals).
- **world_models** — `JEPA`, LeCun's Joint-Embedding Predictive Architecture, as
  an energy-based model that predicts in representation space, not pixel space.
- **interpretability** — `superposition`, Anthropic's toy model of why neurons are
  polysemantic and how sparse features are recoverable.
- CLI: `backprop`, `train`, `jepa`, `superposition`, and 11 new `learn` topics.
- `PHILOSOPHY.md` and an expanded test suite (77 tests, 93% coverage).

## [0.1.0] — 2026-07-03

The first release ships the spine of the SDK: the tracer engine plus the full
arc from a dot product to transformer attention, each runnable with
`explain=True`.

### Added

- **core** — `Trace`/`Step` computation-trace model, `ExplainLevel`
  (beginner → intermediate → engineer → researcher), and the `BaseOp` contract.
- **algebra** — `Vector` (`dot`, `norm`, `cosine_similarity`) and `Matrix`
  (`matmul`, transpose), each producing a step-by-step trace.
- **probability** — `softmax` with temperature control and the
  numerically-stable max-subtraction trick, explained.
- **transformers** — single-head scaled dot-product `Attention`,
  `softmax(QKᵀ/√dₖ)·V`, traced across all four stages.
- **visualization** — Rich terminal renderer with a consistent visual grammar.
- **cli** — the `optimumai` command: `algebra dot|matmul|cosine`, `softmax`,
  `attention --demo`, and `learn`.
- Tooling: hatchling build, pytest suite (40 tests, 91% coverage), ruff, a
  GitHub Actions CI matrix (Python 3.10–3.13), and a PyPI trusted-publishing
  workflow.

[0.2.0]: https://github.com/muhammadyahiya/optimumai/releases/tag/v0.2.0
[0.1.0]: https://github.com/muhammadyahiya/optimumai/releases/tag/v0.1.0
