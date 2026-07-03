# Changelog

All notable changes to OptimumAI are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project adheres to
[Semantic Versioning](https://semver.org/).

## [0.4.0] — 2026-07-03

Foundations of the stack — the math, frameworks, and hardware modern AI runs on,
each with the same `explain=True` treatment and folded into the course as three
new tracks (Math Foundations, Framework Internals, Systems & Hardware). The
course now spans **28 lessons across 9 tracks**.

### Added

- **foundations.math_foundations** — `tensor_intro_trace` (rank/shape/broadcasting)
  and `integrate` (trapezoid + Monte Carlo; expectations are integrals).
- **foundations.pytorch_foundations** — `pytorch_autograd_trace` maps
  `torch.Tensor`/`requires_grad`/dynamic-graph/`backward` onto OptimumAI's own
  `Value` engine (the micrograd → PyTorch line).
- **foundations.jax_foundations** — `grad`, `vmap`, and `pytree` traces:
  composable transformations of pure functions.
- **foundations.gpu_foundations** — `thread_hierarchy_trace` (grid → block → warp
  → thread) and `memory_hierarchy_trace` (registers → shared → global).
- **foundations.cuda_kernel** — `tiled_matmul_trace`: naive vs tiled matmul,
  shared-memory reuse, and memory coalescing (verified against NumPy).
- **foundations.kv_cache** — `kv_cache_size`/`kv_cache_trace`: why context length
  eats VRAM, with MHA vs GQA vs MQA comparisons.
- **foundations.vram** — `vram_estimate`/`vram_trace`: weights + gradients +
  optimizer states + activations + KV cache, training vs inference.
- CLI: `kvcache` and `vram` calculators, plus 9 new `learn` topics.
- 15 new tests (128 total).

## [0.3.0] — 2026-07-03

Turns OptimumAI into a first-principles **AI learning path** you can walk one
step at a time, with progress tracked across sessions — plus the applied-AI
modules from the roadmap.

### Added

- **curriculum** — a `Course` of ordered `Lesson`s grouped into tracks (linear
  algebra → calculus & autograd → optimization & neural nets → transformers →
  applied AI → world models & interpretability), each a runnable, explained
  `Trace`. Lessons declare prerequisites that always precede them.
- **progress** — `ProgressTracker` persists completed lessons to
  `~/.optimumai/progress.json` (override with `OPTIMUMAI_PROGRESS_PATH`), shared
  by the CLI and dashboard.
- **dashboard** — a Streamlit app (`optimumai dashboard`, `optimumai[dashboard]`)
  to browse the course, run any lesson, and track progress visually.
- **embeddings** — token → dense-vector lookup and nearest-neighbour search.
- **rag** — a full retrieval-augmented-generation pipeline trace: embed → cosine
  search → top-k → prompt assembly.
- **diffusion** — the DDPM forward noising schedule and the reverse denoising idea.
- **tutor** — an optional LLM tutor (`optimumai[llm]`) that degrades gracefully
  to a helpful offline message when litellm/an API key is absent.
- CLI: `course`, `learn <topic>` (auto-tracks completion), `progress`,
  `dashboard`, `ask`. `learn` is now backed by the curriculum.
- New extras: `[dashboard]`, `[all]`. 32 new tests (113 total).

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

[0.4.0]: https://github.com/muhammadyahiya/optimumai/releases/tag/v0.4.0
[0.3.0]: https://github.com/muhammadyahiya/optimumai/releases/tag/v0.3.0
[0.2.0]: https://github.com/muhammadyahiya/optimumai/releases/tag/v0.2.0
[0.1.0]: https://github.com/muhammadyahiya/optimumai/releases/tag/v0.1.0
