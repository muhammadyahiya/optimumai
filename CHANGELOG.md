# Changelog

All notable changes to OptimumAI are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project adheres to
[Semantic Versioning](https://semver.org/).

## [1.0.0] — 2026-07-03

The **stable release**. From here the top-level `optimumai` API follows semantic
versioning (see `docs/stability.md`).

### Added

- **llm** — real token generation: `generate` / `generate_trace` auto-detect a
  local **Ollama** server (zero keys), then **Hugging Face** (`HF_TOKEN`),
  **Anthropic** (via the tutor), and finally a built-in **toy** bigram so a demo
  always produces tokens. Stdlib-only HTTP (no new deps). CLI: `generate`,
  `providers`.
- **visualization.concepts** — a `render_concept(name, fmt)` registry: **PNG for
  every concept, GIF where motion helps**, across 14 concepts. CLI: `visualize`.
- **circuit.interactive** — drag-the-inputs circuits as standalone HTML: live
  **softmax** (drag logits → probabilities move) and **backprop** (drag a/b/c/f →
  forward values + gradients recompute). CLI: `playground`.
- **notebooks** — the three notebooks now ship *inside* the wheel; `optimumai
  notebooks` copies them locally and launches Jupyter (`optimumai[notebooks]`).
- **docs site** — MkDocs Material + mkdocstrings, auto-deployed to GitHub Pages.
- **API stability** — a documented semver guarantee on `optimumai.__all__`.
- The tutor now auto-detects the Anthropic SDK + `ANTHROPIC_API_KEY` so `ask`
  answers out of the box. New extras: `[notebooks]`, `[docs]`.

### Notes

- The core install stays light (numpy, rich, click, pydantic-settings); heavy
  features are optional extras. If `pip` can't find a just-released version, that
  is PyPI index propagation lag — wait a moment and retry (or `--no-cache-dir`).

## [0.10.0] — 2026-07-03

The hands-on release — build, edit, animate, and grade, not just watch. Adds a
"GPU Kernels" course track (now **39 lessons across 12 tracks**).

### Added

- **kernels** — GPU kernels from scratch: a pure-Python CUDA-style simulator
  (`GpuSim`) that models the thread grid + memory hierarchy and instruments every
  access; a progression of real kernels (scalar/vector add → tiled matmul →
  softmax → **exact** flash attention), each checked vs NumPy; optional real
  backends (Numba/CuPy/Triton, auto-detected, graceful fallback); and a
  `KernelWorkbench` where you write a kernel and it's graded.
- **visualization.interactive** — `editable_plot`: a self-contained HTML editor
  where editing the equation replots the curve and dragging parameter sliders
  updates both the curve and the equation (bidirectional).
- **visualization.animate** — downloadable GIFs: gradient descent, diffusion
  noising, and softmax-vs-temperature (matplotlib + Pillow, headless).
- **exercises** — compute-the-answer exercises with tolerance grading (`Workbook`).
- **notebooks/** — three runnable Jupyter notebooks (quickstart, GPU kernels,
  hands-on).
- **tutor** — now auto-detects the Anthropic SDK + `ANTHROPIC_API_KEY` (and
  `OPENAI_API_KEY`), so `ask` actually answers once `optimumai[llm]` is installed.
- CLI: `kernel`, `animate`, `editor`, `exercise`. New extras: `[gpu]`.
- 19 new tests (218 total).

## [0.9.0] — 2026-07-03

Turns OptimumAI into a real learning product, grounded in cognitive science:
active recall (the testing effect) and spaced repetition.

### Added

- **quiz** — an active-recall engine: `Quiz(topic)` with a 57-question bank across
  20 topics; `optimumai quiz <topic>` reveals answers + explanations and scores you.
- **review** — spaced-repetition scheduling (SM-2): `ReviewScheduler` grades a
  review 0–5 and computes the next due date; `optimumai review` quizzes whatever's
  due. Quiz scores feed the scheduler automatically.
- **onboarding & discovery** — `optimumai start` (a 30-second guided first run)
  and `optimumai search <query>` (find lessons by keyword).
- `ProgressTracker` now persists spaced-repetition state alongside progress.
- 20 new tests (199 total).

## [0.8.0] — 2026-07-03

Frontier concepts — how today's large models are actually built and run — as a
new "Frontier" course track (now **35 lessons across 11 tracks**).

### Added

- **frontier.flash_attention** — IO-aware tiled attention with online softmax;
  proven **exact** vs standard attention (error ~1e-16), with the HBM/SRAM IO story.
- **frontier.quantization** — `quantize`/`dequantize` to int8/int4, symmetric or
  asymmetric, per-tensor or per-channel (scale + zero-point), with the error and
  compression ratio.
- **frontier.lora** — low-rank adaptation `W = W₀ + BA` (B init 0, r ≪ d): the
  trainable-parameter reduction and why fine-tuning starts from the base model.
- **frontier.rlhf** — the DPO objective on a preference pair (implicit reward,
  margin, loss) and the SFT→reward-model→PPO pipeline it collapses.
- CLI: `optimumai quantize "[...]" --bits 4` plus `learn` topics for all four.
- 8 new tests (185 total).

## [0.7.0] — 2026-07-03

The **circuit** — render any expression or `Value` graph as a computation-graph
"circuit" where each node carries its forward data and backward gradient, like
current through wires. Karpathy's `draw_dot` meets Anthropic's circuits.

### Added

- **circuit.graph** — `FlowGraph`/`FlowNode`, `build_from_value` (from an autograd
  DAG, inserting operator pseudo-nodes like micrograd's `draw_dot`), and
  `build_from_expression` which turns a user arithmetic expression into a real
  `Value` graph behind an **AST allow-list** (only names, numbers, + - * / **).
- **circuit.render** — `to_dot` (Graphviz, `rankdir=LR`), `to_terminal` (a Rich
  table with blue data / orange grad), `to_html` (a self-contained interactive
  vis-network graph), and a `render(source, fmt, out)` dispatcher.
- CLI: `optimumai circuit "<expr>" --vars "a=2,b=-3" --fmt terminal|html|dot`.
- Dashboard: a live **Circuit playground** — type an expression, see the graph.
- 13 new tests (177 total).

## [0.6.0] — 2026-07-03

Adds real **graphs** — matplotlib figures rendered headlessly to PNG (via the
`optimumai[viz]` extra; matplotlib is imported lazily so the base package is
unaffected).

### Added

- **visualization.plots** — `plot_activation` (activation + derivative),
  `plot_softmax_temperature`, `plot_heatmap`, `plot_attention` (attention map for
  your own text), `plot_embeddings` (PCA scatter), `plot_training_curve`.
- **visualization.landscape** — `plot_loss_landscape`: a 2D contour and/or 3D
  surface of a preset (bowl/saddle/rosenbrock) or a custom `x,y` expression, with
  the numeric **gradient-descent trajectory** overlaid; safe AST-checked
  expression parsing.
- CLI: `plot <kind> --out fig.png` and `landscape <func> --out land.png`.
- Every plotting function returns the saved path (with `out=`) or the Figure.
- 19 new tests (164 total; auto-skipped when matplotlib is absent).

## [0.5.0] — 2026-07-03

Makes OptimumAI interactive — feed it your own numbers, text, and equations and
watch them flow. Adds an "Interactive Playground" course track (now **31 lessons
across 10 tracks**).

### Added

- **interactive** — `parse_vector`/`parse_matrix` prompt helpers and an
  `optimumai repl` (prompt_toolkit when installed via `[repl]`, else plain input;
  session vars auto-applied to commands).
- **transformers.TextPipeline** — pass your own text and watch it flow: tokenize
  → embed → +positional → N transformer blocks → next-token distribution.
  `optimumai trace-text "your sentence"`.
- **symbolic.differentiate** — symbolically differentiate your own equation via
  SymPy (`[symbolic]` extra), checked numerically. `optimumai diff "x**3+2*x" --at 3`.
- **analysis.compare / sweep** — put two activations side by side on your input,
  or sweep a parameter (e.g. softmax temperature) and watch the output evolve.
- CLI: `repl`, `trace-text`, `diff`, `compare`, `sweep`, plus `-i/--interactive`
  on `algebra dot|cosine|matmul` and `softmax` to enter values at a prompt.
- New extras: `[repl]`, `[symbolic]`, `[tokenize]`. 32 new tests (145 total).

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

[1.0.0]: https://github.com/muhammadyahiya/optimumai/releases/tag/v1.0.0
[0.10.0]: https://github.com/muhammadyahiya/optimumai/releases/tag/v0.10.0
[0.9.0]: https://github.com/muhammadyahiya/optimumai/releases/tag/v0.9.0
[0.8.0]: https://github.com/muhammadyahiya/optimumai/releases/tag/v0.8.0
[0.7.0]: https://github.com/muhammadyahiya/optimumai/releases/tag/v0.7.0
[0.6.0]: https://github.com/muhammadyahiya/optimumai/releases/tag/v0.6.0
[0.5.0]: https://github.com/muhammadyahiya/optimumai/releases/tag/v0.5.0
[0.4.0]: https://github.com/muhammadyahiya/optimumai/releases/tag/v0.4.0
[0.3.0]: https://github.com/muhammadyahiya/optimumai/releases/tag/v0.3.0
[0.2.0]: https://github.com/muhammadyahiya/optimumai/releases/tag/v0.2.0
[0.1.0]: https://github.com/muhammadyahiya/optimumai/releases/tag/v0.1.0
