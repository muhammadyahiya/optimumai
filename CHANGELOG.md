# Changelog

All notable changes to OptimumAI are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project adheres to
[Semantic Versioning](https://semver.org/).

## [1.6.0] — 2026-07-05

The **Concept Explorer** release — 30 interactive AI/ML concept explainers with
formula + Python code panels, a searchable concept browser, and CLI commands.

### Added

- **30 concept explainers** (`optimumai explain <concept>`) — DAG-style
  interactive HTML, each rendered step-by-step with a KaTeX formula *and* a
  runnable `optimumai` code snippet: `attention`, `backpropagation`,
  `gradient`, `gradient_descent`, `activation_functions`, `adam_optimizer`,
  `adamw_optimizer`, `softmax`, `cross_entropy_loss`, `layer_normalization`,
  `multi_agentic_workflow`, `weights_bias_neuron`, `linear_regression`,
  `logistic_regression`, `bias_variance_tradeoff`, `embedding_lookup`,
  `kmeans_clustering`, `kv_cache`, `model_drift`, `pca`, `q_learning`,
  `reinforcement_learning_overview`, `sum_and_dot_product`, `supervised_ml`,
  `unsupervised_ml`, `tokenizer`, `transformer_block`, `variance`, `dropout`,
  `tfidf`.
- **Code panel** — the DAG explainer's side pane now shows a
  `Python · optimumai` code block right next to the formula, in sync with the
  current step.
- **Concept Explorer** (`optimumai explore`) — a searchable, dark-themed
  landing page listing all 30 concepts as cards with direct launch links.
- **CLI**: `optimumai explain <concept>` (omit to list all) and
  `optimumai explore`.
- New `optimumai.visualization.explain` module exporting `explain`,
  `explore_concepts`, and `list_explain_concepts` from the top-level package.

## [1.5.0] — 2026-07-04

The **OptiX** release — a TypeScript/JavaScript foundation that hardens the
interactive layer. The widgets' JS was previously hand-written inline inside
Python strings (untyped, untested); OptiX replaces that with a typed,
unit-tested kit compiled to one self-contained asset. Suite grows to **589 tests**
(Python) + **32 TypeScript tests**. Base `pip install` stays node-free.

### Added

- **OptiX** (`web/`) — a small TypeScript widget kit (math, a seeded MLP with
  real backprop, a stage stepper, canvas helpers, and DOM mounts), **type-checked
  (`tsc`), linted (ESLint), and unit-tested (Vitest + happy-dom, 32 tests)**,
  bundled by **esbuild** into a single self-contained IIFE
  (`visualization/_static/optix.js`, ~7 KB, zero runtime deps / zero CDN),
  committed and shipped in the wheel. Python inlines it via
  `visualization.assets.optix_js()` — so the browser code is finally typed +
  tested, and node is only needed at dev/build time.
- **Neural-net playground** — `optimumai playground nn` (and `nn_playground()`):
  a TensorFlow-Playground-style widget built on OptiX — pick a 2-D dataset
  (XOR / circle / spiral), set the learning rate and hidden width, and **train a
  tiny MLP live while its decision boundary forms**. Self-contained, offline HTML.
- **CI** — a `web.yml` workflow type-checks, lints, tests, and rebuilds the TS,
  and fails if the committed bundle is stale (`git diff --exit-code`).

## [1.4.0] — 2026-07-04

The **learn-the-tools** release — runnable, explained tutorials for the tools you
actually type, plus in-depth docs for each. Suite grows to **585 tests**.

### Added

- **tutorials** — a tutorial engine (`Tutorial` / `Cell`, `get_tutorial`,
  `list_tutorials`) where each lesson is prose → a short code cell → its live
  output, executed in a shared namespace. Four tutorials:
  - **numpy** — arrays, dtypes, indexing, boolean/fancy indexing, **broadcasting**,
    vectorization vs loops, `axis` aggregations, linear algebra, seeded RNG, and a
    vectorization capstone. Runs fully on the base install.
  - **matplotlib** — the Figure/Axes model, line/bar/scatter/hist/subplots,
    styling, and saving (headless-safe: Agg + `savefig`). Needs `[viz]`.
  - **pytorch** — taught by **building each idea on OptimumAI's own engine so it
    runs torch-free** (autograd via `Value.backward()`, an MLP training loop, …),
    with the real `torch` code shown side-by-side (runs if you add `[torch]`).
  - **finetuning** — a runnable numpy **SFT → LoRA → QLoRA → DPO** toy pipeline
    (reusing `frontier.lora` / `quantization` / `rlhf`, and `rl.ppo`), plus the
    production **HuggingFace + PEFT + TRL** code as labeled reference.
  - CLI: `optimumai tutorial numpy` (or `matplotlib` / `pytorch` / `finetuning`),
    `--notebook out.ipynb` to export, and bare `optimumai tutorial` to list.
- **docs** — four detailed, Feynman-style guides on the site (Learn NumPy /
  matplotlib / PyTorch / LLM fine-tuning), every runnable snippet verified.
- **`[torch]` extra** — optional (`pip install optimumai[torch]`) so the PyTorch
  tutorial's real-torch cells can run; deliberately kept out of `all`/`dev` so the
  base install stays clean.

## [1.3.0] — 2026-07-04

The **see-it-flow** release — visualize your own data, watch concepts flow, and
read the whole feature set. Suite grows to **562 tests**. No new dependencies.

### Added

- **visualization.plotstudio** — a Plot Studio: feed numbers and get a chart
  (`bar`, `hist`, `scatter`, `box`, `line`, `pie`, `violin`) **plus the exact
  matplotlib + numpy code on screen**. Python: `describe`, `plot_code`,
  `plot_data`, `plot_studio_trace`; a self-contained interactive HTML playground
  (`plot_studio_playground`) with live chart + numpy stats + copy-able code. CLI:
  `optimumai plot-studio "[3,1,4,1,5,9]" --kind hist` and `optimumai playground
  plots`.
- **flows** — a subpackage of [distill.pub](https://distill.pub/2016/augmented-rnns/)
  / [Transformer-Explainer](https://poloclub.github.io/transformer-explainer/)-style
  **interactive circuit-flow diagrams** as self-contained, offline HTML (inline
  SVG + vanilla JS, zero CDN): `transformer_flow` (10-stage forward pass with a
  live attention heatmap), `attention_flow` (scaled dot-product attention),
  `tfidf_flow`, and `word2vec_flow`, plus a `flow(name)` dispatcher. CLI:
  `optimumai flow transformer` (or `attention` / `tfidf` / `word2vec`).
- **docs** — a comprehensive, Feynman-style **Features** guide covering every
  capability, and a complete **CLI reference** (every command + a copy-safe
  example), both on the live site.

### Changed

- **CLI** — `optimumai prompt` and `optimumai augrnn` with no argument now print
  the valid subcommands one per line (copy-safe), instead of assuming a default.
  (Running `optimumai augrnn attention|ntm|act` in a shell pipes into programs
  named `ntm`/`act`; use one word at a time, e.g. `optimumai augrnn ntm`.)

## [1.2.0] — 2026-07-04

The **interactive & explained** release — turning traces into things you can *play
with*, and filling the last conceptual gaps. Course grows to **76 lessons / 20
tracks**; suite to **522 tests**. No new dependencies (the playgrounds are
self-contained HTML; the gallery uses the existing `[viz]` extra).

### Added

- **prompting** — prompt-engineering patterns as explainable prompt-assembly
  traces: `zero_shot`, `few_shot`, `chain_of_thought`, `react`,
  `self_consistency`, `structured_output`. Each shows how the prompt is built and
  why it works / how it fails. CLI: `optimumai prompt <pattern>`.
- **augmented_rnns** — the pre-transformer lineage from
  [distill.pub](https://distill.pub/2016/augmented-rnns/): content-based
  `attention_read` (attention as differentiable memory), a Neural Turing Machine
  head (`ntm_read` / `ntm_write`, cosine addressing + erase/add), and
  `adaptive_computation_time` (halting probability + ponder cost). CLI:
  `optimumai augrnn attention|ntm|act`.
- **visualization.playgrounds** — self-contained interactive HTML (inline vanilla
  JS, **no server, no build, works offline**), inspired by
  [Transformer Explainer](https://poloclub.github.io/transformer-explainer/) and
  [TensorFlow Playground](https://playground.tensorflow.org): a Transformer-style
  **attention** widget (hover a query token, drag a temperature slider to
  re-softmax live), a **k-means** playground (click points, watch Lloyd's iterate),
  and an **A\*** playground (draw walls, watch the frontier expand). CLI:
  `optimumai playground attention|kmeans|astar` (softmax/backprop unchanged).
- **visualization.gallery** — per-concept matplotlib plots and animated GIFs for
  the v1.1 modules (k-means, decision boundary, A\* grid, gridworld value
  function, conv feature map, calibration reliability, PPO clip), registered in the
  `visualize` concept registry: `optimumai visualize kmeans|astar|value_iteration|
  conv2d|calibration|ppo_clip|decision_boundary --fmt png|gif`.
- **docs** — the site now covers the v1.1 + v1.2 packages (Classical AI / ML / RL
  and Interactive & Explained pages), plus a dashboard deploy guide. The
  GitHub Pages site is live at <https://muhammadyahiya.github.io/optimumai/>.

### Fixed

- **GitHub Pages** — the docs workflow was deploying to the `gh-pages` branch, but
  Pages had never been enabled in repo settings (a one-time step); the site
  returned 404. Pages is now enabled and serving.

## [1.1.0] — 2026-07-04

The **breadth release** — OptimumAI grows from the deep-learning/LLM stack out to
the *whole field* of AI. Six new packages, each concept an explainable `Trace`
with tests, a CLI command, and a course lesson. The course grows from 39 lessons
/ 12 tracks to **67 lessons / 18 tracks**; the suite from 234 to **455 tests**.

### Added

- **ml** — classical machine learning from scratch (numpy only): `LinearRegression`
  (OLS via the normal equation), `LogisticRegression` (sigmoid + cross-entropy +
  gradient descent), `KMeans` (Lloyd's), `KNN`, `DecisionTree` (Gini/entropy
  information gain), `GaussianNB`, `PCA` (covariance eigendecomposition), and a
  `metrics` module (accuracy, precision/recall/F1, confusion matrix, MSE, R²,
  ROC-AUC). CLI: `optimumai ml {linreg,logreg,kmeans,knn,tree,nb,pca,metrics}`.
- **search** — classical AI search: `bfs`, `dfs`, `uniform_cost_search` (Dijkstra),
  `greedy_best_first`, `astar` (with admissibility notes), `minimax` and
  `alpha_beta` (pruning), over reusable `Graph` / `GridWorld` problems. Each trace
  shows the frontier, expansion order, path, cost, and nodes expanded/pruned.
  CLI: `optimumai algo {bfs,astar,minimax}` (a new group — `search` stays the
  course full-text search).
- **rl** — reinforcement learning: `MDP` + `value_iteration` / `policy_iteration`
  (Bellman backups), tabular `q_learning` / `sarsa` on a gridworld, `reinforce`
  (policy gradient), and `ppo_clip` (the clipped surrogate objective behind
  RLHF, contrasted with the existing DPO). CLI: `optimumai rl
  {mdp,q-learning,reinforce,ppo}`.
- **nlp** — statistical NLP fundamentals: `BPETokenizer`, `TfidfVectorizer`,
  `NGramModel` (+ perplexity, add-k smoothing), `edit_distance` (Levenshtein DP
  with backtrace), and skip-gram `word2vec`. CLI: `optimumai nlp
  {bpe,tfidf,ngram,edit-distance,word2vec}`.
- **vision** — computer-vision primitives: `conv2d` (stride/padding, with the
  output-size formula), `max_pool2d` / `avg_pool2d`, `sobel_edges`, and a tiny
  `cnn_forward` (conv → relu → pool → dense → softmax, narrating the tensor
  shapes). CLI: `optimumai vision {conv,pool,sobel,cnn}`.
- **evaluation** — LLM evaluation metrics: `bleu`, `rouge_n` / `rouge_l`,
  `exact_match`, `token_f1`, `perplexity`, `ece` (calibration), and a candid
  `faithfulness_score` hallucination *heuristic* (documented as an educational
  proxy, not solved detection). CLI: `optimumai eval
  {bleu,rouge,perplexity,calibration,faithfulness}`.
- **notebook** — `04_classical_ai.ipynb`: a runnable tour of all six new
  packages, bundled in the wheel and launchable via `optimumai notebooks`.

### Fixed

- **core._fmt.num** — non-finite floats (`±inf`, `nan`) now render as `∞ / -∞ /
  nan` instead of raising (they occur legitimately in A\*/α-β windows and
  perplexity).
- **visualization.terminal** — step/`why_ai` text is now Rich-escaped, so literal
  brackets in the math (a state `[s0]`, a token `[the]`, `relu[x]`) are no longer
  swallowed as style tags.

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
