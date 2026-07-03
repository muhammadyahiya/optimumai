# OptimumAI

**Unlock the math behind AI.**

Every mathematical operation in modern AI — from a dot product to a full
attention block — can be run with `explain=True` to produce a **step-by-step
computation trace**, a **terminal visualization**, and the intuition for **why
AI actually uses it**.

The same code runs fast in production *or* teaches you exactly what it's doing.
micrograd shows you scalar backprop; EpyNN walks you through MLPs — OptimumAI
gives you a single, traceable API that runs the whole way from `a · b` up to
`softmax(QKᵀ/√dₖ)·V`.

```python
from optimumai import Vector

Vector([1, 2, 3]).dot(Vector([4, 5, 6]), explain=True)
```

```
╭───────────────────────── OptimumAI ──────────────────────────╮
│ DOT                                                           │
│ a · b = Σᵢ aᵢ·bᵢ                                              │
╰───────────────────────────────────────────────────────────────╯
 #  Step                 Computation
 1  Multiply component 0 1 × 4 = 4
 2  Multiply component 1 2 × 5 = 10
 3  Multiply component 2 3 × 6 = 18
 4  Sum the products     4 + 10 + 18 = 32
╭──────── Result · scalar ────────╮
│ 32                              │
╰─────────────────────────────────╯
╭──────────── Why AI uses this ────────────╮
│ • Similarity between two embedding vectors │
│ • The raw attention score q · k            │
│ • The inner loop of every matrix multiply  │
╰────────────────────────────────────────────╯
```

---

## Install

```bash
pip install optimumai
```

Optional extras:

```bash
pip install "optimumai[llm]"   # LLM tutor (Q&A over concepts)
pip install "optimumai[viz]"   # extra plotting backends
```

## Quickstart — Python

```python
from optimumai import Vector, Matrix, softmax, Attention

# Linear algebra
Vector([1, 2, 3]).cosine_similarity(Vector([2, 4, 6]), explain=True)   # → 1.0
Matrix([[1, 2], [3, 4]]).matmul(Matrix([[5, 6], [7, 8]]), explain=True)

# Probability
softmax([2.0, 1.0, 0.1], temperature=0.5, explain=True)

# Transformers — the headline
Attention(d_k=4).forward(Q, K, V, explain=True)
```

Every `explain=True` call returns the numeric result *and* prints the trace, so
it drops straight into notebooks, scripts, and tests. Prefer the data over the
print-out? Use the `*_trace` variants:

```python
trace = Vector([1, 2, 3]).dot_trace(Vector([4, 5, 6]))
trace.result      # 32.0
trace.steps       # [Step(...), Step(...), ...]
trace.why_ai      # ['Similarity between two embedding vectors', ...]
```

## The fundamentals (v0.2)

The same `explain=True` philosophy now runs all the way down to the atoms of
modern AI. Inspired by Karpathy's [micrograd](https://github.com/karpathy/micrograd)/[nanoGPT](https://github.com/karpathy/nanoGPT),
Yann LeCun's world models, and Anthropic's interpretability research
(see [PHILOSOPHY.md](PHILOSOPHY.md)):

```python
from optimumai import Value, MLP, MultiHeadAttention, JEPA, superposition

# Autograd — a scalar computation graph that differentiates itself (micrograd)
a = Value(2.0, label="a"); b = Value(-3.0, label="b")
L = (a * b).tanh(); L.label = "L"
L.backprop(explain=True)          # watch the chain rule flow backwards

# Neural net — real backprop, trained by gradient descent
from optimumai.neural_networks import train_demo
train_demo(steps=150).render("intermediate")   # loss falls to ~0

# Transformers — multi-head attention with a causal mask (the GPT decoder)
MultiHeadAttention.demo().render("engineer")

# World models — LeCun's JEPA: predict in latent space, not pixels
JEPA.demo().render("engineer")    # energy = ‖predicted embedding − target embedding‖²

# Interpretability — Anthropic's superposition: why neurons are polysemantic
superposition(n_features=5, n_neurons=2, explain=True)
```

## Quickstart — CLI

```bash
optimumai algebra dot "[1,2,3]" "[4,5,6]"
optimumai softmax "[2,1,0.1]" --temperature 0.5
optimumai attention --demo --level engineer
optimumai backprop                    # chain rule through a scalar graph
optimumai train --steps 150           # train a tiny MLP, watch loss fall
optimumai jepa --demo                 # LeCun's world-model energy
optimumai superposition               # Anthropic's polysemantic neurons
optimumai learn                       # list every topic (35 across 11 tracks)
optimumai learn transformer --level researcher
```

## Learn it as a course (v0.3)

OptimumAI isn't just a library — it's a **first-principles AI learning path** you
can walk one step at a time, with your progress tracked across sessions.

```bash
optimumai course              # the full path, grouped by track, with ✓/○ progress
optimumai learn dot           # run a lesson — it's marked complete automatically
optimumai progress            # a progress bar + what to learn next
optimumai dashboard           # a Streamlit dashboard to browse + track visually
optimumai ask "why LayerNorm after attention?"   # optional LLM tutor
```

The lessons build on each other — linear algebra → calculus & autograd →
optimization & neural nets → transformers → applied AI (embeddings, RAG,
diffusion) → world models & interpretability — each one a runnable, explained
`Trace`. From Python:

```python
from optimumai import COURSE, ProgressTracker

for lesson in COURSE:
    print(lesson.track, lesson.id, "—", lesson.summary)

COURSE.get("rag").run("engineer")        # retrieval-augmented generation, traced
ProgressTracker().mark_complete("rag")   # track your progress
```

Install the extras you want:

```bash
pip install "optimumai[dashboard]"   # Streamlit progress dashboard
pip install "optimumai[llm]"         # LLM tutor (set OPTIMUMAI_API_KEY)
pip install "optimumai[all]"         # everything
```

## Frontier concepts (v0.8)

How today's large models are actually built and run — each with the same
step-by-step trace:

```bash
optimumai learn flash_attention   # IO-aware tiling + online softmax (exact!)
optimumai learn lora              # low-rank adapters — 10,000× fewer trainable params
optimumai learn dpo               # align to preferences without a reward model or RL
optimumai quantize "[0.1,-2.3,4.5,3.14]" --bits 4    # quantize YOUR values, see the error
```

```python
from optimumai.frontier import flash_attention, quantize, lora, dpo
import numpy as np

Q, K, V = (np.random.default_rng(0).normal(size=(6, 4)) for _ in range(3))
flash_attention(Q, K, V, block_size=2, explain=True)   # exact vs standard attention (error ~1e-16)
```

## Watch it flow — the circuit (v0.7)

Type an expression and see it as a **computation-graph circuit**: every node
shows its forward **data** and backward **gradient**, like current through wires
(Karpathy's `draw_dot` meets Anthropic's circuits).

```bash
optimumai circuit "(a*b + c) * f" --vars "a=2,b=-3,c=10,f=-2"          # terminal
optimumai circuit "(a*b + c) * f" --vars "a=2,b=-3,c=10,f=-2" --fmt html --out circuit.html
optimumai circuit "a*b + c" --fmt dot                                   # Graphviz DOT
```

```python
from optimumai.circuit import build_from_expression, to_html, to_terminal

value, graph = build_from_expression("(a*b + c) * f", {"a": 2, "b": -3, "c": 10, "f": -2})
to_terminal(graph)             # value=-8; leaf grads a=6, b=-4, c=-2, f=4
to_html(graph, "circuit.html") # interactive, self-contained (vis-network)
```

The dashboard has a live **Circuit playground** too: `optimumai dashboard`.

## See the graphs (v0.6)

Beyond terminal traces, render real **matplotlib** figures (needs `optimumai[viz]`):

```bash
optimumai plot activation --name gelu --out gelu.png     # activation + its derivative
optimumai plot softmax --out temps.png                   # distribution vs temperature
optimumai plot attention --text "the cat sat" --out att.png
optimumai plot embeddings --out emb.png                  # PCA scatter
optimumai landscape rosenbrock --out land.png            # 3D loss surface + descent path
```

```python
from optimumai.visualization import plot_loss_landscape, plot_attention

plot_loss_landscape("rosenbrock", kind="both", out="landscape.png")  # 2D contour + 3D surface
plot_attention(text="attention is all you need", out="heatmap.png")
```

## Give it your own input (v0.5)

Stop watching demos — feed OptimumAI *your* numbers, text, and equations and
watch them flow:

```bash
optimumai repl                              # interactive session (arrow keys w/ [repl] extra)
optimumai trace-text "why is the sky blue"  # your words → tokens → transformer → next token
optimumai algebra dot -i                    # prompts you for the vectors
optimumai compare relu gelu --input "[-2,-1,0,1,2]"
optimumai sweep softmax --values "[0.25,0.5,1,2]"    # watch temperature sharpen the distribution
optimumai diff "x**3 + 2*x" --at 3          # symbolic derivative of YOUR equation (→ 29)
```

```python
from optimumai import TextPipeline, compare, differentiate

TextPipeline("attention is all you need", layers=2).forward(explain=True)
compare("relu", "gelu", input=[-2, -1, 0, 1, 2], explain=True)
differentiate("sin(x) * x**2", at=1.0, explain=True)   # needs optimumai[symbolic]
```

## Foundations of the stack (v0.4)

Beyond the math, OptimumAI now explains the *systems* modern AI runs on — the
same `explain=True` treatment for the frameworks and hardware:

```bash
optimumai learn tensors           # rank, shape, broadcasting
optimumai learn integration       # trapezoid & Monte Carlo (expectations are integrals)
optimumai learn pytorch           # what torch.autograd does under the hood
optimumai learn jax               # grad / jit / vmap / pytrees
optimumai learn cuda_matmul       # naive vs tiled matmul + memory coalescing
optimumai kvcache --seq-len 8192  # why context length eats VRAM (MHA vs GQA vs MQA)
optimumai vram --params 70        # VRAM budget to train a 70B model
```

```python
from optimumai import kv_cache_size, vram_estimate, integrate

kv_cache_size(n_layers=32, n_heads=32, head_dim=128, seq_len=4096)  # bytes
vram_estimate(params_billions=7, training=True)                      # GB
integrate(lambda x: x**2, 0, 1, method="monte_carlo")                # ≈ 1/3
```

## Explain levels

The same math, revealed for four audiences (`--level` on the CLI, `level=` in
Python):

| Level          | Adds                                              |
| -------------- | ------------------------------------------------- |
| `beginner`     | The steps and plain-English "why"                 |
| `intermediate` | Per-step detail notes (default)                   |
| `engineer`     | Intermediate values + complexity                  |
| `researcher`   | Everything                                        |

## What's inside

```
optimumai/
├── core/            # Tracer, Step/Trace model, ExplainLevel, BaseOp
├── algebra/         # Vector (dot, norm, cosine), Matrix (matmul)
├── probability/     # softmax (with temperature + stability)
├── autograd/        # Value — a micrograd-style scalar autograd engine  ✨v0.2
├── calculus/        # derivatives, gradients, the chain rule            ✨v0.2
├── optimization/    # SGD, Adam, the training loop                      ✨v0.2
├── neural_networks/ # Neuron/Layer/MLP + full backprop                  ✨v0.2
├── transformers/    # Attention, MultiHeadAttention (causal), PE, Block ✨v0.2
├── world_models/    # JEPA — LeCun's predict-in-latent-space energy     ✨v0.2
├── interpretability/# superposition — Anthropic's polysemantic neurons  ✨v0.2
├── embeddings/      # token → dense vector lookup, nearest neighbours    ✨v0.3
├── rag/             # retrieval-augmented generation pipeline trace      ✨v0.3
├── diffusion/       # forward noising schedule + reverse denoising       ✨v0.3
├── curriculum/      # the Course: a first-principles AI learning path    ✨v0.3
├── progress/        # ProgressTracker — how far you've come              ✨v0.3
├── tutor/           # optional LLM tutor (optimumai[llm])                ✨v0.3
├── dashboard/       # Streamlit progress dashboard (optimumai[dashboard])✨v0.3
├── foundations/     # tensors, integration, PyTorch/JAX, GPU/CUDA, KV, VRAM ✨v0.4
├── interactive/     # prompts + REPL — give it your own input             ✨v0.5
├── symbolic/        # differentiate your own equations (SymPy)            ✨v0.5
├── analysis/        # compare ops & sweep parameters                      ✨v0.5
├── visualization/   # Rich terminal renderer + matplotlib graphs          ✨v0.6
├── circuit/         # computation-graph "circuit" — HTML / Graphviz / TUI ✨v0.7
├── frontier/        # FlashAttention, quantization, LoRA, DPO/RLHF         ✨v0.8
└── cli/             # the `optimumai` command
```

## Roadmap

**v0.1** — the spine: algebra → probability → attention, plus the tracer, CLI,
and terminal visualization.

**v0.2** ✅ — the fundamentals: a micrograd-style autograd engine, calculus,
SGD/Adam, neural networks with real backprop, multi-head attention + causal mask
+ positional encoding + a full transformer block, LeCun's JEPA world model, and
Anthropic-style superposition. See [PHILOSOPHY.md](PHILOSOPHY.md).

**v0.3** ✅ — the learning path: a structured `Course` with progress tracking, a
Streamlit dashboard, embeddings, a RAG pipeline trace, diffusion schedules, and
an optional LLM tutor (`optimumai[llm]`).

**v0.4** ✅ — foundations of the stack: tensors & numerical integration, PyTorch
& JAX internals (autograd, `grad`/`jit`/`vmap`, pytrees), and the systems layer —
the CUDA execution & memory model, tiled matmul kernels + coalescing, the KV
cache (MHA/GQA/MQA), and a VRAM budget calculator. The course now spans **28
lessons across 9 tracks**.

**v0.5** ✅ — interactive input: a REPL, `trace-text` (your words → transformer),
`--interactive` prompts, op `compare`, parameter `sweep`, and symbolic `diff` of
your own equations. The course now spans **31 lessons across 10 tracks**.

**v0.6** ✅ — visualization: matplotlib 2D/3D → PNG. Activation curves, softmax
vs temperature, attention heatmaps, embedding scatter, training curves, and a
**3D loss landscape with the gradient-descent trajectory** carved across it.

**v0.7** ✅ — the **circuit**: render any expression / `Value` graph as an
interactive computation graph (HTML + Graphviz + terminal), with data and
gradients lighting up the wires, plus a live Circuit playground in the dashboard.

**v0.8** ✅ — frontier concepts: FlashAttention (IO-aware tiling + online softmax,
exact), quantization (int8/int4), LoRA (parameter-efficient fine-tuning), and DPO
(preference alignment). The course now spans **35 lessons across 11 tracks**.

**v0.9** (next) — the learning experience, grounded in cognitive science: a
**quiz / active-recall** mode (the testing effect), **spaced-repetition review**
(SM-2), guided onboarding, and lesson search.

## Development

```bash
git clone https://github.com/muhammadyahiya/optimumai
cd optimumai
uv venv && uv pip install -e ".[dev]"
pytest
```

## License

MIT © 2026 Muhammad Yahiya
