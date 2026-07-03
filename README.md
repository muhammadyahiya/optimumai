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
optimumai learn                       # list every topic (16 and counting)
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
├── visualization/   # Rich terminal renderer
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

**v0.4** (next) — foundations of the stack: tensors & numerical integration,
PyTorch & JAX internals (autograd, `grad`/`jit`/`vmap`), and GPU/systems math —
the CUDA execution & memory model, tiled matmul kernels, the KV cache, and a
VRAM budget calculator — all as a new "Systems & Foundations" course track.

## Development

```bash
git clone https://github.com/muhammadyahiya/optimumai
cd optimumai
uv venv && uv pip install -e ".[dev]"
pytest
```

## License

MIT © 2026 Muhammad Yahiya
