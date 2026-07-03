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

## Quickstart — CLI

```bash
optimumai algebra dot "[1,2,3]" "[4,5,6]"
optimumai algebra matmul "[[1,2],[3,4]]" "[[5,6],[7,8]]"
optimumai softmax "[2,1,0.1]" --temperature 0.5
optimumai attention --demo --level engineer
optimumai learn                       # list every topic
optimumai learn attention --level researcher
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
├── transformers/    # scaled dot-product Attention
├── visualization/   # Rich terminal renderer
└── cli/             # the `optimumai` command
```

## Roadmap

`v0.1` ships the spine — algebra → probability → attention — plus the tracer,
CLI, and terminal visualization. Next up:

- **Calculus & optimization** — derivatives, gradients, SGD/Adam convergence
- **Neural networks** — dense layers, activations, full backprop trace
- **Multi-head attention, positional encoding, a full transformer block**
- **Embeddings, RAG pipeline traces, diffusion schedules**
- **LLM tutor** — `Tutor().ask("Why is LayerNorm after attention?")`
- **Streamlit explorer** for visual, interactive pipelines

## Development

```bash
git clone https://github.com/muhammadyahiya/optimumai
cd optimumai
uv venv && uv pip install -e ".[dev]"
pytest
```

## License

MIT © 2026 Muhammad Yahiya
