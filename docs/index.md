# OptimumAI

**Unlock the math behind AI.**

Every operation — from a dot product to a full transformer block — runs with
`explain=True` to produce a **step-by-step computation trace**, a **terminal
visualization**, and the intuition for *why AI actually uses it*. The same code
path runs fast in production *and* teaches you exactly what it's doing.

```bash
pip install optimumai
optimumai start          # 30-second guided tour — start here
optimumai course         # the full learning path (76 lessons across 20 tracks)
```

```python
from optimumai import Vector
Vector([1, 2, 3]).dot(Vector([4, 5, 6]), explain=True)
# ╭──────────────────── OptimumAI ────────────────────╮
# │ DOT  a · b = Σᵢ aᵢ·bᵢ                             │
# ╰────────────────────────────────────────────────────╯
#  1  Multiply component 0   1 × 4 = 4
#  2  Multiply component 1   2 × 5 = 10
#  3  Multiply component 2   3 × 6 = 18
#  4  Sum the products        4 + 10 + 18 = 32
# Result: 32  |  Why AI uses this: cosine similarity, attention score, matmul
```

---

## What's inside

<div class="grid cards" markdown>

-   :material-book-open-variant: **76-lesson Course**

    ---

    First-principles AI from linear algebra to FlashAttention, LoRA, and DPO.
    Every lesson is a runnable, explained trace.

    [:octicons-arrow-right-24: Learning path](course.md)

-   :material-code-braces: **explain=True for everything**

    ---

    Vectors, matrices, softmax, attention, backprop, optimizers, embeddings,
    diffusion, RAG — all traceable. Four detail levels: `beginner` → `researcher`.

    [:octicons-arrow-right-24: All features](features.md)

-   :material-chip: **GPU kernels from scratch**

    ---

    Write per-thread kernels on a pure-Python CUDA simulator. Grade your own
    work. Upgrade to real Numba/CuPy/Triton if a GPU is available.

    [:octicons-arrow-right-24: GPU kernels](gpu-kernels.md)

-   :material-robot: **Real token generation**

    ---

    Local Ollama, Hugging Face, Anthropic, or a built-in toy fallback — so a
    demo always produces tokens, even offline.

    [:octicons-arrow-right-24: Token generation](generation.md)

-   :material-chart-line: **Visualization**

    ---

    PNGs and GIFs for 21+ concepts, an editable equation↔graph, animated
    gradient descent, and interactive drag-the-inputs circuits.

    [:octicons-arrow-right-24: Visualization](visualization.md)

-   :material-layers: **The whole field**

    ---

    Classical ML, AI search, RL, NLP, computer vision, LLM evaluation — each an
    explainable trace, not just a black-box result.

    [:octicons-arrow-right-24: Classical AI, ML & RL](classical-ai.md)

</div>

---

## Install

=== "Core (recommended start)"

    ```bash
    pip install optimumai          # numpy + rich + click — fast, no GPU needed
    ```

=== "With plots"

    ```bash
    pip install "optimumai[viz]"   # adds matplotlib, Pillow — PNGs and GIFs
    ```

=== "With LLM"

    ```bash
    pip install "optimumai[llm]"   # LLM tutor + generation clients
    ```

=== "With notebooks"

    ```bash
    pip install "optimumai[notebooks]"   # JupyterLab launcher
    ```

=== "Everything"

    ```bash
    pip install "optimumai[all]"   # all extras: viz + llm + notebooks + dashboard
    ```

!!! tip "pip can't find the latest release?"
    PyPI's index takes a couple of minutes to propagate after a release.
    Wait a moment and retry, or add `--no-cache-dir`.

---

## 60-second tour

```python
from optimumai import (
    Vector, Matrix, softmax, Value,
    MLP, Attention, MultiHeadAttention, TransformerBlock,
    JEPA, generate
)

# ── Linear algebra ──────────────────────────────────────────────────────────
Vector([1, 2, 3]).cosine_similarity(Vector([2, 4, 6]), explain=True)   # 1.0
Matrix([[1, 2], [3, 4]]).matmul(Matrix([[5, 6], [7, 8]]), explain=True)

# ── Probability ──────────────────────────────────────────────────────────────
softmax([2.0, 1.0, 0.1], temperature=0.5, explain=True)

# ── Autograd ─────────────────────────────────────────────────────────────────
a, b = Value(2.0, label="a"), Value(-3.0, label="b")
L = (a * b).tanh()
L.backprop(explain=True)              # chain rule, step by step

# ── Neural net ───────────────────────────────────────────────────────────────
mlp = MLP(3, [4, 4, 1], activation="tanh", seed=0)
mlp([2.0, 3.0, -1.0])                # forward pass

# ── Transformers ─────────────────────────────────────────────────────────────
Attention.demo().render("engineer")
MultiHeadAttention.demo().render("engineer")
TransformerBlock.demo().render("researcher")

# ── World models ─────────────────────────────────────────────────────────────
JEPA.demo().render("engineer")        # LeCun's energy-based world model

# ── Token generation ─────────────────────────────────────────────────────────
print(generate("Attention is", max_tokens=32))
```

---

## CLI at a glance

```bash
optimumai start                        # guided tour
optimumai course                       # full 76-lesson path with progress
optimumai learn attention              # run any lesson
optimumai quiz softmax                 # active recall
optimumai review                       # spaced repetition (SM-2)

optimumai algebra dot "[1,2,3]" "[4,5,6]"
optimumai softmax "[2,1,0.1]" --temperature 0.5
optimumai attention --demo --level engineer
optimumai backprop
optimumai train --steps 150 --lr 0.05

optimumai kernel matmul                # GPU kernel on the simulator
optimumai visualize attention --fmt gif --out attn.gif
optimumai playground softmax           # drag-the-inputs circuit
optimumai generate "The key insight behind attention is"
optimumai dashboard                    # Streamlit progress dashboard
```

---

## The explain=True pattern

```python
# Two shapes — everything in the library is one of these:
Thing(...).op(args, explain=True, level="engineer")   # prints + returns result
op_trace(args).render("engineer")                      # build Trace, then render
```

Four explanation levels reveal progressively more detail:

| Level | What you see |
|---|---|
| `beginner` | steps + plain-English "why AI uses this" |
| `intermediate` | per-step detail notes (CLI default) |
| `engineer` | intermediate values + algorithmic complexity |
| `researcher` | everything: formulas, proofs, references |

`explain=False` (Python default) skips printing and returns the numeric result
on the same code path — the trace can never lie about what ran.

---

## Version history at a glance

| Version | What shipped |
|---|---|
| v0.2 | Autograd engine, calculus, optimizers, MLP, multi-head attention, JEPA, superposition |
| v0.3 | Structured course, progress tracking, Streamlit dashboard, embeddings, RAG, diffusion, LLM tutor |
| v0.4 | Tensors & integration, PyTorch/JAX models, CUDA memory model, KV cache, VRAM calculator |
| v0.5 | REPL, text-to-transformer pipeline, op comparisons, sweeps, symbolic differentiation |
| v0.6 | matplotlib figures: activation curves, attention heatmaps, embedding scatter, loss landscapes |
| v0.7 | Computation circuit: interactive HTML, Graphviz DOT, terminal (data + gradients on wires) |
| v0.8 | Frontier: FlashAttention, int8/int4 quantization, LoRA, DPO |
| v0.9 | Quiz / active-recall engine, SM-2 spaced repetition, course search |
| v1.0 | Real token generation, drag-the-inputs circuits, visualize-any-concept registry, notebooks |
| v1.1 | Classical ML, AI search, RL, NLP, computer vision, LLM evaluation |
| v1.2 | Prompt engineering patterns, augmented RNNs, interactive HTML playgrounds, concept gallery |
| v1.3 | Plot Studio, distill.pub-style circuit-flow HTML diagrams (transformer, attention, TF-IDF, word2vec) |
| v1.4 | Runnable tutorials: NumPy, matplotlib, PyTorch, LLM fine-tuning |
| v1.5 | OptiX typed TypeScript widget kit; hardened interactive layer |
