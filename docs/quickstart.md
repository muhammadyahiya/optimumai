# Quickstart

Get up and running in under five minutes.

## Installation

```bash
pip install optimumai                  # core — numpy + rich + click, no GPU needed
pip install "optimumai[viz]"           # add matplotlib plots and GIF export
pip install "optimumai[llm]"           # add LLM tutor and token generation clients
pip install "optimumai[notebooks]"     # add JupyterLab launcher
pip install "optimumai[dashboard]"     # add Streamlit progress dashboard
pip install "optimumai[all]"           # everything at once
```

Verify the install:

```bash
python -c "import optimumai; print(optimumai.__version__)"
optimumai --version
```

---

## First steps in Python

Every operation returns the numeric result **and** can explain itself. Add
`explain=True` to see the step-by-step trace:

```python
from optimumai import Vector, Matrix, softmax, Attention

# Linear algebra
Vector([1, 2, 3]).dot(Vector([4, 5, 6]), explain=True)          # 32
Vector([1, 2, 3]).cosine_similarity(Vector([2, 4, 6]), explain=True)  # 1.0
Matrix([[1, 2], [3, 4]]).matmul(Matrix([[5, 6], [7, 8]]), explain=True)

# Probability
softmax([2.0, 1.0, 0.1], temperature=0.5, explain=True)

# Transformers
Attention.demo().render("engineer")
```

Prefer structured data over printed output? Use the `*_trace` variants:

```python
from optimumai import Vector

trace = Vector([1, 2, 3]).dot_trace(Vector([4, 5, 6]))
trace.result    # 32.0
trace.steps     # [Step(label=..., computation=...), ...]
trace.why_ai    # ['Similarity between two embedding vectors', ...]
trace.render("beginner")   # print at any level later
```

---

## First steps in the CLI

```bash
optimumai start                        # guided 30-second tour (start here)
optimumai course                       # full learning path with progress bars
optimumai learn dot                    # run any lesson (auto-marks complete)
optimumai learn attention --level researcher

optimumai algebra dot "[1,2,3]" "[4,5,6]"
optimumai softmax "[2,1,0.1]" --temperature 0.5
optimumai attention --demo
optimumai backprop
optimumai train --steps 150 --lr 0.05

optimumai quiz softmax                 # active-recall quiz
optimumai review                       # spaced-repetition review (SM-2)
optimumai exercise backprop            # compute-the-answer exercise

optimumai kernel matmul                # GPU kernel on the pure-Python simulator
optimumai generate "The key insight behind attention is"
optimumai playground softmax           # interactive drag-the-inputs circuit
optimumai dashboard                    # Streamlit progress dashboard
```

---

## Explain levels

The same math at four levels of detail. Pass `--level` on the CLI or `level=`
in Python:

| Level | What you see |
|---|---|
| `beginner` | steps + plain-English "why AI uses this" |
| `intermediate` | per-step detail notes (CLI default) |
| `engineer` | intermediate values + algorithmic complexity |
| `researcher` | everything: formulas, proofs, source references |

```bash
optimumai algebra dot "[1,2,3]" "[4,5,6]" --level beginner
optimumai algebra dot "[1,2,3]" "[4,5,6]" --level researcher
```

```python
from optimumai import softmax
softmax([2.0, 1.0, 0.1], level="researcher", explain=True)
```

---

## Next steps

- **Full feature tour** → [Features](features.md)
- **76-lesson learning path** → [Course](course.md)
- **All CLI commands** → [CLI reference](cli.md)
- **GPU kernels from scratch** → [GPU kernels](gpu-kernels.md)
- **Visualization & circuits** → [Visualization](visualization.md)
- **Classical ML, AI search, RL** → [Classical AI, ML & RL](classical-ai.md)
- **Token generation** → [Token generation](generation.md)
