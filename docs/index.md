# OptimumAI

**Unlock the math behind AI.** Every operation — from a dot product to a
transformer block — runs with `explain=True` to produce a step-by-step
computation trace, a terminal visualization, and the intuition for *why* AI uses
it. Then go hands-on: write GPU kernels, generate tokens, and build circuits.

```bash
pip install optimumai
optimumai start          # a 30-second guided tour
optimumai course         # the full learning path (39 lessons, 12 tracks)
```

```python
from optimumai import Vector
Vector([1, 2, 3]).dot(Vector([4, 5, 6]), explain=True)   # 32.0, spelled out
```

## What's inside

- **Learn** — a 39-lesson course from linear algebra to FlashAttention, LoRA, and
  DPO, each a runnable, explained trace. Track progress, quiz yourself (active
  recall), and review on a spaced-repetition schedule.
- **Build** — write GPU kernels from scratch on a pure-Python simulator (checked
  against NumPy), then run them on real backends if you have a GPU.
- **Generate** — real token generation via local Ollama, Hugging Face, or
  Anthropic (`optimumai generate "..."`).
- **Visualize** — PNGs and GIFs for any concept, an editable equation↔graph, and
  interactive drag-the-inputs circuits.

## Install

```bash
pip install optimumai                 # core (numpy, rich, click) — light & fast
pip install "optimumai[viz]"          # matplotlib plots + GIFs
pip install "optimumai[llm]"          # the LLM tutor / generation clients
pip install "optimumai[notebooks]"    # Jupyter, for `optimumai notebooks`
pip install "optimumai[all]"          # everything
```

!!! tip "Just published a new version and pip can't find it?"
    PyPI's index takes a couple of minutes to propagate after a release. If
    `pip install optimumai==<new>` fails immediately after a release, wait a
    moment and retry, or add `--no-cache-dir`.
