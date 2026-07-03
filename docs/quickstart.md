# Quickstart

Every operation takes `explain=True` and prints a step-by-step trace.

```python
from optimumai import Vector, Matrix, softmax, Attention

Vector([1, 2, 3]).dot(Vector([4, 5, 6]), explain=True)          # 32
Vector([1, 2, 3]).cosine_similarity(Vector([2, 4, 6]), explain=True)  # 1.0
Matrix([[1, 2], [3, 4]]).matmul(Matrix([[5, 6], [7, 8]]), explain=True)
softmax([2.0, 1.0, 0.1], temperature=0.5, explain=True)
Attention.demo().render("engineer")
```

Prefer the data over the print-out? Use the `*_trace` variants, which return a
`Trace` (`.result`, `.steps`, `.why_ai`).

## CLI

```bash
optimumai learn dot            # run any lesson
optimumai softmax "[2,1,0.1]"  # explain your own input
optimumai quiz softmax         # test yourself (active recall)
optimumai generate "Attention is"   # real token generation
optimumai kernel matmul        # a GPU kernel on the simulator
optimumai playground softmax   # an interactive drag-the-inputs circuit
```

## Explain levels

`beginner → intermediate → engineer → researcher` — the same math, revealing more
(formulas, complexity) as you go. Pass `--level` on the CLI or `level=` in Python.
