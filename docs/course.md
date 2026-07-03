# The learning path

OptimumAI is a first-principles AI course — 39 lessons across 12 tracks, each a
runnable, explained trace.

```bash
optimumai course              # the full path, grouped by track, with progress
optimumai learn attention     # run a lesson (auto-marks it complete)
optimumai progress            # a progress bar + what's next
optimumai search embedding    # find lessons by keyword
optimumai dashboard           # a visual Streamlit dashboard  (optimumai[dashboard])
```

The tracks build on each other:

1. Linear Algebra · 2. Calculus & Autograd · 3. Optimization & Neural Nets ·
4. Probability & Transformers · 5. Applied AI (embeddings, RAG, diffusion) ·
6. World Models & Interpretability · 7. Math Foundations · 8. Framework Internals
(PyTorch/JAX) · 9. Systems & Hardware (CUDA, KV cache, VRAM) · 10. Interactive
Playground · 11. Frontier (FlashAttention, quantization, LoRA, DPO) · 12. GPU
Kernels.

## Retention: quiz + spaced repetition

```bash
optimumai quiz backprop        # active recall — you answer, it grades + explains
optimumai review               # spaced repetition (SM-2): whatever's due
optimumai exercise backprop    # compute-the-answer exercises
```

```python
from optimumai import COURSE, ProgressTracker
for lesson in COURSE:
    print(lesson.track, lesson.id, "—", lesson.summary)
```
