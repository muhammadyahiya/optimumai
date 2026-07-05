# Learning path

OptimumAI is a first-principles AI course — **76 lessons across 20 tracks**,
each a runnable, explained trace. Every lesson auto-marks itself complete when
you run it. Progress is stored in `~/.optimumai/progress.json` and shared
between the CLI and the Streamlit dashboard.

```bash
optimumai start              # 30-second guided tour — start here
optimumai course             # the full path, grouped by track, with ✓/○ progress
optimumai learn attention    # run any lesson (auto-marks complete)
optimumai progress           # progress bar + what's next
optimumai search embedding   # find lessons by keyword (id/title/summary/track)
optimumai dashboard          # visual Streamlit dashboard (needs [dashboard])
```

---

## The 20 tracks

The tracks build on each other in this order:

| # | Track | Topics covered |
|---|---|---|
| 1 | **Linear Algebra** | Dot product, cosine similarity, matrix multiply, norms |
| 2 | **Calculus & Autograd** | Derivatives, chain rule, gradient, `Value` scalar autograd |
| 3 | **Optimization** | SGD, momentum, Adam, minimize |
| 4 | **Neural Networks** | Neurons, MLP, backprop, training loop |
| 5 | **Probability** | Softmax, temperature, sampling, distributions |
| 6 | **Transformers** | Scaled attention, multi-head attention, positional encoding |
| 7 | **Transformer Block** | Pre-norm, FFN, causal mask, `TransformerBlock` |
| 8 | **Applied AI** | Embeddings, RAG, diffusion, word2vec |
| 9 | **World Models** | JEPA, energy-based models |
| 10 | **Interpretability** | Superposition, polysemanticity, circuits |
| 11 | **Math Foundations** | Integration, tensors, expectations |
| 12 | **Framework Internals** | PyTorch, JAX programming models |
| 13 | **Systems & Hardware** | CUDA execution model, tiled matmul kernels, KV cache, VRAM |
| 14 | **Interactive Playground** | REPL, TextPipeline, op comparisons, sweeps |
| 15 | **Frontier** | FlashAttention, quantization, LoRA, DPO |
| 16 | **Classical ML** | Linear/logistic regression, k-means, KNN, decision trees, PCA |
| 17 | **Classical AI Search** | BFS, DFS, UCS, greedy, A*, minimax, alpha-beta |
| 18 | **Reinforcement Learning** | MDPs, value iteration, Q-learning, SARSA, REINFORCE, PPO |
| 19 | **NLP** | BPE tokenization, TF-IDF, n-gram LMs, edit distance, word2vec |
| 20 | **GPU Kernels** | Writing, grading, and running CUDA-style kernels |

---

## Running lessons

```bash
optimumai learn dot                    # linear algebra: the dot product
optimumai learn chain_rule             # calculus: the chain rule
optimumai learn softmax                # probability: softmax + temperature
optimumai learn attention              # transformers: scaled dot-product attention
optimumai learn positional             # transformers: sinusoidal positional encoding
optimumai learn embeddings             # applied AI: embedding lookup + nearest neighbors
optimumai learn rag                    # applied AI: retrieval-augmented generation
optimumai learn diffusion              # applied AI: forward diffusion process
optimumai learn flash_attention        # frontier: FlashAttention IO-aware tiling
optimumai learn lora                   # frontier: low-rank adaptation
optimumai learn dpo                    # frontier: direct preference optimization
optimumai learn tensors                # foundations: tensors and integration
optimumai learn cuda_matmul            # systems: tiled CUDA matmul
optimumai learn pytorch                # framework internals: PyTorch model
optimumai learn jax                    # framework internals: JAX functional style
optimumai learn transformer --level researcher   # with full depth
optimumai learn attention --no-track            # run without recording completion
```

---

## Progress tracking

```bash
optimumai progress                     # bar chart + percentage done + what's next
optimumai progress --reset             # clear all recorded progress
```

Progress is stored at `~/.optimumai/progress.json`.
Override the path with the `OPTIMUMAI_PROGRESS_PATH` environment variable.

```python
from optimumai import COURSE, ProgressTracker

tracker = ProgressTracker()
tracker.mark_complete("attention")
print(tracker.summary())               # {'completed': 1, 'total': 76, 'pct': 1.3}

for lesson in COURSE:
    print(lesson.track, lesson.id, "—", lesson.summary)
```

---

## Active recall — quiz

Studying → passive. Testing yourself → active recall, roughly 2× better
retention. Twenty quizzes, 57 questions total:

```bash
optimumai quiz                         # list every quiz topic
optimumai quiz softmax                 # answer, get graded + explained
optimumai quiz backprop
optimumai quiz attention
optimumai quiz transformer
```

```python
from optimumai import Quiz

q = Quiz("softmax")
q.ask()                  # prints a question, reads your answer, grades it
```

---

## Spaced repetition — review

Quiz scores feed an SM-2 scheduler: after a review graded 0–5, the next
interval grows by the current ease factor. Easy concepts get reviewed less
often; hard ones come back sooner.

```bash
optimumai review                       # whatever the SM-2 scheduler says is due
```

---

## Compute-the-answer exercises

Fill-in-the-blank numerical exercises, tolerance-graded (your answer just has
to be close enough):

```bash
optimumai exercise                     # list exercise topics
optimumai exercise backprop            # compute the gradient, enter a number
optimumai exercise attention
optimumai exercise softmax
```

---

## Course search

```bash
optimumai search attention             # find lessons by keyword (id/title/summary/track)
optimumai search "positional encoding"
optimumai search lora
```

---

## Streamlit dashboard

A visual progress map with per-track breakdowns:

```bash
optimumai dashboard                    # launch on localhost:8501
optimumai dashboard --port 8888
```

Needs `pip install "optimumai[dashboard]"`.
To share publicly, deploy to Streamlit Community Cloud or Hugging Face Spaces —
see [Deploy the dashboard](deploy.md).

---

## LLM tutor (optional)

```bash
optimumai ask "why LayerNorm after attention?"
optimumai ask "explain the difference between RoPE and sinusoidal embeddings"
```

Needs `pip install "optimumai[llm]"` and `ANTHROPIC_API_KEY` (or another
supported provider). Degrades gracefully without it: prints a friendly message
explaining what to install.
