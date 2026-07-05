# Interactive & Explained

Four interactive layers built on top of the core library:

1. **RAG flow diagram** — a D3 + KaTeX pipeline explainer that proves the
   `FlowTrace` schema generalises across concepts
2. **Prompt engineering** — offline, deterministic traces for every standard
   prompting pattern
3. **Augmented RNNs** — the distill.pub lineage from soft attention to NTMs
   to adaptive computation
4. **Interactive playgrounds** — self-contained, offline HTML widgets

---

## RAG flow diagram — `optimumai.rag.flow`

A Transformer-Explainer-style progressive pipeline diagram for
Retrieval-Augmented Generation. Built on the concept-agnostic `FlowTrace`
schema (`optimumai.core.flow_trace`) — the renderer never knows what "RAG"
means; it only reads nodes, edges, and steps.

**Pipeline stages visualised:**

| Step | Stage | What you see |
|---|---|---|
| 1 | Chunking | Document splits into 3 chunks; edges activate |
| 2 | Embed chunks | Each chunk mapped to a vector; formula rendered via KaTeX |
| 3 | Index | Vectors added to the vector store |
| 4 | Embed query | Query vector computed; formula $\vec{q} = \text{Embed}(\text{query})$ |
| 5 | Retrieve | Cosine scores computed (real values from `RAGPipeline`); top-k selected |
| 6 | Rerank | Cross-encoder reranking stage |
| 7 | Assemble | Top chunks concatenated into context |
| 8 | Generate | LLM produces answer conditioned on context |

```bash
optimumai flow rag                                     # Eiffel Tower demo query
optimumai flow rag --query "What year was it built?"  # custom query, real scores
optimumai flow rag --out rag.html
```

```python
from optimumai.rag.flow import rag_flow
from optimumai.rag.trace import build_rag_trace
from optimumai.rag.explainer import render_flow_trace_html, RAG_LAYOUT
from optimumai.core.flow_trace import FlowTrace

# One-liner: build trace + render HTML
path = rag_flow(out="rag_explainer.html")

# Or inspect the trace programmatically
trace = build_rag_trace(query="How tall is the Eiffel Tower?", k=2)
assert isinstance(trace, FlowTrace)
problems = trace.validate()   # [] — referential integrity checks pass

# Real cosine scores from RAGPipeline._embed()
print(trace.steps[4].metrics)
# {'chunk_0_score': 0.6734, 'chunk_1_score': 0.3087, 'chunk_2_score': 0.6847}

# Render any FlowTrace — the renderer is concept-agnostic
html = render_flow_trace_html(trace, RAG_LAYOUT, out="rag.html")
```

**Why the abstraction holds:** The JavaScript in the generated HTML never
imports `optimumai.rag`. It reads `TRACE.nodes`, `TRACE.edges`,
`TRACE.steps[i].formula` (→ KaTeX), `TRACE.steps[i].metrics` (→ table).
Produce a `FlowTrace` from `value_iteration` or `quantization` and this same
renderer draws it, unmodified.

!!! note "Internet required for the RAG diagram"
    The RAG diagram loads **D3 v7** and **KaTeX** from CDN (the other flow
    diagrams — transformer, attention, tfidf, word2vec — are fully offline
    inline SVG). For offline use, download the two CDN files and replace the
    `<script>`/`<link>` tags in the generated HTML.

---

## Prompt engineering — `optimumai.prompting`

Each pattern builds the prompt step by step and explains *why* it works and
how it fails. No API key needed — these are deterministic offline traces.

| Pattern | Command | What it teaches |
|---|---|---|
| `zero-shot` | `optimumai prompt zero-shot` | Role + instruction + task, no examples |
| `few-shot` | `optimumai prompt few-shot` | In-context learning from K exemplars |
| `chain-of-thought` | `optimumai prompt chain-of-thought` | Elicit reasoning before the final answer |
| `react` | `optimumai prompt react` | Thought / Action / Observation with a tool |
| `self-consistency` | `optimumai prompt self-consistency` | Sample N chains, majority-vote the answer |
| `structured-output` | `optimumai prompt structured-output` | Constrain to a validated JSON schema |

```python
from optimumai.prompting import (
    zero_shot, few_shot, chain_of_thought,
    react, self_consistency, structured_output
)

chain_of_thought(
    "If a train travels 60 miles in 2 hours, what is its speed?",
    explain=True
)

self_consistency(
    "What is 2 + 2?",
    sampled_answers=["4", "4", "5"],
    explain=True
)   # -> "4"
```

```bash
optimumai prompt zero-shot
optimumai prompt few-shot
optimumai prompt chain-of-thought
optimumai prompt react
optimumai prompt self-consistency
optimumai prompt structured-output
```

::: optimumai.prompting

---

## Augmented RNNs — `optimumai.augmented_rnns`

The pre-transformer ideas that made attention mainstream, traced from
[distill.pub's 2016 post](https://distill.pub/2016/augmented-rnns/).

### Attention as differentiable memory read

Content-based soft attention: score each memory slot by cosine similarity to
a query, softmax the scores, blend the slots — the direct ancestor of
transformer attention.

```python
from optimumai.augmented_rnns import attention_read
import numpy as np

memory = np.array([[1., 0., -1.], [0., 1., 0.], [-1., 0., 1.], [.5, .5, .5]])
query = np.array([1., 0., -1.])
attention_read(query, memory, explain=True)
```

```bash
optimumai augrnn attention
```

### Neural Turing Machine — NTM

A full NTM read/write head: cosine-addressed soft attention for reading,
and an erase/add mechanism for writing to memory. The first neural system
with an explicit external memory that could be addressed by content.

```python
from optimumai.augmented_rnns import ntm_read, ntm_write
import numpy as np

memory = np.random.default_rng(0).normal(size=(8, 4))
key = np.array([0.5, -1., 0., 0.3])
ntm_read(key, memory, beta=3.0, explain=True)
```

```bash
optimumai augrnn ntm
```

### Adaptive Computation Time — ACT

The model itself decides when to stop computing: it emits a halting
probability at each step, and stops when the cumulative probability exceeds
`1 − ε`. It pays a "ponder cost" for extra steps. This is the idea behind
variable-depth computation in PonderNet and Universal Transformers.

```python
from optimumai.augmented_rnns import adaptive_computation_time
import numpy as np

halting_probs = np.array([0.5, 1.2, 2.0, -0.3, 3.0])
adaptive_computation_time(halting_probs, eps=0.01, explain=True)
```

```bash
optimumai augrnn act
```

::: optimumai.augmented_rnns

---

## Interactive playgrounds — `optimumai.visualization.playgrounds`

Self-contained HTML files with inline vanilla JS — **no server, no build, works
offline.** Generate one with `optimumai playground <name>`:

### Attention playground

Inspired by [Transformer Explainer](https://poloclub.github.io/transformer-explainer/):

- Hover a query token → the attention heatmap updates live
- Drag the temperature slider → scores re-softmax instantly

```bash
optimumai playground attention
```

### Softmax playground

- Drag any logit slider left/right → the probability bars recompute instantly
- Shows how temperature changes the sharpness/flatness of the distribution

```bash
optimumai playground softmax
```

### Backprop playground

Drag any input (a, b, c, f) in the expression `(a·b + c)·f`:

- Forward values update (blue)
- Gradients update (orange)

```bash
optimumai playground backprop
```

### k-means playground

- Click anywhere on the canvas to add a data point
- Watch Lloyd's algorithm re-run: assign to nearest centroid → recompute centroids → repeat
- See which points swap cluster assignments with each step

```bash
optimumai playground kmeans
```

### A* playground

- Click to draw/erase walls on a grid
- A* expands the frontier toward the goal in real time
- Open/closed sets highlighted; path shown in green

```bash
optimumai playground astar
```

::: optimumai.visualization.playgrounds

---

## Concept gallery — `optimumai.visualization.gallery`

Per-concept matplotlib plots and animated GIFs for all 21+ registered concepts.
Also reachable via the `optimumai visualize <concept> --fmt png|gif` registry.

```bash
optimumai visualize                              # list every concept + its formats
optimumai visualize attention --fmt png --out attn.png
optimumai visualize gradient_descent --fmt gif --out gd.gif
```

```python
from optimumai.visualization.concepts import render_concept, list_concepts

list_concepts()
render_concept("attention", fmt="gif", out="attn.gif")
```

Needs `pip install "optimumai[viz]"`.

::: optimumai.visualization.gallery
