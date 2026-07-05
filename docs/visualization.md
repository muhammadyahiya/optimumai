# Visualization

OptimumAI has three visualization layers:

1. **Concept registry** — any-concept PNG or GIF, via `optimumai visualize`
2. **Matplotlib plots & landscapes** — static figures and 3-D surfaces
3. **Circuits & interactive HTML** — editable equations, drag-the-inputs
   playgrounds, and computation graphs with data/gradients flowing the wires

All three are in `optimumai.visualization` (needs `pip install "optimumai[viz]"`
for plots and GIFs; the interactive HTML is self-contained and works without it).

---

## Concept registry — any concept, PNG or GIF

```bash
optimumai visualize                              # list every concept + its formats
optimumai visualize attention --fmt png --out attn.png
optimumai visualize attention --fmt gif --out attn.gif
optimumai visualize gradient_descent --fmt gif --out gd.gif
optimumai visualize kmeans --fmt gif --out kmeans.gif
optimumai visualize softmax --fmt png --out softmax.png
optimumai visualize decision_boundary --fmt png --out db.png
optimumai visualize conv2d --fmt png --out conv.png
optimumai visualize calibration --fmt png --out cal.png
optimumai visualize ppo_clip --fmt png --out ppo.png
optimumai visualize value_iteration --fmt gif --out vi.gif
optimumai visualize astar --fmt gif --out astar.gif
```

```python
from optimumai.visualization.concepts import render_concept, list_concepts

list_concepts()                                  # 21+ registered concepts
render_concept("attention", fmt="gif", out="attn.gif")
render_concept("kmeans", fmt="png", out="km.png")
```

21+ concepts as of v1.2, including all the original static/animated concepts
plus the new classical-AI additions. Run `optimumai visualize` with no argument
to see the live, authoritative list.

---

## Matplotlib plots (needs `[viz]`)

### Activation curves

```bash
optimumai plot activation --name relu --out relu.png
optimumai plot activation --name gelu --out gelu.png
optimumai plot activation --name tanh --out tanh.png
```

### Softmax temperature curves

```bash
optimumai plot softmax --out temps.png
```

### Attention heatmap

```bash
optimumai plot attention --text "the cat sat on the mat" --out att.png
```

### Embedding scatter

```bash
optimumai plot embeddings --out emb.png
```

### Training loss curve

```bash
optimumai plot training --out curve.png
```

### 3-D loss landscapes

```bash
optimumai landscape rosenbrock --out land.png
optimumai landscape bowl --out bowl.png
optimumai landscape bowl --kind contour --out bowl.png
```

### Animated GIFs

```bash
optimumai animate descent --out descent.gif       # gradient descent on a 3-D surface
optimumai animate diffusion --out diffusion.gif   # forward diffusion noising
optimumai animate softmax --out softmax.gif       # softmax over temperatures
```

### Python API

```python
from optimumai.visualization import (
    plot_activation, plot_attention, plot_loss_landscape,
    plot_embeddings, plot_training_curve,
)

plot_activation("gelu", out="gelu.png")
plot_attention(["the", "cat", "sat"], out="att.png")
plot_loss_landscape("rosenbrock", out="land.png")
```

---

## Editable equation ↔ graph (browser)

```bash
optimumai editor "a*x^2 + b*x + c"   # → editable_plot.html
```

Open the HTML file in any browser:

- Edit the equation → the curve replots live
- Drag a parameter slider → the curve and equation both update

No server, no dependencies — all vanilla JS, works offline.

---

## Interactive circuits (HTML playgrounds)

Self-contained HTML files with inline vanilla JS — **no server, no build, works
offline.** Generate any with `optimumai playground <name>`:

```bash
optimumai playground attention    # Transformer-Explainer-style attention heatmap
                                   # hover a query token, drag a temperature slider
optimumai playground softmax      # drag logits → distribution recomputes live
optimumai playground backprop     # drag a/b/c/f → forward values + gradients update
optimumai playground kmeans       # click to add points, Lloyd's algorithm iterates
optimumai playground astar        # draw walls on a grid, A* expands the frontier
```

---

## Computation graph as a circuit

```bash
optimumai circuit "(a*b + c) * f" --vars "a=2,b=-3,c=10,f=-2" --fmt html --out circuit.html
optimumai circuit "(a*b + c) * f" --vars "a=2,b=-3,c=10,f=-2" --fmt dot
optimumai circuit "(a*b + c) * f" --vars "a=2,b=-3,c=10,f=-2" --fmt terminal
```

Renders the computation graph with each node's:

- **Blue** — forward data value
- **Orange** — backward gradient

Available formats: `html` (vis-network), `dot` (Graphviz), `terminal` (rich table).

```python
from optimumai.circuit import build_from_expression, to_terminal, to_html

value, graph = build_from_expression("(a*b + c) * f", {"a": 2, "b": -3, "c": 10, "f": -2})
to_terminal(graph)   # value=-8; leaf grads a=6, b=-4, c=-2, f=4
html = to_html(graph, out="circuit.html")
```

!!! note "Expression security"
    `build_from_expression` walks the parsed AST against an explicit allow-list
    (arithmetic operators only — no function calls, attribute access, or
    subscripts) before evaluating. `__import__(...)`, `open(...)`, and similar
    are rejected with `ValueError`.

---

## distill.pub-style circuit flow diagrams (v1.3+)

Concept-agnostic D3 + KaTeX pipeline diagrams. Each concept module emits a
`FlowTrace` (nodes, edges, steps); the renderer reads only that — it never
knows what concept it's drawing.

```python
from optimumai.flows import transformer_flow, attention_flow, tfidf_flow, word2vec_flow
from optimumai.rag.flow import rag_flow

transformer_flow(out="transformer.html")   # transformer forward pass
attention_flow(out="attention.html")       # scaled dot-product attention
tfidf_flow(out="tfidf.html")               # TF-IDF pipeline
word2vec_flow(out="word2vec.html")         # word2vec skip-gram
rag_flow(out="rag_explainer.html")         # RAG: chunk → embed → retrieve → rerank → generate
```

```bash
optimumai flow transformer
optimumai flow attention
optimumai flow tfidf
optimumai flow word2vec
optimumai flow rag
optimumai flow rag --query "What year was the Eiffel Tower built?"
optimumai flow rag --out my_rag.html
```

### RAG flow diagram — `optimumai.rag.flow`

The RAG diagram is built on the `FlowTrace` schema
(`optimumai.core.flow_trace`):

- **15 nodes** — source document, 3 chunks, 3 chunk embeddings, query,
  query embedding, vector index, retrieved set, reranked set, context, LLM,
  answer
- **16 edges** — each with `active_from_step` so they appear progressively
- **8 steps** — chunking → embed chunks → index → embed query → retrieve →
  rerank → assemble → generate
- **Real cosine scores** — computed by the actual `RAGPipeline._embed()`,
  not hardcoded toy values
- **KaTeX formulas** — `sim(q, vᵢ) = (q·vᵢ)/(‖q‖‖vᵢ‖)` rendered in the
  side panel at the retrieval step

```python
from optimumai.rag.trace import build_rag_trace
from optimumai.rag.explainer import render_flow_trace_html, RAG_LAYOUT

# Build the trace (validates on construction)
trace = build_rag_trace(query="How tall is the Eiffel Tower?", k=2)
print(f"{len(trace.nodes)} nodes, {len(trace.edges)} edges, {len(trace.steps)} steps")
# 15 nodes, 16 edges, 8 steps

# Cosine scores are real — computed by RAGPipeline._embed()
print(trace.steps[4].metrics)
# {'chunk_0_score': 0.6734, 'chunk_1_score': 0.3087, 'chunk_2_score': 0.6847}

# Render to HTML (requires D3 v7 + KaTeX CDN)
path = render_flow_trace_html(trace, RAG_LAYOUT, out="rag.html")
```

**The renderer is concept-agnostic.** The JS in the HTML file never reads
the word "RAG" — it only reads `nodes`, `edges`, `steps`, `formula`, and
`metrics`. Swap the trace JSON for any other `FlowTrace` (e.g. a
`value_iteration_trace`) and this same HTML renders it unmodified.

!!! note "Internet required"
    The RAG flow diagram loads D3 v7 and KaTeX from CDN. For fully offline
    use, download the JS/CSS locally and replace the two CDN `<script>`/
    `<link>` tags in the generated HTML.

---

## Plot Studio (v1.3+)

Feed numbers, get any chart type — plus the exact matplotlib + numpy code:

```python
from optimumai.visualization.plot_studio import PlotStudio

studio = PlotStudio()
studio.bar(data=[3, 7, 2, 8, 4], labels=["A","B","C","D","E"], title="My bar chart")
studio.scatter(x=[1,2,3,4,5], y=[2,4,3,5,4], title="My scatter")
studio.histogram(data=[1,1,2,2,2,3,4,4,5], bins=5)
# Returns the figure + prints the matplotlib + numpy code that produced it
```

Chart types: `bar`, `histogram`, `scatter`, `box`, `line`, `pie`, `violin`
