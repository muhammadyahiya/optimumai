# Visualization & circuits

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

```python
from optimumai.flows import transformer_flow, attention_flow, tfidf_flow, word2vec_flow

transformer_flow(out="transformer.html")   # transformer forward pass, animated
attention_flow(out="attention.html")       # scaled dot-product attention circuit
tfidf_flow(out="tfidf.html")               # TF-IDF circuit
word2vec_flow(out="word2vec.html")         # word2vec skip-gram
```

Self-contained, offline HTML. Open in any browser.

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
