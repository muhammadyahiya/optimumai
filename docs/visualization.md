# Visualization & circuits

## Any concept → PNG or GIF

```bash
optimumai visualize                       # list every concept + its formats
optimumai visualize attention --fmt png --out attn.png
optimumai visualize gradient_descent --fmt gif --out gd.gif
```

```python
from optimumai.visualization.concepts import render_concept, list_concepts
list_concepts()
render_concept("softmax", fmt="gif", out="softmax.gif")
```

Static plots and 3-D loss landscapes also live in `optimumai.visualization`
(`plot_activation`, `plot_attention`, `plot_loss_landscape`, ...).

## Editable equation ↔ graph

```bash
optimumai editor "a*x^2 + b*x + c"   # → editable_plot.html
```

Open the HTML: edit the equation and the curve replots; drag a parameter slider
and both the curve and the equation update.

## Interactive circuits — drag the inputs, watch the math

```bash
optimumai playground softmax    # drag the logits → the distribution recomputes live
optimumai playground backprop   # drag a/b/c/f → forward values + gradients update
```

## The computation graph as a circuit

```bash
optimumai circuit "(a*b + c) * f" --vars "a=2,b=-3,c=10,f=-2" --fmt html
```

Renders the graph with each node's forward **data** (blue) and backward
**gradient** (orange), as HTML (vis-network), Graphviz DOT, or a terminal table.
