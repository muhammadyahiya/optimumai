# Scratchpad boards — declarative, checked, and offline-capable

The first scratchpad shipped two boards as two hand-written JavaScript
functions. That works at two and fails at twenty. This page describes the Tier 2
architecture and, more usefully, *why each piece is shaped the way it is* — each
one fixes a specific defect.

## A board is data

A concept declares a `BoardSpec`; one generic renderer per **kind** interprets
it. Four kinds cover the current set:

| kind | what it draws |
|---|---|
| `vectors` | draggable arrows from the origin; dot product and cosine |
| `function` | a curve from a Python-owned expression, with a tangent glider |
| `matrix` | draggable basis vectors warping a grid; determinant as area |
| `descent` | gradient descent stepping on a curve, with a learning-rate knob |

```python
"gradient_descent": ScratchpadConcept(
    concept_id="gradient_descent",
    lesson_id="descent",                     # the course lesson this *is*
    prerequisites=["tangent_line"],          # a real edge, not prose
    assumed_knowledge=["derivatives"],       # prose, kept separate
    board=BoardSpec(
        kind="descent",
        bounding_box=(-5, 10, 5, -4),
        expression="0.35*x**2 + 0.5",        # Python differentiates this
        params=(Param("lr", "learning rate", 0.4, 0.01, 3.2, 0.01),),
        readouts=(("final_x", "final x"), ("status", "behaviour")),
        snapshots=(Snapshot("Divergence", "Past lr = 1/0.35 ...", {"lr": 3.1}),),
    ),
    ...
)
```

Adding a concept is a dict entry. A test asserts **no concept id appears in the
JavaScript at all**, so the renderers cannot quietly become per-concept again.

## Python owns the mathematics

The old tangent board defined its own curve in JavaScript:

```js
const f      = (x) => 0.3 * x * x * x - 2 * x;
const fPrime = (x) => 0.9 * x * x - 2;
```

while `optimumai diff` computed derivatives in Python. Two implementations of
the same mathematics drift, and the browser copy drifts *silently* — a wrong
slope looks exactly like a correct one.

Now a board declares the expression once. `expressions.compile_expression()`
parses it with SymPy, differentiates it, and emits JavaScript for both via
SymPy's own `jscode` printer. The browser evaluates arithmetic; it never defines
a function. A typo fails at build time rather than rendering a blank board:

```python
compile_expression("0.3*x**3 - 2*y")   # ValueError: unbound symbol(s) y
compile_expression("0.3*x**3 - ")      # ValueError: could not parse
```

## The prerequisite graph is executable

Prerequisites used to be free text (`["vector algebra"]`), which reads well and
enforces nothing. They are now edges between real concept ids, so the graph can
be checked, ordered, and enforced:

```bash
optimumai scratchpad --order        # print the graph with lock/done state
optimumai scratchpad matrix_transform
# Error: 'matrix_transform' has unmet prerequisites: dot_product.
#        Do those first, or pass --force to open it anyway.
```

`validate_dag()` rejects dangling edges, self-loops and cycles, and
`create_app()` calls it at startup so a bad graph fails loudly instead of
producing a confusing sidebar. `learning_order()` topologically sorts with
alphabetical tie-breaking, so the sidebar order is stable across runs.

Human-readable background lives on `assumed_knowledge`. It was never the same
thing as an edge, and conflating the two is what made the graph unenforceable.

## One progress store, two front doors

Each board declares the curriculum `lesson_id` it corresponds to —
`dot_product` → `dot`, `tangent_line` → `derivative`, `matrix_transform` →
`matmul`, `gradient_descent` → `descent`. Marking a board complete writes that
lesson id through `ProgressTracker`, so:

* `optimumai learn dot` makes the `dot_product` board show as done, and
* finishing the board counts toward `optimumai progress` and spaced review.

A test asserts every `lesson_id` exists in `COURSE`, so the mapping cannot rot
into a private second store.

## Only the declared library loads

A board declares `library` and `needs_katex`, and the template emits tags for
exactly those. A vectors board does not download a maths typesetter it never
renders, and no board pays for a library it does not use.

## Offline is real, not aspirational

The server was always local-first, but the *page* pulled JSXGraph and KaTeX from
a CDN, so the first load of a board needed the network. On a plane it did not
draw.

```bash
optimumai scratchpad --vendor      # download assets into static/vendor/, once
```

After that `resolve_assets()` prefers the local copies and the sidebar reads
"offline · vendored". Partial vendoring is deliberately *not* treated as offline
— a half-populated directory falls back to CDN rather than serving a broken page.

## Snapshots are checked claims

A slider with no guidance is a dead control; most learners move it once and
learn nothing. Each board ships snapshots — authored parameter settings with a
one-line reason:

> **Divergence** — Past lr = 1/0.35 = 2.857 each overshoot is bigger than the
> last, so the loss runs away instead of settling.

Those claims are **tested**, by replicating the iteration in Python and
asserting the labelled behaviour actually occurs. Two authored bugs were caught
this way and fixed:

* an "Oscillating" snapshot at `lr = 2.6` that in fact converges, because the
  threshold for L = 0.35x² is `1/0.35 = 2.857`, not 2.5;
* a "Steep descent" snapshot at `x = -3` where f'(x) = 0.9x² - 2 is *positive*
  (+6.1) — a steep climb. f' is most negative at x = 0.

If the numbers drift, the label becomes a lie, so the label is pinned.
