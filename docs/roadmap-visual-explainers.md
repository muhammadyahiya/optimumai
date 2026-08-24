# OptimumAI — Visual Explanation Roadmap
Research-driven, 4 sprints. Sources: WolframAlpha, 3Blue1Brown/Manim, explorable-explanation products.

## RESEARCHER — the transferable findings

### From WolframAlpha
| # | Pattern | Mechanic |
|---|---------|----------|
| W1 | Pod stack | Titled independent blocks, sparse integer `position` (multiples of 100), exactly one `primary`. Per-pod `error` so one failure never blanks the page. |
| W2 | Tier order | Identity 100 → Answer 200 → Procedure 250 → Equivalence 300 → Picture 400 → Structure 500 → Neighborhood 600 → Concept 700. Answer-first, then progressively less committal. |
| W3 | Hint-before-step | Cursor is `(index, phase)`, `phase ∈ {hint, revealed}`. "Next" on a hint reveals THAT step. Hint names the technique; step performs it. |
| W4 | Two-level granularity | 3–8 top-level *strategy* steps; all length pushed to level-2 *mechanics*. Hard rule: only ONE drill-down open at a time. |
| W5 | Addressable state | Every disclosure has a stable name (`PodId__State`), serializable to a flag AND a URL hash. Step-by-step is a *state of the answer*, not a separate page. |
| W6 | Interpretation echo | Restate canonically before answering; doubles as API discovery (`≡ optimumai.algebra.dot(...)`). |
| W7 | Type-driven viz | The *shape* of the answer selects the picture (Interval → number line). Not "author remembered a chart". |
| W8 | `--check` mode | THE gap: WA explains its own derivation, never reads yours. Validate a student's chain, report first invalid transition. |
| W9 | One expandable token | A single visual token means "more inside", everywhere. Never mix `>` / `...` / `(more)`. |

### From 3Blue1Brown / Manim (exact values, from source — not eyedropped)
- Canvas `#000000` (Grant's production config). NOTE: his ffmpeg applies `saturation=1.5`, so published frames do NOT match library hexes.
- Basis triple, stable since 2016: `X=GREEN_C #83C167`, `Y=RED_C #FC6255`, `Z=BLUE_D #29ABCA`. Matrix columns tinted to match the basis vectors they encode.
- **`YELLOW` reserved for attention only** — every Manim indication animation defaults to it. Never give it a semantic role.
- Signed params: sign → hue family, magnitude → lightness (`BLUE_E→BLUE_B` / `RED_E→RED_B`), sign glyph printed redundantly.
- **Weights (blue/red) vs data (greyscale `GREY_C→WHITE`)** — tells an activation from a weight at a glance.
- Two-tier grid: solid `BLUE_D` w2 + faded same-hue w1 @ opacity .25, `faded_line_ratio=4`.
- Faded reference of prior state at 30% opacity (`fade(0.7)`) — "keep a copy of the grid in the background to track where everything ends up relative to where it starts".
- Easing: quintic smootherstep `6t⁵−15t⁴+10t³` (zero velocity AND acceleration at both ends). Durations 1s move / 2s equation morph / 3s spatial deformation / 1s beat.
- Stagger enumerable things (`lag_ratio≈.05–.2`), NEVER a continuum (`lag_ratio=0`). When a stagger supplies the envelope, per-element easing is `linear`.
- NN: activation → `fill_opacity`; edge stroke `min(3·(|w|/max|w|)³,3)` — the cube suppresses near-zero weights.
- Honest truncation: `⋮` + braced true count; `⋯` in penultimate row/col.

### Pedagogy that constrains the API
- "Never start with definitions — definitions are an ending point." → primary constructor takes the *object*, formula is a derived view.
- "Every movement deliberate, with an identifiable purpose." → animation primitives must name the claim the motion makes.
- Productive failure (Kapur): problem-first beat instruction-first by ~2 SD on *conceptual* understanding at equal procedural scores. Independently corroborates W3.
- Always pair a spatial metaphor with its non-spatial reading (real ML objects are 12,288-D).

## BA — user stories (Sprint 1)
- **US-1** As a learner I want a *hint* before each step so I can attempt the reasoning first (productive failure).
- **US-2** As a learner I want to know *why* a step is licensed, not just what it does.
- **US-3** As a learner I want a 3–8 step overview, drilling in only where I'm lost.
- **US-4** As a learner I want to link someone to the exact step I'm stuck on.
- **US-5** As a maintainer I want ONE palette source so surfaces stop drifting.
- **US-6** As a learner on a plane I want it to work offline.

## ARCHITECT — decisions
- **AD-1 Design tokens are Python, not CSS.** `optimumai/design/tokens.py` is the single source; emits CSS custom properties and ANSI. Evidence: 64 distinct hardcoded hexes across 6 surfaces, zero shared.
- **AD-2 Extend `_steps()`, don't fork it.** New keys (`hint`, `justification`, `substeps`) via `e.get()` → all 30 existing concepts keep working untouched.
- **AD-3 Cursor is `(index, phase)`.** Pressing Next on a hint reveals that same step. This is W3's load-bearing detail; getting it wrong makes hints just another step.
- **AD-4 Pre-render every disclosure state.** No fetch, no CDN for our own content. Disclosure = CSS/JS over already-present DOM.
- **AD-5 Renderers hold no pedagogy.** If a renderer must decide what to show, that decision belongs in the model.
- **AD-6 Mutual exclusion enforced in the model,** so terminal and HTML agree.

## SPRINTS
**Sprint 1 — "The Guided Step" (foundation + the distinctive mechanic)**
1. `design/tokens.py`: Manim-derived palette + motion contract; `to_css_vars()`, `to_ansi()`.
2. Extend step model: `hint`, `justification`, `substeps` (backwards-compatible).
3. Explain HTML: `(index,phase)` cursor, hints toggle, Show-all, Start-over, mutual-exclusion drill-down, `?why` justification, `#step=N` deep link, keyboard.
4. Author the new fields for 3 concepts (attention, backpropagation, gradient_descent).
5. Tests + docs + CHANGELOG.

**Sprint 2 — "Pods & Terminal"** — Pod/tier model (W1,W2), interpretation echo + reproducing code (W6), terminal renderer so `explain` works without a browser, `--state` flags (W5).

**Sprint 3 — "Manipulate"** — scratchpad boards become declarative (`Param` types: Range/Choice/Bool/Point), snapshots as authored parameter tours, grid-deformation board (determinant = unit square, eigenvector = span line), token adoption across all surfaces.

**Sprint 4 — "Check My Work"** — `--check` (W8), worksheet/answer-key generator, multiple methods (W-methods), type-driven number-line dispatch (W7).

## From explorable-explanation products (verified)
| # | Pattern | Origin |
|---|---------|--------|
| E1 | Pre-computed trace → JSON → browser replay (Python computes, HTML renders) | AnimatedLLM, PAIR, Google grokking — 3 independent convergences |
| E2 | Interactive formula view: formula is an *indexed view of the data* | CNN Explainer |
| E3 | Reversible abstraction: collapsed-by-default, click → animated derivation, click → collapse | Transformer Explainer |
| E4 | Cognitive gates: `goals:` field hides next step until an event fires | Nicky Case, Mathigon |
| E5 | Place Your Bets: predict *before* reveal (strongest in a terminal) | Case, NYT "You Draw It" |
| E6 | Ladder of abstraction — and stepping *down* matters as much as up | Bret Victor |
| E7 | Tensor-as-colored-box-row; hue = token, held constant article-wide; length ∝ real dim | Jay Alammar |
| E8 | Named-phase stepper / slow-motion (name the 5 phases of an update) | GAN Lab |
| E9 | Same-diagram-one-part-highlighted + declared legend up front | colah |
| E10 | Progressive hints in the authoring DSL: `[[100 (First hint. | Second hint.)]]` | Mathigon |

### Caveats to internalize
- **Most readers never touch the widgets** (NYT). The *default* state must already teach.
- **Interactivity ≠ learning**: readers report engagement without measurable learning gains. Test recall/application, not felt engagement.
- **Guided path AND sandbox** — CNN Explainer's beginners explicitly asked for a step-by-step first run.
- Distill burned out at 100+ hrs/article and concluded the *template + community* mattered more than the venue → invest in the **authoring format**, not hand-built explainers.

## VERIFIED repo findings (checked, not assumed)
1. `core/flow_trace.py` already has the E1 architecture — `SCHEMA_VERSION`, `DataRef`, `FlowStep.inputs/outputs`, `to_json()`, `validate()` — but `grep -rl FlowTrace src/` returns only itself + 3 `rag/` files. **Paid for, wired to one feature.**
2. `core/trace.py` `Step` = `(index,title,expression,value,detail)` — **no input refs**, so E2/E6-style linked highlighting has no data to bind to. `FlowStep` already solves this; the two schemas want unifying, not duplicating.
3. **12 CDN references** (`cdnjs` ×8, `jsdelivr` ×4). The "offline / local-first" claim is currently false.
4. **64 distinct hardcoded hex colors** across 6 interactive surfaces, no shared source.

## Sprint 1 final scope — "The Guided Step"
Chosen because 4 independent sources converge on hint-before-step (WA C2, Kapur productive failure, Case "Place Your Bets", Mathigon progressive hints), and because it needs no schema migration.
1. `design/tokens.py` — Manim-derived palette + motion contract; `to_css_vars()` / `to_ansi()`.
2. Step model += `hint`, `justification`, `substeps` (via `.get()`, so all 30 concepts keep working).
3. Explain HTML: `(index,phase)` cursor · hints toggle · Show-all · Start-over · mutual-exclusion drill-down · `?why` justification · `#step=N` deep link · keyboard.
4. Author the new fields for attention, backpropagation, gradient_descent.
5. Tests + docs + CHANGELOG.

DEFERRED (with reason): FlowTrace/Trace unification → Sprint 2 (schema migration, needs its own sprint). CDN inlining → Sprint 2. `--check` → Sprint 4 (needs sympy equivalence).
