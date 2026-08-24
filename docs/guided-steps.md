# Guided steps — hints, justifications, and two-level detail

`optimumai explain <concept>` renders a concept as a DAG you step through. As of
Sprint 1 each step can also carry three optional fields that change it from a
slideshow into something you have to think in front of.

The design is taken from patterns that several independent sources converge on:
WolframAlpha's step-by-step solver, 3Blue1Brown's stated pedagogy, Mathigon's
progressive hints, and Manu Kapur's *productive failure* result — that
attempting a problem *before* instruction produced substantially better
conceptual understanding at equal procedural scores.

## The three fields

```python
CONCEPTS["attention"] = {
    ...
    "steps": _steps(
        {
            "title": "Scale by sqrt(d_k)",
            "narration": "Dividing by sqrt(dimension) prevents ...",
            "formula": r"\frac{QK^T}{\sqrt{d_k}}",

            # names the technique WITHOUT performing it
            "hint": "These dot products get larger as the vectors get longer ...",

            # the rule that LICENSES the move -- "why is this allowed"
            "justification": "If q and k have independent, unit-variance entries ...",

            # level-2 mechanics, collapsed by default
            "substeps": [
                {"title": "Which dimension is d_k", "detail": "..."},
                {"title": "Why saturation stops learning", "detail": "...",
                 "formula": r"..."},
            ],
        },
    ),
}
```

All three are read with `.get()`, so concepts authored before Sprint 1 keep
working untouched — they simply render as they always did.

## The `(index, phase)` cursor

This is the detail that makes hints work, and it is easy to get wrong. The
cursor is **a pair**, not an integer:

```
phase in {hint, revealed}

(3, hint)  --Next-->  (3, revealed)  --Next-->  (4, hint)
```

Pressing **Next** while a hint is showing reveals *that same step*. It does not
advance to the next one. If you implement hints as extra list entries instead,
the reader just presses Next twice as fast and the pause where the thinking was
supposed to happen never happens.

While in the hint phase the narration, formula, code, justification, substeps
and metrics are all hidden. Only the hint is visible.

## Authoring guidance

**Hints name the technique, they do not perform it.** "Isolate the variable
terms on one side" is a hint; "add 3x to both sides" is the step. A test asserts
a hint is never identical to its own narration.

**Keep the top level to 3–8 steps.** Level 1 is *strategy* — the decisions a
human would narrate out loud. Level 2 is *mechanics* — the algebra that produced
the level-1 result. Push length downward rather than adding top-level steps; a
test enforces the 3–8 bound on guided concepts.

**Justification answers "why is this allowed", not "what does it do."** It
should name the theorem, identity, or definition the step relies on.

## Controls

| Control | Key | Behaviour |
|---|---|---|
| Next / Reveal step | `→` or `Space` | Reveals the hint's step, or advances |
| Prev | `←` | Back one step, already revealed |
| Show all steps | `a` | Every step stacked in one scrollable list |
| Hints on/off | `h` | Off skips the hint phase entirely |
| Start over | `r` | Back to step 1, hints on |

At the last step the Next button becomes **Start over**.

The current step is mirrored into the URL as `#step=4`, so you can link someone
to the exact step they are stuck on, and reloading returns there.

## Only one drill-down open at a time

Opening a substep closes any other open one. This is deliberate: the point of
the two-level split is that the top level stays readable as an outline, and
letting several drill-downs accumulate destroys exactly that.

## Which concepts are authored

Sprint 1 authored the fields for `attention`, `backpropagation`, and
`gradient_descent`. The remaining 27 concepts render normally and can be
enriched one at a time — each is a self-contained edit to its own entry.
