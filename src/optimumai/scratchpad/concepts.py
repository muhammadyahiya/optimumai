"""Scratchpad concepts, declared as data.

Tier 1 hardcoded one JavaScript function per board. That does not scale: the
course has 82 lessons, and one bespoke ``initXBoard()`` each would mean 82
hand-maintained renderers. Here a concept declares a :class:`BoardSpec` -- a
*kind*, a bounding box, some draggable points, some parameters -- and one
generic renderer per kind builds it. Adding a concept is a dict entry; adding a
*kind* is the rare case.

Four kinds cover the current set:

``vectors``  draggable arrows from the origin; dot product and cosine
``function`` a curve from a Python-owned expression, with a tangent glider
``matrix``   draggable basis vectors warping a grid; determinant as area
``descent``  gradient descent stepping on a curve, with a learning-rate knob

Three things are deliberately *not* in the JavaScript:

* the mathematics of any curve -- see :mod:`optimumai.scratchpad.expressions`
* the colours -- they come from :mod:`optimumai.design`
* the prerequisite graph -- see :mod:`optimumai.scratchpad.dag`
"""

from __future__ import annotations

from dataclasses import dataclass, field

from optimumai.design import PALETTE

from .expressions import compile_expression


@dataclass(frozen=True)
class Param:
    """A continuous knob, rendered as a slider and as a CLI flag."""

    name: str
    label: str
    default: float
    min: float
    max: float
    step: float

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "label": self.label,
            "default": self.default,
            "min": self.min,
            "max": self.max,
            "step": self.step,
        }


@dataclass(frozen=True)
class DraggablePoint:
    """A point the learner can drag. ``role`` names a colour in the palette."""

    name: str
    x: float
    y: float
    role: str = "attention"

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "x": self.x,
            "y": self.y,
            "color": getattr(PALETTE, self.role),
        }


@dataclass(frozen=True)
class Snapshot:
    """An authored parameter setting worth visiting, with a one-line reason.

    A slider with no guidance is a dead control -- most learners wiggle it once
    and learn nothing. Snapshots are the author's claim about *where something
    interesting happens*, and they double as the golden cases in tests.
    """

    label: str
    note: str
    values: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"label": self.label, "note": self.note, "values": self.values}


@dataclass(frozen=True)
class BoardSpec:
    """Everything the browser needs to build one board, and nothing more."""

    kind: str
    bounding_box: tuple[float, float, float, float]
    library: str = "jsxgraph"
    needs_katex: bool = True
    expression: str | None = None
    var: str = "x"
    points: tuple[DraggablePoint, ...] = ()
    params: tuple[Param, ...] = ()
    readouts: tuple[tuple[str, str], ...] = ()
    snapshots: tuple[Snapshot, ...] = ()

    def to_dict(self) -> dict:
        """Serialise for the page, compiling the expression via SymPy if present."""
        payload: dict = {
            "kind": self.kind,
            "bounding_box": list(self.bounding_box),
            "library": self.library,
            "needs_katex": self.needs_katex,
            "points": [p.to_dict() for p in self.points],
            "params": [p.to_dict() for p in self.params],
            "readouts": [{"key": k, "label": lbl} for k, lbl in self.readouts],
            "snapshots": [s.to_dict() for s in self.snapshots],
            "palette": {
                "basis_x": PALETTE.basis_x,
                "basis_y": PALETTE.basis_y,
                "grid": PALETTE.grid,
                "grid_faded": PALETTE.grid_faded,
                "attention": PALETTE.attention,
                "result": PALETTE.result,
                "given": PALETTE.given,
                "derivative": PALETTE.derivative,
                "counterexample": PALETTE.counterexample,
            },
            "ghost_opacity": PALETTE.ghost_opacity,
        }
        payload["expression"] = (
            compile_expression(self.expression, self.var).to_dict()
            if self.expression
            else None
        )
        return payload


@dataclass(frozen=True)
class ScratchpadConcept:
    """One interactive board, plus the teaching material that frames it."""

    concept_id: str
    title: str
    track: str
    board: BoardSpec
    why_ai_uses_it: str
    cli_hook: str
    quiz_question: str
    quiz_answer: str
    #: Curriculum lesson this board *is*, so progress writes to one store.
    lesson_id: str
    #: Edges in the prerequisite graph -- other ``concept_id`` values.
    prerequisites: list[str] = field(default_factory=list)
    #: Prose background. Never an edge; it is not machine-checkable.
    assumed_knowledge: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "concept_id": self.concept_id,
            "title": self.title,
            "track": self.track,
            "board": self.board.to_dict(),
            "why_ai_uses_it": self.why_ai_uses_it,
            "cli_hook": self.cli_hook,
            "quiz_question": self.quiz_question,
            "quiz_answer": self.quiz_answer,
            "lesson_id": self.lesson_id,
            "prerequisites": list(self.prerequisites),
            "assumed_knowledge": list(self.assumed_knowledge),
        }


CONCEPTS: dict[str, ScratchpadConcept] = {
    "dot_product": ScratchpadConcept(
        concept_id="dot_product",
        title="Vector algebra — dot product & cosine similarity",
        track="Spine -> Branch C (Transformers)",
        lesson_id="dot",
        prerequisites=[],
        assumed_knowledge=["vector algebra"],
        board=BoardSpec(
            kind="vectors",
            bounding_box=(-6, 6, 6, -6),
            points=(
                DraggablePoint("a", 3, 2, role="basis_x"),
                DraggablePoint("b", 2, -2, role="given"),
            ),
            readouts=(("magA", "|a|"), ("magB", "|b|"), ("cos", "cos similarity")),
            snapshots=(
                Snapshot(
                    "Perpendicular",
                    "Dot product is exactly 0 -- no shared direction at all.",
                    {"ax": 3, "ay": 0, "bx": 0, "by": 3},
                ),
                Snapshot(
                    "Opposed",
                    "Cosine hits -1: same line, opposite ends.",
                    {"ax": 3, "ay": 1, "bx": -3, "by": -1},
                ),
                Snapshot(
                    "Aligned but unequal",
                    "Cosine is 1 while the dot product is not -- length is not direction.",
                    {"ax": 1, "ay": 1, "bx": 4, "by": 4},
                ),
            ),
        ),
        why_ai_uses_it=(
            "A dot product measures how much two vectors point in the same "
            "direction. Every attention score in a transformer is exactly "
            "this: the dot product between a query vector and a key vector. "
            "Cosine similarity is the same operation normalized by "
            "magnitude, which is what embedding search and RAG retrieval "
            "use to rank 'closest meaning' independent of vector length."
        ),
        cli_hook="optimumai algebra dot '[3,2]' '[2,-2]' --level intermediate",
        quiz_question=(
            "If vector a and vector b point in exactly opposite directions, "
            "what is their cosine similarity?"
        ),
        quiz_answer="-1 (the angle between them is 180 degrees, cos(180) = -1)",
    ),
    "tangent_line": ScratchpadConcept(
        concept_id="tangent_line",
        title="Derivatives — the tangent line as instantaneous slope",
        track="Spine -> Branch B (Deep Learning)",
        lesson_id="derivative",
        prerequisites=[],
        assumed_knowledge=["algebra", "coordinate geometry"],
        board=BoardSpec(
            kind="function",
            bounding_box=(-5, 8, 5, -8),
            expression="0.3*x**3 - 2*x",
            readouts=(("x", "x"), ("fx", "f(x)"), ("slope", "slope f'(x)")),
            snapshots=(
                Snapshot(
                    "Local maximum",
                    "Slope passes through 0 going positive-to-negative.",
                    {"x": -1.4907},
                ),
                Snapshot(
                    "Local minimum",
                    "Slope is 0 again -- gradient descent would stop dead here.",
                    {"x": 1.4907},
                ),
                Snapshot(
                    "Steepest descent",
                    "f' is most negative at the inflection point, so this is the "
                    "biggest downhill step this curve offers.",
                    {"x": 0.0},
                ),
                Snapshot(
                    "Steep climb",
                    "Slope is large and positive here -- gradient descent would "
                    "step left, and hard.",
                    {"x": -3.0},
                ),
            ),
        ),
        why_ai_uses_it=(
            "The derivative at a point is the slope of the tangent line "
            "there. Backpropagation is nothing more than computing this "
            "slope for the loss function with respect to every weight, "
            "then nudging each weight a small step in the direction that "
            "decreases the loss: w_new = w_old - learning_rate * dL/dw."
        ),
        cli_hook="optimumai diff '0.3*x**3 - 2*x' --at 1.5",
        quiz_question=(
            "At the very bottom of a U-shaped curve, what is the slope of "
            "the tangent line, and what does that mean for gradient "
            "descent?"
        ),
        quiz_answer=(
            "The slope is 0 — gradient descent stops updating the weight "
            "because dL/dw = 0, meaning it has reached a minimum."
        ),
    ),
    "matrix_transform": ScratchpadConcept(
        concept_id="matrix_transform",
        title="Matrices — a transformation of space, and its determinant",
        track="Spine -> Branch C (Transformers)",
        lesson_id="matmul",
        prerequisites=["dot_product"],
        assumed_knowledge=["vector algebra", "area of a parallelogram"],
        board=BoardSpec(
            kind="matrix",
            bounding_box=(-5, 5, 5, -5),
            points=(
                DraggablePoint("i", 1, 0, role="basis_x"),
                DraggablePoint("j", 0, 1, role="basis_y"),
            ),
            readouts=(("det", "determinant"), ("area", "unit square area")),
            snapshots=(
                Snapshot(
                    "Identity",
                    "Nothing moves; the determinant is 1.",
                    {"ix": 1, "iy": 0, "jx": 0, "jy": 1},
                ),
                Snapshot(
                    "Collapsed to a line",
                    "Both basis vectors are parallel: determinant 0, and the "
                    "transformation is not invertible.",
                    {"ix": 2, "iy": 1, "jx": 4, "jy": 2},
                ),
                Snapshot(
                    "Orientation flipped",
                    "j has crossed to the other side of i, so the determinant "
                    "is negative -- space has been turned over.",
                    {"ix": 1, "iy": 0, "jx": 0, "jy": -1},
                ),
            ),
        ),
        why_ai_uses_it=(
            "A matrix is not a grid of numbers to memorize rules about — it "
            "is a transformation of space, and its columns are simply where "
            "the basis vectors land. Every linear layer in a network is this "
            "operation. The determinant is the factor by which areas scale, "
            "so a determinant of zero means the layer has collapsed a "
            "dimension and destroyed information that no later layer can "
            "recover."
        ),
        cli_hook="optimumai algebra matmul '[[1,0],[0,1]]' '[[2,1],[1,1]]'",
        quiz_question=(
            "You drag the basis vectors until they lie on the same line. What "
            "is the determinant, and what does that say about whether the "
            "transformation can be undone?"
        ),
        quiz_answer=(
            "The determinant is 0. All of space has been squashed onto a line, "
            "so area is destroyed and the map cannot be inverted — you cannot "
            "unsquish a line back into a plane."
        ),
    ),
    "gradient_descent": ScratchpadConcept(
        concept_id="gradient_descent",
        title="Gradient descent — the learning rate decides everything",
        track="Spine -> Branch B (Deep Learning)",
        lesson_id="descent",
        prerequisites=["tangent_line"],
        assumed_knowledge=["derivatives", "iteration"],
        board=BoardSpec(
            kind="descent",
            bounding_box=(-5, 10, 5, -4),
            expression="0.35*x**2 + 0.5",
            params=(
                Param("lr", "learning rate", 0.4, 0.01, 3.2, 0.01),
                Param("x0", "start x", -4.0, -4.5, 4.5, 0.1),
                Param("steps", "steps", 8, 1, 25, 1),
            ),
            readouts=(("final_x", "final x"), ("final_y", "final loss"),
                      ("status", "behaviour")),
            snapshots=(
                Snapshot(
                    "Healthy convergence",
                    "Steps shrink as the slope flattens; it settles in the valley.",
                    {"lr": 0.4, "x0": -4.0, "steps": 20},
                ),
                Snapshot(
                    "Too small",
                    "Correct direction, but it runs out of steps far from the minimum.",
                    {"lr": 0.05, "x0": -4.0, "steps": 8},
                ),
                Snapshot(
                    "Oscillating",
                    "It overshoots the minimum every step and lands on the far wall, "
                    "but still closes in -- |1 - lr*0.7| is under 1.",
                    {"lr": 2.0, "x0": -4.0, "steps": 12},
                ),
                Snapshot(
                    "Divergence",
                    "Past lr = 1/0.35 = 2.857 each overshoot is bigger than the last, "
                    "so the loss runs away instead of settling.",
                    {"lr": 3.1, "x0": -4.0, "steps": 10},
                ),
            ),
        ),
        why_ai_uses_it=(
            "Training is this loop and almost nothing else: measure the slope, "
            "step downhill, repeat. The learning rate is the one knob that "
            "decides whether the loss falls, crawls, oscillates, or explodes — "
            "which is why it is the first hyperparameter anyone tunes. Drag it "
            "past the stability threshold and you can watch a training run "
            "diverge in real time."
        ),
        cli_hook="optimumai train --steps 150",
        quiz_question=(
            "Raising the learning rate makes each step bigger. Why does making "
            "it too big cause the loss to *increase* rather than merely "
            "converge more slowly?"
        ),
        quiz_answer=(
            "The gradient is only accurate right where it was measured. A step "
            "that is too long lands past the minimum on the far wall, where the "
            "slope is steeper, so the next step is longer still and the "
            "iterates run away."
        ),
    ),
}


def get_concept(concept_id: str) -> ScratchpadConcept:
    if concept_id not in CONCEPTS:
        raise KeyError(
            f"Unknown scratchpad concept '{concept_id}'. "
            f"Available: {', '.join(CONCEPTS.keys())}"
        )
    return CONCEPTS[concept_id]


def list_concepts() -> list[ScratchpadConcept]:
    """Concepts in prerequisite order, so the sidebar teaches a sequence."""
    from .dag import learning_order

    return [CONCEPTS[cid] for cid in learning_order(CONCEPTS)]
