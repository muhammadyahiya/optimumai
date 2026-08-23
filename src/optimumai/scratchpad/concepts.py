"""
optimumai.scratchpad.concepts
------------------------------
Static metadata for each interactive scratchpad concept. Every entry conforms
to the same output contract already used by `explain=True` elsewhere in
optimumai: prerequisites, a "why AI uses this" bridge, a CLI hook, and a quiz
question. The live numeric trace itself is computed client-side in
static/scratchpad.js so dragging a point never round-trips to the server.

Adding a new concept = adding one entry here + one `initXBoard()` function in
scratchpad.js. Nothing else needs to change.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ScratchpadConcept:
    concept_id: str
    title: str
    track: str
    prerequisites: list[str]
    why_ai_uses_it: str
    cli_hook: str
    quiz_question: str
    quiz_answer: str
    board_type: str  # matches an init*Board() function name in scratchpad.js


CONCEPTS = {
    "dot_product": ScratchpadConcept(
        concept_id="dot_product",
        title="Vector algebra — dot product & cosine similarity",
        track="Spine -> Branch C (Transformers)",
        prerequisites=["vector algebra"],
        why_ai_uses_it=(
            "A dot product measures how much two vectors point in the same "
            "direction. Every attention score in a transformer is exactly "
            "this: the dot product between a query vector and a key vector. "
            "Cosine similarity is the same operation normalized by "
            "magnitude, which is what embedding search and RAG retrieval "
            "use to rank 'closest meaning' independent of vector length."
        ),
        cli_hook="optimumai learn dot_product --level intermediate",
        quiz_question=(
            "If vector a and vector b point in exactly opposite directions, "
            "what is their cosine similarity?"
        ),
        quiz_answer="-1 (the angle between them is 180 degrees, cos(180) = -1)",
        board_type="dotProduct",
    ),
    "tangent_line": ScratchpadConcept(
        concept_id="tangent_line",
        title="Derivatives — the tangent line as instantaneous slope",
        track="Spine -> Branch B (Deep Learning)",
        prerequisites=["algebra", "coordinate geometry"],
        why_ai_uses_it=(
            "The derivative at a point is the slope of the tangent line "
            "there. Backpropagation is nothing more than computing this "
            "slope for the loss function with respect to every weight, "
            "then nudging each weight a small step in the direction that "
            "decreases the loss: w_new = w_old - learning_rate * dL/dw."
        ),
        cli_hook="optimumai learn derivatives --level intermediate",
        quiz_question=(
            "At the very bottom of a U-shaped curve, what is the slope of "
            "the tangent line, and what does that mean for gradient "
            "descent?"
        ),
        quiz_answer=(
            "The slope is 0 — gradient descent stops updating the weight "
            "because dL/dw = 0, meaning it has reached a minimum."
        ),
        board_type="tangentLine",
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
    return list(CONCEPTS.values())
