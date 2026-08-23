"""Simulate one annotated agent-loop run — the ``optimumai trace-loop`` command.

Combines the convergence math with a narrative: for a given subject and
per-attempt approval probability, it shows the cumulative chance the loop has
approved by each attempt and names the expected outcome. It performs no live LLM
calls — the model names are shown only to make the maker/checker split concrete.
"""

from __future__ import annotations

from optimumai.core._fmt import num
from optimumai.core.trace import Trace


def simulate_loop_trace(
    subject: str = "AAPL",
    iterations: int = 3,
    p_approve: float = 0.5,
    maker_model: str = "claude-sonnet-4-6",
    checker_model: str = "claude-haiku-4-5",
) -> Trace:
    """Trace a simulated loop run: evidence → attempts → gate, with convergence math."""
    if iterations < 1:
        raise ValueError(f"iterations must be ≥ 1, got {iterations}")
    if not 0 < p_approve <= 1:
        raise ValueError(f"p_approve must be in (0, 1], got {p_approve}")
    if maker_model == checker_model:
        raise ValueError("maker_model and checker_model must differ (independent verification).")

    q = 1.0 - p_approve
    t = Trace(
        op="simulate_loop",
        formula="evidence → [maker → checker → gate]×N;  P(approved by k) = 1 − (1 − p)^k",
        complexity="O(iterations)",
        why_ai=[
            "Shows the bounded evidence→maker→checker→gate loop end to end",
            "Maker and checker are different models — independent verification",
            "Cumulative approval probability predicts when the loop will publish",
        ],
        meta={"subject": subject, "maker_model": maker_model, "checker_model": checker_model,
              "p_approve": p_approve, "iterations": iterations},
    )
    t.add(
        "Gather evidence",
        f"tools → evidence table for {subject!r} (no LLM yet)",
        subject,
        detail="A number not in this table is a hallucination by definition.",
    )

    cdf = 0.0
    for k in range(1, iterations + 1):
        cdf = 1.0 - q**k
        t.add(
            f"Attempt {k}: maker ({maker_model}) → checker ({checker_model})",
            f"P(approved by attempt {k}) = 1 − (1 − {num(p_approve)})^{k} = {num(cdf)}",
            cdf,
            detail="Reject feeds the checker's violations back into the next attempt.",
        )

    p_exhaust = q**iterations
    likely = "APPROVE (publish artifact)" if cdf >= 0.5 else "EXHAUST → escalate"
    t.add(
        "Gate outcome",
        f"P(exhaust) = (1 − {num(p_approve)})^{iterations} = {num(p_exhaust)} → likely {likely}",
        cdf,
    )
    t.result = cdf
    return t
