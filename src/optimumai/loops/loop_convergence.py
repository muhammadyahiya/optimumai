"""Loop convergence — how many attempts until the checker approves?

If each attempt clears the gate independently with probability ``p``, the number
of attempts to the first APPROVE is a Geometric random variable. This lesson
computes the cumulative approval probability attempt-by-attempt, the chance the
loop exhausts its ``max_attempts`` budget without approving, and the expected
number of attempts — the math that tells you whether a loop will actually
converge or just burn its budget.
"""

from __future__ import annotations

from optimumai.core._fmt import num
from optimumai.core.explain import ExplainLevel
from optimumai.core.trace import Trace


def loop_convergence_trace(p_approve: float, max_attempts: int = 3) -> Trace:
    """Trace Geometric convergence of a loop with per-attempt approval prob ``p``."""
    if not 0 < p_approve <= 1:
        raise ValueError(f"p_approve must be in (0, 1], got {p_approve}")
    if max_attempts < 1:
        raise ValueError(f"max_attempts must be ≥ 1, got {max_attempts}")

    q = 1.0 - p_approve
    t = Trace(
        op="loop_convergence",
        formula="P(approve by attempt k) = 1 − (1 − p)^k",
        complexity="O(max_attempts)",
        why_ai=[
            "Tells you if a loop converges within budget or just exhausts attempts",
            "P(exhaust) = (1 − p)^N is the escalation rate you must staff for",
            "Low p ⇒ raise attempts, improve evidence/prompts, or accept more escalations",
        ],
        meta={"p_approve": p_approve, "max_attempts": max_attempts},
    )

    cdf = 0.0
    for k in range(1, max_attempts + 1):
        cdf = 1.0 - q**k
        t.add(
            f"P(approve by attempt {k})",
            f"1 − (1 − {num(p_approve)})^{k} = {num(cdf)}",
            cdf,
        )

    p_exhaust = q**max_attempts
    t.add(
        "P(exhaust without approval)",
        f"(1 − {num(p_approve)})^{max_attempts} = {num(p_exhaust)}",
        p_exhaust,
        detail="This fraction of runs hits max_attempts and escalates.",
    )

    # Expected attempts conditioned on approving within N: Σ k·p·q^(k-1) / P(approve≤N).
    approve_by_n = 1.0 - q**max_attempts
    exp_num = sum(k * p_approve * q ** (k - 1) for k in range(1, max_attempts + 1))
    expected = exp_num / approve_by_n if approve_by_n > 0 else float("inf")
    t.add(
        "Expected attempts (given approval within N)",
        f"Σ k·p·q^(k−1) / P(approve≤N) = {num(expected)}",
        expected,
    )
    t.result = 1.0 - q**max_attempts
    return t


def loop_convergence(
    p_approve: float,
    max_attempts: int = 3,
    explain: bool = False,
    level: str | ExplainLevel = ExplainLevel.INTERMEDIATE,
) -> float:
    """Return P(approve within ``max_attempts``). Set ``explain=True`` to print the trace."""
    t = loop_convergence_trace(p_approve, max_attempts=max_attempts)
    return t.render(level) if explain else t.result


def demo(seed: int = 0) -> Trace:
    """A checker that approves 55% of attempts, over a 3-attempt budget."""
    return loop_convergence_trace(0.55, max_attempts=3)
