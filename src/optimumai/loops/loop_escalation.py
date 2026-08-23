"""Loop escalation — the decision theory of retry vs hand to a human.

When an attempt is rejected, the loop faces a choice: retry (cheap, might approve
with probability ``p``) or escalate to a human (costs ``c_h`` but resolves it).
Comparing expected values gives a threshold approval probability ``p*`` below
which escalation is the rational move. This is why a loop escalates instead of
retrying forever — and how to set ``max_attempts``/``escalate_after``.
"""

from __future__ import annotations

from optimumai.core._fmt import num
from optimumai.core.explain import ExplainLevel
from optimumai.core.trace import Trace


def loop_escalation_trace(
    p_approve: float,
    value_approved: float = 1.0,
    attempt_cost: float = 0.05,
    human_cost: float = 0.30,
) -> Trace:
    """Trace the retry-vs-escalate decision and its break-even approval probability."""
    if not 0 <= p_approve <= 1:
        raise ValueError(f"p_approve must be in [0, 1], got {p_approve}")
    if value_approved <= 0:
        raise ValueError(f"value_approved must be > 0, got {value_approved}")

    t = Trace(
        op="loop_escalation",
        formula="retry if p·V − c_a > V − c_h  ⇔  p > p* = (V − c_h + c_a) / V",
        complexity="O(1)",
        why_ai=[
            "Escalation must always fire on exhaustion — never fail silently",
            "Turns 'retry forever' into a principled stopping decision",
            "Sets max_attempts / escalate_after from real costs, not a guess",
        ],
        meta={"p_approve": p_approve, "V": value_approved, "c_a": attempt_cost, "c_h": human_cost},
    )

    ev_retry = p_approve * value_approved - attempt_cost
    t.add(
        "EV(retry one more attempt)",
        f"p·V − c_a = {num(p_approve)}·{num(value_approved)} − "
        f"{num(attempt_cost)} = {num(ev_retry)}",
        ev_retry,
        detail="A retry pays off only if it likely approves and is cheap.",
    )

    ev_escalate = value_approved - human_cost
    t.add(
        "EV(escalate to a human)",
        f"V − c_h = {num(value_approved)} − {num(human_cost)} = {num(ev_escalate)}",
        ev_escalate,
    )

    p_star = (value_approved - human_cost + attempt_cost) / value_approved
    t.add(
        "Break-even approval probability p*",
        f"(V − c_h + c_a)/V = {num(p_star)}",
        p_star,
        detail="Below p*, escalate; above p*, another retry is worth it.",
    )

    decision = "retry" if p_approve > p_star else "escalate"
    t.add(
        "Decision",
        f"p {'>' if decision == 'retry' else '≤'} p* → {decision.upper()}",
        decision,
    )
    t.result = p_star
    return t


def loop_escalation(
    p_approve: float,
    value_approved: float = 1.0,
    attempt_cost: float = 0.05,
    human_cost: float = 0.30,
    explain: bool = False,
    level: str | ExplainLevel = ExplainLevel.INTERMEDIATE,
) -> float:
    """Return the break-even approval probability ``p*``. Set ``explain=True`` to print."""
    t = loop_escalation_trace(
        p_approve, value_approved=value_approved, attempt_cost=attempt_cost, human_cost=human_cost
    )
    return t.render(level) if explain else t.result


def demo(seed: int = 0) -> Trace:
    """A struggling attempt (p=0.2) weighed against a moderately costly human handoff."""
    return loop_escalation_trace(
        p_approve=0.20, value_approved=1.0, attempt_cost=0.05, human_cost=0.30
    )
