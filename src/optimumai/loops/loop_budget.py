"""Loop budget — the token accounting behind a bounded agent loop.

An agent loop retries until an adversarial checker approves, so its cost is
*unbounded in principle*. Two numbers bound it in practice: a hard cap
``max_tokens_per_run`` and a **soft brake** that halts new attempts once usage
crosses ``pause_at_budget_pct`` of that cap. This lesson accumulates the
per-attempt token cost, finds where the soft brake trips, and reports how much
of the budget was actually spent.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from optimumai.core._fmt import num
from optimumai.core.explain import ExplainLevel
from optimumai.core.trace import Trace


def loop_budget_trace(
    attempt_tokens: Iterable[float],
    max_tokens: float = 120_000,
    pause_at_budget_pct: float = 80.0,
) -> Trace:
    """Trace cumulative token usage against the soft brake / hard cap.

    Args:
        attempt_tokens: Tokens consumed by each successive attempt (maker+checker).
        max_tokens: The hard per-run budget cap.
        pause_at_budget_pct: Soft-brake threshold as a percent of ``max_tokens``.
    """
    costs = [float(c) for c in attempt_tokens]
    if not costs:
        raise ValueError("attempt_tokens must be non-empty")
    if max_tokens <= 0:
        raise ValueError(f"max_tokens must be > 0, got {max_tokens}")
    if not 0 < pause_at_budget_pct <= 100:
        raise ValueError(f"pause_at_budget_pct must be in (0, 100], got {pause_at_budget_pct}")

    ceiling = max_tokens * pause_at_budget_pct / 100.0
    t = Trace(
        op="loop_budget",
        formula="brake when Σ tokensᵢ ≥ max_tokens · pause_pct/100",
        complexity="O(attempts)",
        why_ai=[
            "A verified loop retries until it passes — cost is unbounded without a cap",
            "The soft brake stops new attempts before the hard cap is breached",
            "Budget is a hard guarantee, not a suggestion: it bounds worst-case spend",
        ],
        meta={"max_tokens": max_tokens, "pause_at_budget_pct": pause_at_budget_pct},
    )
    t.add(
        "Compute the soft-brake ceiling",
        f"{num(max_tokens)} · {num(pause_at_budget_pct)}/100 = {num(ceiling)} tokens",
        ceiling,
        detail="New attempts stop once cumulative usage reaches this ceiling.",
    )

    cumulative = 0.0
    attempts_run = 0
    braked = False
    for i, cost in enumerate(costs, start=1):
        # The brake is checked *before* an attempt: if we're already at/over the
        # ceiling, this attempt does not run.
        if cumulative >= ceiling:
            t.add(
                f"Before attempt {i}: soft brake trips",
                f"cumulative {num(cumulative)} ≥ ceiling {num(ceiling)} → halt",
                cumulative,
                detail="The loop escalates rather than spend past the brake.",
            )
            braked = True
            break
        cumulative += cost
        attempts_run += 1
        t.add(
            f"Run attempt {i}",
            f"+{num(cost)} → cumulative {num(cumulative)} "
            f"({num(cumulative / max_tokens * 100)}% of cap)",
            cumulative,
        )

    pct_used = cumulative / max_tokens * 100.0
    t.add(
        "Report budget outcome",
        f"{attempts_run} attempt(s) ran, {num(pct_used)}% of the hard cap used"
        + ("" if braked else "; brake not reached"),
        attempts_run,
        detail=f"Hard cap {num(max_tokens)}; ceiling {num(ceiling)}; spent {num(cumulative)}.",
    )
    t.result = attempts_run
    return t


def loop_budget(
    attempt_tokens: Iterable[float],
    max_tokens: float = 120_000,
    pause_at_budget_pct: float = 80.0,
    explain: bool = False,
    level: str | ExplainLevel = ExplainLevel.INTERMEDIATE,
) -> int:
    """Return the number of attempts that run before the soft brake trips."""
    t = loop_budget_trace(
        attempt_tokens, max_tokens=max_tokens, pause_at_budget_pct=pause_at_budget_pct
    )
    return t.render(level) if explain else t.result


def demo(seed: int = 0) -> Trace:
    """A loop whose attempts each cost ~4k tokens against a 20k cap, 80% brake."""
    rng = np.random.default_rng(seed)
    costs = (rng.normal(4000, 300, size=6)).round().tolist()
    return loop_budget_trace(costs, max_tokens=20_000, pause_at_budget_pct=80.0)
