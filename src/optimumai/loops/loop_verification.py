"""Loop verification — why the checker must be a *different* model.

The maker fabricates on some fraction ``f`` of attempts. A checker catches a
fabrication with probability ``c`` — but only if it doesn't share the maker's
blind spot. Model the shared blind spot as an overlap ``ρ``: a same-model
checker's effective catch rate falls to ``c·(1 − ρ)``. This lesson computes the
probability a fabrication slips through for an independent checker (ρ = 0) vs a
correlated one, and the ratio between them — the quantitative case for
"checker ≠ maker".
"""

from __future__ import annotations

from optimumai.core._fmt import num
from optimumai.core.explain import ExplainLevel
from optimumai.core.trace import Trace


def loop_verification_trace(
    fabricate_rate: float,
    catch_rate: float,
    blind_spot_overlap: float,
) -> Trace:
    """Trace slip-through probability for an independent vs same-model checker."""
    for name, val in (("fabricate_rate", fabricate_rate), ("catch_rate", catch_rate),
                      ("blind_spot_overlap", blind_spot_overlap)):
        if not 0 <= val <= 1:
            raise ValueError(f"{name} must be in [0, 1], got {val}")

    t = Trace(
        op="loop_verification",
        formula="P(slip) = f · (1 − c_eff),   c_eff = c · (1 − ρ)",
        complexity="O(1)",
        why_ai=[
            "Same model = same training = same blind spots = fabrications approved",
            "An independent verifier (ρ→0) is the loop's core correctness guarantee",
            "Quantifies why you spend a second, different model on checking",
        ],
        meta={"f": fabricate_rate, "c": catch_rate, "rho": blind_spot_overlap},
    )

    slip_indep = fabricate_rate * (1.0 - catch_rate)
    t.add(
        "Independent checker (ρ = 0)",
        f"{num(fabricate_rate)} · (1 − {num(catch_rate)}) = {num(slip_indep)}",
        slip_indep,
        detail="A fresh model catches at its full rate c.",
    )

    c_eff = catch_rate * (1.0 - blind_spot_overlap)
    t.add(
        "Same-model checker: effective catch rate",
        f"c·(1 − ρ) = {num(catch_rate)} · (1 − {num(blind_spot_overlap)}) = {num(c_eff)}",
        c_eff,
        detail="Shared blind spots mean it cannot catch what it would also miss.",
    )

    slip_corr = fabricate_rate * (1.0 - c_eff)
    t.add(
        "Same-model checker: slip-through",
        f"{num(fabricate_rate)} · (1 − {num(c_eff)}) = {num(slip_corr)}",
        slip_corr,
    )

    ratio = slip_corr / slip_indep if slip_indep > 0 else float("inf")
    t.add(
        "How much worse is same-model checking?",
        f"slip_corr / slip_indep = {num(ratio)}×",
        ratio,
        detail="Fabrications slip through this many times more often with a shared model.",
    )
    t.result = ratio
    return t


def loop_verification(
    fabricate_rate: float,
    catch_rate: float,
    blind_spot_overlap: float,
    explain: bool = False,
    level: str | ExplainLevel = ExplainLevel.INTERMEDIATE,
) -> float:
    """Return the slip-through ratio (same-model / independent)."""
    t = loop_verification_trace(fabricate_rate, catch_rate, blind_spot_overlap)
    return t.render(level) if explain else t.result


def demo(seed: int = 0) -> Trace:
    """Maker fabricates 30% of the time; a good checker catches 90%; 70% blind-spot overlap."""
    return loop_verification_trace(fabricate_rate=0.30, catch_rate=0.90, blind_spot_overlap=0.70)
