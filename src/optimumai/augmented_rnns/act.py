"""Adaptive Computation Time — letting an RNN decide how long to think.

A standard RNN spends exactly one computation step per input token, no matter
how easy or hard that token is to process. Graves (2016), the third pillar of
Distill's *Attention and Augmented Recurrent Neural Networks*, asked: why not
let the network *ponder* — run extra internal steps on hard inputs and stop
early on easy ones — and learn *when to stop* the same way it learns
everything else, by gradient descent?

The mechanism: at each ponder step ``t`` the network computes a scalar halting
probability from its state, ``hₜ = sigmoid(Wₕ · sₜ)``. These halting
probabilities accumulate across steps. The network keeps pondering — running
more internal steps on the *same* input, without consuming a new token — until
the running sum first reaches ``1 − ε`` (a small ``ε`` avoids needing
infinitely many steps to hit exactly 1):

    Σₜ hₜ  ≥  1 − ε

The step where this happens is the halt step ``N``. Because the partial sums
almost never land exactly on 1, there is a **remainder** ``R = 1 − Σ_{t<N} hₜ``
left over — this remainder, not the raw ``h_N``, is used as the final step's
weight, so all the weights across the ponder sequence still sum to exactly 1
(like a softmax, but built by accumulation instead of normalization, and
producing a *variable-length* stop instead of a fixed-length distribution).

The **ponder cost** ``N + R`` is added to the training loss, penalizing the
network for thinking longer than necessary — this is what makes "stop early
when possible" something the network is pushed to learn rather than exploit
(without it, always pondering the maximum number of steps would be free).

Where it led: ACT was one of the first mechanisms to make *compute itself* a
learned, differentiable, per-example quantity instead of a fixed
architectural constant. That idea — spend variable compute depending on
input difficulty — reappears throughout modern AI: early-exit networks,
mixture-of-depths transformers, and (in spirit) test-time / "reasoning"
compute scaling in today's LLMs all inherit ACT's core bet that a network
should decide, and be rewarded for deciding, how hard to think.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from optimumai.core._fmt import arr, num
from optimumai.core.trace import Trace


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def adaptive_computation_time(
    halting_logits: Iterable[float], eps: float = 0.01
) -> dict[str, float | int | np.ndarray]:
    """Run ACT's halting mechanism over a sequence of ponder-step logits.

    Args:
        halting_logits: Raw (pre-sigmoid) halting score at each ponder step,
            in order. Must be non-empty.
        eps: Small tolerance; pondering stops once the cumulative halting
            probability reaches ``1 - eps`` (or the logits run out).

    Returns:
        A dict with ``halting_probs`` (sigmoid of each logit up to and
        including the halt step), ``cumulative`` (running sum at each step),
        ``halt_step`` (1-based index of the step that triggered halting),
        ``remainder`` (weight given to the final step), and ``ponder_cost``
        (``halt_step + remainder``).
    """
    logits = np.asarray(list(halting_logits), dtype=float)
    if logits.ndim != 1 or logits.size == 0:
        raise ValueError(
            f"halting_logits must be a non-empty 1-D sequence, got shape {logits.shape}"
        )
    if not (0.0 < eps < 1.0):
        raise ValueError(f"eps must be in (0, 1), got {eps}")

    probs = _sigmoid(logits)
    threshold = 1.0 - eps

    cumulative = np.cumsum(probs)
    reach = np.nonzero(cumulative >= threshold)[0]
    halt_idx = int(reach[0]) if reach.size > 0 else probs.size - 1
    halt_step = halt_idx + 1  # 1-based

    prior_sum = float(cumulative[halt_idx - 1]) if halt_idx > 0 else 0.0
    remainder = float(np.clip(1.0 - prior_sum, 0.0, 1.0))
    ponder_cost = halt_step + remainder

    return {
        "halting_probs": probs[:halt_step],
        "cumulative": cumulative[:halt_step],
        "halt_step": halt_step,
        "remainder": remainder,
        "ponder_cost": ponder_cost,
    }


def adaptive_computation_time_trace(halting_logits: Iterable[float], eps: float = 0.01) -> Trace:
    """Build the full trace of ACT's ponder-and-halt mechanism.

    Shows the per-step halting probabilities, the cumulative sum, the step at
    which pondering halts, the leftover remainder, and the resulting ponder
    cost (the term added to the training loss).
    """
    logits = np.asarray(list(halting_logits), dtype=float)
    if logits.ndim != 1 or logits.size == 0:
        raise ValueError(
            f"halting_logits must be a non-empty 1-D sequence, got shape {logits.shape}"
        )
    if not (0.0 < eps < 1.0):
        raise ValueError(f"eps must be in (0, 1), got {eps}")

    t = Trace(
        op="adaptive_computation_time",
        formula="hₜ=σ(Wₕsₜ); halt at min N s.t. Σhₜ≥1−ε; R=1−Σ_{t<N}hₜ; cost=N+R",
        complexity="O(T) for up to T ponder steps",
        why_ai=[
            "Lets a recurrent network spend more internal computation steps on "
            "harder inputs and fewer on easy ones",
            "The ponder cost N+R is added to the loss, so 'think less when you "
            "can' becomes a learned, gradient-driven incentive",
            "An early example of *learned, variable* compute — the ancestor in "
            "spirit of early-exit nets and today's test-time compute scaling",
        ],
        meta={"eps": eps, "n_logits": int(logits.size)},
    )

    t.add(
        "Halting logits at each ponder step",
        f"Wₕ · sₜ (raw, pre-sigmoid)\n{arr(logits)}",
        logits,
        detail="One scalar logit per internal ponder step on this input.",
    )

    probs = _sigmoid(logits)
    t.add(
        "Halting probabilities: hₜ = sigmoid(logitₜ)",
        f"hₜ  →  {arr(probs)}",
        probs,
        detail="Each hₜ ∈ (0, 1) — how much the network 'wants' to stop at step t.",
    )

    threshold = 1.0 - eps
    cumulative = np.cumsum(probs)
    t.add(
        "Accumulate: cumulative sum of hₜ",
        f"Σ_{{i≤t}} hᵢ  →  {arr(cumulative)}",
        cumulative,
        detail=f"Pondering stops once this running sum reaches 1 − ε = {num(threshold)}.",
    )

    reach = np.nonzero(cumulative >= threshold)[0]
    halt_idx = int(reach[0]) if reach.size > 0 else probs.size - 1
    halt_step = halt_idx + 1
    ran_out = reach.size == 0
    t.add(
        "Find the halt step",
        f"first t with cumulative[t] ≥ {num(threshold)}  →  step {halt_step}",
        halt_step,
        detail=(
            "Logits were exhausted before crossing the threshold; halting at "
            "the last available step."
            if ran_out
            else f"Cumulative sum first reaches {num(float(cumulative[halt_idx]))} "
            f"at step {halt_step}."
        ),
    )

    prior_sum = float(cumulative[halt_idx - 1]) if halt_idx > 0 else 0.0
    remainder = float(np.clip(1.0 - prior_sum, 0.0, 1.0))
    t.add(
        "Compute the remainder: R = 1 − Σ_{t<N} hₜ",
        f"R = 1 − {num(prior_sum)} = {num(remainder)}",
        remainder,
        detail="R replaces h_N as the halt step's weight, so all weights across "
        "the ponder sequence sum to exactly 1.",
    )

    ponder_cost = halt_step + remainder
    t.add(
        "Ponder cost: N + R",
        f"{halt_step} + {num(remainder)} = {num(ponder_cost)}",
        ponder_cost,
        detail="Added to the training loss — the network is penalized for "
        "pondering longer than it needs to.",
    )
    t.result = {
        "halting_probs": probs[:halt_step],
        "cumulative": cumulative[:halt_step],
        "halt_step": halt_step,
        "remainder": remainder,
        "ponder_cost": ponder_cost,
    }
    return t


def demo(seed: int = 0) -> Trace:
    """A tiny, reproducible 6-step ponder sequence for docs and the CLI."""
    rng = np.random.default_rng(seed)
    logits = rng.normal(loc=0.5, scale=1.0, size=6).round(2)
    return adaptive_computation_time_trace(logits, eps=0.01)
