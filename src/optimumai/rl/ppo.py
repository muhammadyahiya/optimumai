"""PPO — the clipped objective that makes policy gradients safe to reuse.

:mod:`optimumai.rl.policy_gradient` (REINFORCE) has a practical problem: its
gradient ``∇log π_θ(a|s) · G`` is only valid for data sampled from the
*current* ``π_θ``. Collect a batch, take one gradient step, and technically
you should throw the batch away and collect fresh data — reusing it, or
taking several optimizer steps on it, means training against a policy that
has since moved, which is unstable at best.

**Proximal Policy Optimization (PPO)** fixes this by explicitly correcting
for the mismatch with an *importance-sampling ratio*, then clamping how much
that correction is allowed to help:

    rₜ(θ) = exp( logπ_θ(aₜ|sₜ) − logπ_θ_old(aₜ|sₜ) )        (= π_θ / π_θ_old)

    L^CLIP(θ) = E[ min( rₜ·Aₜ,  clip(rₜ, 1−ε, 1+ε)·Aₜ ) ]

``Aₜ`` is the *advantage* — how much better this action was than the state's
average (a positive advantage means "do more of this," negative means "do
less"). Reading the ``min`` of two terms case by case:

- If ``Aₜ > 0`` (good action) and the ratio has already grown past ``1+ε``,
  the clipped term caps the objective — there is no extra reward for pushing
  the policy *even further* toward this action in a single update.
- If ``Aₜ < 0`` (bad action) and the ratio has already shrunk past ``1−ε``,
  the clip again caps the objective — no extra reward for suppressing it
  further right now.
- Inside the trust region ``rₜ ∈ [1−ε, 1+ε]``, clipping never engages and
  ``L^CLIP`` exactly equals the plain (unclipped) surrogate ``rₜ·Aₜ``.

The `min` (not the clipped term alone) matters too: it makes the objective a
*pessimistic* (lower) bound, so clipping only ever removes an incentive to
move further, never creates a new one to move backward. Net effect: many
optimizer steps, even multiple epochs, over the same batch of (old_logprob,
new_logprob, advantage) triples — safely, because the objective's gradient
vanishes once ``θ`` has moved "far enough" from ``θ_old`` on any given sample.

Why AI cares — and how this differs from DPO: PPO is stage 3 of classic RLHF
(see :mod:`optimumai.frontier.rlhf`) — the step that actually reinforcement-
learns the language model's sampling policy against a learned reward model,
token by token, with this exact clipped objective (plus a KL penalty to the
reference model). **DPO** was invented specifically to *avoid* this PPO loop:
it reparameterizes the same RLHF optimum into one closed-form classification
loss over preference pairs, trading PPO's on-policy-sampling machinery
(rollouts, advantages, clipping, reward-hacking risk) for a single supervised
loss — at the cost of PPO's ability to explore beyond the labeled pairs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from optimumai.core._fmt import num
from optimumai.core.explain import ExplainLevel
from optimumai.core.trace import Trace


@dataclass(frozen=True)
class PPOSample:
    """One (state, action) observation replayed from a rollout batch.

    Attributes:
        old_logprob: logπ_θ_old(a|s) — log-prob under the policy that
            generated this action (frozen for the whole batch).
        new_logprob: logπ_θ(a|s) — log-prob under the policy currently being
            optimized (changes across optimizer steps/epochs).
        advantage: Aₜ, how much better this action was than the state's
            baseline (e.g. from a critic / GAE — not recomputed here).
    """

    old_logprob: float
    new_logprob: float
    advantage: float


def _default_batch() -> list[PPOSample]:
    """A hand-built batch that exercises every clipping case at least once.

    Ratios and which case each sample hits (all with ``old_logprob=-1.0``):
        0. new=-0.95  ->  r≈1.05, A>0   — inside the trust region
        1. new=-1.05  ->  r≈0.95, A<0   — inside the trust region
        2. new=-0.30  ->  r≈2.01, A>0   — clip caps the upside
        3. new=-1.80  ->  r≈0.45, A<0   — clip caps the downside
        4. new=-0.30  ->  r≈2.01, A<0   — ratio clamped, but min() keeps unclipped
        5. new=-1.80  ->  r≈0.45, A>0   — ratio clamped, but min() keeps unclipped
    """
    return [
        PPOSample(old_logprob=-1.0, new_logprob=-0.95, advantage=1.0),
        PPOSample(old_logprob=-1.0, new_logprob=-1.05, advantage=-1.0),
        PPOSample(old_logprob=-1.0, new_logprob=-0.30, advantage=1.0),
        PPOSample(old_logprob=-1.0, new_logprob=-1.80, advantage=-1.0),
        PPOSample(old_logprob=-1.0, new_logprob=-0.30, advantage=-1.0),
        PPOSample(old_logprob=-1.0, new_logprob=-1.80, advantage=1.0),
    ]


def ppo_clip_trace(batch: list[PPOSample] | None = None, epsilon: float = 0.2) -> Trace:
    """Build the full trace of the PPO clipped surrogate objective on ``batch``.

    Walks every sample through the ratio, the unclipped and clipped terms,
    whether/why clipping engaged, and the final batch-averaged loss
    (``L = -mean(L^CLIP)`` since optimizers minimize).
    """
    if not (0.0 < epsilon < 1.0):
        raise ValueError(f"epsilon must be in (0, 1), got {epsilon}")

    samples = batch if batch is not None else _default_batch()
    if not samples:
        raise ValueError("batch must contain at least one sample")

    t = Trace(
        op="ppo_clip",
        formula=(
            "L^CLIP = E[ min( rₜ·Aₜ, clip(rₜ, 1−ε, 1+ε)·Aₜ ) ]      "
            "rₜ = exp(logπ_new − logπ_old)"
        ),
        complexity="O(batch size) — no rollout cost, pure objective evaluation",
        why_ai=[
            "This is the objective PPO actually maximizes during RLHF's "
            "reinforcement-learning stage, sample by sample, token by token",
            "Clipping lets you take multiple gradient steps (even epochs) over "
            "one batch of rollouts instead of one-and-done REINFORCE updates",
            "Contrast with DPO (frontier/rlhf.py): DPO replaces this entire "
            "ratio/advantage/clipping machinery with one closed-form loss "
            "over preference pairs — no rollouts, no clipping, no reward model",
        ],
        meta={"epsilon": epsilon, "batch_size": len(samples)},
    )
    t.add(
        "The batch",
        "\n".join(
            f"sample {i}: old_logp={num(s.old_logprob)}, new_logp={num(s.new_logprob)}, "
            f"A={num(s.advantage)}"
            for i, s in enumerate(samples)
        ),
        detail="Each row is one replayed (state, action) pair: its log-prob "
        "under the policy that generated it, its log-prob under the policy "
        "being optimized right now, and its advantage.",
    )

    unclipped_terms = []
    surrogate_terms = []
    n_clipped = 0
    for i, s in enumerate(samples):
        ratio = float(np.exp(s.new_logprob - s.old_logprob))
        unclipped = ratio * s.advantage
        clipped_ratio = float(np.clip(ratio, 1.0 - epsilon, 1.0 + epsilon))
        clipped = clipped_ratio * s.advantage
        surrogate = min(unclipped, clipped)
        ratio_clamped = clipped_ratio != ratio
        # The ratio can be clamped without the *objective* being capped: the min()
        # only bites when clamping the ratio actually lowers the term (see module
        # docstring — min() makes this a one-sided, pessimistic bound).
        objective_capped = surrogate == clipped and clipped < unclipped
        n_clipped += objective_capped

        unclipped_terms.append(unclipped)
        surrogate_terms.append(surrogate)

        if objective_capped:
            direction = (
                "upside (ratio too high, A>0)" if ratio > 1.0 else "downside (ratio too low, A<0)"
            )
            why = f"clipping engaged — caps {direction}: L^CLIP = clipped < unclipped"
        elif ratio_clamped:
            why = (
                "ratio is outside the trust region, but the min() still picks the "
                "*unclipped* term here (clipping would only help, not hurt, this "
                "sample, and the min is a pessimistic bound) — no effect on L^CLIP"
            )
        else:
            why = "inside the trust region — clip has no effect, L^CLIP = unclipped"

        t.add(
            f"Sample {i}: ratio, both terms, and the min",
            f"r = exp({num(s.new_logprob)} − {num(s.old_logprob)}) = {num(ratio)}\n"
            f"unclipped = r·A = {num(ratio)}·{num(s.advantage)} = {num(unclipped)}\n"
            f"clipped   = clip(r, {num(1 - epsilon)}, {num(1 + epsilon)})·A "
            f"= {num(clipped_ratio)}·{num(s.advantage)} = {num(clipped)}\n"
            f"L^CLIP_t = min(unclipped, clipped) = {num(surrogate)}",
            float(surrogate),
            detail=why,
        )

    mean_surrogate = float(np.mean(surrogate_terms))
    loss = -mean_surrogate
    t.add(
        "Batch objective and loss",
        f"L^CLIP = mean(L^CLIP_t) = {num(mean_surrogate)}   →   loss = -L^CLIP = {num(loss)}",
        float(loss),
        detail=f"{n_clipped}/{len(samples)} sample(s) had clipping engaged. "
        "Optimizers minimize, so the trainable loss is the negative of the "
        "objective PPO wants to maximize.",
    )

    vanilla_pg = float(np.mean(unclipped_terms))
    t.add(
        "Compare to vanilla policy gradient (no clip, no ratio)",
        f"REINFORCE-style objective (∇log π·A, ratio-free) ≈ {num(vanilla_pg)}   "
        f"vs. PPO's {num(mean_surrogate)}",
        detail="Without clipping, a large ratio (policy moved far from θ_old) "
        "lets one sample dominate the batch update — PPO's min/clip bounds "
        "how much any single sample's drift can move the objective.",
    )

    t.result = float(loss)
    t.meta["ratios"] = [float(np.exp(s.new_logprob - s.old_logprob)) for s in samples]
    t.meta["n_clipped"] = n_clipped
    t.meta["mean_objective"] = mean_surrogate
    return t


def ppo_clip(
    batch: list[PPOSample] | None = None,
    epsilon: float = 0.2,
    explain: bool = False,
    level: str | ExplainLevel = ExplainLevel.ENGINEER,
) -> float:
    """Return the PPO clipped loss for ``batch``. ``explain=True`` prints the trace."""
    t = ppo_clip_trace(batch, epsilon=epsilon)
    return t.render(level) if explain else t.result


def demo(seed: int = 0) -> Trace:
    """Return a ready-to-render PPO clipped-objective trace on a hand-built batch."""
    del seed  # the default batch is fixed on purpose: it exercises every clip case
    return ppo_clip_trace()
