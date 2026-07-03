"""REINFORCE — the simplest way to differentiate *through* an agent's actions.

Value iteration and Q-learning both learn a *value* function and act greedily
w.r.t. it — the policy is implicit. **Policy gradient** methods instead
parameterize the policy directly, ``π_θ(a|s)``, and adjust ``θ`` by gradient
ascent on expected return. The question is: how do you get a gradient through
a *sampling* step (the agent's action is drawn from a distribution, not
computed by a differentiable formula)?

The **REINFORCE** (score-function) estimator answers this with one identity:

    ∇_θ E[R] = E[ ∇_θ log π_θ(a|s) · R ]

Intuition: you cannot differentiate "which action got sampled," but you *can*
differentiate "how likely was the action I happen to have sampled" — that's
``∇_θ log π_θ(a|s)``, the **score function**. Weighting that gradient by the
return ``R`` (or, per-timestep, the return-to-go ``Gₜ``) means: nudge up the
log-probability of actions that led to high return, nudge down the ones that
led to low return. No value function, no model of the environment — just
"try things, reweight by how well they went."

The concrete recipe (Monte-Carlo policy gradient) for one trajectory:

1. Roll out the current policy: sample ``a_t ~ π_θ(·|s_t)`` for each step.
2. Compute the return-to-go ``Gₜ = Σ_{k≥t} γ^{k−t} r_k`` for every timestep.
3. Estimate the gradient ``Σ_t ∇_θ log π_θ(a_t|s_t) · Gₜ``.
4. Take a gradient-ascent step: ``θ ← θ + α · gradient``.

For a **softmax policy** over logits ``θ_s`` (one logit per action in state
``s``), the score function has a clean closed form:
``∇_{θ_a} log π(a|s) = 1[a taken] − π(a|s)`` — push the taken action's logit
up, every other action's logit down, both scaled by ``Gₜ``.

Why AI cares: REINFORCE is the ancestor of every LLM alignment method that
touches sampled text — RLHF's PPO stage (see :mod:`optimumai.rl.ppo` and
:mod:`optimumai.frontier.rlhf`) is a variance-reduced, more stable descendant
of exactly this score-function trick, applied to next-token sampling instead
of grid moves.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from optimumai.core._fmt import arr, num
from optimumai.core.explain import ExplainLevel
from optimumai.core.trace import Trace
from optimumai.probability.softmax import softmax


@dataclass(frozen=True)
class BanditEnv:
    """A stateless k-armed bandit: pick an arm, get a fixed reward.

    Deterministic rewards keep the lesson focused on how REINFORCE shifts
    probability mass toward the best action, without the extra variance a
    stochastic reward would add on top of the policy's own sampling noise.
    """

    rewards: tuple[float, ...] = (0.1, 1.0, -0.5)

    @property
    def n_actions(self) -> int:
        return len(self.rewards)

    def step(self, action: int) -> float:
        return self.rewards[action]


def reinforce_trace(
    env: BanditEnv | None = None,
    episodes: int = 200,
    alpha: float = 0.1,
    gamma: float = 0.99,
    seed: int = 0,
) -> Trace:
    """Build the full trace of REINFORCE learning a softmax policy on ``env``.

    Each "episode" here is a single-step trajectory (state, action, reward) —
    enough to show the sample → return → score-function-gradient → update
    loop without the bookkeeping of multi-step return-to-go, which
    :func:`optimumai.rl.ppo.ppo_trace` picks up with real advantages instead.
    """
    if alpha <= 0:
        raise ValueError(f"alpha must be > 0, got {alpha}")
    if not (0.0 <= gamma <= 1.0):
        raise ValueError(f"gamma must be in [0, 1], got {gamma}")
    if episodes <= 0:
        raise ValueError(f"episodes must be > 0, got {episodes}")

    bandit = env or BanditEnv()
    rng = np.random.default_rng(seed)
    logits = np.zeros(bandit.n_actions)

    t = Trace(
        op="reinforce",
        formula="∇_θ J(θ) = E[ ∇_θ log π_θ(a|s) · Gₜ ]      θ ← θ + α·∇_θ J(θ)",
        complexity=f"O(|A|) per step over {episodes} episodes",
        why_ai=[
            "The score-function trick ∇log π(a)·R is how you get a gradient "
            "through a *sampled* action — no value function required",
            "This is the mathematical ancestor of the PPO stage of RLHF: same "
            "idea, applied to sampled tokens instead of bandit arms",
            "Softmax-over-logits policies are exactly how an LLM samples its "
            "next token, and 1[a taken] − π(a) is that softmax's gradient",
        ],
        meta={"alpha": alpha, "gamma": gamma, "episodes": episodes, "rewards": bandit.rewards},
    )

    initial_probs = softmax(logits)
    t.add(
        "Initial policy π_θ(a) = softmax(θ)",
        f"θ = {arr(logits)}  →  π = {arr(initial_probs)}",
        initial_probs.copy(),
        detail="Uniform logits start as a uniform (maximum-entropy) policy — "
        "no action is preferred yet.",
    )
    t.add(
        "The bandit's rewards",
        f"R(a) = {arr(np.asarray(bandit.rewards))}",
        detail=f"Arm {int(np.argmax(bandit.rewards))} is best; REINFORCE never "
        "sees this list directly, only sampled (action, reward) pairs.",
    )

    n_logged = 3
    for ep in range(1, episodes + 1):
        probs = softmax(logits)
        action = int(rng.choice(bandit.n_actions, p=probs))
        reward = bandit.step(action)  # single-step episode: Gₜ = r_t (no future to discount)

        # Score function for a softmax policy: d/dθ_a' log π(a) = 1[a'=a] − π(a').
        grad_log_pi = -probs.copy()
        grad_log_pi[action] += 1.0
        gradient = grad_log_pi * reward
        logits = logits + alpha * gradient

        if ep <= n_logged:
            t.add(
                f"Episode {ep}: sample, score, update",
                f"a ~ π  →  a={action} (r={num(reward)})\n"
                f"∇log π(a|s) = 1[a] − π = {arr(grad_log_pi)}\n"
                f"θ ← θ + α·∇log π·G  →  θ += {num(alpha)}·{arr(gradient)}  →  θ = {arr(logits)}",
                logits.copy(),
                detail="G = r here since each episode is one step (no discounting "
                "needed); a multi-step episode would use the return-to-go Gₜ instead.",
            )

    final_probs = softmax(logits)
    t.add(
        f"Policy after {episodes} episodes",
        f"θ = {arr(logits)}  →  π = {arr(final_probs)}",
        final_probs.copy(),
        detail="Probability mass has shifted toward the highest-reward arm — "
        "exactly what ascending E[∇log π · R] should do.",
    )

    t.result = final_probs
    t.meta["initial_probs"] = initial_probs
    t.meta["final_logits"] = logits
    t.meta["best_action"] = int(np.argmax(bandit.rewards))
    return t


def reinforce(
    env: BanditEnv | None = None,
    episodes: int = 200,
    alpha: float = 0.1,
    gamma: float = 0.99,
    seed: int = 0,
    explain: bool = False,
    level: str | ExplainLevel = ExplainLevel.ENGINEER,
) -> np.ndarray:
    """Return the learned action-probability vector. ``explain=True`` prints the trace."""
    t = reinforce_trace(env, episodes, alpha, gamma, seed)
    return t.render(level) if explain else t.result


def demo(seed: int = 0) -> Trace:
    """Return a ready-to-render REINFORCE trace on the default 3-armed bandit."""
    return reinforce_trace(seed=seed)
