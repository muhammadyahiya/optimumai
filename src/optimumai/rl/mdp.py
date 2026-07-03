"""Markov Decision Processes — the formal object reinforcement learning solves.

An MDP is a tuple ``(S, A, P, R, γ)``: a set of *states*, a set of *actions*, a
*transition model* ``P(s'|s, a)`` (how the world responds to an action), a
*reward function* ``R(s, a, s')``, and a *discount factor* ``γ ∈ [0, 1)`` that
trades off immediate vs. future reward. The agent does not choose rewards
directly — it chooses actions, and the environment (via ``P`` and ``R``)
decides what happens next. "Solving" an MDP means finding a *policy*
``π(a|s)`` that maximizes expected discounted return.

The key insight that makes MDPs solvable is the **Bellman optimality
equation**. The value of a state under the best possible behavior decomposes
into "the best immediate move" plus "the discounted value of wherever that
move leads":

    V*(s) = maxₐ Σₛ' P(s'|s, a) [ R(s, a, s') + γ V*(s') ]

This is a fixed-point equation — ``V*`` appears on both sides — and it has a
unique solution when ``γ < 1``. Two classic dynamic-programming algorithms find
it:

- **Value iteration** repeatedly applies the Bellman *backup* (the right-hand
  side, without the max being satisfied yet) as an update rule until the
  values stop changing (``Δ < θ``), then reads off the greedy policy.
- **Policy iteration** alternates between *policy evaluation* (solve for
  ``V^π`` exactly under the current policy) and *policy improvement* (make the
  policy greedy w.r.t. that ``V^π``), which also converges to ``V*, π*``.

Why AI cares: every RL algorithm — tabular Q-learning, deep Q-networks, PPO —
is, at its core, an approximate way to satisfy this same equation when ``P``
and ``R`` are unknown or the state space is too large to enumerate. The MDP is
the whiteboard model; the rest of RL is "how do we solve this without ever
writing down ``P``?"
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from optimumai.core._fmt import arr, num
from optimumai.core.explain import ExplainLevel
from optimumai.core.trace import Trace


@dataclass
class MDP:
    """A finite, discounted Markov Decision Process.

    Attributes:
        states: State labels, e.g. ``["s0", "s1", "s2"]``.
        actions: Action labels, e.g. ``["left", "right"]``.
        transition: ``P[s_idx][a_idx]`` is a probability vector over next-state
            indices — ``transition[s][a][s']`` = P(s'|s, a). Must sum to 1 per
            ``(s, a)``.
        reward: ``R[s_idx][a_idx][s'_idx]`` = immediate reward for that
            transition.
        gamma: Discount factor, ``0 <= gamma < 1``.
        terminal: Optional set of state indices with no outgoing value (their
            value is fixed at 0 and they are excluded from the max/backup).
    """

    states: list[str]
    actions: list[str]
    transition: np.ndarray
    reward: np.ndarray
    gamma: float = 0.9
    terminal: frozenset[int] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        self.transition = np.asarray(self.transition, dtype=float)
        self.reward = np.asarray(self.reward, dtype=float)
        n_s, n_a = len(self.states), len(self.actions)
        if self.transition.shape != (n_s, n_a, n_s):
            raise ValueError(
                f"transition must have shape ({n_s}, {n_a}, {n_s}), "
                f"got {self.transition.shape}"
            )
        if self.reward.shape != (n_s, n_a, n_s):
            raise ValueError(
                f"reward must have shape ({n_s}, {n_a}, {n_s}), got {self.reward.shape}"
            )
        if not (0.0 <= self.gamma < 1.0):
            raise ValueError(f"gamma must be in [0, 1), got {self.gamma}")
        row_sums = self.transition.sum(axis=-1)
        if not np.allclose(row_sums, 1.0):
            bad = np.argwhere(~np.isclose(row_sums, 1.0))
            raise ValueError(f"transition probabilities must sum to 1 per (s, a); bad at {bad}")

    @property
    def n_states(self) -> int:
        return len(self.states)

    @property
    def n_actions(self) -> int:
        return len(self.actions)

    def expected_backup(self, values: np.ndarray, s: int, a: int) -> float:
        """The Bellman backup Σₛ' P(s'|s,a)[R(s,a,s') + γV(s')] for one (s, a)."""
        p = self.transition[s, a]
        r = self.reward[s, a]
        return float(np.sum(p * (r + self.gamma * values)))


def value_iteration_trace(mdp: MDP, theta: float = 1e-6, max_iterations: int = 1000) -> Trace:
    """Build the full trace of value iteration converging to ``V*`` on ``mdp``.

    Repeatedly applies the Bellman optimality backup to every state, tracking
    the largest change ``Δ`` each sweep, and stops once ``Δ < theta``. The
    greedy policy w.r.t. the converged values is then extracted.
    """
    if theta <= 0:
        raise ValueError(f"theta must be > 0, got {theta}")
    if max_iterations <= 0:
        raise ValueError(f"max_iterations must be > 0, got {max_iterations}")

    n_s, n_a = mdp.n_states, mdp.n_actions
    values = np.zeros(n_s)

    t = Trace(
        op="value_iteration",
        formula="V*(s) = maxₐ Σₛ' P(s'|s,a) [ R(s,a,s') + γV*(s') ]",
        complexity="O(|S|²|A|) per sweep",
        why_ai=[
            "The Bellman optimality equation is the fixed point every RL "
            "algorithm is approximating, tabular or deep",
            "Value iteration is the textbook way to solve a *known* MDP "
            "exactly — the yardstick tabular Q-learning is checked against",
            "The same backup, applied to a learned/approximate model, underlies "
            "planning-based agents (MuZero-style world models)",
        ],
        meta={
            "gamma": mdp.gamma,
            "theta": theta,
            "states": list(mdp.states),
            "actions": list(mdp.actions),
        },
    )
    t.add(
        "Initialize",
        f"V(s) = 0 for all s  →  {arr(values)}",
        values.copy(),
        detail="Any starting point works; the Bellman backup is a contraction "
        "under γ < 1, so it converges regardless of initialization.",
    )

    n_iters = 0
    for i in range(1, max_iterations + 1):
        new_values = values.copy()
        deltas = []
        for s in range(n_s):
            if s in mdp.terminal:
                new_values[s] = 0.0
                continue
            q_sa = np.array([mdp.expected_backup(values, s, a) for a in range(n_a)])
            new_values[s] = float(np.max(q_sa))
            deltas.append(abs(new_values[s] - values[s]))
        delta = max(deltas) if deltas else 0.0
        n_iters = i
        if i <= 3 or delta < theta:
            t.add(
                f"Sweep {i}: backup every state",
                f"V(s) ← maxₐ backup(s,a)  →  {arr(new_values)}   (Δ = {num(delta)})",
                new_values.copy(),
                detail="Δ is the largest change across all states this sweep — "
                "the standard value-iteration stopping statistic.",
            )
        values = new_values
        if delta < theta:
            break

    q_final = np.array(
        [[mdp.expected_backup(values, s, a) for a in range(n_a)] for s in range(n_s)]
    )
    greedy_actions = np.argmax(q_final, axis=1)
    policy = {
        mdp.states[s]: mdp.actions[greedy_actions[s]] for s in range(n_s) if s not in mdp.terminal
    }

    t.add(
        f"Converged after {n_iters} sweeps",
        f"‖ΔV‖∞ < θ = {num(theta)}  →  V* = {arr(values)}",
        values.copy(),
        detail="Once no state's value changes by more than θ, we have (to "
        "numerical precision) the fixed point V*.",
    )
    t.add(
        "Extract the greedy policy π*(s) = argmaxₐ backup(s,a)",
        "\n".join(f"π*({s}) = {a}" for s, a in policy.items()),
        detail="Acting greedily w.r.t. V* is guaranteed optimal — this is the "
        "policy-improvement half of the Bellman equation.",
    )

    t.result = values
    t.meta["policy"] = policy
    t.meta["q_values"] = q_final
    t.meta["iterations"] = n_iters
    return t


def value_iteration(
    mdp: MDP,
    theta: float = 1e-6,
    max_iterations: int = 1000,
    explain: bool = False,
    level: str | ExplainLevel = ExplainLevel.ENGINEER,
) -> np.ndarray:
    """Return ``V*`` for ``mdp``. ``explain=True`` prints the full trace."""
    t = value_iteration_trace(mdp, theta=theta, max_iterations=max_iterations)
    return t.render(level) if explain else t.result


def policy_iteration_trace(mdp: MDP, theta: float = 1e-6, max_iterations: int = 1000) -> Trace:
    """Build the full trace of policy iteration converging to ``π*`` on ``mdp``.

    Alternates *policy evaluation* — solve the linear system ``V^π = R + γPV^π``
    exactly for the current (possibly suboptimal) policy — with *policy
    improvement* — make the policy greedy w.r.t. that ``V^π`` — until the
    policy stops changing. Guaranteed to converge in a finite number of
    iterations for a finite MDP.
    """
    if theta <= 0:
        raise ValueError(f"theta must be > 0, got {theta}")
    if max_iterations <= 0:
        raise ValueError(f"max_iterations must be > 0, got {max_iterations}")

    n_s, n_a = mdp.n_states, mdp.n_actions
    policy = np.zeros(n_s, dtype=int)  # start: always take action 0

    t = Trace(
        op="policy_iteration",
        formula="Evaluate: V^π = R^π + γP^πV^π   |   Improve: π(s) ← argmaxₐ backup(s,a)",
        complexity="O(|S|³) per evaluation (linear solve) × iterations",
        why_ai=[
            "Two-phase evaluate/improve loops recur throughout RL (e.g. actor-"
            "critic alternates a critic update with a policy update)",
            "Policy improvement is monotonic — V^{π_{k+1}} ≥ V^{π_k} everywhere "
            "— which is why the loop is guaranteed to terminate",
            "Exact policy evaluation (solving a linear system) is the small-MDP "
            "analogue of a learned critic/value network in deep RL",
        ],
        meta={"gamma": mdp.gamma, "theta": theta},
    )
    t.add(
        "Initialize an arbitrary policy",
        f"π(s) = {mdp.actions[0]!r} for all s",
        detail="Any starting policy works — improvement only ever helps.",
    )

    n_iters = 0
    for i in range(1, max_iterations + 1):
        n_iters = i
        # --- policy evaluation: solve V = R^pi + gamma * P^pi @ V exactly ---
        p_pi = np.array(
            [
                mdp.transition[s, policy[s]] if s not in mdp.terminal else np.zeros(n_s)
                for s in range(n_s)
            ]
        )
        r_pi = np.array(
            [
                mdp.expected_backup(np.zeros(n_s), s, policy[s]) if s not in mdp.terminal else 0.0
                for s in range(n_s)
            ]
        )
        a_mat = np.eye(n_s) - mdp.gamma * p_pi
        values = np.linalg.solve(a_mat, r_pi)
        t.add(
            f"Iteration {i}: evaluate π exactly",
            f"Solve (I − γP^π)V = R^π  →  V^π = {arr(values)}",
            values.copy(),
            detail="This is a linear system, not an iterative sweep — policy "
            "evaluation for a *fixed* policy has a closed-form solution.",
        )

        # --- policy improvement ---
        q_sa = np.array(
            [[mdp.expected_backup(values, s, a) for a in range(n_a)] for s in range(n_s)]
        )
        new_policy = np.argmax(q_sa, axis=1)
        changed = int(np.sum(new_policy != policy))
        t.add(
            f"Iteration {i}: improve π ← argmaxₐ backup(s,a)",
            f"π = {[mdp.actions[a] for a in new_policy]}   ({changed} state(s) changed action)",
            detail="If no action changes, π is already greedy w.r.t. its own "
            "value function, which is exactly the definition of optimal.",
        )
        if changed == 0:
            policy = new_policy
            break
        policy = new_policy

    policy_map = {
        mdp.states[s]: mdp.actions[policy[s]] for s in range(n_s) if s not in mdp.terminal
    }
    t.add(
        f"Converged after {n_iters} iteration(s)",
        f"π* = {policy_map}",
        detail="Policy iteration always terminates in finitely many steps for "
        "a finite MDP, since there are finitely many policies and each "
        "iteration strictly improves or stops.",
    )

    t.result = values
    t.meta["policy"] = policy_map
    t.meta["iterations"] = n_iters
    return t


def policy_iteration(
    mdp: MDP,
    theta: float = 1e-6,
    max_iterations: int = 1000,
    explain: bool = False,
    level: str | ExplainLevel = ExplainLevel.ENGINEER,
) -> np.ndarray:
    """Return ``V^π*`` for ``mdp``. ``explain=True`` prints the full trace."""
    t = policy_iteration_trace(mdp, theta=theta, max_iterations=max_iterations)
    return t.render(level) if explain else t.result


def make_corridor_mdp(gamma: float = 0.9) -> MDP:
    """A tiny 1-D corridor MDP used across the RL demos.

    Three non-terminal states ``s0 → s1 → s2`` plus a terminal ``goal``. Two
    actions: ``right`` (progress toward the goal, reward 0, +10 on entering
    ``goal``) and ``stay`` (no progress, reward -1, to make dawdling costly).
    ``right`` succeeds 90% of the time and fails to move 10% of the time
    (mirroring the classic "slippery" gridworld), so the MDP is genuinely
    stochastic — the reason we need the *expected* Bellman backup at all.
    """
    states = ["s0", "s1", "s2", "goal"]
    actions = ["right", "stay"]
    n_s, n_a = len(states), len(actions)
    p = np.zeros((n_s, n_a, n_s))
    r = np.zeros((n_s, n_a, n_s))

    for s in range(3):  # s0, s1, s2 (goal is index 3, terminal)
        nxt = s + 1
        p[s, 0, nxt] = 0.9  # "right" mostly succeeds
        p[s, 0, s] = 0.1  # ...and sometimes slips in place
        r[s, 0, nxt] = 10.0 if nxt == 3 else 0.0
        r[s, 0, s] = 0.0

        p[s, 1, s] = 1.0  # "stay" always stays
        r[s, 1, s] = -1.0

    p[3, 0, 3] = 1.0  # goal is absorbing regardless of action
    p[3, 1, 3] = 1.0

    return MDP(
        states=states,
        actions=actions,
        transition=p,
        reward=r,
        gamma=gamma,
        terminal=frozenset({3}),
    )


def demo(seed: int = 0) -> Trace:
    """Return a ready-to-render value-iteration trace on the corridor MDP."""
    del seed  # value iteration on this MDP is fully deterministic
    return value_iteration_trace(make_corridor_mdp())
