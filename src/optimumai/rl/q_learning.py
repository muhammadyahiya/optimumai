"""Q-learning and SARSA — learning to act *without* ever seeing P or R.

:mod:`optimumai.rl.mdp` solves an MDP when the transition model ``P`` and
reward function ``R`` are fully known. Real environments rarely hand you
those — an agent only gets to *interact*: take an action, observe the next
state and a reward, repeat. **Temporal-difference (TD) learning** estimates
the same Bellman fixed point from that raw experience, one transition at a
time, with no model of the world required.

Both algorithms below maintain a table ``Q(s, a)`` — the expected discounted
return of taking action ``a`` in state ``s`` and then acting well — and update
it toward a *bootstrapped target*: the observed reward plus the discounted
value of wherever the agent landed.

- **Q-learning** (off-policy):
  ``Q(s,a) ← Q(s,a) + α[ r + γ maxₐ' Q(s',a') − Q(s,a) ]``
  The target always uses the *best* next action, regardless of what the agent
  actually does next. It is learning about the greedy policy while *behaving*
  according to an exploratory one (e.g. ε-greedy) — hence "off-policy."

- **SARSA** (on-policy):
  ``Q(s,a) ← Q(s,a) + α[ r + γ Q(s',a') − Q(s,a) ]``
  The target uses the action the agent *actually takes* next (sampled from
  the same ε-greedy policy), not the best one. It is learning about — and
  therefore stays a little more cautious around — the policy it is actually
  following, which makes SARSA safer near penalties an ε-greedy explorer
  might stumble into.

Both converge to the optimal ``Q*`` (and hence the optimal greedy policy) in
the tabular setting under standard conditions (every state-action pair
visited infinitely often, a suitably decayed step size). Why AI cares: this
sample-based bootstrapping — no model, just "try it and update toward what
you observed" — is exactly what Deep Q-Networks (DQN) scale up with a neural
network in place of the table, and the on-policy/off-policy distinction
reappears throughout modern RL (PPO is on-policy; the replay buffers in DQN
exploit being off-policy).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from optimumai.core._fmt import arr, num
from optimumai.core.explain import ExplainLevel
from optimumai.core.trace import Trace

# Actions in the gridworld, laid out so index order is intuitive to read.
_ACTIONS = ("up", "down", "left", "right")
_MOVES = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}


@dataclass(frozen=True)
class GridWorld:
    """A tiny, fully deterministic grid: step off the edge and you don't move.

    The agent starts at ``start`` and the episode ends on reaching ``goal``
    (reward ``+goal_reward``) or a cell in ``traps`` (reward ``trap_reward``,
    also terminal). Every other step costs ``step_reward`` (encourages short
    paths). Deterministic transitions isolate the TD-learning behavior being
    taught here from the extra variance a stochastic MDP would add.
    """

    height: int = 3
    width: int = 3
    start: tuple[int, int] = (0, 0)
    goal: tuple[int, int] = (2, 2)
    traps: tuple[tuple[int, int], ...] = ((1, 1),)
    step_reward: float = -0.1
    goal_reward: float = 1.0
    trap_reward: float = -1.0

    @property
    def n_states(self) -> int:
        return self.height * self.width

    def state_index(self, pos: tuple[int, int]) -> int:
        row, col = pos
        return row * self.width + col

    def is_terminal(self, pos: tuple[int, int]) -> bool:
        return pos == self.goal or pos in self.traps

    def step(self, pos: tuple[int, int], action: str) -> tuple[tuple[int, int], float, bool]:
        """Apply ``action`` at ``pos``; return ``(next_pos, reward, done)``."""
        if self.is_terminal(pos):
            return pos, 0.0, True
        dr, dc = _MOVES[action]
        row, col = pos[0] + dr, pos[1] + dc
        row = min(max(row, 0), self.height - 1)
        col = min(max(col, 0), self.width - 1)
        new_pos = (row, col)
        if new_pos == self.goal:
            return new_pos, self.goal_reward, True
        if new_pos in self.traps:
            return new_pos, self.trap_reward, True
        return new_pos, self.step_reward, False


def _epsilon_greedy(q_row: np.ndarray, epsilon: float, rng: np.random.Generator) -> int:
    """Explore with probability ``epsilon``, else exploit the current best action."""
    if rng.random() < epsilon:
        return int(rng.integers(len(q_row)))
    return int(np.argmax(q_row))


def _greedy_policy(q_table: np.ndarray, world: GridWorld) -> dict[str, str]:
    policy = {}
    for row in range(world.height):
        for col in range(world.width):
            pos = (row, col)
            if world.is_terminal(pos):
                continue
            best = int(np.argmax(q_table[world.state_index(pos)]))
            policy[str(pos)] = _ACTIONS[best]
    return policy


def _run_td(
    world: GridWorld,
    algorithm: str,
    episodes: int,
    alpha: float,
    gamma: float,
    epsilon: float,
    seed: int,
    n_logged_updates: int,
) -> tuple[np.ndarray, list[str]]:
    """Shared TD-learning loop for both Q-learning and SARSA.

    Returns the final Q-table and a list of pre-formatted log lines for the
    first ``n_logged_updates`` TD updates (used by the trace builders below).
    """
    rng = np.random.default_rng(seed)
    q_table = np.zeros((world.n_states, len(_ACTIONS)))
    logs: list[str] = []
    total_updates = 0

    for _episode in range(episodes):
        pos = world.start
        s = world.state_index(pos)
        a = _epsilon_greedy(q_table[s], epsilon, rng)
        done = False
        while not done:
            new_pos, r, done = world.step(pos, _ACTIONS[a])
            s_next = world.state_index(new_pos)

            if algorithm == "q_learning":
                target = r + (0.0 if done else gamma * float(np.max(q_table[s_next])))
                a_next = None
            else:  # sarsa
                a_next = _epsilon_greedy(q_table[s_next], epsilon, rng)
                target = r + (0.0 if done else gamma * q_table[s_next, a_next])

            td_error = target - q_table[s, a]
            q_before = q_table[s, a]
            q_table[s, a] += alpha * td_error
            total_updates += 1

            if total_updates <= n_logged_updates:
                bootstrap = num(0.0) if done else num(target - r)
                logs.append(
                    f"({pos}, {_ACTIONS[a]}) → r={num(r)}, next={new_pos}   "
                    f"target = {num(r)} + γ·{bootstrap} = {num(target)}   "
                    f"Q ← {num(q_before)} + {num(alpha)}·({num(target)} − {num(q_before)}) "
                    f"= {num(q_table[s, a])}"
                )

            pos, s = new_pos, s_next
            a = a_next if algorithm == "sarsa" else _epsilon_greedy(q_table[s], epsilon, rng)

    return q_table, logs


def _td_trace(
    algorithm: str,
    world: GridWorld,
    episodes: int,
    alpha: float,
    gamma: float,
    epsilon: float,
    seed: int,
) -> Trace:
    if not (0.0 <= gamma < 1.0):
        raise ValueError(f"gamma must be in [0, 1), got {gamma}")
    if not (0.0 < alpha <= 1.0):
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")
    if not (0.0 <= epsilon <= 1.0):
        raise ValueError(f"epsilon must be in [0, 1], got {epsilon}")
    if episodes <= 0:
        raise ValueError(f"episodes must be > 0, got {episodes}")

    is_q_learning = algorithm == "q_learning"
    label = "Q-learning" if is_q_learning else "SARSA"
    bootstrap_term = (
        "r + γ·maxₐ' Q(s',a')" if is_q_learning else "r + γ·Q(s', a')  (a' actually taken)"
    )
    target_formula = f"target = {bootstrap_term}"

    t = Trace(
        op=algorithm,
        formula=f"Q(s,a) ← Q(s,a) + α[ {bootstrap_term} − Q(s,a) ]",
        complexity=f"O(|S||A|) memory; O(1) per step over {episodes} episodes",
        why_ai=[
            "Model-free TD learning — no P or R needed, only lived experience "
            "— is how Deep Q-Networks (Atari, and beyond) actually learn",
            "Off-policy (Q-learning) vs on-policy (SARSA) is a core RL design "
            "axis: can you learn from old/other-policy data, or only your own?",
            "The bootstrapped TD target (reward + discounted next value, not "
            "the full return) is what makes RL learn from partial episodes",
        ],
        meta={
            "algorithm": label,
            "alpha": alpha,
            "gamma": gamma,
            "epsilon": epsilon,
            "episodes": episodes,
            "grid": f"{world.height}x{world.width}",
        },
    )
    t.add(
        f"{label} target",
        target_formula,
        detail=(
            "Off-policy: the target bootstraps off the *greedy* next action, "
            "regardless of what ε-greedy actually does next."
            if is_q_learning
            else "On-policy: the target bootstraps off the *actual* next action "
            "sampled from the same ε-greedy policy being followed."
        ),
    )
    t.add(
        "The gridworld",
        f"{world.height}x{world.width} grid, start={world.start}, goal={world.goal} "
        f"(r={num(world.goal_reward)}), traps={world.traps} (r={num(world.trap_reward)}), "
        f"step reward={num(world.step_reward)}",
        detail="Deterministic transitions isolate the learning-rule behavior "
        "from environment stochasticity.",
    )

    q_table, logs = _run_td(
        world, algorithm, episodes, alpha, gamma, epsilon, seed, n_logged_updates=5
    )
    for i, line in enumerate(logs, start=1):
        t.add(f"TD update {i}", line, detail="One (s, a, r, s') transition and its Q update.")

    policy = _greedy_policy(q_table, world)
    t.add(
        f"Final Q-table after {episodes} episodes",
        arr(q_table),
        q_table.copy(),
        detail=f"Rows = states (row-major over the {world.height}x{world.width} grid), "
        f"columns = actions {_ACTIONS}.",
    )
    t.add(
        "Greedy policy π(s) = argmaxₐ Q(s,a)",
        "\n".join(f"π{pos} = {act}" for pos, act in policy.items()),
        detail="Acting greedily w.r.t. the learned Q-table — this is what the "
        "agent actually does once training stops (ε → 0).",
    )

    t.result = q_table
    t.meta["policy"] = policy
    return t


def q_learning_trace(
    world: GridWorld | None = None,
    episodes: int = 500,
    alpha: float = 0.5,
    gamma: float = 0.9,
    epsilon: float = 0.1,
    seed: int = 0,
) -> Trace:
    """Build the full trace of tabular Q-learning on ``world`` (default: a 3x3 grid)."""
    return _td_trace("q_learning", world or GridWorld(), episodes, alpha, gamma, epsilon, seed)


def sarsa_trace(
    world: GridWorld | None = None,
    episodes: int = 500,
    alpha: float = 0.5,
    gamma: float = 0.9,
    epsilon: float = 0.1,
    seed: int = 0,
) -> Trace:
    """Build the full trace of tabular SARSA on ``world`` (default: a 3x3 grid)."""
    return _td_trace("sarsa", world or GridWorld(), episodes, alpha, gamma, epsilon, seed)


def q_learning(
    world: GridWorld | None = None,
    episodes: int = 500,
    alpha: float = 0.5,
    gamma: float = 0.9,
    epsilon: float = 0.1,
    seed: int = 0,
    explain: bool = False,
    level: str | ExplainLevel = ExplainLevel.ENGINEER,
) -> np.ndarray:
    """Return the learned Q-table. ``explain=True`` prints the full trace."""
    t = q_learning_trace(world, episodes, alpha, gamma, epsilon, seed)
    return t.render(level) if explain else t.result


def sarsa(
    world: GridWorld | None = None,
    episodes: int = 500,
    alpha: float = 0.5,
    gamma: float = 0.9,
    epsilon: float = 0.1,
    seed: int = 0,
    explain: bool = False,
    level: str | ExplainLevel = ExplainLevel.ENGINEER,
) -> np.ndarray:
    """Return the learned Q-table. ``explain=True`` prints the full trace."""
    t = sarsa_trace(world, episodes, alpha, gamma, epsilon, seed)
    return t.render(level) if explain else t.result


def demo(seed: int = 0) -> Trace:
    """Return a ready-to-render Q-learning trace on the default 3x3 gridworld."""
    return q_learning_trace(seed=seed)
