"""Reinforcement learning — how an agent learns to act from reward alone.

Four building blocks, each handed off to the next:

- :mod:`optimumai.rl.mdp` — the formal object (states, actions, transitions,
  rewards, discount) and the Bellman equation that "solves" it exactly when
  the environment model is known (value iteration, policy iteration).
- :mod:`optimumai.rl.q_learning` — model-free temporal-difference learning
  (Q-learning, SARSA) when the model is *not* known, only lived experience.
- :mod:`optimumai.rl.policy_gradient` — REINFORCE, which parameterizes the
  policy directly and differentiates through sampled actions via the
  score-function trick.
- :mod:`optimumai.rl.ppo` — PPO's clipped surrogate objective, the stabilized
  policy-gradient update that powers the reinforcement-learning stage of
  RLHF (contrasted with the closed-form DPO loss in
  :mod:`optimumai.frontier.rlhf`).
"""

from optimumai.rl.mdp import (
    MDP,
    policy_iteration,
    policy_iteration_trace,
    value_iteration,
    value_iteration_trace,
)
from optimumai.rl.policy_gradient import BanditEnv, reinforce, reinforce_trace
from optimumai.rl.ppo import PPOSample, ppo_clip, ppo_clip_trace
from optimumai.rl.q_learning import GridWorld, q_learning, q_learning_trace, sarsa, sarsa_trace

__all__ = [
    "MDP",
    "BanditEnv",
    "GridWorld",
    "PPOSample",
    "policy_iteration",
    "policy_iteration_trace",
    "ppo_clip",
    "ppo_clip_trace",
    "q_learning",
    "q_learning_trace",
    "reinforce",
    "reinforce_trace",
    "sarsa",
    "sarsa_trace",
    "value_iteration",
    "value_iteration_trace",
]
