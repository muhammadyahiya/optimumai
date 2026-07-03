import numpy as np
import pytest

from optimumai.rl import (
    MDP,
    BanditEnv,
    GridWorld,
    PPOSample,
    policy_iteration_trace,
    ppo_clip_trace,
    q_learning_trace,
    reinforce_trace,
    sarsa_trace,
    value_iteration_trace,
)
from optimumai.rl.mdp import make_corridor_mdp

# Independently hand/loop-computed fixed point for the corridor MDP (gamma=0.9),
# solved via the linear system for the "always right" policy: V = (I - gamma P)^-1 R.
# Cross-checked against 10,000 sweeps of value iteration from a very different
# initialization ([100, -100, 50, 0]); both agree to 1e-6.
_CORRIDOR_V_STAR = np.array([7.83589071, 8.80328463, 9.89010989, 0.0])


# --------------------------------------------------------------------------- mdp
def test_value_iteration_converges_to_known_optimal_values():
    mdp = make_corridor_mdp()
    t = value_iteration_trace(mdp)
    assert np.allclose(t.result, _CORRIDOR_V_STAR, atol=1e-4)


def test_value_iteration_extracts_optimal_policy():
    mdp = make_corridor_mdp()
    t = value_iteration_trace(mdp)
    # "right" progresses toward the goal in every non-terminal corridor state.
    assert t.meta["policy"] == {"s0": "right", "s1": "right", "s2": "right"}


def test_policy_iteration_matches_value_iteration():
    mdp = make_corridor_mdp()
    t = policy_iteration_trace(mdp)
    assert np.allclose(t.result, _CORRIDOR_V_STAR, atol=1e-4)
    assert t.meta["policy"] == {"s0": "right", "s1": "right", "s2": "right"}


def test_mdp_rejects_bad_gamma():
    states, actions = ["s0"], ["a0"]
    p = np.ones((1, 1, 1))
    r = np.zeros((1, 1, 1))
    with pytest.raises(ValueError):
        MDP(states=states, actions=actions, transition=p, reward=r, gamma=1.0)


def test_mdp_rejects_non_stochastic_transitions():
    states, actions = ["s0", "s1"], ["a0"]
    p = np.array([[[0.5, 0.4]], [[1.0, 0.0]]])  # first row sums to 0.9, not 1
    r = np.zeros((2, 1, 2))
    with pytest.raises(ValueError):
        MDP(states=states, actions=actions, transition=p, reward=r)


def test_value_iteration_trace_shape_and_metadata():
    t = value_iteration_trace(make_corridor_mdp())
    assert len(t) >= 3
    assert t.result is not None
    assert "Bellman" not in t.formula  # formula is the equation itself, not a name
    assert "maxₐ" in t.formula
    assert len(t.why_ai) >= 2


def test_value_iteration_rejects_bad_theta():
    with pytest.raises(ValueError):
        value_iteration_trace(make_corridor_mdp(), theta=0.0)


# --------------------------------------------------------------------- q_learning
def test_q_learning_learns_optimal_policy_on_gridworld():
    # Oracle: exact value iteration run directly against GridWorld.step (ground
    # truth transitions/rewards), independent of the tabular TD-learning code
    # path being tested. Every non-tied state has a *unique* best action; (0, 0)
    # has a genuine tie between "down" and "right" (two equally short safe paths
    # around the trap at (1, 1)), so it is excluded from the exact-match check.
    world = GridWorld()
    actions = ("up", "down", "left", "right")
    gamma = 0.9
    values = np.zeros(world.n_states)
    for _ in range(2000):
        new_values = values.copy()
        for row in range(world.height):
            for col in range(world.width):
                pos = (row, col)
                s = world.state_index(pos)
                if world.is_terminal(pos):
                    new_values[s] = 0.0
                    continue
                q_sa = []
                for a in actions:
                    nxt, r, done = world.step(pos, a)
                    ns = world.state_index(nxt)
                    q_sa.append(r + (0.0 if done else gamma * values[ns]))
                new_values[s] = max(q_sa)
        values = new_values

    oracle_policy = {}
    for row in range(world.height):
        for col in range(world.width):
            pos = (row, col)
            if world.is_terminal(pos):
                continue
            q_sa = []
            for a in actions:
                nxt, r, done = world.step(pos, a)
                ns = world.state_index(nxt)
                q_sa.append(r + (0.0 if done else gamma * values[ns]))
            oracle_policy[str(pos)] = actions[int(np.argmax(q_sa))]

    t = q_learning_trace(world, episodes=800, seed=0)
    learned_policy = t.meta["policy"]
    for state, action in oracle_policy.items():
        if state == "(0, 0)":
            continue  # genuine tie between "down" and "right"; skip
        assert learned_policy[state] == action, f"{state}: {learned_policy[state]} != {action}"


def test_sarsa_learns_optimal_greedy_policy_on_gridworld():
    world = GridWorld()
    t = sarsa_trace(world, episodes=800, seed=0)
    # Unambiguous (non-tied) states: SARSA should match Q-learning's greedy result.
    expected = {
        "(0, 1)": "right",
        "(0, 2)": "down",
        "(1, 0)": "down",
        "(1, 2)": "down",
        "(2, 0)": "right",
        "(2, 1)": "right",
    }
    for state, action in expected.items():
        assert t.meta["policy"][state] == action


def test_q_learning_off_policy_target_uses_max_next_q():
    t = q_learning_trace(episodes=10, seed=1)
    assert "maxₐ" in t.formula


def test_sarsa_on_policy_target_uses_actual_next_action():
    t = sarsa_trace(episodes=10, seed=1)
    assert "actually taken" in t.formula


def test_q_learning_trace_result_is_full_q_table():
    world = GridWorld()
    t = q_learning_trace(world, episodes=50, seed=0)
    assert t.result.shape == (world.n_states, 4)
    assert len(t.why_ai) >= 2


def test_q_learning_rejects_bad_alpha():
    with pytest.raises(ValueError):
        q_learning_trace(alpha=0.0)


def test_q_learning_rejects_bad_epsilon():
    with pytest.raises(ValueError):
        q_learning_trace(epsilon=1.5)


# ----------------------------------------------------------------- policy_gradient
def test_reinforce_shifts_probability_to_best_arm():
    env = BanditEnv(rewards=(0.1, 1.0, -0.5))
    t = reinforce_trace(env, episodes=300, alpha=0.1, seed=0)
    assert np.argmax(t.result) == 1
    assert t.result[1] > t.meta["initial_probs"][1]


def test_reinforce_result_is_a_valid_distribution():
    t = reinforce_trace(episodes=50, seed=0)
    assert np.sum(t.result) == pytest.approx(1.0)
    assert np.all(t.result > 0)


def test_reinforce_trace_shape_and_formula():
    t = reinforce_trace(episodes=20, seed=0)
    assert len(t) >= 4
    assert "∇_θ log π_θ(a|s)" in t.formula
    assert len(t.why_ai) >= 2


def test_reinforce_rejects_bad_alpha():
    with pytest.raises(ValueError):
        reinforce_trace(alpha=0.0)


def test_reinforce_rejects_bad_episodes():
    with pytest.raises(ValueError):
        reinforce_trace(episodes=0)


# --------------------------------------------------------------------------- ppo
def test_ppo_clip_matches_unclipped_inside_trust_region():
    # ratio = exp(-0.95 - (-1.0)) ≈ 1.051, inside [0.8, 1.2] -> clip has no effect.
    batch = [PPOSample(old_logprob=-1.0, new_logprob=-0.95, advantage=1.0)]
    t = ppo_clip_trace(batch, epsilon=0.2)
    ratio = float(np.exp(-0.95 - (-1.0)))
    unclipped = ratio * 1.0
    assert t.result == pytest.approx(-unclipped, abs=1e-9)
    assert t.meta["n_clipped"] == 0


def test_ppo_clip_caps_objective_outside_trust_region_positive_advantage():
    # ratio = exp(-0.30 - (-1.0)) ≈ 2.014, clipped to 1.2 -> clipped < unclipped,
    # so min() picks the clipped term and the objective *is* capped.
    batch = [PPOSample(old_logprob=-1.0, new_logprob=-0.30, advantage=1.0)]
    t = ppo_clip_trace(batch, epsilon=0.2)
    assert t.result == pytest.approx(-1.2, abs=1e-9)
    assert t.meta["n_clipped"] == 1


def test_ppo_clip_caps_objective_outside_trust_region_negative_advantage():
    # ratio = exp(-1.80 - (-1.0)) ≈ 0.449, clipped to 0.8 -> clipped(=-0.8) <
    # unclipped(=-0.449), so min() picks the clipped term: objective is capped.
    batch = [PPOSample(old_logprob=-1.0, new_logprob=-1.80, advantage=-1.0)]
    t = ppo_clip_trace(batch, epsilon=0.2)
    assert t.result == pytest.approx(0.8, abs=1e-9)
    assert t.meta["n_clipped"] == 1


def test_ppo_min_stays_pessimistic_when_clamping_would_only_help():
    # ratio ≈ 2.014 but advantage < 0: clipping the ratio down would *raise* the
    # term (less negative), so min() keeps the unclipped term -> objective is
    # NOT capped, even though the ratio itself sits outside [0.8, 1.2].
    batch = [PPOSample(old_logprob=-1.0, new_logprob=-0.30, advantage=-1.0)]
    t = ppo_clip_trace(batch, epsilon=0.2)
    ratio = float(np.exp(-0.30 - (-1.0)))
    unclipped = ratio * -1.0
    assert t.result == pytest.approx(-unclipped, abs=1e-9)
    assert t.meta["n_clipped"] == 0


def test_ppo_clip_trace_on_default_batch_hand_checked():
    t = ppo_clip_trace()
    # Hand-computed: of the 6 default samples, exactly indices 2 and 3 have the
    # ratio clamped *and* the min() select the clipped term.
    assert t.meta["n_clipped"] == 2
    assert t.result == pytest.approx(0.17739701191299084, abs=1e-8)


def test_ppo_clip_trace_shape_and_formula():
    t = ppo_clip_trace()
    assert len(t) >= 6  # batch intro + 6 samples + loss (+ comparison)
    assert "clip(rₜ" in t.formula
    assert len(t.why_ai) >= 2


def test_ppo_rejects_bad_epsilon():
    with pytest.raises(ValueError):
        ppo_clip_trace(epsilon=0.0)
    with pytest.raises(ValueError):
        ppo_clip_trace(epsilon=1.0)


def test_ppo_rejects_empty_batch():
    with pytest.raises(ValueError):
        ppo_clip_trace(batch=[])


# ---------------------------------------------------------------------- demo()s
def test_all_demos_are_deterministic_and_return_traces():
    # Import each concept module's demo directly by name (not via
    # `optimumai.rl.<name>` attribute access): every rl submodule's primary
    # function shares its module's name (q_learning.py -> q_learning()), which
    # rebinds the package attribute to the function once optimumai.rl is
    # imported — the same pattern used throughout the rest of the codebase
    # (see e.g. probability/softmax.py -> softmax()).
    from optimumai.rl.mdp import demo as mdp_demo
    from optimumai.rl.policy_gradient import demo as policy_gradient_demo
    from optimumai.rl.ppo import demo as ppo_demo
    from optimumai.rl.q_learning import demo as q_learning_demo

    for demo_fn in (mdp_demo, q_learning_demo, policy_gradient_demo, ppo_demo):
        t1 = demo_fn(seed=0)
        t2 = demo_fn(seed=0)
        assert t1.result is not None
        if isinstance(t1.result, np.ndarray):
            assert np.allclose(t1.result, t2.result)
        else:
            assert t1.result == pytest.approx(t2.result)
