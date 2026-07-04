import numpy as np
import pytest

from optimumai.augmented_rnns import act as act_mod
from optimumai.augmented_rnns import attention as attention_mod
from optimumai.augmented_rnns import ntm as ntm_mod
from optimumai.augmented_rnns.act import adaptive_computation_time, adaptive_computation_time_trace
from optimumai.augmented_rnns.attention import attention_read, attention_read_trace
from optimumai.augmented_rnns.ntm import NTMMemory, ntm_read, ntm_trace, ntm_write


# --- attention: content-based memory access -----------------------------------
def test_attention_read_matches_reference_implementation():
    rng = np.random.default_rng(0)
    memory = rng.normal(size=(5, 4))
    query = rng.normal(size=4)

    scores = memory @ query
    ref_weights = np.exp(scores - scores.max())
    ref_weights = ref_weights / ref_weights.sum()
    ref_read = ref_weights @ memory

    out = attention_read(query, memory)
    assert np.allclose(out, ref_read)


def test_attention_weights_sum_to_one():
    t = attention_read_trace(np.array([1.0, 0.0]), np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]))
    weights = t.steps[1].value
    assert np.sum(weights) == pytest.approx(1.0)
    assert np.all(weights > 0)


def test_attention_prefers_more_similar_row():
    # Row 0 is parallel to the query; row 1 is orthogonal -> row 0 should dominate.
    query = np.array([1.0, 0.0, 0.0])
    memory = np.array([[2.0, 0.0, 0.0], [0.0, 5.0, 0.0]])
    t = attention_read_trace(query, memory)
    weights = t.steps[1].value
    assert weights[0] > weights[1]


def test_attention_trace_shape_and_result():
    query = np.array([0.5, 0.5])
    memory = np.array([[1.0, 0.0], [0.0, 1.0]])
    t = attention_read_trace(query, memory)
    assert len(t) == 3  # score, softmax, blend
    assert t.formula
    assert len(t.why_ai) >= 1
    assert np.allclose(t.result, attention_read(query, memory))


def test_attention_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        attention_read_trace(np.array([1.0, 2.0, 3.0]), np.array([[1.0, 0.0], [0.0, 1.0]]))
    with pytest.raises(ValueError):
        attention_read_trace(np.array([[1.0, 2.0]]), np.array([[1.0, 0.0], [0.0, 1.0]]))


def test_attention_demo_runs():
    t = attention_mod.demo(seed=1)
    assert t.result.shape == (3,)


# --- ntm: content-based addressing, read, write -------------------------------
def _ref_cosine_weights(key, memory, beta):
    sims = (memory @ key) / (np.linalg.norm(memory, axis=1) * np.linalg.norm(key))
    scaled = beta * sims
    w = np.exp(scaled - scaled.max())
    return w / w.sum()


def test_ntm_read_matches_reference_cosine_addressing():
    rng = np.random.default_rng(2)
    memory = rng.normal(size=(4, 3))
    key = rng.normal(size=3)
    beta = 2.0

    ref_weights = _ref_cosine_weights(key, memory, beta)
    ref_read = ref_weights @ memory

    out = ntm_read(key, memory, beta=beta)
    assert np.allclose(out, ref_read)


def test_ntm_write_erase_then_add_matches_reference():
    rng = np.random.default_rng(3)
    memory = rng.normal(size=(3, 2))
    key = rng.normal(size=2)
    erase = np.array([0.5, 1.0])
    add = np.array([1.0, -1.0])
    beta = 1.5

    w = _ref_cosine_weights(key, memory, beta)
    ref_new_memory = memory * (1 - w[:, None] * erase[None, :]) + w[:, None] * add[None, :]

    out = ntm_write(memory, key, erase, add, beta=beta)
    assert np.allclose(out, ref_new_memory)


def test_ntm_memory_class_read_then_write_updates_state():
    memory = np.array([[1.0, 0.0], [0.0, 1.0]])
    mem = NTMMemory(memory, beta=1.0)
    key = np.array([1.0, 0.0])
    read_before = mem.read(key)
    mem.write(key, erase=np.array([1.0, 1.0]), add=np.array([5.0, 5.0]))
    assert not np.allclose(mem.memory, memory)
    assert read_before.shape == (2,)


def test_ntm_content_addressing_favors_closest_row():
    memory = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    key = np.array([1.0, 0.0])
    weights = _ref_cosine_weights(key, memory, beta=5.0)
    assert np.argmax(weights) == 0


def test_ntm_trace_shape_result_and_meta():
    memory = np.array([[1.0, 0.0], [0.0, 1.0]])
    read_key = np.array([1.0, 0.0])
    write_key = np.array([0.0, 1.0])
    erase = np.array([0.5, 0.5])
    add = np.array([2.0, 2.0])

    t = ntm_trace(memory, read_key, write_key, erase, add, beta=2.0)
    assert len(t) == 7
    assert t.formula
    assert len(t.why_ai) >= 1
    assert "read" in t.result and "memory" in t.result
    assert t.result["memory"].shape == memory.shape


def test_ntm_rejects_bad_beta_and_shapes():
    memory = np.array([[1.0, 0.0], [0.0, 1.0]])
    with pytest.raises(ValueError):
        ntm_read(np.array([1.0, 0.0]), memory, beta=0.0)
    with pytest.raises(ValueError):
        ntm_read(np.array([1.0, 0.0, 0.0]), memory)
    with pytest.raises(ValueError):
        ntm_write(memory, np.array([1.0, 0.0]), np.array([1.0]), np.array([1.0, 1.0]))


def test_ntm_demo_runs():
    t = ntm_mod.demo(seed=1)
    assert "read" in t.result


# --- act: adaptive computation time --------------------------------------------
def _ref_act(logits, eps):
    probs = 1.0 / (1.0 + np.exp(-logits))
    threshold = 1.0 - eps
    cumulative = np.cumsum(probs)
    idx = np.argmax(cumulative >= threshold) if np.any(cumulative >= threshold) else len(probs) - 1
    halt_step = idx + 1
    prior = cumulative[idx - 1] if idx > 0 else 0.0
    remainder = max(0.0, min(1.0, 1.0 - prior))
    return halt_step, remainder, halt_step + remainder


def test_act_matches_reference_halting():
    logits = np.array([-2.0, -1.0, 0.5, 3.0, 3.0])
    halt_step, remainder, ponder_cost = _ref_act(logits, eps=0.01)

    out = adaptive_computation_time(logits, eps=0.01)
    assert out["halt_step"] == halt_step
    assert out["remainder"] == pytest.approx(remainder)
    assert out["ponder_cost"] == pytest.approx(ponder_cost)


def test_act_weights_sum_to_one_including_remainder():
    logits = np.array([-1.0, 0.2, 0.8, 2.0])
    out = adaptive_computation_time(logits, eps=0.01)
    probs = out["halting_probs"].copy()
    probs[-1] = out["remainder"]
    assert np.sum(probs) == pytest.approx(1.0, abs=1e-9)


def test_act_halts_earlier_for_confident_logits():
    confident = np.array([5.0, 5.0, 5.0, 5.0])
    unsure = np.array([-3.0, -3.0, -3.0, -3.0, -3.0, -3.0])
    out_confident = adaptive_computation_time(confident, eps=0.01)
    out_unsure = adaptive_computation_time(unsure, eps=0.01)
    assert out_confident["halt_step"] < out_unsure["halt_step"]


def test_act_halts_at_last_step_when_logits_run_out():
    logits = np.array([-5.0, -5.0, -5.0])
    out = adaptive_computation_time(logits, eps=0.01)
    assert out["halt_step"] == 3


def test_act_trace_shape_result_and_meta():
    logits = np.array([-1.0, 0.5, 2.0])
    t = adaptive_computation_time_trace(logits, eps=0.01)
    assert len(t) == 6
    assert t.formula
    assert len(t.why_ai) >= 1
    ref_halt, ref_remainder, ref_cost = _ref_act(logits, eps=0.01)
    assert t.result["halt_step"] == ref_halt
    assert t.result["remainder"] == pytest.approx(ref_remainder)
    assert t.result["ponder_cost"] == pytest.approx(ref_cost)


def test_act_rejects_bad_input():
    with pytest.raises(ValueError):
        adaptive_computation_time_trace([], eps=0.01)
    with pytest.raises(ValueError):
        adaptive_computation_time_trace([1.0, 2.0], eps=0.0)
    with pytest.raises(ValueError):
        adaptive_computation_time_trace([1.0, 2.0], eps=1.0)


def test_act_demo_runs():
    t = act_mod.demo(seed=2)
    assert t.result["halt_step"] >= 1
