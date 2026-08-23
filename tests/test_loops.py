"""Tests for the Agent Loops track (Loop Engineering math)."""

import pytest

from optimumai.core.trace import Trace
from optimumai.loops import (
    loop_budget,
    loop_budget_trace,
    loop_convergence,
    loop_convergence_trace,
    loop_escalation,
    loop_escalation_trace,
    loop_memory,
    loop_memory_trace,
    loop_state,
    loop_state_trace,
    loop_verification,
    loop_verification_trace,
)
from optimumai.loops.simulate import simulate_loop_trace

# ----- loop_budget -----

def test_loop_budget_soft_brake_stops_early():
    # 400-token attempts, 1000 cap, 50% brake → ceiling 500: run attempts 1 and 2,
    # then before attempt 3 cumulative 800 ≥ 500 → brake.
    assert loop_budget([400, 400, 400, 400], max_tokens=1000, pause_at_budget_pct=50) == 2


def test_loop_budget_no_brake_runs_all():
    assert loop_budget([100, 100, 100], max_tokens=100_000, pause_at_budget_pct=80) == 3


def test_loop_budget_validation():
    with pytest.raises(ValueError):
        loop_budget_trace([], max_tokens=1000)
    with pytest.raises(ValueError):
        loop_budget_trace([100], max_tokens=0)


# ----- loop_convergence -----

def test_loop_convergence_cdf():
    # 1 - (1-0.5)^3 = 0.875
    assert loop_convergence(0.5, max_attempts=3) == pytest.approx(0.875)


def test_loop_convergence_certain_when_p_one():
    assert loop_convergence(1.0, max_attempts=1) == pytest.approx(1.0)


def test_loop_convergence_validation():
    with pytest.raises(ValueError):
        loop_convergence_trace(0.0)
    with pytest.raises(ValueError):
        loop_convergence_trace(0.5, max_attempts=0)


# ----- loop_verification -----

def test_loop_verification_same_model_is_worse():
    # Independent slip = 0.3*(1-0.9)=0.03; correlated with rho=0.7:
    # c_eff=0.9*0.3=0.27, slip=0.3*0.73=0.219; ratio=7.3
    ratio = loop_verification(0.3, 0.9, 0.7)
    assert ratio == pytest.approx(7.3, rel=1e-3)


def test_loop_verification_independent_ratio_is_one():
    assert loop_verification(0.3, 0.9, 0.0) == pytest.approx(1.0)


def test_loop_verification_validation():
    with pytest.raises(ValueError):
        loop_verification_trace(1.5, 0.9, 0.0)


# ----- loop_memory -----

def test_loop_memory_ratio_between_zero_and_one():
    ratio = loop_memory([1000] * 10, keep_verbatim=3, digest_tokens=40)
    # verbatim 3000 + digest 7*40=280 → 3280/10000 = 0.328
    assert ratio == pytest.approx(0.328)


def test_loop_memory_validation():
    with pytest.raises(ValueError):
        loop_memory_trace([])


# ----- loop_escalation -----

def test_loop_escalation_break_even():
    # p* = (V - c_h + c_a)/V = (1 - 0.3 + 0.05)/1 = 0.75
    p_star = loop_escalation(0.2, value_approved=1.0, attempt_cost=0.05, human_cost=0.30)
    assert p_star == pytest.approx(0.75)


def test_loop_escalation_validation():
    with pytest.raises(ValueError):
        loop_escalation_trace(0.5, value_approved=0)


# ----- loop_state -----

def test_loop_state_approval_rate():
    log = [{"verdict": "APPROVE", "tokens": 1000}, {"verdict": "REJECT", "tokens": 2000},
           {"verdict": "APPROVE", "tokens": 1500}]
    assert loop_state(log) == pytest.approx(2 / 3)


def test_loop_state_validation():
    with pytest.raises(ValueError):
        loop_state_trace([])


# ----- simulate (trace-loop) -----

def test_simulate_loop_trace():
    t = simulate_loop_trace("AAPL", iterations=3, p_approve=0.55)
    assert isinstance(t, Trace)
    assert t.result == pytest.approx(1 - 0.45**3)


def test_simulate_rejects_identical_models():
    with pytest.raises(ValueError, match="must differ"):
        simulate_loop_trace(maker_model="m", checker_model="m")


# ----- every lesson returns a Trace with steps (mirrors test_curriculum) -----

@pytest.mark.parametrize("trace_fn", [
    lambda: loop_budget_trace([400, 400], max_tokens=1000),
    lambda: loop_convergence_trace(0.5),
    lambda: loop_verification_trace(0.3, 0.9, 0.7),
    lambda: loop_memory_trace([1000, 1000, 1000]),
    lambda: loop_escalation_trace(0.2),
    lambda: loop_state_trace([{"verdict": "APPROVE", "tokens": 1000}]),
])
def test_traces_have_steps(trace_fn):
    t = trace_fn()
    assert isinstance(t, Trace)
    assert len(t) >= 1
