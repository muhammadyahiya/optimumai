import pytest

from optimumai.autograd import Value
from optimumai.optimization import SGD, Adam, descent_demo, minimize, minimize_trace


def test_adam_finds_bowl_minimum():
    # f(x,y) = (x-3)² + (y+1)² has its minimum at (3, -1)
    final = descent_demo("adam", steps=100).result
    assert final[0] == pytest.approx(3.0, abs=0.05)
    assert final[1] == pytest.approx(-1.0, abs=0.05)


def test_sgd_reduces_loss():
    trace = descent_demo("sgd", steps=60)
    assert trace.steps[0].value > trace.last().value  # loss went down


def test_minimize_returns_params_and_lowers_loss():
    x = Value(5.0, label="x")

    def loss_fn():
        return (x - 2.0) ** 2

    params = minimize(loss_fn, [x], SGD([x], lr=0.1), steps=50)
    assert params[0] == pytest.approx(2.0, abs=0.1)


def test_adam_moment_state_advances():
    x = Value(0.0)
    opt = Adam([x], lr=0.1)

    def loss_fn():
        return (x - 1.0) ** 2

    minimize_trace(loss_fn, [x], opt, steps=10)
    assert opt.t == 10  # ten update steps recorded


def test_minimize_trace_has_convergence_step():
    trace = descent_demo("adam", steps=30)
    assert any(s.title == "converged" for s in trace)
