import pytest

from optimumai.calculus import chain_rule_trace, derivative, derivative_trace, gradient


def test_derivative_of_cube():
    # d/dx x³ = 3x² → at x=2, = 12
    assert derivative(lambda x: x**3, 2.0) == pytest.approx(12.0, abs=1e-3)


def test_derivative_trace_steps():
    t = derivative_trace(lambda x: x * x, 3.0)
    assert len(t) == 3  # f(x+h), f(x-h), central difference
    assert t.result == pytest.approx(6.0, abs=1e-3)


def test_gradient_of_sum_of_squares():
    # ∇(x²+y²) = [2x, 2y] → at (3,4) = [6, 8]
    g = gradient(lambda p: p[0] ** 2 + p[1] ** 2, [3.0, 4.0])
    assert g[0] == pytest.approx(6.0, abs=1e-3)
    assert g[1] == pytest.approx(8.0, abs=1e-3)


def test_chain_rule_matches_analytic():
    x = 1.5
    t = chain_rule_trace(x)
    # dy/dx = (1 - tanh²(x²)) * 2x
    import math

    expected = (1 - math.tanh(x * x) ** 2) * 2 * x
    assert t.result == pytest.approx(expected, abs=1e-4)
