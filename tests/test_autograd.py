import math

import pytest

from optimumai.autograd import Value


def test_add_mul_grads():
    a = Value(2.0, label="a")
    b = Value(-3.0, label="b")
    c = Value(10.0, label="c")
    loss = a * b + c
    loss.backward()
    assert a.grad == pytest.approx(-3.0)  # dL/da = b
    assert b.grad == pytest.approx(2.0)   # dL/db = a
    assert c.grad == pytest.approx(1.0)   # dL/dc = 1


def test_tanh_grad():
    x = Value(0.8)
    y = x.tanh()
    y.backward()
    assert x.grad == pytest.approx(1 - math.tanh(0.8) ** 2)


def test_div_and_pow_grads():
    a = Value(6.0)
    b = Value(3.0)
    f = a / b
    f.backward()
    assert a.grad == pytest.approx(1 / 3)      # d(a/b)/da = 1/b
    assert b.grad == pytest.approx(-6 / 9)     # d(a/b)/db = -a/b²


def test_gradient_accumulates_for_reused_node():
    a = Value(3.0)
    y = a + a  # a used twice → dy/da = 2
    y.backward()
    assert a.grad == pytest.approx(2.0)


def test_numeric_gradient_check():
    def f(av, bv):
        A, B = Value(av), Value(bv)
        return A * B + A.tanh() - B**2

    out = f(1.3, -2.1)
    out.backward()
    # rebuild to grab the leaves' grads
    A, B = Value(1.3), Value(-2.1)
    out2 = A * B + A.tanh() - B**2
    out2.backward()
    h = 1e-6
    base = f(1.3, -2.1).data
    da = (f(1.3 + h, -2.1).data - base) / h
    db = (f(1.3, -2.1 + h).data - base) / h
    assert A.grad == pytest.approx(da, abs=1e-3)
    assert B.grad == pytest.approx(db, abs=1e-3)


def test_backward_trace_records_steps_and_result():
    a = Value(2.0, label="a")
    b = Value(3.0, label="b")
    loss = (a * b).tanh()
    loss.label = "L"
    t = loss.backward_trace()
    assert t.op == "backprop"
    assert len(t) >= 2          # at least a seed + some chain-rule steps
    assert t.result == pytest.approx(loss.data)


def test_zero_grad():
    a = Value(1.0)
    (a * 3).backward()
    assert a.grad != 0.0
    a.zero_grad()
    assert a.grad == 0.0


def test_exp_and_log_grads():
    x = Value(1.5)
    y = x.exp()
    y.backward()
    assert x.grad == pytest.approx(math.exp(1.5))  # d(eˣ)/dx = eˣ

    z = Value(4.0)
    w = z.log()
    w.backward()
    assert z.grad == pytest.approx(1 / 4.0)  # d(ln x)/dx = 1/x


def test_relu_gates():
    pos = Value(2.0)
    pos.relu().backward()
    assert pos.grad == pytest.approx(1.0)
    neg = Value(-2.0)
    neg.relu().backward()
    assert neg.grad == pytest.approx(0.0)


def test_reverse_and_neg_ops():
    a = Value(4.0)
    assert (2 + a).data == pytest.approx(6.0)
    assert (10 - a).data == pytest.approx(6.0)
    assert (3 * a).data == pytest.approx(12.0)
    assert (8 / a).data == pytest.approx(2.0)
    assert (-a).data == pytest.approx(-4.0)


def test_backprop_convenience_returns_self():
    a = Value(2.0, label="a")
    loss = a * a
    assert loss.backprop() is loss
    assert a.grad == pytest.approx(4.0)  # d(a²)/da = 2a

