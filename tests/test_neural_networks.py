import pytest

from optimumai.autograd import Value
from optimumai.neural_networks import MLP, Neuron, forward_backward_trace, train_demo


def test_neuron_forward_is_value():
    n = Neuron(3, activation="tanh", seed=0)
    out = n([1.0, 2.0, 3.0])
    assert isinstance(out, Value)
    assert -1.0 <= out.data <= 1.0  # tanh range
    assert len(n.parameters()) == 4  # 3 weights + bias


def test_mlp_shape_and_param_count():
    mlp = MLP(3, [4, 4, 1], seed=0)
    out = mlp([1.0, 2.0, 3.0])
    assert isinstance(out, Value)
    # params: (3*4+4) + (4*4+4) + (4*1+1) = 16 + 20 + 5 = 41
    assert len(mlp.parameters()) == 41


def test_training_reduces_loss():
    trace = train_demo(steps=100, lr=0.05)
    assert trace.last().value < trace.steps[0].value * 0.5  # at least halved


def test_forward_backward_produces_gradients():
    mlp = MLP(3, [4, 1], seed=1)
    t = forward_backward_trace(mlp, [0.5, -1.0, 2.0], target=1.0)
    grad_norm = t.steps[3].value  # the "Backward pass" step logs ‖grad‖
    assert grad_norm > 0.0


def test_neuron_rejects_bad_activation():
    with pytest.raises(ValueError):
        Neuron(2, activation="sigmoid")
