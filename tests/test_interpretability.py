import numpy as np
import pytest

from optimumai.interpretability import superposition, superposition_trace


def test_superposition_recovers_shape():
    h_hat = superposition(5, 2)
    assert h_hat.shape == (5,)


def test_superposition_requires_more_features_than_neurons():
    with pytest.raises(ValueError):
        superposition_trace(2, 5)  # fits without packing → not superposition


def test_features_interfere_when_packed():
    t = superposition_trace(6, 2, seed=0)
    # more features than neurons ⇒ they cannot be orthogonal ⇒ Gram off-diagonals ≠ 0
    assert t.meta["max_offdiag"] > 0.0


def test_superposition_rejects_bad_sparsity():
    with pytest.raises(ValueError):
        superposition_trace(5, 2, sparsity=1.5)


def test_superposition_trace_shape_and_steps():
    t = superposition_trace(5, 2)
    assert len(t) == 4  # setup, gram, encode, decode
    assert np.asarray(t.result).shape == (5,)
