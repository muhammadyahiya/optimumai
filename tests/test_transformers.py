import numpy as np
import pytest

from optimumai.transformers import Attention


def _reference_attention(Q, K, V):
    scores = Q @ K.T / np.sqrt(Q.shape[1])
    shifted = scores - scores.max(axis=-1, keepdims=True)
    weights = np.exp(shifted) / np.exp(shifted).sum(axis=-1, keepdims=True)
    return weights @ V, weights


def test_attention_matches_reference():
    rng = np.random.default_rng(1)
    Q, K, V = (rng.normal(size=(3, 4)) for _ in range(3))
    out = Attention(d_k=4).forward(Q, K, V)
    expected, _ = _reference_attention(Q, K, V)
    assert np.allclose(out, expected)


def test_attention_weights_rows_sum_to_one():
    t = Attention.demo()
    weights_step = next(s for s in t if s.title.startswith("Softmax"))
    assert np.allclose(np.sum(weights_step.value, axis=-1), 1.0)


def test_attention_output_shape():
    Q = np.zeros((2, 8))
    K = np.zeros((5, 8))
    V = np.zeros((5, 3))
    out = Attention().forward(Q, K, V)
    assert out.shape == (2, 3)


def test_attention_has_four_stages():
    assert len(Attention.demo()) == 4


def test_attention_rejects_feature_mismatch():
    with pytest.raises(ValueError):
        Attention().trace(np.zeros((2, 4)), np.zeros((2, 5)), np.zeros((2, 4)))


def test_attention_rejects_token_mismatch():
    with pytest.raises(ValueError):
        Attention().trace(np.zeros((2, 4)), np.zeros((3, 4)), np.zeros((2, 4)))
