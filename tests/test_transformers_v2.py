import numpy as np
import pytest

from optimumai.transformers import (
    MultiHeadAttention,
    TransformerBlock,
    positional_encoding,
    positional_encoding_trace,
)


def test_multihead_output_shape():
    X = np.zeros((5, 8))
    out = MultiHeadAttention(n_heads=2, d_model=8).forward(X)
    assert out.shape == (5, 8)


def test_multihead_rejects_indivisible_dims():
    with pytest.raises(ValueError):
        MultiHeadAttention(n_heads=3, d_model=8)


def test_multihead_weights_rows_sum_to_one():
    t = MultiHeadAttention.demo()
    head_steps = [s for s in t if s.title.startswith("Head")]
    assert head_steps
    for s in head_steps:
        assert np.allclose(np.sum(s.value, axis=-1), 1.0)


def test_causal_mask_blocks_future():
    t = MultiHeadAttention.demo()  # demo uses causal=True
    head_steps = [s for s in t if s.title.startswith("Head")]
    for s in head_steps:
        weights = s.value
        # strictly-upper-triangular weights (future positions) must be ~0
        assert np.allclose(np.triu(weights, k=1), 0.0)


def test_positional_encoding_shape_and_range():
    pe = positional_encoding(6, 8)
    assert pe.shape == (6, 8)
    assert pe.max() <= 1.0 and pe.min() >= -1.0  # sin/cos bounded
    assert len(positional_encoding_trace(6, 8)) >= 3


def test_transformer_block_shape():
    out = TransformerBlock(d_model=8, n_heads=2).forward(np.zeros((4, 8)))
    assert out.shape == (4, 8)
    assert len(TransformerBlock.demo()) == 6  # LN, MHA, add, LN, FFN, add
