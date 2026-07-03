import numpy as np
import pytest

from optimumai.frontier.flash_attention import demo as flash_demo
from optimumai.frontier.flash_attention import flash_attention
from optimumai.frontier.lora import lora_trace
from optimumai.frontier.quantization import dequantize, quantize, quantize_trace
from optimumai.frontier.rlhf import dpo, dpo_trace


def _standard_attention(Q, K, V):
    scale = 1.0 / np.sqrt(Q.shape[1])
    scores = Q @ K.T * scale
    shifted = scores - scores.max(axis=-1, keepdims=True)
    w = np.exp(shifted) / np.exp(shifted).sum(axis=-1, keepdims=True)
    return w @ V


# --- FlashAttention: must be EXACT --------------------------------------------
def test_flash_attention_is_exact():
    rng = np.random.default_rng(1)
    Q, K, V = (rng.normal(size=(6, 4)) for _ in range(3))
    out = flash_attention(Q, K, V, block_size=2)
    assert np.allclose(out, _standard_attention(Q, K, V), atol=1e-9)


def test_flash_attention_error_reported_tiny():
    assert flash_demo().meta["max_abs_error"] < 1e-9


# --- Quantization -------------------------------------------------------------
def test_quantize_roundtrip_error_bounded_by_scale():
    x = np.array([0.1, -2.3, 4.5, -0.02, 3.14])
    q, scale, zp = quantize(x, bits=8)
    x_hat = dequantize(q, scale, zp)
    assert np.max(np.abs(x - x_hat)) <= float(np.max(scale))


def test_int4_coarser_than_int8():
    x = np.linspace(-3, 3, 12)
    e8 = quantize_trace(x, bits=8).meta["max_error"]
    e4 = quantize_trace(x, bits=4).meta["max_error"]
    assert e4 >= e8  # fewer bits → coarser


def test_quantize_validation():
    with pytest.raises(ValueError):
        quantize_trace(np.array([1.0, 2.0]), bits=3)
    with pytest.raises(ValueError):
        quantize_trace(np.array([1.0, 2.0]), scheme="banana")


# --- LoRA ---------------------------------------------------------------------
def test_lora_reduces_parameters():
    t = lora_trace(d_in=8, d_out=8, rank=2)
    assert t.meta["lora_params"] < t.meta["full_params"]
    assert t.meta["reduction_factor"] > 1.0
    # r·(d_in+d_out) = 2·16 = 32 vs 64 → 2x
    assert t.meta["reduction_factor"] == pytest.approx(2.0)


# --- DPO ----------------------------------------------------------------------
def test_dpo_loss_nonnegative():
    assert dpo() >= 0.0


def test_dpo_trace_has_result_and_meta():
    t = dpo_trace()
    assert t.result >= 0.0
    assert "margin" in t.meta
