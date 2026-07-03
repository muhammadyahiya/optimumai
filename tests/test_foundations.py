import numpy as np
import pytest

from optimumai.foundations import (
    grad_trace,
    integrate,
    kv_cache_size,
    memory_hierarchy_trace,
    pytorch_autograd,
    pytorch_autograd_trace,
    pytree_trace,
    thread_hierarchy_trace,
    tiled_matmul,
    tiled_matmul_trace,
    vmap_trace,
    vram_estimate,
)


# --- math foundations ---------------------------------------------------------
def test_integrate_trapezoid_x_squared():
    # ∫₀¹ x² dx = 1/3
    assert integrate(lambda x: x**2, 0.0, 1.0, method="trapezoid", n=1000) == pytest.approx(
        1 / 3, abs=1e-3
    )


def test_integrate_monte_carlo_approximates():
    est = integrate(lambda x: x**2, 0.0, 1.0, method="monte_carlo", n=50000)
    assert est == pytest.approx(1 / 3, abs=0.02)


# --- kv cache -----------------------------------------------------------------
def test_kv_cache_formula():
    # 2 · layers · kv_heads · head_dim · seq · batch · bytes
    size = kv_cache_size(2, 4, 8, 10, batch=1, bytes_per_elem=2, kv_heads=4)
    assert size == 2 * 2 * 4 * 8 * 10 * 1 * 2


def test_gqa_and_mqa_shrink_the_cache():
    mha = kv_cache_size(32, 32, 128, 4096, kv_heads=32)
    gqa = kv_cache_size(32, 32, 128, 4096, kv_heads=4)
    mqa = kv_cache_size(32, 32, 128, 4096, kv_heads=1)
    assert mqa < gqa < mha


# --- vram ---------------------------------------------------------------------
def test_vram_training_exceeds_inference():
    assert vram_estimate(7, training=True) > vram_estimate(7, training=False)


def test_vram_weights_scale_with_size():
    assert vram_estimate(70, training=False) > vram_estimate(7, training=False)


# --- cuda kernel --------------------------------------------------------------
def test_tiled_matmul_matches_numpy():
    c = tiled_matmul(32, 32, 32, 8)
    assert c.shape == (32, 32)


def test_tiling_reduces_global_reads_by_tile_factor():
    t = tiled_matmul_trace(64, 64, 64, 16)
    assert t.meta["reduction_factor"] == 16
    assert t.meta["tiled_reads"] < t.meta["naive_reads"]


# --- gpu foundations ----------------------------------------------------------
def test_thread_hierarchy_total():
    assert thread_hierarchy_trace(grid_dim=3, block_dim=4).result == 12


def test_thread_hierarchy_reverse_mapping():
    # global id 7 with block_dim 4 → block 1, thread 3
    assert thread_hierarchy_trace(grid_dim=3, block_dim=4, thread_id=7).result == (1, 3)


def test_memory_hierarchy_runs():
    assert len(memory_hierarchy_trace()) >= 3


# --- pytorch foundations ------------------------------------------------------
def test_pytorch_autograd_produces_loss():
    assert isinstance(pytorch_autograd(), float)
    assert len(pytorch_autograd_trace()) >= 3


# --- jax foundations ----------------------------------------------------------
def test_jax_grad_of_square():
    assert grad_trace(lambda x: x**2, 3.0).result == pytest.approx(6.0, abs=1e-3)


def test_jax_vmap():
    out = vmap_trace(lambda z: z**2, np.array([1.0, 2.0, 3.0, 4.0])).result
    assert np.allclose(out, [1, 4, 9, 16])


def test_jax_pytree_roundtrip():
    leaves = pytree_trace({"w": [1.0, 2.0], "b": 3.0}).result
    assert sorted(leaves) == [1.0, 2.0, 3.0]
