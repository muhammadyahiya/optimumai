import numpy as np
import pytest

from optimumai.vision import (
    avg_pool2d,
    cnn_forward,
    cnn_forward_trace,
    conv2d,
    conv2d_trace,
    dense,
    max_pool2d,
    pool2d_trace,
    relu,
    sobel_edges,
    sobel_edges_trace,
)
from optimumai.vision.cnn import demo as cnn_demo
from optimumai.vision.convolution import demo as conv_demo
from optimumai.vision.edges import GX, GY
from optimumai.vision.edges import demo as edges_demo
from optimumai.vision.pooling import demo as pool_demo


# --- independent inline references --------------------------------------------
def _ref_cross_correlate(x: np.ndarray, w: np.ndarray, stride: int, padding: int) -> np.ndarray:
    """Hand-rolled sliding window, computed independently of conv2d's implementation."""
    xp = np.pad(x, padding) if padding else x
    kh, kw = w.shape
    out_h = (x.shape[0] - kh + 2 * padding) // stride + 1
    out_w = (x.shape[1] - kw + 2 * padding) // stride + 1
    out = np.zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            r, c = i * stride, j * stride
            total = 0.0
            for a in range(kh):
                for b in range(kw):
                    total += xp[r + a, c + b] * w[a, b]
            out[i, j] = total
    return out


def _ref_pool(x: np.ndarray, k: int, stride: int, how: str) -> np.ndarray:
    out_h = (x.shape[0] - k) // stride + 1
    out_w = (x.shape[1] - k) // stride + 1
    out = np.zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            r, c = i * stride, j * stride
            vals = [x[r + a, c + b] for a in range(k) for b in range(k)]
            out[i, j] = max(vals) if how == "max" else sum(vals) / len(vals)
    return out


# --- convolution ----------------------------------------------------------------
def test_conv2d_matches_hand_rolled_reference_no_padding():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(5, 5))
    w = rng.normal(size=(3, 3))
    out = conv2d(x, w, stride=1, padding=0)
    ref = _ref_cross_correlate(x, w, stride=1, padding=0)
    assert np.allclose(out, ref)


def test_conv2d_matches_hand_rolled_reference_with_stride_and_padding():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(6, 6))
    w = rng.normal(size=(3, 3))
    out = conv2d(x, w, stride=2, padding=1)
    ref = _ref_cross_correlate(x, w, stride=2, padding=1)
    assert np.allclose(out, ref)


def test_conv2d_output_shape_matches_formula():
    # out = floor((H - k + 2p) / s) + 1
    x = np.zeros((7, 7))
    w = np.zeros((3, 3))
    out = conv2d(x, w, stride=2, padding=1)
    expected = (7 - 3 + 2 * 1) // 2 + 1
    assert out.shape == (expected, expected) == (4, 4)


def test_conv2d_convolve_flips_kernel_180():
    x = np.arange(9, dtype=float).reshape(3, 3)
    w = np.array([[1.0, 2.0], [3.0, 4.0]])
    cross_correlated = conv2d(x, w, mode="cross-correlate")
    convolved = conv2d(x, w, mode="convolve")
    # convolve(x, w) == cross-correlate(x, flip180(w))
    assert np.allclose(convolved, conv2d(x, w[::-1, ::-1], mode="cross-correlate"))
    assert not np.allclose(cross_correlated, convolved)


def test_conv2d_rejects_kernel_larger_than_image():
    with pytest.raises(ValueError):
        conv2d(np.zeros((3, 3)), np.zeros((5, 5)))


def test_conv2d_rejects_non_2d_input():
    with pytest.raises(ValueError):
        conv2d(np.zeros((3, 3, 3)), np.zeros((2, 2)))


def test_conv2d_rejects_bad_stride_padding_mode():
    with pytest.raises(ValueError):
        conv2d(np.zeros((4, 4)), np.zeros((2, 2)), stride=0)
    with pytest.raises(ValueError):
        conv2d(np.zeros((4, 4)), np.zeros((2, 2)), padding=-1)
    with pytest.raises(ValueError):
        conv2d(np.zeros((4, 4)), np.zeros((2, 2)), mode="banana")


def test_conv2d_trace_has_formula_why_ai_and_result():
    t = conv2d_trace(np.eye(4), np.array([[1.0, 0.0], [0.0, 1.0]]))
    assert t.formula
    assert len(t.why_ai) >= 3
    assert t.result.shape == (3, 3)
    # kernel step + output-size step + one step per output window + feature map step
    assert len(t) == 2 + 3 * 3 + 1


def test_conv_demo_detects_vertical_edge():
    t = conv_demo()
    # every row of the demo image has the identical 0->1 boundary, so the
    # feature map must be constant down each column (translation invariance
    # along the edge) and the strongest-magnitude response must sit at the
    # boundary rather than in the flat interior regions.
    assert np.all(t.result == t.result[0, :])
    boundary_response = np.abs(t.result[:, 0])
    flat_region_response = np.abs(t.result[:, -1])
    assert np.all(boundary_response > flat_region_response)


# --- pooling --------------------------------------------------------------------
def test_max_pool2d_matches_reference():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(6, 6))
    out = max_pool2d(x, kernel_size=2, stride=2)
    assert np.allclose(out, _ref_pool(x, 2, 2, "max"))


def test_avg_pool2d_matches_reference():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(6, 6))
    out = avg_pool2d(x, kernel_size=3, stride=1)
    assert np.allclose(out, _ref_pool(x, 3, 1, "avg"))


def test_pool2d_output_shape_matches_formula():
    x = np.zeros((7, 7))
    out = max_pool2d(x, kernel_size=3, stride=2)
    expected = (7 - 3) // 2 + 1
    assert out.shape == (expected, expected) == (3, 3)


def test_pool2d_default_stride_equals_kernel_size_nonoverlapping():
    x = np.arange(16, dtype=float).reshape(4, 4)
    out = max_pool2d(x, kernel_size=2)  # stride defaults to kernel_size
    assert out.shape == (2, 2)


def test_max_pool_keeps_strongest_value_translation_tolerant():
    # a single hot pixel anywhere inside the same 2x2 window yields the same output
    base = np.zeros((4, 4))
    shifted = np.zeros((4, 4))
    base[0, 0] = 9.0
    shifted[0, 1] = 9.0
    assert np.array_equal(max_pool2d(base, 2, 2), max_pool2d(shifted, 2, 2))


def test_pool2d_rejects_kernel_larger_than_image():
    with pytest.raises(ValueError):
        max_pool2d(np.zeros((3, 3)), kernel_size=5)


def test_pool2d_rejects_non_2d_input():
    with pytest.raises(ValueError):
        avg_pool2d(np.zeros((3, 3, 3)), kernel_size=2)


def test_pool2d_trace_rejects_bad_how():
    with pytest.raises(ValueError):
        pool2d_trace(np.zeros((4, 4)), how="median")


def test_pool2d_trace_has_result_and_why_ai():
    t = pool2d_trace(np.arange(16, dtype=float).reshape(4, 4), kernel_size=2, how="max")
    assert len(t.why_ai) >= 3
    assert t.result.shape == (2, 2)
    # output-size step + one step per window + pooled-output step
    assert len(t) == 1 + 2 * 2 + 1


def test_pool_demo_max_values_are_correct():
    t = pool_demo()
    assert t.result.shape == (2, 2)
    assert t.result[0, 0] == 4.0  # max of [[1,3],[4,2]]
    assert t.result[1, 1] == 5.0  # max of [[5,3],[2,4]]


# --- sobel edges ------------------------------------------------------------------
def test_sobel_gx_gy_kernels_are_standard():
    assert np.array_equal(GX, np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]))
    assert np.array_equal(GY, np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]))


def test_sobel_matches_independent_convolution_reference():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(5, 5))
    magnitude, orientation = sobel_edges(x, padding=1)

    gx_ref = _ref_cross_correlate(x, GX, stride=1, padding=1)
    gy_ref = _ref_cross_correlate(x, GY, stride=1, padding=1)
    assert np.allclose(magnitude, np.sqrt(gx_ref**2 + gy_ref**2))
    assert np.allclose(orientation, np.arctan2(gy_ref, gx_ref))


def test_sobel_flat_image_has_zero_gradient():
    # padding=0 avoids the padding boundary itself reading as an edge, isolating
    # the property under test: uniform brightness has no gradient anywhere.
    flat = np.full((5, 5), 3.0)
    magnitude, _ = sobel_edges(flat, padding=0)
    assert np.allclose(magnitude, 0.0)


def test_sobel_zero_padding_creates_a_border_gradient():
    # zero-padding a flat image DOES introduce a real brightness discontinuity
    # at the border against the padded zeros; only the interior stays flat.
    flat = np.full((5, 5), 3.0)
    magnitude, _ = sobel_edges(flat, padding=1)
    assert np.allclose(magnitude[1:-1, 1:-1], 0.0)
    assert np.all(magnitude[0, :] > 0)


def test_sobel_output_shape_with_padding_one_matches_input():
    x = np.zeros((6, 6))
    magnitude, orientation = sobel_edges(x, padding=1)
    assert magnitude.shape == orientation.shape == (6, 6)


def test_sobel_detects_strong_vertical_edge():
    x = np.zeros((6, 6))
    x[:, 3:] = 1.0  # sharp dark-to-bright vertical boundary at column 3
    magnitude, _ = sobel_edges(x, padding=1)
    edge_col_energy = magnitude[:, 2:4].sum()
    other_energy = magnitude.sum() - edge_col_energy
    assert edge_col_energy > other_energy


def test_sobel_rejects_too_small_image():
    with pytest.raises(ValueError):
        sobel_edges(np.zeros((2, 2)))


def test_sobel_rejects_non_2d_input():
    with pytest.raises(ValueError):
        sobel_edges_trace(np.zeros((3, 3, 3)))


def test_sobel_trace_has_formula_result_and_why_ai():
    t = sobel_edges_trace(np.eye(5), padding=1)
    assert t.formula
    assert len(t.why_ai) >= 3
    assert t.result.shape == (5, 5)
    assert len(t) == 6  # Gx kernel, Gx map, Gy kernel, Gy map, magnitude, orientation


def test_edges_demo_peak_is_on_the_boundary():
    t = edges_demo()
    peak_col = t.meta["peak_position"][1]
    assert peak_col in (2, 3)  # boundary sits between input columns 2 and 3


# --- tiny CNN forward pass --------------------------------------------------------
def test_relu_zeroes_negatives_only():
    z = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
    assert np.array_equal(relu(z), np.array([0.0, 0.0, 0.0, 0.5, 2.0]))


def test_dense_matches_matrix_reference():
    rng = np.random.default_rng(5)
    flat = rng.normal(size=6)
    w = rng.normal(size=(3, 6))
    b = rng.normal(size=3)
    assert np.allclose(dense(flat, w, b), w @ flat + b)


def test_dense_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        dense(np.zeros(4), np.zeros((2, 5)), np.zeros(2))
    with pytest.raises(ValueError):
        dense(np.zeros(4), np.zeros((2, 4)), np.zeros(3))


def test_cnn_forward_matches_independent_reference():
    rng = np.random.default_rng(6)
    image = rng.normal(size=(6, 6))
    kernel = rng.normal(size=(3, 3))
    w = rng.normal(size=(2, 4))
    b = rng.normal(size=2)

    probs = cnn_forward(image, kernel, w, b, pool_size=2)

    # independent reference: conv (hand-rolled) -> relu -> pool (hand-rolled)
    # -> flatten -> dense -> softmax, computed without calling any vision code.
    feature_map = _ref_cross_correlate(image, kernel, stride=1, padding=0)
    activated = np.maximum(0.0, feature_map)
    pooled = _ref_pool(activated, 2, 2, "max")
    flat = pooled.flatten()
    logits = w @ flat + b
    shifted = logits - logits.max()
    ref_probs = np.exp(shifted) / np.exp(shifted).sum()

    assert np.allclose(probs, ref_probs)
    assert probs.sum() == pytest.approx(1.0)
    assert np.all(probs > 0)


def test_cnn_forward_shape_pipeline_matches_formulas():
    image = np.zeros((6, 6))
    kernel = np.zeros((3, 3))
    # conv: (6-3+0)/1+1 = 4  -> feature_map (4,4)
    # pool: (4-2)/2+1 = 2    -> pooled (2,2) -> flat len 4
    w = np.zeros((3, 4))
    b = np.zeros(3)
    t = cnn_forward_trace(image, kernel, w, b, pool_size=2)
    assert t.meta["feature_map_shape"] == (4, 4)
    assert t.meta["pooled_shape"] == (2, 2)
    assert t.meta["flat_len"] == 4
    assert t.meta["n_classes"] == 3
    assert t.result.shape == (3,)


def test_cnn_forward_rejects_non_2d_image():
    with pytest.raises(ValueError):
        cnn_forward_trace(np.zeros((6, 6, 1)), np.zeros((3, 3)), np.zeros((2, 4)), np.zeros(2))


def test_cnn_forward_rejects_dense_shape_mismatch():
    image = np.zeros((6, 6))
    kernel = np.zeros((3, 3))
    with pytest.raises(ValueError):
        cnn_forward_trace(image, kernel, np.zeros((2, 999)), np.zeros(2))


def test_cnn_forward_trace_has_shape_flow_formula_and_why_ai():
    t = cnn_demo()
    assert t.formula
    assert len(t.why_ai) >= 3
    assert "shape_flow" in t.meta
    assert "-->" in t.meta["shape_flow"]
    assert t.result.sum() == pytest.approx(1.0)
    assert len(t) == 8


def test_cnn_demo_predicted_class_is_valid_index():
    t = cnn_demo()
    assert t.meta["predicted_class"] in range(t.meta["n_classes"])
