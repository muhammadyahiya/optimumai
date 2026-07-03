"""Pooling — downsampling a feature map while keeping what matters.

Intuition
---------
After a convolution produces a feature map, pooling shrinks it by summarizing
each small window with a single number. **Max pooling** keeps the strongest
activation in the window ("was this pattern present *somewhere* nearby?");
**average pooling** keeps the mean ("how strongly, on average, nearby?"). The
network throws away exact pixel-level position and keeps only coarse spatial
layout — a deliberate trade of precision for robustness and compute.

Math
----
For a ``k×k`` window with stride ``s`` (no learned parameters, no padding by
default), at output position ``(i, j)`` the window covers input rows
``[i·s, i·s+k)`` and columns ``[j·s, j·s+k)``:

    max_pool[i, j] = max( x[i·s : i·s+k, j·s : j·s+k] )
    avg_pool[i, j] = mean( x[i·s : i·s+k, j·s : j·s+k] )

The same output-size formula as convolution applies (with ``p=0`` unless
padding is requested): ``out = ⌊(H − k)/s⌋ + 1``. A common convention is
``stride = k`` (non-overlapping windows), which this module defaults to.

Why AI uses it
---------------
* **Translation tolerance**: if the pattern a kernel detects shifts by a
  couple of pixels — a cat's ear moves from column 4 to column 5 — max
  pooling over a window spanning both columns still reports the same
  activation. The network becomes robust to small, irrelevant shifts instead
  of memorizing exact pixel coordinates.
* **Dimensionality reduction**: a 2×2 pool with stride 2 divides both spatial
  dimensions by 2, i.e. the feature map shrinks 4×, which shrinks the compute
  and parameter count of every layer downstream.
* **A cheap, parameter-free way to grow the receptive field**: after pooling,
  the same size kernel in the next layer "sees" a proportionally larger
  region of the original image.
* Modern architectures increasingly replace pooling with strided
  convolutions or attention downsampling, but max/avg pooling is still the
  simplest way to teach the concept and shows up throughout classic CNNs
  (LeNet, AlexNet, VGG) and many vision backbones today.
"""

from __future__ import annotations

import numpy as np

from optimumai.core._fmt import arr, num
from optimumai.core.trace import Trace

_REDUCERS = {"max": np.max, "avg": np.mean}


def _out_size(size: int, k: int, stride: int) -> int:
    return (size - k) // stride + 1


def _validate(x: np.ndarray, kernel_size: int, stride: int) -> None:
    if x.ndim != 2:
        raise ValueError(f"pooling expects a 2-D feature map, got shape {x.shape}")
    if kernel_size < 1:
        raise ValueError(f"kernel_size must be >= 1, got {kernel_size}")
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    if kernel_size > x.shape[0] or kernel_size > x.shape[1]:
        raise ValueError(f"kernel_size {kernel_size} does not fit inside image {x.shape}")


def _pool(x: np.ndarray, kernel_size: int, stride: int, how: str) -> np.ndarray:
    reduce = _REDUCERS[how]
    out_h = _out_size(x.shape[0], kernel_size, stride)
    out_w = _out_size(x.shape[1], kernel_size, stride)
    out = np.zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            r, c = i * stride, j * stride
            window = x[r : r + kernel_size, c : c + kernel_size]
            out[i, j] = float(reduce(window))
    return out


def max_pool2d(x, kernel_size: int = 2, stride: int | None = None) -> np.ndarray:
    """Downsample ``x`` by taking the max of each ``kernel_size×kernel_size`` window."""
    x = np.asarray(x, dtype=float)
    stride = kernel_size if stride is None else stride
    _validate(x, kernel_size, stride)
    return _pool(x, kernel_size, stride, "max")


def avg_pool2d(x, kernel_size: int = 2, stride: int | None = None) -> np.ndarray:
    """Downsample ``x`` by averaging each ``kernel_size×kernel_size`` window."""
    x = np.asarray(x, dtype=float)
    stride = kernel_size if stride is None else stride
    _validate(x, kernel_size, stride)
    return _pool(x, kernel_size, stride, "avg")


def pool2d_trace(x, kernel_size: int = 2, stride: int | None = None, how: str = "max") -> Trace:
    """Build the full trace of pooling ``x``, window by window."""
    if how not in _REDUCERS:
        raise ValueError(f"how must be 'max' or 'avg', got {how!r}")
    x = np.asarray(x, dtype=float)
    stride = kernel_size if stride is None else stride
    _validate(x, kernel_size, stride)

    out_h = _out_size(x.shape[0], kernel_size, stride)
    out_w = _out_size(x.shape[1], kernel_size, stride)
    label = "max" if how == "max" else "mean"

    t = Trace(
        op=f"{how}_pool2d",
        formula=f"{how}_pool[i,j] = {label}( x[i·s : i·s+k, j·s : j·s+k] )",
        complexity=f"O(out_h · out_w · k²) = O({out_h}·{out_w}·{kernel_size}²), no parameters",
        why_ai=[
            "Translation tolerance: a pattern shifting a pixel or two inside the "
            "window still yields the same (or a similar) pooled value",
            f"Dimensionality reduction: a {kernel_size}×{kernel_size} window with "
            f"stride {stride} shrinks each spatial dimension by {stride}x, cutting "
            "downstream compute",
            "Parameter-free: unlike convolution, pooling has no learned weights, so "
            "it adds robustness without adding capacity to overfit",
            "Grows the effective receptive field of later layers 'for free'",
        ],
        meta={
            "input_shape": x.shape,
            "kernel_size": kernel_size,
            "stride": stride,
            "how": how,
            "output_shape": (out_h, out_w),
        },
    )

    t.add(
        "Output size",
        f"⌊({x.shape[0]} − {kernel_size}) / {stride}⌋ + 1 = {out_h}  "
        f"(same formula gives {out_w} for width)",
        (out_h, out_w),
    )

    reduce = _REDUCERS[how]
    out = np.zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            r, c = i * stride, j * stride
            window = x[r : r + kernel_size, c : c + kernel_size]
            val = float(reduce(window))
            out[i, j] = val
            t.add(
                f"Window (i={i}, j={j})",
                f"window =\n{arr(window)}\n{label}(window) = {num(val)}",
                val,
                detail=f"top-left corner of the window is input pixel ({r}, {c})",
            )

    t.add(f"{how.title()}-pooled output", f"y =\n{arr(out)}", out)
    t.result = out
    return t


def demo(seed: int = 0) -> Trace:
    """Max-pool a 4×4 feature map with a 2×2 window, stride 2 (non-overlapping)."""
    x = np.array(
        [
            [1.0, 3.0, 2.0, 0.0],
            [4.0, 2.0, 1.0, 1.0],
            [0.0, 1.0, 5.0, 3.0],
            [2.0, 0.0, 2.0, 4.0],
        ]
    )
    return pool2d_trace(x, kernel_size=2, stride=2, how="max")
