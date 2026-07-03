"""2D convolution — the operation that lets a network learn local patterns.

Intuition
---------
A kernel (a.k.a. filter) is a small grid of numbers, e.g. 3×3, that encodes a
local pattern: an edge, a corner, a blob of one color next to another. Sliding
that kernel over every position of an image and, at each stop, multiplying it
element-wise against the patch of image underneath and summing the result,
produces one number per position — a new, smaller image called a **feature
map**. High values mean "this patch looks like what the kernel is looking
for." A CNN *learns* the kernel's numbers instead of hand-designing them.

Math
----
For an ``H×W`` image ``x`` and a ``k×k`` kernel ``w`` (no padding, stride 1),
cross-correlation at output position ``(i, j)`` is:

    y[i, j] = Σ_{a=0}^{k-1} Σ_{b=0}^{k-1} x[i+a, j+b] · w[a, b]

Two extra knobs change the geometry, not the arithmetic:

* **stride** ``s`` — skip ``s`` pixels between stops instead of 1, so the
  kernel visits every ``s``-th position. Larger stride ⇒ smaller output,
  cheaper compute, coarser spatial detail.
* **padding** ``p`` — add ``p`` rings of zeros around the border first, so
  edge pixels get looked at as many times as interior ones and the output
  can stay the same size as the input ("same" padding).

Together they fix the output size exactly:

    out = floor((H − k + 2p) / s) + 1          (same formula for W)

Convolution vs. cross-correlation
----------------------------------
Signal-processing "convolution" flips the kernel 180° (``w[k-1-a, k-1-b]``)
before sliding it; "cross-correlation" does not. Every deep learning
framework (PyTorch's ``Conv2d``, TensorFlow's ``Conv2D``) actually implements
**cross-correlation** and calls it "convolution" — the flip makes no
difference to a *learned* kernel, since gradient descent will simply learn
the flipped pattern if that's what minimizes the loss. This module defaults
to cross-correlation (``mode="cross-correlate"``, the AI convention) and
offers `mode="convolve"` for the textbook-flipped version so the distinction
is explicit rather than glossed over.

Why AI uses it
---------------
* Convolution is the core operator of every CNN (LeNet, ResNet, U-Net) and
  the vision backbone hiding inside CLIP, Stable Diffusion's U-Net, and
  video models.
* **Weight sharing**: one 3×3 kernel has 9 numbers no matter how big the
  image is, and it is applied at every position — a colossal parameter
  saving over a fully-connected layer that would need one weight per
  (input pixel, output pixel) pair.
* **Translation equivariance**: shifting the input shifts the output feature
  map by the same amount, which is exactly the right inductive bias for
  "find this pattern wherever it appears in the image."
"""

from __future__ import annotations

import numpy as np

from optimumai.core._fmt import arr, num
from optimumai.core.trace import Trace


def _out_size(size: int, k: int, stride: int, padding: int) -> int:
    """``floor((size − k + 2·padding) / stride) + 1``."""
    return (size - k + 2 * padding) // stride + 1


def _pad(x: np.ndarray, padding: int) -> np.ndarray:
    if padding == 0:
        return x
    return np.pad(x, padding, mode="constant", constant_values=0.0)


def _validate(x: np.ndarray, w: np.ndarray, stride: int, padding: int, mode: str) -> None:
    if x.ndim != 2:
        raise ValueError(f"conv2d expects a 2-D image, got shape {x.shape}")
    if w.ndim != 2:
        raise ValueError(f"conv2d expects a 2-D kernel, got shape {w.shape}")
    if w.shape[0] > x.shape[0] + 2 * padding or w.shape[1] > x.shape[1] + 2 * padding:
        raise ValueError(
            f"kernel {w.shape} does not fit inside padded image "
            f"{(x.shape[0] + 2 * padding, x.shape[1] + 2 * padding)}"
        )
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    if padding < 0:
        raise ValueError(f"padding must be >= 0, got {padding}")
    if mode not in ("cross-correlate", "convolve"):
        raise ValueError(f"mode must be 'cross-correlate' or 'convolve', got {mode!r}")


def conv2d(
    x, w, stride: int = 1, padding: int = 0, mode: str = "cross-correlate"
) -> np.ndarray:
    """Slide kernel ``w`` over image ``x`` and return the output feature map.

    ``mode="cross-correlate"`` (default) is what every deep learning framework
    calls "convolution": no kernel flip. ``mode="convolve"`` flips the kernel
    180° first, matching the classical signal-processing definition.
    """
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    _validate(x, w, stride, padding, mode)
    if mode == "convolve":
        w = w[::-1, ::-1]

    xp = _pad(x, padding)
    kh, kw = w.shape
    out_h = _out_size(x.shape[0], kh, stride, padding)
    out_w = _out_size(x.shape[1], kw, stride, padding)
    out = np.zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            r, c = i * stride, j * stride
            patch = xp[r : r + kh, c : c + kw]
            out[i, j] = float(np.sum(patch * w))
    return out


def conv2d_trace(
    x, w, stride: int = 1, padding: int = 0, mode: str = "cross-correlate"
) -> Trace:
    """Build the full trace of sliding ``w`` over ``x``, window by window."""
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    _validate(x, w, stride, padding, mode)
    w_applied = w[::-1, ::-1] if mode == "convolve" else w

    xp = _pad(x, padding)
    kh, kw = w.shape
    out_h = _out_size(x.shape[0], kh, stride, padding)
    out_w = _out_size(x.shape[1], kw, stride, padding)

    t = Trace(
        op="conv2d",
        formula="y[i,j] = ΣₐΣᵦ x[i·s+a, j·s+b] · w[a,b];  out = ⌊(H − k + 2p)/s⌋ + 1",
        complexity=f"O(out_h · out_w · k²) = O({out_h}·{out_w}·{kh}·{kw})",
        why_ai=[
            "Weight sharing: one small kernel scans the whole image instead of a "
            "full weight per pixel pair, cutting parameters by orders of magnitude",
            "Translation equivariance: shift the input, the feature map shifts with "
            "it — exactly the bias you want for 'find this pattern anywhere'",
            "Stacked convolutions build a receptive-field hierarchy: edges → "
            "textures → parts → objects, the backbone of every CNN",
            "mode='cross-correlate' (no kernel flip) is what PyTorch/TensorFlow "
            "actually compute and call 'Conv2d' — gradient descent learns whichever "
            "orientation minimizes the loss, so the flip is a historical footnote",
        ],
        meta={
            "input_shape": x.shape,
            "kernel_shape": w.shape,
            "stride": stride,
            "padding": padding,
            "mode": mode,
            "output_shape": (out_h, out_w),
        },
    )

    t.add(
        "Kernel",
        f"w =\n{arr(w)}" + ("  (flip 180° for true convolution)" if mode == "convolve" else ""),
        w_applied,
        detail="cross-correlation uses w as-is; convolution flips it 180° first, "
        "reversing both the row order and the column order.",
    )
    if padding:
        t.add(
            "Zero-pad the input",
            f"pad by {padding} on every side: {x.shape} → {xp.shape}",
            xp,
            detail="Padding lets border pixels get visited as many times as "
            "interior ones, and can keep the output the same size as the input.",
        )
    t.add(
        "Output size",
        f"⌊({x.shape[0]} − {kh} + 2·{padding}) / {stride}⌋ + 1 = {out_h}  "
        f"(same formula gives {out_w} for width)",
        (out_h, out_w),
    )

    out = np.zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            r, c = i * stride, j * stride
            patch = xp[r : r + kh, c : c + kw]
            prod = patch * w_applied
            val = float(np.sum(prod))
            out[i, j] = val
            t.add(
                f"Window (i={i}, j={j})",
                f"patch =\n{arr(patch)}\nΣ(patch ⊙ w) = {num(val)}",
                val,
                detail=f"top-left corner of the patch is input pixel ({r}, {c})",
            )

    t.add("Feature map", f"y =\n{arr(out)}", out)
    t.result = out
    return t


def demo(seed: int = 0) -> Trace:
    """Detect a vertical edge in a small 5×5 image with a 3×3 kernel."""
    x = np.array(
        [
            [0.0, 0.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0, 1.0],
        ]
    )
    vertical_edge_kernel = np.array(
        [
            [1.0, 0.0, -1.0],
            [1.0, 0.0, -1.0],
            [1.0, 0.0, -1.0],
        ]
    )
    return conv2d_trace(x, vertical_edge_kernel, stride=1, padding=0)
