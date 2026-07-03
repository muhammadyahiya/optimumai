"""Quantization — how a model shrinks from fp32 to int8/int4.

A trained weight matrix is a pile of ``float32`` numbers: 4 bytes each. Most of
that precision is wasted — the values sit in a narrow range and inference is
tolerant of small rounding. Quantization maps those floats onto a small grid of
integers so each weight costs 1 byte (int8) or half a byte (int4), then rebuilds
an approximate float on the fly with a stored ``scale`` (and ``zero_point``).

The whole scheme is two numbers per group of weights:

    scale       — the size of one integer step in float space
    zero_point  — which integer code represents 0.0 (asymmetric only)

Think of them as the *std* and *mean* of a normalization: ``scale`` stretches
the integer grid to cover the data, ``zero_point`` shifts it so real 0 lands on
a grid point. Quantize with ``q = round(x/scale) + zero_point`` (clipped to the
integer range) and dequantize with ``x̂ = (q − zero_point)·scale``. The gap
``x − x̂`` is the quantization error, and this module surfaces its max magnitude
plus the memory saving so the trade-off is explicit.

Schemes and granularity
------------------------
* **symmetric** — grid centred on 0, ``zero_point = 0``, range
  ``[-(2^(b-1)-1), 2^(b-1)-1]``; ``scale = max|x| / qmax``. Great for weights,
  which are roughly zero-mean.
* **asymmetric** — grid spans ``[0, 2^b-1]`` with a non-zero ``zero_point``;
  ``scale = (max−min)/(qmax−qmin)``. Better for skewed tensors like ReLU
  activations.
* **per-tensor** — one ``scale`` for the whole array (cheapest).
* **per-channel** — one ``scale`` per row of a 2-D matrix; preserves accuracy
  far better because a single fat channel no longer blows up everyone's scale.
"""

from __future__ import annotations

import numpy as np

from optimumai.core._fmt import arr, num
from optimumai.core.explain import ExplainLevel
from optimumai.core.trace import Trace

_BYTES_FP32 = 4


def _qrange(bits: int, scheme: str) -> tuple[int, int]:
    """Return ``(qmin, qmax)`` integer bounds for the given bit width/scheme."""
    if scheme == "symmetric":
        return -(2 ** (bits - 1) - 1), 2 ** (bits - 1) - 1
    return 0, 2**bits - 1


def _validate(bits: int, scheme: str, granularity: str) -> None:
    if bits not in (4, 8):
        raise ValueError(f"bits must be 4 or 8, got {bits}")
    if scheme not in ("symmetric", "asymmetric"):
        raise ValueError(f"scheme must be 'symmetric' or 'asymmetric', got {scheme!r}")
    if granularity not in ("tensor", "channel"):
        raise ValueError(f"granularity must be 'tensor' or 'channel', got {granularity!r}")


def _params(x: np.ndarray, bits: int, scheme: str, granularity: str):
    """Compute ``scale`` and ``zero_point`` broadcastable against ``x``.

    Per-tensor params are scalars; per-channel params have shape ``(rows, 1)`` so
    they broadcast over the columns of a 2-D matrix.
    """
    qmin, qmax = _qrange(bits, scheme)
    if granularity == "channel":
        if x.ndim != 2:
            raise ValueError("per-channel granularity requires a 2-D matrix")
        axis, keepdims = 1, True
    else:
        axis, keepdims = None, False

    if scheme == "symmetric":
        max_abs = np.max(np.abs(x), axis=axis, keepdims=keepdims)
        scale = np.where(max_abs == 0, 1.0, max_abs / qmax)
        zero_point = np.zeros_like(scale, dtype=float)
    else:  # asymmetric
        x_min = np.min(x, axis=axis, keepdims=keepdims)
        x_max = np.max(x, axis=axis, keepdims=keepdims)
        span = x_max - x_min
        scale = np.where(span == 0, 1.0, span / (qmax - qmin))
        zero_point = np.round(qmin - x_min / scale)
    return scale, zero_point


def quantize(x, bits: int = 8, scheme: str = "symmetric", granularity: str = "tensor"):
    """Quantize ``x`` to integer codes.

    Returns:
        ``(q, scale, zero_point)`` where ``q`` are the clipped integer codes and
        ``scale``/``zero_point`` are the params needed to dequantize.
    """
    _validate(bits, scheme, granularity)
    x = np.asarray(x, dtype=float)
    qmin, qmax = _qrange(bits, scheme)
    scale, zero_point = _params(x, bits, scheme, granularity)
    q = np.clip(np.round(x / scale) + zero_point, qmin, qmax).astype(int)
    return q, scale, zero_point


def dequantize(q, scale, zero_point) -> np.ndarray:
    """Rebuild approximate floats: ``x̂ = (q − zero_point)·scale``."""
    return (np.asarray(q, dtype=float) - np.asarray(zero_point, dtype=float)) * np.asarray(
        scale, dtype=float
    )


def quantize_trace(
    x, bits: int = 8, scheme: str = "symmetric", granularity: str = "tensor"
) -> Trace:
    """Build the full trace of quantizing (and dequantizing) ``x``."""
    _validate(bits, scheme, granularity)
    x = np.asarray(x, dtype=float)
    if x.ndim not in (1, 2):
        raise ValueError(f"x must be 1-D or 2-D, got shape {x.shape}")

    qmin, qmax = _qrange(bits, scheme)
    scale, zero_point = _params(x, bits, scheme, granularity)
    q = np.clip(np.round(x / scale) + zero_point, qmin, qmax).astype(int)
    x_hat = dequantize(q, scale, zero_point)
    error = x - x_hat
    max_error = float(np.max(np.abs(error)))
    compression_ratio = _BYTES_FP32 / (bits / 8.0)

    t = Trace(
        op="quantization",
        formula="q = clip(round(x/scale) + zero_point, qmin, qmax);  x̂ = (q − zero_point)·scale",
        complexity="O(n) elementwise; storage drops from 32 bits/param to "
        f"{bits} bits/param",
        why_ai=[
            "fp16 already halves fp32 to 2 bytes/param; int8 halves that again to "
            "1 byte, int4 quarters it to half a byte",
            "scale + zero_point act like the std + mean of a normalization: stretch "
            "the integer grid over the data, shift so real 0 lands on a code",
            "per-channel scales preserve accuracy far better than per-tensor — one "
            "outlier channel no longer inflates everyone's step size",
            "weight-only int8/int4 is standard for LLM inference: LLM.int8(), GPTQ "
            "and AWQ all quantize weights while keeping compute in higher precision",
        ],
        meta={
            "bits": bits,
            "scheme": scheme,
            "granularity": granularity,
            "qmin": qmin,
            "qmax": qmax,
            "scale": scale,
            "zero_point": zero_point,
            "max_error": max_error,
            "compression_ratio": compression_ratio,
            "shape": x.shape,
        },
    )

    t.add(
        "Integer range from bit width",
        f"{bits}-bit {scheme} grid → codes in [{qmin}, {qmax}] ({qmax - qmin + 1} levels)",
        detail="int8 gives 255 levels; int4 only 15 — fewer levels, coarser rounding.",
    )
    t.add(
        f"Scale (and zero_point), granularity = {granularity}",
        f"scale = {arr(scale)}\nzero_point = {arr(zero_point)}",
        detail=(
            "symmetric: scale = max|x|/qmax, zero_point = 0.  "
            "asymmetric: scale = (max−min)/(qmax−qmin), zero_point = round(qmin − min/scale)."
        ),
    )
    t.add(
        "Quantize → integer codes",
        f"q = clip(round(x/scale) + zero_point)\n{arr(q)}",
        q,
        detail="Each float is snapped to the nearest grid point and stored as a small int.",
    )
    t.add(
        "Dequantize → approximate floats",
        f"x̂ = (q − zero_point)·scale\n{arr(x_hat)}",
        x_hat,
        detail=f"Original x = {arr(x)}",
    )
    t.add(
        "Quantization error + memory saving",
        f"max |x − x̂| = {num(max_error)};  compression = {num(compression_ratio)}x "
        f"smaller than fp32",
        max_error,
        detail=(
            f"Storing {bits} bits/param instead of 32 makes the tensor "
            f"{num(compression_ratio)}x smaller; the cost is at most {num(max_error)} "
            "of rounding error per value."
        ),
    )

    t.result = x_hat
    return t


def quantize_explain(
    x,
    bits: int = 8,
    scheme: str = "symmetric",
    granularity: str = "tensor",
    explain: bool = False,
    level: str | ExplainLevel = ExplainLevel.ENGINEER,
) -> np.ndarray:
    """Return the dequantized array. Set ``explain=True`` to print the trace."""
    t = quantize_trace(x, bits=bits, scheme=scheme, granularity=granularity)
    return t.render(level) if explain else t.result


def demo(seed: int = 0) -> Trace:
    """Quantize a handful of floats to int8 for docs and the CLI."""
    x = np.array([-0.9, -0.3, 0.0, 0.15, 0.62, 1.4])
    return quantize_trace(x, bits=8)
