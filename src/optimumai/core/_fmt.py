"""Small, dependency-light number/array formatting helpers.

Kept in ``core`` so both the math ops (which build human-readable expression
strings) and the visualization layer can share one formatting convention
without importing each other.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def num(x: Any, precision: int = 4) -> str:
    """Format a scalar compactly: integers stay integers, floats get trimmed."""
    val = float(x)
    if val == int(val) and abs(val) < 1e15:
        return str(int(val))
    return f"{val:.{precision}g}"


def arr(x: Any, precision: int = 4) -> str:
    """Format a scalar, vector, or matrix as a compact string."""
    a = np.asarray(x, dtype=float)
    if a.ndim == 0:
        return num(float(a), precision)
    return np.array2string(
        a,
        precision=precision,
        suppress_small=True,
        separator=", ",
        floatmode="maxprec",
    )


def shape_of(x: Any) -> str:
    """Return a short shape label like ``(2, 3)`` or ``scalar``."""
    a = np.asarray(x)
    return "scalar" if a.ndim == 0 else str(tuple(a.shape))
