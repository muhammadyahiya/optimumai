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
    """Format a scalar, vector, or matrix as a compact string.

    Non-numeric values (strings, dicts, lists of tuples, ...) that cannot be
    coerced to a float array fall back to ``str(x)`` so any trace renders safely.
    """
    if isinstance(x, str):
        return x
    try:
        a = np.asarray(x, dtype=float)
    except (ValueError, TypeError):
        return str(x)
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
    if isinstance(x, str):
        return "text"
    try:
        a = np.asarray(x, dtype=float)
    except (ValueError, TypeError):
        return type(x).__name__
    return "scalar" if a.ndim == 0 else str(tuple(a.shape))
