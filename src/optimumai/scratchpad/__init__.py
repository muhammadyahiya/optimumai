"""
optimumai.scratchpad
---------------------
Interactive, local-first math scratchpad for the optimumai learning library.
Drag points, watch the trace update live, read why AI uses the concept, and
quiz yourself — all in the browser, all offline (JSXGraph + KaTeX, no API
calls).

Usage:
    from optimumai.scratchpad import launch
    launch("dot_product")

Or from the CLI:
    optimumai scratchpad dot_product

``launch`` is resolved lazily so the concept metadata stays importable — and
``optimumai scratchpad`` can still list the boards — without the optional
[scratchpad] extra (Flask) installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .concepts import CONCEPTS, BoardSpec, Param, Snapshot, get_concept, list_concepts
from .dag import is_locked, learning_order, unmet_prerequisites, validate_dag

if TYPE_CHECKING:  # pragma: no cover
    from .cli import launch

__all__ = [
    "CONCEPTS",
    "BoardSpec",
    "Param",
    "Snapshot",
    "get_concept",
    "is_locked",
    "launch",
    "learning_order",
    "list_concepts",
    "unmet_prerequisites",
    "validate_dag",
]


def __getattr__(name: str):
    """Import the Flask-dependent launcher only when it is actually asked for."""
    if name == "launch":
        from .cli import launch as _launch

        return _launch
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
