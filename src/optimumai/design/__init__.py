"""Design tokens — one source of truth for color and motion across every surface.

    from optimumai.design import PALETTE, MOTION, to_css_vars

Every interactive surface (``explain``, ``explore``, ``flow``, ``playground``,
``circuit``, ``scratchpad``) used to declare its own hex codes; 64 distinct
colors had accumulated with no shared source. Import from here instead.
"""

from __future__ import annotations

from .tokens import (
    MOTION,
    PALETTE,
    Motion,
    Palette,
    to_ansi,
    to_css_vars,
)

__all__ = ["MOTION", "PALETTE", "Motion", "Palette", "to_ansi", "to_css_vars"]
