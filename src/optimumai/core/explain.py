"""Explanation verbosity levels.

The same computation can be explained for four audiences. Higher levels do not
*change* the math — they reveal more of it (formulas, complexity, edge cases).
"""

from __future__ import annotations

from enum import Enum


class ExplainLevel(str, Enum):
    """How much detail a rendered trace should surface."""

    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ENGINEER = "engineer"
    RESEARCHER = "researcher"

    @classmethod
    def parse(cls, value: str | ExplainLevel) -> ExplainLevel:
        """Coerce a string (case-insensitive) or enum into an ``ExplainLevel``."""
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).strip().lower())
        except ValueError as exc:  # pragma: no cover - defensive
            valid = ", ".join(level.value for level in cls)
            raise ValueError(
                f"Unknown explain level {value!r}. Choose one of: {valid}."
            ) from exc

    @property
    def rank(self) -> int:
        """Ordinal used to gate optional sections (higher shows more)."""
        order = [
            ExplainLevel.BEGINNER,
            ExplainLevel.INTERMEDIATE,
            ExplainLevel.ENGINEER,
            ExplainLevel.RESEARCHER,
        ]
        return order.index(self)

    def at_least(self, other: ExplainLevel) -> bool:
        """True when this level is as detailed as ``other`` (or more)."""
        return self.rank >= other.rank
