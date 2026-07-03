"""The computation trace — the beating heart of OptimumAI.

Every operation, whether a 3-element dot product or a full attention block,
produces a :class:`Trace`: an ordered list of :class:`Step` objects plus the
final result and the "why AI cares" context. A trace can be rendered to a
terminal, inspected programmatically, or asserted against in tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from optimumai.core.explain import ExplainLevel


@dataclass
class Step:
    """One intermediate move in a computation.

    Attributes:
        index: 1-based position in the trace.
        title: Short label, e.g. ``"Multiply component 0"``.
        expression: Human-readable computation, e.g. ``"1 × 4 = 4"``.
        value: The numeric result of this step (scalar / vector / matrix).
        detail: Optional longer note shown at higher explain levels.
    """

    index: int
    title: str
    expression: str
    value: Any = None
    detail: str = ""


@dataclass
class Trace:
    """A complete, replayable record of an operation.

    Attributes:
        op: Name of the operation, e.g. ``"dot"`` or ``"attention"``.
        result: The final output value.
        steps: Ordered intermediate steps.
        formula: The closed-form formula, e.g. ``"a · b = Σ aᵢbᵢ"``.
        why_ai: Bullet points on where this shows up in real AI systems.
        complexity: Big-O note, surfaced at engineer/researcher levels.
        meta: Free-form extra data (shapes, hyper-params, ...).
    """

    op: str
    result: Any = None
    steps: list[Step] = field(default_factory=list)
    formula: str = ""
    why_ai: list[str] = field(default_factory=list)
    complexity: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def add(
        self, title: str, expression: str, value: Any = None, detail: str = ""
    ) -> Trace:
        """Append a step and return ``self`` for fluent chaining."""
        self.steps.append(
            Step(
                index=len(self.steps) + 1,
                title=title,
                expression=expression,
                value=value,
                detail=detail,
            )
        )
        return self

    def render(
        self,
        level: str | ExplainLevel = ExplainLevel.INTERMEDIATE,
        console: Any = None,
    ) -> Any:
        """Pretty-print this trace to the terminal and return the result.

        Imported lazily so the math layer never hard-depends on the UI layer.
        """
        from optimumai.visualization.terminal import render_trace

        render_trace(self, level=ExplainLevel.parse(level), console=console)
        return self.result

    def last(self) -> Step:
        """Return the final recorded step (raises if the trace is empty)."""
        if not self.steps:
            raise IndexError("trace has no steps")
        return self.steps[-1]

    def __len__(self) -> int:
        return len(self.steps)

    def __iter__(self):
        return iter(self.steps)
