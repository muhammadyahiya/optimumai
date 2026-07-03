"""Base class for composable, explainable operations.

Most simple ops live as methods on :class:`~optimumai.algebra.vector.Vector`
etc. and just return a :class:`~optimumai.core.trace.Trace`. Multi-stage ops
(attention, a transformer block, an optimizer step) are cleaner as objects that
carry configuration, so they subclass :class:`BaseOp`.

The contract is deliberately tiny: implement :meth:`trace` to build a
:class:`Trace`, and callers get :meth:`run` (fast path) and calling the object
directly (``op(x, explain=True)``) for free.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from optimumai.core.explain import ExplainLevel
from optimumai.core.trace import Trace


class BaseOp(ABC):
    """An operation that can compute quietly or explain itself step by step."""

    name: str = "op"

    @abstractmethod
    def trace(self, *args: Any, **kwargs: Any) -> Trace:
        """Build and return a full :class:`Trace` for the given inputs."""
        raise NotImplementedError

    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Compute and return just the result (no rendering)."""
        return self.trace(*args, **kwargs).result

    def explain(
        self,
        *args: Any,
        level: str | ExplainLevel = ExplainLevel.INTERMEDIATE,
        console: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Compute, render the trace, and return the result."""
        return self.trace(*args, **kwargs).render(level=level, console=console)

    def __call__(
        self,
        *args: Any,
        explain: bool = False,
        level: str | ExplainLevel = ExplainLevel.INTERMEDIATE,
        console: Any = None,
        **kwargs: Any,
    ) -> Any:
        """``op(x)`` computes; ``op(x, explain=True)`` also prints the trace."""
        t = self.trace(*args, **kwargs)
        if explain:
            return t.render(level=level, console=console)
        return t.result
