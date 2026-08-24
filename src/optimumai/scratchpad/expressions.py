"""Python is the source of truth for every expression a board plots.

A board declares one expression as a string. This module parses it with SymPy,
differentiates it symbolically, and emits **JavaScript** for both via SymPy's
own ``jscode`` printer. The browser therefore never defines a function -- it
only evaluates arithmetic that Python handed it.

That matters for correctness, not elegance. The previous board hardcoded

    const f      = (x) => 0.3 * x * x * x - 2 * x;
    const fPrime = (x) => 0.9 * x * x - 2;

in JavaScript while ``optimumai diff`` computed derivatives in Python. Two
implementations of the same mathematics drift, and the browser copy drifts
silently -- a wrong slope looks like a correct slope. With one source there is
nothing to disagree with.

SymPy is imported lazily and is only needed by boards that actually declare an
expression; vector boards work without it.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any


def _require_sympy() -> Any:
    try:
        import sympy
    except ImportError as exc:  # pragma: no cover - exercised via the CLI path
        raise ImportError(
            "plotting a function board needs SymPy: "
            'pip install "optimumai[scratchpad]"'
        ) from exc
    return sympy


@dataclass(frozen=True)
class CompiledExpression:
    """One expression, compiled once in Python, ready to hand to the browser.

    Attributes:
        source: The expression exactly as authored, e.g. ``"0.3*x**3 - 2*x"``.
        var: The free variable.
        latex: KaTeX-ready LaTeX for display.
        js: JavaScript for ``f(var)``.
        derivative_source: SymPy's simplified derivative, in Python syntax.
        derivative_latex: LaTeX for the derivative.
        derivative_js: JavaScript for ``f'(var)``.
    """

    source: str
    var: str
    latex: str
    js: str
    derivative_source: str
    derivative_latex: str
    derivative_js: str

    def to_dict(self) -> dict[str, str]:
        return {
            "source": self.source,
            "var": self.var,
            "latex": self.latex,
            "js": self.js,
            "derivative_source": self.derivative_source,
            "derivative_latex": self.derivative_latex,
            "derivative_js": self.derivative_js,
        }


@lru_cache(maxsize=128)
def compile_expression(source: str, var: str = "x") -> CompiledExpression:
    """Parse, differentiate, and emit JS for ``source``.

    Raises:
        ValueError: if ``source`` does not parse, so a typo fails at import
            time rather than rendering a silently blank board.
    """
    sympy = _require_sympy()
    from sympy.printing.jscode import jscode

    symbol = sympy.Symbol(var)
    try:
        f = sympy.sympify(source, locals={var: symbol})
    except (sympy.SympifyError, SyntaxError, TypeError) as exc:
        raise ValueError(f"could not parse expression {source!r}") from exc

    free = f.free_symbols - {symbol}
    if free:
        names = ", ".join(sorted(str(s) for s in free))
        raise ValueError(
            f"expression {source!r} has unbound symbol(s) {names}; "
            f"a board plots one variable ({var!r})"
        )

    derivative = sympy.simplify(sympy.diff(f, symbol))
    return CompiledExpression(
        source=source,
        var=var,
        latex=sympy.latex(f),
        js=jscode(f),
        derivative_source=str(derivative),
        derivative_latex=sympy.latex(derivative),
        derivative_js=jscode(derivative),
    )
