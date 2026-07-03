"""Symbolic differentiation — the ground truth behind every autograd engine.

Give it an equation as a string (``"x**3 + 2*x"``) and it returns the exact
derivative, computed the way a calculus student would: parse the expression,
apply the symbolic rules, simplify, and (optionally) plug in a number. This is
the *reference* you check a hand-written backward pass against.

SymPy is an optional dependency (``pip install "optimumai[symbolic]"``); it is
imported lazily so the rest of OptimumAI never hard-depends on it.
"""

from __future__ import annotations

from typing import Any

from optimumai.core._fmt import num
from optimumai.core.explain import ExplainLevel
from optimumai.core.trace import Trace


def _require_sympy() -> Any:
    """Import and return the ``sympy`` module, or raise a friendly ImportError."""
    try:
        import sympy
    except ImportError as exc:
        raise ImportError(
            'symbolic differentiation needs SymPy: pip install "optimumai[symbolic]"'
        ) from exc
    return sympy


def differentiate_trace(
    expression: str, var: str = "x", at: float | None = None
) -> Trace:
    """Build the full trace of ``d/dvar`` applied to a user-supplied equation."""
    sympy = _require_sympy()

    symbol = sympy.Symbol(var)
    try:
        f = sympy.sympify(expression, locals={var: symbol})
    except (sympy.SympifyError, SyntaxError, TypeError) as exc:
        raise ValueError(f"could not parse expression {expression!r}") from exc

    deriv = sympy.diff(f, symbol)
    simplified = sympy.simplify(deriv)
    deriv_str = str(simplified)

    t = Trace(
        op="differentiate",
        formula=f"d/d{var} f({var})",
        complexity="O(size of expression tree)",
        why_ai=[
            "Gradients drive learning — every weight update follows a derivative",
            "Autograd automates exactly this symbolic rule over millions of ops",
            "Symbolic diff is the ground truth you check autograd against",
        ],
        meta={"expression": expression, "var": var, "derivative": deriv_str},
    )

    t.add(
        f"Parse the function f({var})",
        f"f({var}) = {sympy.sstr(f)}",
        detail=f"Parsed {expression!r} into a symbolic expression in {var}.",
    )
    t.add(
        f"Differentiate symbolically  f'({var})",
        f"d/d{var} [{sympy.sstr(f)}] = {sympy.sstr(deriv)}",
        detail="Applies the power, product, chain, and quotient rules automatically.",
    )
    t.add(
        "Simplify",
        f"f'({var}) = {deriv_str}",
        detail="Collect like terms so the derivative reads cleanly.",
    )

    result: Any
    if at is not None:
        f_at = float(f.subs(symbol, at))
        fp_at = float(simplified.subs(symbol, at))
        t.add(
            f"Evaluate f({num(at)})",
            f"f({num(at)}) = {num(f_at)}",
            f_at,
            detail="Numeric value of the original function at the point.",
        )
        t.add(
            f"Evaluate f'({num(at)})",
            f"f'({num(at)}) = {num(fp_at)}",
            fp_at,
            detail=f"The slope of f at {var} = {num(at)} — the gradient signal.",
        )
        result = fp_at
    else:
        result = deriv_str

    t.result = result
    return t


def differentiate(
    expression: str,
    var: str = "x",
    at: float | None = None,
    explain: bool = False,
    level: str | ExplainLevel = ExplainLevel.INTERMEDIATE,
) -> Any:
    """Differentiate ``expression`` w.r.t. ``var``. Set ``explain=True`` to print."""
    t = differentiate_trace(expression, var=var, at=at)
    return t.render(level) if explain else t.result


def demo() -> Trace:
    """Differentiate ``x**3 + 2*x`` and evaluate at 3 (f'(x) = 3x² + 2 → 29)."""
    return differentiate_trace("x**3 + 2*x", at=3.0)
