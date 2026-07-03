"""Render a :class:`~optimumai.core.trace.Trace` to the terminal with Rich.

The visual grammar is intentionally consistent across every operation:

    ┌ formula ┐  →  step table  →  result  →  why AI uses this

so that a dot product and a full attention block feel like the same tool.
"""

from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from optimumai.core._fmt import arr, shape_of
from optimumai.core.explain import ExplainLevel
from optimumai.core.trace import Trace

_default_console = Console()


def _fmt_step_value(value: Any) -> str:
    if value is None:
        return ""
    text = arr(value)
    # Multi-line arrays already carry their own layout; keep scalars terse.
    return text


def render_trace(
    trace: Trace,
    level: str | ExplainLevel = ExplainLevel.INTERMEDIATE,
    console: Console | None = None,
) -> None:
    """Pretty-print ``trace`` at the requested detail ``level``."""
    level = ExplainLevel.parse(level)
    console = console or _default_console

    # ---- Header: name + formula ------------------------------------------
    header = Text(trace.op.replace("_", " ").upper(), style="bold cyan")
    if trace.formula:
        header.append("\n")
        header.append(trace.formula, style="italic")
    console.print(Panel(header, border_style="cyan", title="[bold]OptimumAI[/bold]"))

    # ---- Steps table ------------------------------------------------------
    table = Table(show_lines=True, expand=False, border_style="grey42")
    table.add_column("#", justify="right", style="dim", no_wrap=True)
    table.add_column("Step", style="bold")
    table.add_column("Computation", style="white")
    if level.at_least(ExplainLevel.ENGINEER):
        table.add_column("Result", style="green")

    show_detail = level.at_least(ExplainLevel.INTERMEDIATE)
    for step in trace.steps:
        computation = step.expression
        if show_detail and step.detail:
            computation += f"\n[dim italic]{step.detail}[/dim italic]"
        row = [str(step.index), step.title, computation]
        if level.at_least(ExplainLevel.ENGINEER):
            row.append(_fmt_step_value(step.value))
        table.add_row(*row)
    console.print(table)

    # ---- Result -----------------------------------------------------------
    result_text = arr(trace.result) if trace.result is not None else "—"
    console.print(
        Panel(
            Text(result_text, style="bold green"),
            title=f"Result  ·  {shape_of(trace.result)}",
            border_style="green",
        )
    )

    # ---- Why AI uses this -------------------------------------------------
    if trace.why_ai:
        bullets = "\n".join(f"• {reason}" for reason in trace.why_ai)
        console.print(
            Panel(bullets, title="Why AI uses this", border_style="magenta")
        )

    # ---- Complexity (engineer+) ------------------------------------------
    if trace.complexity and level.at_least(ExplainLevel.ENGINEER):
        console.print(Text(f"Complexity: {trace.complexity}", style="dim yellow"))
