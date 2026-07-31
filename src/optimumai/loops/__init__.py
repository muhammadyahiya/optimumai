"""Agent loops — the math behind Loop Engineering.

A Loop-Engineered agent runs a bounded, verified, stateful cycle
``evidence → triage → maker → adversarial checker → gate → state``. This track
explains the quantitative backbone of that loop, each lesson built and traced
offline (no live LLM calls): the token **budget** and its soft brake,
**convergence** to approval, why **verification** needs an independent model,
the **memory** compaction ratio, the **escalation** stopping rule, and the
run-log **state** statistics.

Each submodule exposes ``<name>_trace`` (returns a
:class:`~optimumai.core.trace.Trace`), a thin ``<name>`` wrapper (returns the
result, or renders the trace if ``explain=True``), and a ``demo`` for the
curriculum/CLI.
"""

from optimumai.loops.loop_budget import loop_budget, loop_budget_trace
from optimumai.loops.loop_convergence import loop_convergence, loop_convergence_trace
from optimumai.loops.loop_escalation import loop_escalation, loop_escalation_trace
from optimumai.loops.loop_memory import loop_memory, loop_memory_trace
from optimumai.loops.loop_state import loop_state, loop_state_trace
from optimumai.loops.loop_verification import loop_verification, loop_verification_trace
from optimumai.loops.simulate import simulate_loop_trace

__all__ = [
    "loop_budget",
    "loop_budget_trace",
    "loop_convergence",
    "loop_convergence_trace",
    "loop_verification",
    "loop_verification_trace",
    "loop_memory",
    "loop_memory_trace",
    "loop_escalation",
    "loop_escalation_trace",
    "loop_state",
    "loop_state_trace",
    "simulate_loop_trace",
]
