"""The circuit view — render a computation graph like an electrical circuit.

Build a :class:`FlowGraph` from a :class:`~optimumai.autograd.value.Value` or a
user expression, then render it as interactive HTML, Graphviz DOT, or a terminal
table — with each node's forward **data** and backward **gradient** on the wires.
"""

from optimumai.circuit.graph import (
    FlowGraph,
    FlowNode,
    build_from_expression,
    build_from_value,
)
from optimumai.circuit.render import render, to_dot, to_html, to_terminal

__all__ = [
    "FlowGraph",
    "FlowNode",
    "build_from_expression",
    "build_from_value",
    "render",
    "to_dot",
    "to_html",
    "to_terminal",
]
