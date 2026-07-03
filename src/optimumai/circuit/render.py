"""Render a :class:`~optimumai.circuit.graph.FlowGraph` three ways.

This is the v0.7 headline — the "electrical circuit" view of AI math. A user
gives an expression and *sees* the computation graph, with each node's forward
**data** (blue) and backward **grad** (orange) flowing through the wires.

Three renderers, one dispatcher:

* :func:`to_dot`      — Graphviz DOT text (records for values, ellipses for ops).
* :func:`to_terminal` — a Rich table, "the circuit" in your terminal.
* :func:`to_html`     — a self-contained interactive vis-network page (CDN only).
* :func:`render`      — dispatch on ``fmt`` over a Value / FlowGraph / expression.

    >>> from optimumai.circuit.render import demo
    >>> "digraph" in demo(fmt="dot")
    True
"""

from __future__ import annotations

import html
import json

from optimumai.circuit.graph import (
    FlowGraph,
    build_from_expression,
    build_from_value,
)
from optimumai.core._fmt import num

# Shared palette — data = forward (blue), grad = backward (orange).
_DATA_COLOR = "#2563eb"
_GRAD_COLOR = "#ea7317"


def to_dot(fg: FlowGraph) -> str:
    """Emit a Graphviz DOT description of ``fg`` (left-to-right layout).

    Value nodes are records showing ``label | data d.dd | grad g.gg``; operator
    pseudo-nodes are small ellipses carrying the op symbol.
    """
    lines: list[str] = [
        "digraph circuit {",
        "  rankdir=LR;",
        '  graph [bgcolor="white"];',
        '  node [fontname="Helvetica"];',
        '  edge [color="#64748b"];',
    ]

    for node in fg.value_nodes():
        label = node.label or "·"
        record = (
            f"{{ {_dot_escape(label)} | "
            f"data {_dot_escape(num(node.data))} | "
            f"grad {_dot_escape(num(node.grad))} }}"
        )
        lines.append(
            f'  "{node.id}" [shape=record, style="filled,rounded", '
            f'fillcolor="#eff6ff", color="{_DATA_COLOR}", label="{record}"];'
        )

    for node in fg.op_nodes():
        lines.append(
            f'  "{node.id}" [shape=ellipse, style=filled, '
            f'fillcolor="#fff7ed", color="{_GRAD_COLOR}", '
            f'label="{_dot_escape(node.label)}"];'
        )

    for src, dst in fg.edges:
        lines.append(f'  "{src}" -> "{dst}";')

    lines.append("}")
    return "\n".join(lines)


def _dot_escape(text: str) -> str:
    """Escape characters that are special inside a DOT record/label."""
    out = text.replace("\\", "\\\\").replace('"', '\\"')
    for ch in "{}|<>":
        out = out.replace(ch, "\\" + ch)
    return out


def to_terminal(fg: FlowGraph, console: object | None = None) -> None:
    """Print ``fg`` as "the circuit" in the terminal using Rich.

    Nodes are listed in topological order (as stored). Each value row shows its
    label, the producing op, its ``data`` (blue) and ``grad`` (orange), plus a
    one-line summary of the wires flowing into it.
    """
    from rich.console import Console
    from rich.table import Table
    from rich.text import Text

    con = console if console is not None else Console()

    # Map op-node id -> the value id it feeds, so we can summarise the wiring
    # of each value node in terms of the child values that feed its op.
    op_to_target: dict[str, str] = {}
    inputs_of_op: dict[str, list[str]] = {}
    for src, dst in fg.edges:
        if dst.endswith("_op"):
            inputs_of_op.setdefault(dst, []).append(src)
        else:
            op_to_target[src] = dst

    id_to_node = {n.id: n for n in fg.nodes}

    def _display(node_id: str) -> str:
        n = id_to_node.get(node_id)
        if n is None:
            return "?"
        return n.label or n.op or "·"

    table = Table(
        title="⚡ Circuit — forward data (blue) · backward grad (orange)",
        header_style="bold",
        expand=False,
    )
    table.add_column("node", style="bold")
    table.add_column("op", style="magenta")
    table.add_column("data", justify="right")
    table.add_column("grad", justify="right")
    table.add_column("wires", style="dim")

    for node in fg.value_nodes():
        # Find the op node (if any) that produces this value, and its inputs.
        wires = ""
        for op_id, tgt in op_to_target.items():
            if tgt == node.id:
                srcs = inputs_of_op.get(op_id, [])
                parts = " , ".join(_display(s) for s in srcs)
                wires = f"{parts} → {id_to_node[op_id].label} → {node.label or '·'}"
                break

        table.add_row(
            node.label or "·",
            node.op or "leaf",
            Text(num(node.data), style=_DATA_COLOR),
            Text(num(node.grad), style=_GRAD_COLOR),
            wires,
        )

    con.print(table)
    con.print(
        Text("● data / forward", style=_DATA_COLOR)
        + Text("    ", style="")
        + Text("● grad / backward", style=_GRAD_COLOR)
    )


def to_html(fg: FlowGraph, out: str) -> str:
    """Write a self-contained interactive circuit page to ``out``; return path.

    Uses vis-network from a CDN (single ``<script>`` tag — no other local
    assets). Value nodes show ``label``, ``data`` (blue) and ``grad`` (orange)
    on multiple lines; operator pseudo-nodes are small circles. Edges are the
    directed "wires". Physics is enabled for a self-organising layout.
    """
    vis_nodes: list[dict] = []
    for node in fg.value_nodes():
        head = node.label or node.op or "·"
        multiline = (
            f"{head}\ndata {num(node.data)}\ngrad {num(node.grad)}"
        )
        vis_nodes.append(
            {
                "id": node.id,
                "label": multiline,
                "shape": "box",
                "color": {
                    "background": "#eff6ff",
                    "border": _DATA_COLOR,
                    "highlight": {"background": "#dbeafe", "border": _DATA_COLOR},
                },
                "font": {"multi": False, "align": "left", "color": "#0f172a"},
                "margin": 10,
            }
        )
    for node in fg.op_nodes():
        vis_nodes.append(
            {
                "id": node.id,
                "label": node.label,
                "shape": "circle",
                "size": 14,
                "color": {"background": "#fff7ed", "border": _GRAD_COLOR},
                "font": {"color": _GRAD_COLOR, "size": 18},
            }
        )

    vis_edges = [
        {"from": src, "to": dst, "arrows": "to", "color": {"color": "#64748b"}}
        for src, dst in fg.edges
    ]

    summary = (
        f"Computation circuit with {len(fg.value_nodes())} value nodes, "
        f"{len(fg.op_nodes())} operators, and {len(fg.edges)} wires. "
        "Blue shows forward data; orange shows backward gradient."
    )

    nodes_json = json.dumps(vis_nodes)
    edges_json = json.dumps(vis_edges)

    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>OptimumAI · Circuit</title>
<script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
<style>
  :root {{ --data: {_DATA_COLOR}; --grad: {_GRAD_COLOR}; }}
  html, body {{ margin: 0; height: 100%; font-family: Helvetica, Arial, sans-serif; }}
  #circuit {{ width: 100%; height: 82vh; border-bottom: 1px solid #e2e8f0; }}
  .sr-only {{
    position: absolute; width: 1px; height: 1px; padding: 0; margin: -1px;
    overflow: hidden; clip: rect(0,0,0,0); white-space: nowrap; border: 0;
  }}
  header {{ padding: 14px 18px 4px; }}
  header h1 {{ margin: 0; font-size: 18px; }}
  .legend {{ padding: 8px 18px 16px; font-size: 14px; color: #334155; }}
  .swatch {{
    display: inline-block; width: 12px; height: 12px; border-radius: 3px;
    margin: 0 6px 0 16px; vertical-align: middle;
  }}
  .swatch:first-child {{ margin-left: 0; }}
  .data {{ background: var(--data); }}
  .grad {{ background: var(--grad); }}
</style>
</head>
<body>
<header>
  <h1>⚡ Computation Circuit</h1>
  <h2 class="sr-only">{html.escape(summary)}</h2>
</header>
<div class="legend">
  <span class="swatch data"></span>blue = data / forward
  <span class="swatch grad"></span>orange = grad / backward
</div>
<div id="circuit"></div>
<script>
  const nodes = new vis.DataSet({nodes_json});
  const edges = new vis.DataSet({edges_json});
  const container = document.getElementById("circuit");
  const data = {{ nodes: nodes, edges: edges }};
  const options = {{
    physics: {{
      enabled: true,
      solver: "forceAtlas2Based",
      stabilization: {{ iterations: 200 }}
    }},
    layout: {{ improvedLayout: true }},
    interaction: {{ hover: true, tooltipDelay: 100 }},
    edges: {{ smooth: {{ type: "cubicBezier", roundness: 0.4 }} }}
  }};
  new vis.Network(container, data, options);
</script>
</body>
</html>
"""

    with open(out, "w", encoding="utf-8") as fh:
        fh.write(doc)
    return out


def render(
    source: object,
    fmt: str = "html",
    out: str | None = None,
) -> str | None:
    """Build a :class:`FlowGraph` from ``source`` and render it as ``fmt``.

    Args:
        source: A :class:`~optimumai.autograd.value.Value`, an already-built
            :class:`FlowGraph`, or an arithmetic expression ``str`` (evaluated
            with default variables via
            :func:`~optimumai.circuit.graph.build_from_expression`).
        fmt: One of ``"html"``, ``"dot"``, ``"terminal"``.
        out: Output path. Required for ``html`` (defaults to ``"circuit.html"``);
            optional for ``dot`` (writes the DOT text there if given).

    Returns:
        The output path (html), the DOT string (dot), or ``None`` (terminal).

    Raises:
        ValueError: If ``fmt`` is not a recognised format.
    """
    fg = _coerce_flowgraph(source)

    if fmt == "html":
        target = out or "circuit.html"
        return to_html(fg, target)
    if fmt == "dot":
        dot = to_dot(fg)
        if out:
            with open(out, "w", encoding="utf-8") as fh:
                fh.write(dot)
        return dot
    if fmt == "terminal":
        to_terminal(fg)
        return None

    raise ValueError(
        f"unknown fmt {fmt!r} (expected one of 'html', 'dot', 'terminal')"
    )


def _coerce_flowgraph(source: object) -> FlowGraph:
    """Normalise ``source`` (FlowGraph / Value / expression str) to a FlowGraph."""
    from optimumai.autograd.value import Value

    if isinstance(source, FlowGraph):
        return source
    if isinstance(source, Value):
        return build_from_value(source)
    if isinstance(source, str):
        _, fg = build_from_expression(source)
        return fg
    raise ValueError(
        f"cannot render {type(source).__name__}; "
        "pass a Value, a FlowGraph, or an expression string"
    )


def demo(out: str | None = None, fmt: str = "dot") -> str | None:
    """Render the classic micrograd example ``(a*b + c) * f`` in ``fmt``.

    Variables: ``a=2, b=-3, c=10, f=-2`` (so ``L = (2·-3+10)·-2 = -8``).
    Runs backward, then dispatches to :func:`render`. Defaults to ``dot`` so it
    is trivial to smoke-test.
    """
    _, fg = build_from_expression(
        "(a*b + c) * f",
        {"a": 2, "b": -3, "c": 10, "f": -2},
        run_backward=True,
    )
    return render(fg, fmt=fmt, out=out)
