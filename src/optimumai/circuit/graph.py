"""Turn an autograd DAG into a renderable *circuit* graph.

This is the data model behind OptimumAI's "electrical circuit" view of AI math:
Karpathy's micrograd ``draw_dot`` fused with Anthropic's "circuits" framing.
A :class:`Value` computation graph is flattened into a :class:`FlowGraph` of
:class:`FlowNode` objects (value nodes + operator pseudo-nodes) and directed
edges — the "wires". Each renderer in :mod:`optimumai.circuit.render` consumes
one of these and paints the forward **data** (blue) and backward **grad**
(orange) flowing through it.

    >>> from optimumai.circuit.graph import build_from_expression
    >>> L, fg = build_from_expression("(a*b + c) * f",
    ...                               {"a": 2, "b": -3, "c": 10, "f": -2})
    >>> L.data
    -8.0
    >>> len(fg.value_nodes())
    6
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field

from optimumai.autograd.value import Value

# The AST node types we allow inside a user-supplied arithmetic expression.
# This is the safety boundary: anything outside this allow-list is rejected
# *before* we hand the string to ``eval`` (with an empty ``__builtins__``).
_ALLOWED_NODES: tuple[type[ast.AST], ...] = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Name,
    ast.Load,
    ast.Constant,
    # binary operators
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Pow,
    # unary operators
    ast.UAdd,
    ast.USub,
)


@dataclass
class FlowNode:
    """A single node in a :class:`FlowGraph`.

    Attributes:
        id: Stable identity string (``str(id(value))`` for value nodes,
            ``"<value_id>_op"`` for operator pseudo-nodes).
        label: Human label — a variable name (``"a"``), the result name
            (``"L"``), or an operator symbol (``"*"``).
        data: Forward value (the number that flows *forward* through the wire).
        grad: Backward gradient (∂L/∂node — flows *backward*).
        op: The op that produced this value (``""`` for leaves / op nodes carry
            the symbol here too).
        is_op: ``True`` for operator pseudo-nodes, ``False`` for value nodes.
    """

    id: str
    label: str
    data: float
    grad: float
    op: str
    is_op: bool


@dataclass
class FlowGraph:
    """A flattened, renderable view of a :class:`Value` computation DAG.

    Attributes:
        nodes: Every node — value nodes and operator pseudo-nodes.
        edges: Directed wires as ``(from_id, to_id)`` tuples.
    """

    nodes: list[FlowNode] = field(default_factory=list)
    edges: list[tuple[str, str]] = field(default_factory=list)

    def value_nodes(self) -> list[FlowNode]:
        """Return only the real value nodes (leaves + intermediates)."""
        return [n for n in self.nodes if not n.is_op]

    def op_nodes(self) -> list[FlowNode]:
        """Return only the operator pseudo-nodes."""
        return [n for n in self.nodes if n.is_op]


def build_from_value(root: Value) -> FlowGraph:
    """Flatten a :class:`Value` DAG into a :class:`FlowGraph`.

    Walks ``root._topo()`` (dependencies before dependents). Every ``Value``
    becomes a value node. Every ``Value`` that was *produced* by an op (has both
    ``_op`` and children) additionally gets an operator pseudo-node wired in
    between it and its children — mirroring micrograd's ``draw_dot``::

        child ─┐
               ├─▶ (op) ─▶ result
        child ─┘

    Leaves (no ``_op`` / no children) get no op node.
    """
    fg = FlowGraph()
    seen_edges: set[tuple[str, str]] = set()

    def add_edge(src: str, dst: str) -> None:
        edge = (src, dst)
        if edge not in seen_edges:
            seen_edges.add(edge)
            fg.edges.append(edge)

    for v in root._topo():
        vid = str(id(v))
        fg.nodes.append(
            FlowNode(
                id=vid,
                label=v.label or "",
                data=v.data,
                grad=v.grad,
                op=v._op,
                is_op=False,
            )
        )
        # Insert an operator pseudo-node when this value was produced by an op.
        if v._op and v._prev:
            op_id = f"{vid}_op"
            fg.nodes.append(
                FlowNode(
                    id=op_id,
                    label=v._op,
                    data=v.data,
                    grad=v.grad,
                    op=v._op,
                    is_op=True,
                )
            )
            for child in v._prev:
                add_edge(str(id(child)), op_id)
            add_edge(op_id, vid)

    return fg


def _validate_ast(tree: ast.AST) -> None:
    """Reject any node type outside the arithmetic allow-list.

    This is what makes evaluating a user string safe: calls, attributes,
    subscripts, comprehensions, etc. never reach ``eval``.
    """
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ValueError(
                f"disallowed syntax in expression: {type(node).__name__} "
                "(only names, numbers, and + - * / ** are permitted)"
            )


def build_from_expression(
    expression: str,
    variables: dict[str, float] | None = None,
    run_backward: bool = True,
) -> tuple[Value, FlowGraph]:
    """Compile a safe arithmetic ``expression`` into a Value graph + FlowGraph.

    Args:
        expression: An arithmetic string using variable names, numbers, and the
            operators ``+ - * / **`` with parentheses, e.g. ``"(a*b + c) * f"``.
        variables: Optional map of name → value. Names not present default to
            ``1.0``. Each distinct name becomes a labelled leaf ``Value``.
        run_backward: If ``True`` (default), run ``.backward()`` on the result so
            the graph carries gradients.

    Returns:
        ``(result_value, flow_graph)`` — the labelled result ``Value`` (label
        ``"L"``) and its flattened :class:`FlowGraph`.

    Raises:
        ValueError: If the expression is empty, cannot be parsed, or contains
            disallowed syntax (anything beyond names / numbers / arithmetic).
    """
    if not expression or not expression.strip():
        raise ValueError("expression is empty")

    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"invalid expression: {exc}") from exc

    _validate_ast(tree)

    variables = variables or {}
    # Collect every identifier referenced in the expression.
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    namespace: dict[str, Value] = {
        name: Value(float(variables.get(name, 1.0)), label=name) for name in names
    }

    code = compile(tree, filename="<expression>", mode="eval")
    result = eval(code, {"__builtins__": {}}, namespace)  # noqa: S307 - AST allow-listed above

    if not isinstance(result, Value):
        # e.g. a bare number like "3 + 4" — wrap so downstream stays uniform.
        result = Value(float(result))
    result.label = "L"

    if run_backward:
        result.backward()

    return result, build_from_value(result)
