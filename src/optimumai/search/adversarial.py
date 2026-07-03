"""Adversarial search — choosing a move when an opponent is choosing against you.

Path-finding (:mod:`optimumai.search.uninformed` /
:mod:`optimumai.search.informed`) assumes a single agent moving through a
static world. Games add a second decision-maker who wants the opposite
outcome. **Minimax** formalizes this: build the tree of every possible
sequence of moves, label each leaf with a score (positive = good for the
*maximizing* player, negative = good for the *minimizing* player), and
propagate scores upward — MAX layers keep the best (largest) child value,
MIN layers keep the worst-for-MAX (smallest) child value. The root's value is
the outcome both players get if they play optimally; the child that achieves
it is the move to make.

**Alpha-beta pruning** computes the *exact same* root value while skipping
branches that provably cannot change it. It tracks a window ``(alpha, beta)``:

* ``alpha`` — the best value the MAX player can already guarantee somewhere
  else in the tree (a lower bound on the final answer).
* ``beta`` — the best value the MIN player can already guarantee elsewhere
  (an upper bound on the final answer).

While descending, a MIN node updates ``beta`` downward as it finds smaller
child values; a MAX node updates ``alpha`` upward as it finds larger ones.
The moment ``alpha >= beta`` at a node, that node's remaining children are
**pruned**: a MAX node facing ``alpha >= beta`` means the MIN parent already
has a move (worth ``beta``) that is at least as good for MIN as anything this
MAX node could still produce, so MIN would never let the game reach this MAX
node in the first place — searching further here cannot change the parent's
decision. The symmetric argument holds for MIN nodes pruning against a
MAX parent. Because a pruned branch is, by construction, one that could never
have supplied a *better* answer than what's already guaranteed elsewhere, the
returned value is provably identical to plain minimax — pruning only removes
provably-irrelevant work, never explores less of the *decision-relevant* tree.

Why AI cares
------------
Minimax with alpha-beta pruning (and its game-specific evaluation function
for positions too deep to search exhaustively) powered classical
superhuman game engines — most famously chess (Deep Blue, 1997) — decades
before neural approaches. Modern systems like AlphaZero still do tree search
at their core; they replace the hand-written evaluation function and
move-ordering heuristics with a learned neural network, but the underlying
game tree and min/max structure is the same idea explained here. Good move
ordering (searching the best move first) is what makes pruning bite hardest,
which is why this module's demo intentionally shows an ordering where
pruning does real work.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from optimumai.core._fmt import num
from optimumai.core.trace import Trace


@dataclass
class GameNode:
    """A node in a small, explicit game tree.

    A leaf has ``value`` set and no ``children``; an internal node has
    ``children`` and no ``value``. ``player`` records whose turn it is to
    move *at* this node (``"max"`` or ``"min"``), used only for the trace
    narration — the recursion itself alternates automatically.

    Attributes:
        name: Short label for tracing, e.g. ``"A"`` or ``"root"``.
        value: The static evaluation, only set on leaves.
        children: Child nodes reachable by one move, only set on internal nodes.
        player: Whose turn it is to move at this node.
    """

    name: str
    value: float | None = None
    children: list[GameNode] = field(default_factory=list)
    player: Literal["max", "min"] = "max"

    @property
    def is_leaf(self) -> bool:
        """True if this node has no children (a terminal position)."""
        return not self.children


def _validate(node: GameNode) -> None:
    if node.is_leaf and node.value is None:
        raise ValueError(f"leaf node {node.name!r} must have a value")
    if not node.is_leaf and node.value is not None:
        raise ValueError(f"internal node {node.name!r} must not have a static value")


def _validate_tree(node: GameNode) -> None:
    _validate(node)
    for child in node.children:
        _validate_tree(child)


def _fmt_bound(x: float) -> str:
    """Format an alpha/beta window bound, which may be genuinely infinite.

    ``optimumai.core._fmt.num`` assumes finite input (it calls ``int(x)``
    internally), so +/-inf — the natural starting window before any node has
    been evaluated — needs its own tiny formatter.
    """
    if x == float("inf"):
        return "+inf"
    if x == float("-inf"):
        return "-inf"
    return num(x)


def minimax_trace(node: GameNode, maximizing: bool = True) -> Trace:
    """Build the full trace of plain minimax evaluation over a game tree."""
    _validate_tree(node)

    t = Trace(
        op="minimax",
        formula="MAX picks max(children); MIN picks min(children); leaves return value",
        complexity="O(b^d) for branching factor b and depth d — every node is visited",
        why_ai=[
            "The formal model of optimal play in a two-player zero-sum game",
            "Every leaf and internal node is visited exactly once: the exhaustive baseline",
            "Alpha-beta (this module) returns the identical value while skipping branches",
        ],
        meta={"root": node.name},
    )
    nodes_visited = 0
    best_child_name: dict[str, str] = {}

    def recurse(n: GameNode, is_max: bool, depth: int) -> float:
        nonlocal nodes_visited
        nodes_visited += 1
        indent = "  " * depth
        if n.is_leaf:
            t.add(
                f"{indent}Leaf {n.name!r}",
                f"value({n.name!r}) = {num(n.value)}",
                n.value,
            )
            return n.value

        role = "MAX" if is_max else "MIN"
        t.add(
            f"{indent}Enter {role} node {n.name!r}",
            f"evaluate {len(n.children)} children of {n.name!r}",
            detail="",
        )
        child_values = []
        for child in n.children:
            v = recurse(child, not is_max, depth + 1)
            child_values.append((child.name, v))

        best_value = max(v for _, v in child_values) if is_max else min(v for _, v in child_values)
        best_name = next(name for name, v in child_values if v == best_value)
        best_child_name[n.name] = best_name
        t.add(
            f"{indent}{role} node {n.name!r} resolves",
            f"{role.lower()}({[num(v) for _, v in child_values]}) = {num(best_value)}  "
            f"(achieved by child {best_name!r})",
            best_value,
        )
        return best_value

    root_value = recurse(node, maximizing, 0)
    t.result = root_value
    t.meta.update(
        value=root_value,
        best_move=best_child_name.get(node.name),
        nodes_visited=nodes_visited,
    )
    return t


def minimax(node: GameNode, maximizing: bool = True) -> float:
    """Return the minimax value of the root of ``node``'s game tree."""
    return minimax_trace(node, maximizing=maximizing).result


def alpha_beta_trace(
    node: GameNode,
    maximizing: bool = True,
    alpha: float = float("-inf"),
    beta: float = float("inf"),
) -> Trace:
    """Build the full trace of minimax with alpha-beta pruning.

    Returns the identical root value :func:`minimax_trace` would (see the
    module docstring for why), while typically visiting far fewer nodes —
    the trace records every prune with the ``(alpha, beta)`` window active
    at the time.
    """
    _validate_tree(node)

    t = Trace(
        op="alpha_beta",
        formula="prune remaining children once alpha >= beta at a node",
        complexity="O(b^(d/2)) best case with good move ordering, vs O(b^d) for minimax",
        why_ai=[
            "Same guaranteed value as plain minimax — pruning skips only irrelevant work",
            "Move ordering matters: searching strong moves first prunes more aggressively",
            "The classical-AI ancestor of the tree search inside AlphaZero-style engines",
        ],
        meta={"root": node.name},
    )
    nodes_visited = 0
    nodes_pruned = 0
    best_child_name: dict[str, str] = {}

    def recurse(n: GameNode, is_max: bool, a: float, b: float, depth: int) -> float:
        nonlocal nodes_visited, nodes_pruned
        nodes_visited += 1
        indent = "  " * depth
        if n.is_leaf:
            t.add(
                f"{indent}Leaf {n.name!r}",
                f"value({n.name!r}) = {num(n.value)}",
                n.value,
            )
            return n.value

        role = "MAX" if is_max else "MIN"
        t.add(
            f"{indent}Enter {role} node {n.name!r}",
            f"window (alpha={_fmt_bound(a)}, beta={_fmt_bound(b)}) on entry",
            detail="",
        )
        best_value = float("-inf") if is_max else float("inf")
        best_name = None
        for i, child in enumerate(n.children):
            v = recurse(child, not is_max, a, b, depth + 1)
            if is_max:
                if v > best_value:
                    best_value, best_name = v, child.name
                a = max(a, best_value)
            else:
                if v < best_value:
                    best_value, best_name = v, child.name
                b = min(b, best_value)

            if a >= b:
                remaining = n.children[i + 1 :]
                if remaining:
                    nodes_pruned += _count_nodes(remaining)
                    t.add(
                        f"{indent}PRUNE at {n.name!r}",
                        f"alpha={_fmt_bound(a)} >= beta={_fmt_bound(b)} after child "
                        f"{child.name!r} → skip {[c.name for c in remaining]}",
                        detail=(
                            f"{role} already guarantees {num(best_value)}; the parent's "
                            "opposite-role choice elsewhere makes exploring the rest pointless."
                        ),
                    )
                break

        best_child_name[n.name] = best_name
        t.add(
            f"{indent}{role} node {n.name!r} resolves",
            f"{role.lower()}(...) = {num(best_value)}  (achieved by child {best_name!r})",
            best_value,
        )
        return best_value

    root_value = recurse(node, maximizing, alpha, beta, 0)
    t.result = root_value
    t.meta.update(
        value=root_value,
        best_move=best_child_name.get(node.name),
        nodes_visited=nodes_visited,
        nodes_pruned=nodes_pruned,
    )
    return t


def alpha_beta(
    node: GameNode,
    maximizing: bool = True,
    alpha: float = float("-inf"),
    beta: float = float("inf"),
) -> float:
    """Return the minimax value of ``node``'s game tree, computed with pruning."""
    return alpha_beta_trace(node, maximizing=maximizing, alpha=alpha, beta=beta).result


def _count_nodes(nodes: list[GameNode]) -> int:
    """Count a list of subtrees' total node count (used to size a prune)."""
    total = 0
    for n in nodes:
        total += 1 + _count_nodes(n.children)
    return total


def demo(seed: int = 0) -> Trace:
    """Alpha-beta over a small hand-built game tree, ordered so pruning fires.

    Tree (MAX at root, values are leaf utilities)::

        root (MAX)
        |-- A (MIN)
        |   |-- 3
        |   |-- 12
        |   `-- 8
        `-- B (MIN)
            |-- 2
            |-- 4
            `-- 6

    MIN(A) = min(3, 12, 8) = 3 and MIN(B) = min(2, 4, 6) = 2, so the root
    (MAX) picks A with value 3 — the highest guaranteed outcome for MAX.
    Exploring A fully sets alpha = 3 at the root before B is even opened.
    Inside B, the first leaf (2) immediately drops B's running beta to 2,
    and since alpha (3) >= beta (2) at that point, B's remaining two leaves
    (4 and 6) are pruned — they could only ever make B look *worse* for
    MIN's opponent's bound, which can't change the root's decision to play
    A. Plain minimax visits all 9 nodes; alpha-beta visits 7 and prunes 2.

    """
    root = GameNode(
        name="root",
        player="max",
        children=[
            GameNode(
                name="A",
                player="min",
                children=[
                    GameNode(name="A1", value=3),
                    GameNode(name="A2", value=12),
                    GameNode(name="A3", value=8),
                ],
            ),
            GameNode(
                name="B",
                player="min",
                children=[
                    GameNode(name="B1", value=2),
                    GameNode(name="B2", value=4),
                    GameNode(name="B3", value=6),
                ],
            ),
        ],
    )
    return alpha_beta_trace(root, maximizing=True)
