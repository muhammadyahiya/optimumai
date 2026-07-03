"""Uninformed search — finding a path with no sense of "closer" to the goal.

Breadth-first search (BFS), depth-first search (DFS), and uniform-cost search
(UCS) all share one skeleton: keep a *frontier* of states discovered but not
yet expanded, repeatedly pop one, check if it's the goal, and otherwise push
its unvisited neighbors. What differs is purely *which state the frontier
hands back next*:

* **BFS** uses a FIFO queue (first discovered, first expanded) → explores the
  graph in layers by hop-count, so it finds the path with the *fewest edges*.
  It is optimal only when every edge costs the same (unit-cost graphs) —
  otherwise a "shorter" path in hops can be more expensive in total cost.
* **DFS** uses a LIFO stack (most recently discovered, first expanded) →
  plunges down one branch before backtracking. Cheap in memory, but gives no
  optimality guarantee at all: the first path it finds could be far from
  shortest.
* **UCS** (uniform-cost search, i.e. Dijkstra's algorithm run until the goal
  is popped) uses a priority queue ordered by accumulated path cost ``g(n)`` →
  always expands the cheapest-so-far frontier state next. This makes it
  optimal for *any* graph with non-negative edge costs, unit or not — the
  first time UCS pops the goal, that cost is provably the cheapest possible.

Why AI cares
------------
These are the textbook baselines every "smarter" search algorithm is measured
against. Planning systems, robot motion planning, and puzzle solvers reach
for BFS when action costs are uniform, UCS (Dijkstra) whenever costs differ,
and DFS-style exploration whenever memory is the bottleneck (e.g. iterative
deepening in game engines). A* in :mod:`optimumai.search.informed` is best
understood as "UCS plus a heuristic that skips the search directions we
already know are hopeless."
"""

from __future__ import annotations

import heapq
from collections import deque
from typing import Protocol

from optimumai.core._fmt import num
from optimumai.core.trace import Trace


class SearchProblem(Protocol):
    """The minimal interface every search algorithm in this package needs."""

    def neighbors(self, state: object) -> dict[object, float]: ...
    def cost(self, a: object, b: object) -> float: ...


def _reconstruct_path(parent: dict, start: object, goal: object) -> list:
    """Walk parent pointers backward from ``goal`` to ``start``, then reverse."""
    path = [goal]
    node = goal
    while node != start:
        node = parent[node]
        path.append(node)
    path.reverse()
    return path


def _path_cost(problem: SearchProblem, path: list) -> float:
    """Sum edge costs along a path (0 for a single-state or empty path)."""
    return sum(problem.cost(path[i], path[i + 1]) for i in range(len(path) - 1))


def _validate_endpoints(problem: SearchProblem, start: object, goal: object) -> None:
    try:
        problem.neighbors(start)
    except KeyError as exc:
        raise ValueError(f"start state {start!r} is not part of the problem") from exc


def bfs_trace(problem: SearchProblem, start: object, goal: object) -> Trace:
    """Build the full trace of breadth-first search from ``start`` to ``goal``."""
    _validate_endpoints(problem, start, goal)

    t = Trace(
        op="bfs",
        formula="frontier = FIFO queue; expand in the order states were discovered",
        complexity="O(V + E) time and space",
        why_ai=[
            "Finds the fewest-edges path — optimal when every action costs the same",
            "The template for level-by-level exploration: puzzle solvers, shortest "
            "hop-count routing, breadth-limited web crawling",
            "Its FIFO discipline is exactly what makes it explore in complete layers",
        ],
        meta={"start": start, "goal": goal},
    )

    frontier: deque = deque([start])
    visited = {start}
    parent: dict = {}
    order: list = []

    if start == goal:
        t.add("Goal test on start", f"{start!r} == goal → done immediately", detail="")
        t.result = [start]
        t.meta.update(path=[start], path_cost=0.0, nodes_expanded=0, expansion_order=[])
        return t

    found = False
    while frontier:
        state = frontier.popleft()
        order.append(state)
        t.add(
            f"Expand {state!r}",
            f"frontier (FIFO) = {list(frontier)!r} before pop → pop {state!r}",
            detail=f"Visited so far: {sorted(map(str, visited))}",
        )
        for neighbor in problem.neighbors(state):
            if neighbor in visited:
                continue
            visited.add(neighbor)
            parent[neighbor] = state
            frontier.append(neighbor)
            t.add(
                f"Discover {neighbor!r} from {state!r}",
                f"push {neighbor!r} onto frontier, parent[{neighbor!r}] = {state!r}",
                detail="Goal test happens at discovery time in this implementation.",
            )
            if neighbor == goal:
                found = True
                break
        if found:
            break

    if not found:
        raise ValueError(f"no path from {start!r} to {goal!r}")

    path = _reconstruct_path(parent, start, goal)
    cost = _path_cost(problem, path)
    t.add(
        "Reconstruct path",
        " → ".join(map(str, path)),
        path,
        detail=f"Path cost = {num(cost)} (BFS optimizes edge COUNT, not necessarily this cost).",
    )
    t.result = path
    t.meta.update(
        path=path,
        path_cost=cost,
        nodes_expanded=len(order),
        expansion_order=order,
    )
    return t


def bfs(problem: SearchProblem, start: object, goal: object) -> list:
    """Return the fewest-edges path from ``start`` to ``goal`` (BFS)."""
    return bfs_trace(problem, start, goal).result


def dfs_trace(problem: SearchProblem, start: object, goal: object) -> Trace:
    """Build the full trace of depth-first search from ``start`` to ``goal``."""
    _validate_endpoints(problem, start, goal)

    t = Trace(
        op="dfs",
        formula="frontier = LIFO stack; expand the most recently discovered state",
        complexity="O(V + E) time, O(V) space (much less than BFS on wide graphs)",
        why_ai=[
            "Cheap in memory: only the current path plus siblings sit on the stack",
            "No optimality guarantee — the first path found may be far from shortest",
            "The basis of backtracking search used in CSP solvers and puzzle solving",
        ],
        meta={"start": start, "goal": goal},
    )

    stack: list = [start]
    visited = {start}
    parent: dict = {}
    order: list = []

    found = start == goal
    while stack and not found:
        state = stack.pop()
        order.append(state)
        t.add(
            f"Expand {state!r}",
            f"pop {state!r} from stack (LIFO); stack now = {stack!r}",
            detail=f"Goal test: {state!r} == {goal!r}? {state == goal}",
        )
        if state == goal:
            found = True
            break
        for neighbor in problem.neighbors(state):
            if neighbor in visited:
                continue
            visited.add(neighbor)
            parent[neighbor] = state
            stack.append(neighbor)
            t.add(
                f"Discover {neighbor!r} from {state!r}",
                f"push {neighbor!r} onto stack, parent[{neighbor!r}] = {state!r}",
                detail="",
            )

    if not found:
        raise ValueError(f"no path from {start!r} to {goal!r}")

    path = [start] if start == goal else _reconstruct_path(parent, start, goal)
    cost = _path_cost(problem, path)
    t.add(
        "Reconstruct path",
        " → ".join(map(str, path)),
        path,
        detail=f"Path cost = {num(cost)}. DFS gives no guarantee this is minimal.",
    )
    t.result = path
    t.meta.update(
        path=path,
        path_cost=cost,
        nodes_expanded=len(order),
        expansion_order=order,
    )
    return t


def dfs(problem: SearchProblem, start: object, goal: object) -> list:
    """Return *a* path from ``start`` to ``goal`` found via DFS (not necessarily optimal)."""
    return dfs_trace(problem, start, goal).result


def uniform_cost_search_trace(problem: SearchProblem, start: object, goal: object) -> Trace:
    """Build the full trace of uniform-cost search (Dijkstra to a single goal)."""
    _validate_endpoints(problem, start, goal)

    t = Trace(
        op="uniform_cost_search",
        formula="frontier = priority queue ordered by g(n) = accumulated path cost",
        complexity="O(E log V) with a binary heap",
        why_ai=[
            "Optimal for any graph with non-negative edge costs, unlike plain BFS",
            "Exactly Dijkstra's algorithm, stopped early once the goal is popped",
            "The 'g(n) only' special case of A* (informed.py) with h(n) = 0 everywhere",
        ],
        meta={"start": start, "goal": goal},
    )

    counter = 0  # tie-breaker so heap never compares states directly
    frontier: list = [(0.0, counter, start)]
    best_g = {start: 0.0}
    parent: dict = {}
    order: list = []
    expanded: set = set()

    while frontier:
        g, _, state = heapq.heappop(frontier)
        if state in expanded:
            continue  # stale entry from a since-improved path
        expanded.add(state)
        order.append(state)
        t.add(
            f"Pop {state!r} (cheapest in frontier)",
            f"g({state!r}) = {num(g)}; remaining frontier = "
            f"{sorted((num(c), str(s)) for c, _, s in frontier)}",
            detail=f"Goal test: {state!r} == {goal!r}? {state == goal}",
        )
        if state == goal:
            break
        for neighbor, edge_cost in problem.neighbors(state).items():
            new_g = g + edge_cost
            if neighbor in expanded:
                continue
            if new_g < best_g.get(neighbor, float("inf")):
                best_g[neighbor] = new_g
                parent[neighbor] = state
                counter += 1
                heapq.heappush(frontier, (new_g, counter, neighbor))
                t.add(
                    f"Relax {neighbor!r} via {state!r}",
                    f"g({neighbor!r}) = {num(g)} + {num(edge_cost)} = {num(new_g)} "
                    "(better than any previous route)",
                    detail=f"parent[{neighbor!r}] = {state!r}",
                )

    if goal not in expanded:
        raise ValueError(f"no path from {start!r} to {goal!r}")

    path = _reconstruct_path(parent, start, goal) if start != goal else [start]
    cost = best_g[goal]
    t.add(
        "Reconstruct optimal path",
        " → ".join(map(str, path)),
        path,
        detail=f"Path cost = {num(cost)} — provably minimal (UCS is optimal here).",
    )
    t.result = path
    t.meta.update(
        path=path,
        path_cost=cost,
        nodes_expanded=len(order),
        expansion_order=order,
    )
    return t


def uniform_cost_search(problem: SearchProblem, start: object, goal: object) -> list:
    """Return the minimum-cost path from ``start`` to ``goal`` (UCS / Dijkstra)."""
    return uniform_cost_search_trace(problem, start, goal).result


def demo(seed: int = 0) -> Trace:
    """Run UCS on a small hand-drawn weighted graph for docs and the CLI.

    Graph (undirected):  A-B=1, A-C=4, B-C=2, B-D=5, C-D=1
    Cheapest A→D route:  A-B-C-D = 1+2+1 = 4  (vs. direct A-C-D = 4+1 = 5,
    or the 2-hop-looking-shorter B-D leg = 1+5 = 6).
    """
    from optimumai.search.problem import Graph

    g = Graph()
    g.add_edge("A", "B", 1)
    g.add_edge("A", "C", 4)
    g.add_edge("B", "C", 2)
    g.add_edge("B", "D", 5)
    g.add_edge("C", "D", 1)
    return uniform_cost_search_trace(g, "A", "D")
