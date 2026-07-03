"""Informed search — using a heuristic to search toward the goal, not just outward.

Uninformed search (:mod:`optimumai.search.uninformed`) treats every direction
as equally promising. Informed search adds a **heuristic** ``h(n)``: an
estimate of the cost remaining from state ``n`` to the goal. Two ways to use
it:

* **Greedy best-first search** expands the frontier state with the smallest
  ``h(n)`` — "always walk toward what looks closest to the goal." Fast and
  often finds *a* path quickly, but ``h`` is only an estimate: greedy can walk
  into a heuristic dead-end and the path it returns is not guaranteed optimal.
* **A\\*** expands the frontier state with the smallest ``f(n) = g(n) + h(n)``,
  combining the *actual* cost so far (``g``, like UCS) with the *estimated*
  cost remaining (``h``, like greedy). This balance is what makes A* both
  efficient (it still prefers promising directions) and correct (it never
  forgets how expensive it truly was to get here).

Admissibility, consistency, and why A* is optimal
--------------------------------------------------
A heuristic ``h`` is **admissible** if it never overestimates the true
remaining cost: ``h(n) <= h*(n)`` for every state ``n``, where ``h*`` is the
actual cheapest cost from ``n`` to the goal. Manhattan distance on a
4-connected unit-cost grid is admissible because you can never reach the goal
in *fewer* moves than the Manhattan distance.

With an admissible ``h``, A* is guaranteed to return an optimal path: suppose
A* popped a suboptimal goal node ``G`` first. Some ancestor ``n`` of the true
optimal path must still be sitting in the frontier (search hasn't reached the
goal along that path yet), with ``f(n) = g(n) + h(n) <= g(n) + h*(n) =
cost(optimal path)`` (the inequality is exactly admissibility). Since
``cost(optimal path) < g(G) = f(G)`` (G is suboptimal, h(G) = 0 at the goal),
we'd have ``f(n) < f(G)``, so a priority queue ordered by ``f`` would have
popped ``n`` before ``G`` — contradiction. So A* can never pop a suboptimal
goal first.

A heuristic is **consistent** (a.k.a. monotone) if, for every edge ``n -> n'``
with cost ``c(n, n')``, ``h(n) <= c(n, n') + h(n')`` — the heuristic itself
obeys the triangle inequality. Consistency is a strictly stronger condition
than admissibility, and it buys something extra: it guarantees ``f(n)`` is
non-decreasing along any path A* explores, which means the *first* time A*
pops a state, that state's ``g`` value is already optimal — exactly like UCS,
no re-expansion or cost-relaxation needed later. Manhattan/Euclidean distance
on a grid with unit-cost moves are both consistent.

Why AI cares
------------
A* (Hart, Nilsson & Raphael, 1968) is the workhorse of path-finding in
robotics and games. Its generalization — best-first search guided by a
learned or hand-crafted evaluation function — reappears throughout AI: beam
search in sequence decoding orders candidates by a score exactly analogous to
``f(n)``, and Monte Carlo Tree Search's selection step is "best-first with an
exploration bonus" in the same spirit.
"""

from __future__ import annotations

import heapq
from typing import Protocol

from optimumai.core._fmt import num
from optimumai.core.trace import Trace


class HeuristicProblem(Protocol):
    """The interface informed search needs: neighbors, edge costs, and a heuristic."""

    def neighbors(self, state: object) -> dict[object, float]: ...
    def cost(self, a: object, b: object) -> float: ...
    def heuristic(self, state: object, goal: object) -> float: ...


def _reconstruct_path(parent: dict, start: object, goal: object) -> list:
    path = [goal]
    node = goal
    while node != start:
        node = parent[node]
        path.append(node)
    path.reverse()
    return path


def _validate_endpoints(problem: HeuristicProblem, start: object) -> None:
    problem.neighbors(start)  # raises KeyError if unknown, matching uninformed.py


def greedy_best_first_trace(problem: HeuristicProblem, start: object, goal: object) -> Trace:
    """Build the full trace of greedy best-first search (order by ``h(n)`` alone)."""
    _validate_endpoints(problem, start)

    t = Trace(
        op="greedy_best_first",
        formula="frontier = priority queue ordered by h(n) alone (ignores g(n))",
        complexity="O(E log V) with a binary heap; NOT optimal in general",
        why_ai=[
            "Fast: always lunges toward what looks nearest to the goal",
            "No optimality guarantee — a misleading h(n) can walk it into a costly detour",
            "The 'h(n) only' extreme that A* fixes by also tracking real cost g(n)",
        ],
        meta={"start": start, "goal": goal},
    )

    counter = 0
    h_start = problem.heuristic(start, goal)
    frontier: list = [(h_start, counter, start)]
    visited = {start}
    parent: dict = {}
    g_cost = {start: 0.0}
    order: list = []

    found = False
    while frontier:
        h, _, state = heapq.heappop(frontier)
        order.append(state)
        t.add(
            f"Expand {state!r}",
            f"h({state!r}) = {num(h)} (smallest in frontier)",
            detail=f"Goal test: {state!r} == {goal!r}? {state == goal}",
        )
        if state == goal:
            found = True
            break
        for neighbor, edge_cost in problem.neighbors(state).items():
            if neighbor in visited:
                continue
            visited.add(neighbor)
            parent[neighbor] = state
            g_cost[neighbor] = g_cost[state] + edge_cost
            h_n = problem.heuristic(neighbor, goal)
            counter += 1
            heapq.heappush(frontier, (h_n, counter, neighbor))
            t.add(
                f"Discover {neighbor!r} from {state!r}",
                f"h({neighbor!r}) = {num(h_n)}  (g not used for ordering here)",
                detail=f"parent[{neighbor!r}] = {state!r}",
            )

    if not found:
        raise ValueError(f"no path from {start!r} to {goal!r}")

    path = _reconstruct_path(parent, start, goal)
    cost = g_cost[goal]
    t.add(
        "Reconstruct path",
        " → ".join(map(str, path)),
        path,
        detail=f"Path cost = {num(cost)}. Greedy gives no guarantee this is minimal.",
    )
    t.result = path
    t.meta.update(path=path, path_cost=cost, nodes_expanded=len(order), expansion_order=order)
    return t


def greedy_best_first(problem: HeuristicProblem, start: object, goal: object) -> list:
    """Return *a* path from ``start`` to ``goal`` via greedy best-first search."""
    return greedy_best_first_trace(problem, start, goal).result


def astar_trace(problem: HeuristicProblem, start: object, goal: object) -> Trace:
    """Build the full trace of A* search (order by ``f(n) = g(n) + h(n)``).

    Optimal whenever ``problem.heuristic`` is admissible (never overestimates
    the true remaining cost) — see the module docstring for the proof sketch.
    """
    _validate_endpoints(problem, start)

    t = Trace(
        op="astar",
        formula="f(n) = g(n) + h(n);  frontier = priority queue ordered by f(n)",
        complexity="O(E log V) with a binary heap; optimal if h is admissible",
        why_ai=[
            "Optimal AND efficient: g(n) guarantees correctness, h(n) guarantees focus",
            "Standard path-finding in robotics, games, and route planners",
            "Best-first-with-a-score reappears as beam search / MCTS selection in modern AI",
        ],
        meta={"start": start, "goal": goal},
    )

    counter = 0
    h_start = problem.heuristic(start, goal)
    frontier: list = [(h_start, counter, start)]
    best_g = {start: 0.0}
    parent: dict = {}
    order: list = []
    expanded: set = set()

    while frontier:
        f, _, state = heapq.heappop(frontier)
        if state in expanded:
            continue  # stale heap entry from a since-improved g
        g = best_g[state]
        expanded.add(state)
        order.append(state)
        h = f - g
        t.add(
            f"Pop {state!r} (smallest f in frontier)",
            f"g({state!r}) = {num(g)}, h({state!r}) = {num(h)}, f = {num(f)}",
            detail=f"Goal test: {state!r} == {goal!r}? {state == goal}",
        )
        if state == goal:
            break
        for neighbor, edge_cost in problem.neighbors(state).items():
            if neighbor in expanded:
                continue
            new_g = g + edge_cost
            if new_g < best_g.get(neighbor, float("inf")):
                best_g[neighbor] = new_g
                parent[neighbor] = state
                h_n = problem.heuristic(neighbor, goal)
                new_f = new_g + h_n
                counter += 1
                heapq.heappush(frontier, (new_f, counter, neighbor))
                t.add(
                    f"Relax {neighbor!r} via {state!r}",
                    f"g({neighbor!r}) = {num(g)} + {num(edge_cost)} = {num(new_g)}, "
                    f"h({neighbor!r}) = {num(h_n)}, f = {num(new_f)}",
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
        detail=f"Path cost = {num(cost)} — optimal because h is admissible.",
    )
    t.result = path
    t.meta.update(path=path, path_cost=cost, nodes_expanded=len(order), expansion_order=order)
    return t


def astar(problem: HeuristicProblem, start: object, goal: object) -> list:
    """Return the optimal path from ``start`` to ``goal`` via A* (admissible ``h`` assumed)."""
    return astar_trace(problem, start, goal).result


def demo(seed: int = 0) -> Trace:
    """Run A* on a small 4x4 grid with a wall, for docs and the CLI.

    Grid (S=start, G=goal, #=wall)::

        S . . .
        # # . .
        . . . .
        . . . G

    The wall forces a detour; A* with Manhattan-distance h should expand
    fewer nodes than plain BFS/UCS to find the same optimal path.
    """
    from optimumai.search.problem import GridWorld

    gw = GridWorld(width=4, height=4, walls={(1, 0), (1, 1)})
    return astar_trace(_GridAdapter(gw), (0, 0), (3, 3))


class _GridAdapter:
    """Binds ``GridWorld.heuristic`` to a fixed goal-agnostic 2-arg call for demos."""

    def __init__(self, grid) -> None:
        self._grid = grid

    def neighbors(self, state):
        return self._grid.neighbors(state)

    def cost(self, a, b):
        return self._grid.cost(a, b)

    def heuristic(self, state, goal):
        return self._grid.heuristic(state, goal, kind="manhattan")
