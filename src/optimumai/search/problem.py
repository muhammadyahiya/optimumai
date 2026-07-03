"""Search problems — the shared vocabulary every search algorithm plugs into.

Every classical search algorithm (BFS, DFS, UCS, greedy best-first, A*, minimax)
answers the same question: *given a state, what can I do next, what does it
cost, and how do I know I'm getting closer to the goal?* This module factors
that question into two tiny, dependency-free problem types so the algorithm
modules can stay generic:

* :class:`Graph` — an explicit set of states connected by weighted, directed
  edges (a city map, a puzzle's state-transition graph, ...). This is the
  textbook "search on a graph" setting: BFS/DFS/UCS/A* all just need
  ``neighbors(state)`` and ``cost(a, b)``.
* :class:`GridWorld` — a 2-D grid with blocked cells, 4-connected moves (no
  diagonals), and a *heuristic* to the goal. This is the classic path-finding
  demo (think: a robot in a warehouse, or a game NPC navigating a level) and
  is where admissible heuristics (Manhattan/Euclidean distance) earn their
  keep in A*.

Why AI cares
------------
Search is one of the oldest ideas in AI (Newell & Simon, 1950s) and it never
left: planning, robotics path-finding, puzzle solvers, game-tree search, and
even LLM decoding strategies like beam search are all instances of "explore a
space of states guided by cost and/or a heuristic." Getting the problem
formulation right — states, actions, costs, a goal test — is most of the work;
the algorithms in :mod:`optimumai.search.uninformed` and
:mod:`optimumai.search.informed` are then almost mechanical.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt

State = str
Coord = tuple[int, int]


@dataclass
class Graph:
    """An explicit directed, weighted graph used as a search problem.

    The adjacency structure is a plain ``dict[state, dict[neighbor, cost]]`` —
    no external graph library needed. Edges are directed: add both
    ``a -> b`` and ``b -> a`` for an undirected edge (see :meth:`add_edge`).

    Attributes:
        adjacency: ``{state: {neighbor: edge_cost, ...}, ...}``.
        heuristics: Optional straight-line-style estimates ``{state: h(state)}``
            for use with greedy best-first / A*. Defaults to 0 everywhere
            (which makes A* degrade gracefully to uniform-cost search).

    Example:
        >>> g = Graph()
        >>> g.add_edge("A", "B", 1)
        >>> g.add_edge("B", "C", 2)
        >>> g.neighbors("A")
        {'B': 1}
        >>> g.cost("B", "C")
        2
    """

    adjacency: dict[State, dict[State, float]] = field(default_factory=dict)
    heuristics: dict[State, float] = field(default_factory=dict)

    def add_edge(self, a: State, b: State, cost: float = 1.0, bidirectional: bool = True) -> None:
        """Add an edge ``a -> b`` with the given cost (and the reverse, by default)."""
        if cost < 0:
            raise ValueError(f"edge cost must be non-negative, got {cost}")
        self.adjacency.setdefault(a, {})[b] = cost
        self.adjacency.setdefault(b, self.adjacency.get(b, {}))
        if bidirectional:
            self.adjacency.setdefault(b, {})[a] = cost

    def neighbors(self, state: State) -> dict[State, float]:
        """Return ``{neighbor: edge_cost}`` reachable in one step from ``state``."""
        if state not in self.adjacency:
            raise KeyError(f"unknown state {state!r}")
        return dict(self.adjacency[state])

    def cost(self, a: State, b: State) -> float:
        """Return the edge cost of the direct move ``a -> b``."""
        try:
            return self.adjacency[a][b]
        except KeyError as exc:
            raise KeyError(f"no edge {a!r} -> {b!r}") from exc

    def heuristic(self, state: State, goal: State) -> float:
        """Return an estimate of the remaining cost from ``state`` to ``goal``.

        Falls back to 0 for states with no registered heuristic, which is
        always admissible (never overestimates) but uninformative.
        """
        if state == goal:
            return 0.0
        return self.heuristics.get(state, 0.0)

    def states(self) -> list[State]:
        """All states that appear as either an edge source or destination."""
        return list(self.adjacency.keys())


@dataclass
class GridWorld:
    """A 2-D grid path-finding problem: 4-connected moves, optional walls.

    Coordinates are ``(row, col)`` with row increasing downward, matching how
    the grid is usually printed. Movement is restricted to the 4 orthogonal
    neighbors (no diagonals), each costing 1 by default, so BFS already finds
    shortest *hop-count* paths here — the more interesting question is how
    much less work A* does than BFS/UCS once a heuristic is added.

    Attributes:
        width: Number of columns.
        height: Number of rows.
        walls: Set of blocked ``(row, col)`` cells that cannot be entered.

    Example:
        >>> gw = GridWorld(width=3, height=3, walls={(1, 1)})
        >>> sorted(gw.neighbors((0, 1)))
        [(0, 0), (0, 2), (1, 1)]
    """

    width: int
    height: int
    walls: set[Coord] = field(default_factory=set)

    def in_bounds(self, state: Coord) -> bool:
        """True if ``state`` lies within the grid's rectangle."""
        r, c = state
        return 0 <= r < self.height and 0 <= c < self.width

    def is_free(self, state: Coord) -> bool:
        """True if ``state`` is in bounds and not a wall."""
        return self.in_bounds(state) and state not in self.walls

    def neighbors(self, state: Coord) -> dict[Coord, float]:
        """Return ``{neighbor: 1.0}`` for the in-bounds, non-wall 4-neighbors.

        Note the returned dict includes cells even when they happen to be
        walls filtered out — walls are excluded entirely, never returned.
        """
        if not self.in_bounds(state):
            raise KeyError(f"state {state} is outside the {self.height}x{self.width} grid")
        r, c = state
        candidates = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
        return {n: 1.0 for n in candidates if self.is_free(n)}

    def cost(self, a: Coord, b: Coord) -> float:
        """Cost of moving from ``a`` to an adjacent free cell ``b`` (always 1)."""
        if b not in self.neighbors(a):
            raise KeyError(f"no move {a} -> {b}")
        return 1.0

    def heuristic(self, state: Coord, goal: Coord, kind: str = "manhattan") -> float:
        """Estimate the remaining cost from ``state`` to ``goal``.

        ``manhattan`` (``|dr| + |dc|``) is admissible here because every move
        costs exactly 1 and can only change row or column by 1 — the true
        remaining cost can never be *less* than the Manhattan distance.
        ``euclidean`` is also admissible (straight-line distance is never
        longer than a path constrained to a grid) but less *tight*, so it
        prunes less and A* typically expands more nodes with it than with
        Manhattan on a 4-connected grid.
        """
        (r1, c1), (r2, c2) = state, goal
        dr, dc = abs(r1 - r2), abs(c1 - c2)
        if kind == "manhattan":
            return float(dr + dc)
        if kind == "euclidean":
            return sqrt(dr**2 + dc**2)
        raise ValueError(f"kind must be 'manhattan' or 'euclidean', got {kind!r}")

    def render(self, path: list[Coord] | None = None) -> str:
        """Render the grid as text: ``#`` walls, ``*`` path, ``.`` free cells."""
        path_set = set(path or [])
        rows = []
        for r in range(self.height):
            row_chars = []
            for c in range(self.width):
                cell = (r, c)
                if cell in self.walls:
                    row_chars.append("#")
                elif cell in path_set:
                    row_chars.append("*")
                else:
                    row_chars.append(".")
            rows.append("".join(row_chars))
        return "\n".join(rows)
