"""Classical AI search — how agents find paths and choose moves.

Two families, four algorithms:

* **Path-finding** (:mod:`optimumai.search.uninformed`,
  :mod:`optimumai.search.informed`) — :func:`bfs`, :func:`dfs`,
  :func:`uniform_cost_search`, :func:`greedy_best_first`, and :func:`astar`
  all search a :class:`~optimumai.search.problem.Graph` or
  :class:`~optimumai.search.problem.GridWorld` for a route from a start state
  to a goal state.
* **Adversarial search** (:mod:`optimumai.search.adversarial`) —
  :func:`minimax` and :func:`alpha_beta` choose a move in a two-player game
  tree, with alpha-beta pruning finding the identical value while visiting
  fewer nodes.

Every function has a ``*_trace`` counterpart (e.g. :func:`astar_trace`) that
returns a full :class:`~optimumai.core.trace.Trace` of the search: frontier
contents, expansion order, and (for informed/adversarial search) the g/h/f
values or the alpha-beta window at every step.
"""

from optimumai.search.adversarial import (
    GameNode,
    alpha_beta,
    alpha_beta_trace,
    minimax,
    minimax_trace,
)
from optimumai.search.informed import (
    astar,
    astar_trace,
    greedy_best_first,
    greedy_best_first_trace,
)
from optimumai.search.problem import Graph, GridWorld
from optimumai.search.uninformed import (
    bfs,
    bfs_trace,
    dfs,
    dfs_trace,
    uniform_cost_search,
    uniform_cost_search_trace,
)

__all__ = [
    "GameNode",
    "Graph",
    "GridWorld",
    "alpha_beta",
    "alpha_beta_trace",
    "astar",
    "astar_trace",
    "bfs",
    "bfs_trace",
    "dfs",
    "dfs_trace",
    "greedy_best_first",
    "greedy_best_first_trace",
    "minimax",
    "minimax_trace",
    "uniform_cost_search",
    "uniform_cost_search_trace",
]
