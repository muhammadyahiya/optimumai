import pytest

from optimumai.search.adversarial import (
    GameNode,
    alpha_beta,
    alpha_beta_trace,
    minimax,
    minimax_trace,
)
from optimumai.search.adversarial import demo as adversarial_demo
from optimumai.search.informed import (
    astar,
    astar_trace,
    greedy_best_first,
    greedy_best_first_trace,
)
from optimumai.search.informed import demo as informed_demo
from optimumai.search.problem import Graph, GridWorld
from optimumai.search.uninformed import (
    bfs,
    bfs_trace,
    dfs,
    dfs_trace,
    uniform_cost_search,
    uniform_cost_search_trace,
)
from optimumai.search.uninformed import demo as uninformed_demo


# --- Fixtures -------------------------------------------------------------
def _weighted_graph() -> Graph:
    """A-B=1, A-C=4, B-C=2, B-D=5, C-D=1 (undirected).

    A -> D routes: A-B-D=6, A-C-D=5, A-B-C-D=4 (cheapest, 3 hops).
    A plain fewest-HOPS route (A-B-D or A-C-D) has only 2 hops but costs more,
    which is exactly why BFS (optimal for hop count) differs from UCS/A*
    (optimal for cost) on this graph.
    """
    g = Graph()
    g.add_edge("A", "B", 1)
    g.add_edge("A", "C", 4)
    g.add_edge("B", "C", 2)
    g.add_edge("B", "D", 5)
    g.add_edge("C", "D", 1)
    return g


def _walled_grid() -> GridWorld:
    """4x4 grid, walls forcing a detour from (0, 0) to (3, 3)."""
    return GridWorld(width=4, height=4, walls={(1, 0), (1, 1)})


class _GridAdapter:
    """Adapts GridWorld's 3-arg heuristic to the 2-arg (state, goal) interface."""

    def __init__(self, grid: GridWorld, kind: str = "manhattan") -> None:
        self._grid = grid
        self._kind = kind

    def neighbors(self, state):
        return self._grid.neighbors(state)

    def cost(self, a, b):
        return self._grid.cost(a, b)

    def heuristic(self, state, goal):
        return self._grid.heuristic(state, goal, kind=self._kind)


# --- Graph problem ----------------------------------------------------------
def test_graph_neighbors_and_cost():
    g = _weighted_graph()
    assert g.neighbors("A") == {"B": 1, "C": 4}
    assert g.cost("B", "C") == 2


def test_graph_rejects_negative_cost():
    g = Graph()
    with pytest.raises(ValueError):
        g.add_edge("A", "B", -1)


def test_graph_rejects_unknown_edge_or_state():
    g = _weighted_graph()
    with pytest.raises(KeyError):
        g.cost("A", "D")  # no direct edge
    with pytest.raises(KeyError):
        g.neighbors("Z")


# --- GridWorld problem --------------------------------------------------------
def test_gridworld_neighbors_respect_walls_and_bounds():
    gw = GridWorld(width=3, height=3, walls={(1, 1)})
    # corner (0, 0) only has 2 in-bounds neighbors, both free
    assert set(gw.neighbors((0, 0))) == {(0, 1), (1, 0)}
    # center wall is never returned as a neighbor of anything
    assert (1, 1) not in gw.neighbors((0, 1))


def test_gridworld_manhattan_heuristic_is_admissible_on_open_grid():
    gw = GridWorld(width=5, height=5)
    goal = (4, 4)
    # On an open (wall-free) grid the true shortest cost IS the Manhattan
    # distance, so h should never exceed it anywhere (h <= h*).
    for r in range(5):
        for c in range(5):
            h = gw.heuristic((r, c), goal, kind="manhattan")
            true_cost = abs(r - goal[0]) + abs(c - goal[1])
            assert h <= true_cost + 1e-9


def test_gridworld_rejects_bad_heuristic_kind():
    gw = GridWorld(width=2, height=2)
    with pytest.raises(ValueError):
        gw.heuristic((0, 0), (1, 1), kind="banana")


def test_gridworld_out_of_bounds_neighbor_lookup_raises():
    gw = GridWorld(width=2, height=2)
    with pytest.raises(KeyError):
        gw.neighbors((5, 5))


# --- BFS ----------------------------------------------------------------------
def test_bfs_finds_fewest_hop_path_not_necessarily_cheapest():
    g = _weighted_graph()
    path = bfs(g, "A", "D")
    # BFS is optimal for hop count: 2 hops (A-B-D or A-C-D), NOT the cheapest
    # 3-hop route A-B-C-D that UCS/A* find.
    assert len(path) - 1 == 2
    assert path[0] == "A" and path[-1] == "D"


def test_bfs_trivial_start_equals_goal():
    g = _weighted_graph()
    assert bfs(g, "A", "A") == ["A"]


def test_bfs_raises_on_unsolvable_graph():
    g = Graph()
    g.add_edge("A", "B", 1)
    g.add_edge("X", "Y", 1)  # disconnected from A/B
    with pytest.raises(ValueError):
        bfs(g, "A", "Y")


def test_bfs_raises_on_unknown_start():
    g = _weighted_graph()
    with pytest.raises(ValueError):
        bfs(g, "not-a-state", "A")


def test_bfs_trace_shape_and_metadata():
    t = bfs_trace(_weighted_graph(), "A", "D")
    assert t.op == "bfs"
    assert len(t) > 0
    assert t.result == t.meta["path"]
    assert t.formula
    assert len(t.why_ai) >= 2
    assert t.meta["nodes_expanded"] > 0


# --- DFS ------------------------------------------------------------------
def test_dfs_finds_a_valid_connected_path():
    g = _weighted_graph()
    path = dfs(g, "A", "D")
    assert path[0] == "A" and path[-1] == "D"
    # every consecutive pair must be a real edge
    for a, b in zip(path, path[1:], strict=False):
        g.cost(a, b)  # raises KeyError if not a real edge


def test_dfs_raises_on_unsolvable_graph():
    g = Graph()
    g.add_edge("A", "B", 1)
    g.add_edge("X", "Y", 1)
    with pytest.raises(ValueError):
        dfs(g, "A", "Y")


def test_dfs_trace_shape():
    t = dfs_trace(_weighted_graph(), "A", "D")
    assert t.op == "dfs"
    assert t.result[0] == "A"
    assert t.why_ai


# --- Uniform-cost search (Dijkstra) ------------------------------------------
def test_ucs_finds_optimal_cost_path():
    g = _weighted_graph()
    path = uniform_cost_search(g, "A", "D")
    assert path == ["A", "B", "C", "D"]  # hand-computed cheapest route, cost 4


def test_ucs_cost_matches_hand_computation():
    t = uniform_cost_search_trace(_weighted_graph(), "A", "D")
    assert t.meta["path_cost"] == pytest.approx(4.0)


def test_ucs_raises_on_unsolvable_graph():
    g = Graph()
    g.add_edge("A", "B", 1)
    g.add_edge("X", "Y", 1)
    with pytest.raises(ValueError):
        uniform_cost_search(g, "A", "Y")


def test_ucs_trace_shape_and_why_ai():
    t = uniform_cost_search_trace(_weighted_graph(), "A", "D")
    assert t.op == "uniform_cost_search"
    assert t.result == t.meta["path"]
    assert t.complexity
    assert len(t.why_ai) >= 2


# --- A* vs UCS: same optimal cost, fewer (or equal) expansions ----------------
def test_astar_matches_ucs_optimal_cost_on_weighted_graph():
    g = _weighted_graph()
    g.heuristics = {"A": 0, "B": 0, "C": 0, "D": 0}  # h=0 -> A* degenerates to UCS
    astar_path = astar(g, "A", "D")
    ucs_path = uniform_cost_search(g, "A", "D")
    assert astar_path == ucs_path == ["A", "B", "C", "D"]


def test_astar_cost_equals_ucs_cost_and_expands_no_more_on_grid():
    gw = _walled_grid()
    adapter = _GridAdapter(gw, kind="manhattan")
    start, goal = (0, 0), (3, 3)

    astar_t = astar_trace(adapter, start, goal)
    ucs_t = uniform_cost_search_trace(gw, start, goal)

    assert astar_t.meta["path_cost"] == pytest.approx(ucs_t.meta["path_cost"])
    assert astar_t.meta["nodes_expanded"] <= ucs_t.meta["nodes_expanded"]
    # both must be genuinely optimal, e.g. 6 moves around the two-cell wall
    assert astar_t.meta["path_cost"] == pytest.approx(6.0)


def test_astar_finds_the_optimal_grid_path():
    gw = _walled_grid()
    adapter = _GridAdapter(gw)
    path = astar(adapter, (0, 0), (3, 3))
    assert path[0] == (0, 0) and path[-1] == (3, 3)
    # every step must be a legal, free-cell move
    for a, b in zip(path, path[1:], strict=False):
        assert b in gw.neighbors(a)


def test_astar_raises_on_unsolvable_grid():
    # fully enclose the goal so it can never be reached
    gw = GridWorld(width=3, height=3, walls={(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)})
    adapter = _GridAdapter(gw)
    with pytest.raises(ValueError):
        astar(adapter, (0, 0), (2, 2))


def test_astar_trace_shows_g_h_f_in_meta_and_formula():
    t = astar_trace(_GridAdapter(_walled_grid()), (0, 0), (3, 3))
    assert t.op == "astar"
    assert "f(n)" in t.formula
    assert t.meta["path"] == t.result
    assert len(t.why_ai) >= 2


# --- Greedy best-first: fast but not guaranteed optimal -----------------------
def test_greedy_reaches_the_goal_with_a_valid_path():
    gw = _walled_grid()
    adapter = _GridAdapter(gw)
    path = greedy_best_first(adapter, (0, 0), (3, 3))
    assert path[0] == (0, 0) and path[-1] == (3, 3)
    for a, b in zip(path, path[1:], strict=False):
        assert b in gw.neighbors(a)


def test_greedy_trace_shape():
    t = greedy_best_first_trace(_GridAdapter(_walled_grid()), (0, 0), (3, 3))
    assert t.op == "greedy_best_first"
    assert t.result == t.meta["path"]


# --- Minimax vs alpha-beta: identical value, less work ------------------------
def _sample_tree() -> GameNode:
    """MAX root -> two MIN children A, B; MIN(A)=3, MIN(B)=2 -> root picks A=3."""
    return GameNode(
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


def test_minimax_hand_computed_value_and_move():
    t = minimax_trace(_sample_tree())
    assert t.result == 3  # max(min(3,12,8)=3, min(2,4,6)=2) = 3
    assert t.meta["best_move"] == "A"
    assert t.meta["nodes_visited"] == 9  # every one of the 9 nodes is visited


def test_alpha_beta_matches_minimax_value_with_fewer_expansions():
    tree_a, tree_b = _sample_tree(), _sample_tree()
    mm = minimax(tree_a)
    ab = alpha_beta(tree_b)
    assert ab == mm == 3

    mm_t = minimax_trace(_sample_tree())
    ab_t = alpha_beta_trace(_sample_tree())
    assert ab_t.result == mm_t.result
    assert ab_t.meta["nodes_visited"] < mm_t.meta["nodes_visited"]
    assert ab_t.meta["nodes_pruned"] > 0
    assert ab_t.meta["best_move"] == mm_t.meta["best_move"] == "A"


def test_alpha_beta_prunes_exactly_the_hand_computed_branches():
    # After exploring A fully, alpha=3 at the root. Entering B, the first
    # leaf (2) drops beta to 2 immediately; alpha(3) >= beta(2), so B's
    # remaining two leaves (B2, B3) are pruned before ever being visited.
    t = alpha_beta_trace(_sample_tree())
    assert t.meta["nodes_pruned"] == 2
    assert t.meta["nodes_visited"] == 7


def test_minimizing_root_flips_every_level_not_just_the_root():
    # maximizing=False flips whose turn it is at EVERY depth (roles alternate
    # from the root), not just the root's own aggregation. So A and B become
    # MAX nodes here: MAX(A)=max(3,12,8)=12, MAX(B)=max(2,4,6)=6, and the
    # MIN root then picks min(12, 6) = 6, achieved by B.
    t = minimax_trace(_sample_tree(), maximizing=False)
    assert t.result == 6
    assert t.meta["best_move"] == "B"


def test_gamenode_validation_rejects_malformed_trees():
    with pytest.raises(ValueError):
        # internal node must not carry a static value
        minimax_trace(GameNode(name="bad", value=5, children=[GameNode(name="c", value=1)]))
    with pytest.raises(ValueError):
        # leaf must carry a value
        minimax_trace(GameNode(name="leafless"))


# --- Demos: deterministic, importable, produce a real Trace -------------------
def test_uninformed_demo_is_deterministic_and_traced():
    t1, t2 = uninformed_demo(), uninformed_demo()
    assert t1.result == t2.result == ["A", "B", "C", "D"]
    assert t1.op == "uniform_cost_search"


def test_informed_demo_is_deterministic_and_traced():
    t1, t2 = informed_demo(), informed_demo()
    assert t1.result == t2.result
    assert t1.op == "astar"
    assert t1.meta["path_cost"] == pytest.approx(6.0)


def test_adversarial_demo_is_deterministic_and_prunes():
    t1, t2 = adversarial_demo(), adversarial_demo()
    assert t1.result == t2.result == 3
    assert t1.meta["nodes_pruned"] == t2.meta["nodes_pruned"] == 2
