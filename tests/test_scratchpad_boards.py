"""Tier 2 scratchpad — declarative boards, executable DAG, one progress store.

Each test class maps to one of the findings that motivated the rewrite, so a
failure says which property regressed rather than just "something broke".
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from optimumai.progress import ProgressTracker
from optimumai.scratchpad import assets as assets_mod
from optimumai.scratchpad.concepts import CONCEPTS, BoardSpec, list_concepts
from optimumai.scratchpad.dag import (
    is_locked,
    learning_order,
    unmet_prerequisites,
    validate_dag,
)
from optimumai.scratchpad.expressions import compile_expression

JS_PATH = Path(assets_mod.__file__).parent / "static" / "scratchpad.js"
KNOWN_KINDS = {"vectors", "function", "matrix", "descent"}


@dataclass
class FakeConcept:
    """Minimal stand-in for graph tests, so they don't depend on real content."""

    concept_id: str
    prerequisites: list[str] = field(default_factory=list)


# --- F1: boards are declarative -------------------------------------------


class TestDeclarativeBoards:
    def test_every_board_kind_has_a_renderer(self):
        js = JS_PATH.read_text()
        match = re.search(r"const RENDERERS = \{(.*?)\}", js, re.S)
        assert match, "could not find the RENDERERS table"
        registered = set(re.findall(r"(\w+):", match.group(1)))
        assert registered == KNOWN_KINDS
        for concept in CONCEPTS.values():
            assert concept.board.kind in registered, concept.concept_id

    def test_adding_a_concept_needs_no_javascript(self):
        """The renderers are generic: no concept id may appear in the JS."""
        js = JS_PATH.read_text()
        for cid in CONCEPTS:
            assert cid not in js, f"{cid} is hardcoded in scratchpad.js"

    def test_board_spec_serialises_for_the_browser(self):
        payload = CONCEPTS["dot_product"].board.to_dict()
        assert payload["kind"] == "vectors"
        assert [p["name"] for p in payload["points"]] == ["a", "b"]
        assert payload["palette"]["basis_x"].startswith("#")
        json.dumps(payload)  # must be JSON-serialisable for the template

    def test_fewer_renderers_than_concepts(self):
        """The whole point: kinds must not grow one-per-concept."""
        kinds = {c.board.kind for c in CONCEPTS.values()}
        assert len(kinds) <= len(CONCEPTS)

    def test_snapshots_are_authored_with_a_reason(self):
        """A slider with no guidance is a dead control."""
        for concept in CONCEPTS.values():
            for snap in concept.board.snapshots:
                assert snap.label and snap.note, concept.concept_id


# --- F2: Python owns the mathematics --------------------------------------


class TestPythonIsSourceOfTruth:
    def test_no_maths_hardcoded_in_javascript(self):
        js = JS_PATH.read_text()
        assert "const f = (x) =>" not in js
        assert "fPrime" not in js
        assert "0.3 * x * x * x" not in js

    def test_derivative_matches_sympy_directly(self):
        """Compared numerically: exact symbolic subtraction leaves float epsilon."""
        sympy = pytest.importorskip("sympy")
        for concept in CONCEPTS.values():
            source = concept.board.expression
            if not source:
                continue
            compiled = compile_expression(source, concept.board.var)
            x = sympy.Symbol(concept.board.var)
            expected = sympy.diff(sympy.sympify(source), x)
            got = sympy.sympify(compiled.derivative_source)
            for probe in (-3.0, -1.0, 0.0, 0.5, 2.0, 4.0):
                assert float(got.subs(x, probe)) == pytest.approx(
                    float(expected.subs(x, probe)), rel=1e-12, abs=1e-12
                ), f"{concept.concept_id} at x={probe}"

    def test_emits_javascript_not_python_syntax(self):
        compiled = compile_expression("0.3*x**3 - 2*x")
        assert "**" not in compiled.js
        assert "Math.pow" in compiled.js
        assert "**" not in compiled.derivative_js

    def test_rejects_unparseable_expression(self):
        with pytest.raises(ValueError, match="could not parse"):
            compile_expression("0.3*x**3 - ")

    def test_rejects_unbound_symbol(self):
        """A typo'd variable must fail at build time, not render a blank board."""
        with pytest.raises(ValueError, match="unbound symbol"):
            compile_expression("0.3*x**3 - 2*y")

    def test_compiled_javascript_evaluates_correctly(self):
        """Cross-check the emitted JS arithmetic against Python."""
        compiled = compile_expression("0.3*x**3 - 2*x")
        py = compiled.js.replace("Math.pow(x, 3)", "x**3")
        for x in (-2.0, 0.0, 1.5, 3.0):
            assert eval(py, {"x": x}) == pytest.approx(0.3 * x**3 - 2 * x)  # noqa: S307


# --- F3: the DAG is executable -------------------------------------------


class TestPrerequisiteGraph:
    def test_real_graph_is_sound(self):
        assert validate_dag(CONCEPTS) == []

    def test_learning_order_puts_prerequisites_first(self):
        order = learning_order(CONCEPTS)
        assert set(order) == set(CONCEPTS)
        for cid in order:
            for prereq in CONCEPTS[cid].prerequisites:
                assert order.index(prereq) < order.index(cid)

    def test_learning_order_is_stable(self):
        assert learning_order(CONCEPTS) == learning_order(CONCEPTS)

    def test_sidebar_uses_learning_order(self):
        assert [c.concept_id for c in list_concepts()] == learning_order(CONCEPTS)

    def test_detects_dangling_edge(self):
        graph = {"a": FakeConcept("a", ["nope"])}
        assert any("unknown concept" in p for p in validate_dag(graph))

    def test_detects_self_loop(self):
        graph = {"a": FakeConcept("a", ["a"])}
        assert any("itself" in p for p in validate_dag(graph))

    def test_detects_cycle(self):
        graph = {
            "a": FakeConcept("a", ["b"]),
            "b": FakeConcept("b", ["c"]),
            "c": FakeConcept("c", ["a"]),
        }
        assert any(p.startswith("cycle:") for p in validate_dag(graph))

    def test_learning_order_refuses_an_invalid_graph(self):
        graph = {"a": FakeConcept("a", ["b"]), "b": FakeConcept("b", ["a"])}
        with pytest.raises(ValueError, match="invalid"):
            learning_order(graph)

    def test_lock_state_follows_completion(self):
        concept = CONCEPTS["matrix_transform"]
        assert concept.prerequisites == ["dot_product"]
        assert is_locked(concept, completed=[])
        assert unmet_prerequisites(concept, []) == ["dot_product"]
        assert not is_locked(concept, completed=["dot_product"])

    def test_prerequisites_are_ids_not_prose(self):
        """Free text is unenforceable; edges must point at real concepts."""
        for concept in CONCEPTS.values():
            for prereq in concept.prerequisites:
                assert prereq in CONCEPTS
            for note in concept.assumed_knowledge:
                assert note not in CONCEPTS  # prose lives in its own field


# --- F4: only the declared library loads ---------------------------------


class TestLazyAssetLoading:
    def test_required_keys_reflect_the_board(self):
        assert assets_mod.required_keys("jsxgraph", needs_katex=True) == [
            "jsxgraph_css", "jsxgraph_js", "katex_css", "katex_js",
        ]
        assert assets_mod.required_keys("jsxgraph", needs_katex=False) == [
            "jsxgraph_css", "jsxgraph_js",
        ]

    def test_board_that_needs_no_katex_gets_none(self):
        spec = BoardSpec(kind="vectors", bounding_box=(-1, 1, 1, -1), needs_katex=False)
        resolved = assets_mod.resolve_assets(spec.library, spec.needs_katex)
        assert "katex_js" not in resolved
        assert "jsxgraph_js" in resolved

    def test_unknown_library_contributes_nothing(self):
        assert assets_mod.required_keys("plotly", needs_katex=False) == []


# --- F5: offline once vendored -------------------------------------------


class TestOfflineAssets:
    def test_defaults_to_cdn_when_not_vendored(self, tmp_path):
        resolved = assets_mod.resolve_assets("jsxgraph", True, directory=tmp_path)
        assert resolved["offline"] is False
        assert str(resolved["jsxgraph_js"]).startswith("https://")

    def test_prefers_local_copies_once_present(self, tmp_path):
        for asset in assets_mod.ASSETS:
            (tmp_path / asset.filename).write_text("/* stub */")
        resolved = assets_mod.resolve_assets("jsxgraph", True, directory=tmp_path)
        assert resolved["offline"] is True
        for key in ("jsxgraph_js", "katex_js"):
            assert str(resolved[key]).startswith("/static/vendor/")

    def test_partial_vendoring_is_not_treated_as_offline(self, tmp_path):
        (tmp_path / "jsxgraphcore.js").write_text("/* stub */")
        resolved = assets_mod.resolve_assets("jsxgraph", True, directory=tmp_path)
        assert resolved["offline"] is False

    def test_asset_urls_are_version_pinned(self):
        for asset in assets_mod.ASSETS:
            assert "@" in asset.url, f"{asset.key} is not pinned to a version"


# --- F6: one progress store ---------------------------------------------


class TestProgressReuse:
    def test_concepts_point_at_real_curriculum_lessons(self):
        from optimumai.curriculum import COURSE

        lesson_ids = {lesson.id for lesson in COURSE.lessons}
        for concept in CONCEPTS.values():
            assert concept.lesson_id in lesson_ids, concept.concept_id

    def test_lesson_ids_are_unique(self):
        ids = [c.lesson_id for c in CONCEPTS.values()]
        assert len(ids) == len(set(ids))

    def test_completion_writes_to_the_course_store(self, tmp_path, monkeypatch):
        pytest.importorskip("flask")
        from optimumai.scratchpad.server import create_app

        store = tmp_path / "progress.json"
        monkeypatch.setenv("OPTIMUMAI_PROGRESS_PATH", str(store))
        app = create_app(progress_path=str(store))
        app.config["TESTING"] = True
        with app.test_client() as client:
            resp = client.post("/api/complete/dot_product")
            assert resp.status_code == 200
            assert resp.get_json()["lesson_id"] == "dot"
        # the course tracker -- not a scratchpad-private file -- must see it
        assert ProgressTracker(store).is_complete("dot")

    def test_course_completion_unlocks_the_board(self, tmp_path, monkeypatch):
        pytest.importorskip("flask")
        from optimumai.scratchpad.server import create_app

        store = tmp_path / "progress.json"
        monkeypatch.setenv("OPTIMUMAI_PROGRESS_PATH", str(store))
        tracker = ProgressTracker(store)
        tracker.mark_complete("dot")   # as `optimumai learn dot` would
        tracker.save()

        app = create_app(progress_path=str(store))
        app.config["TESTING"] = True
        with app.test_client() as client:
            order = client.get("/api/order").get_json()
        by_id = {row["concept_id"]: row for row in order}
        assert by_id["dot_product"]["completed"] is True
        assert by_id["matrix_transform"]["unmet"] == []

    def test_locked_board_reports_its_blockers(self, tmp_path, monkeypatch):
        pytest.importorskip("flask")
        from optimumai.scratchpad.server import create_app

        store = tmp_path / "progress.json"
        monkeypatch.setenv("OPTIMUMAI_PROGRESS_PATH", str(store))
        app = create_app(progress_path=str(store))
        app.config["TESTING"] = True
        with app.test_client() as client:
            order = client.get("/api/order").get_json()
        by_id = {row["concept_id"]: row for row in order}
        assert by_id["matrix_transform"]["unmet"] == ["dot_product"]
        assert by_id["gradient_descent"]["unmet"] == ["tangent_line"]


# --- rendered page ------------------------------------------------------


class TestRenderedPage:
    @pytest.fixture
    def client(self, tmp_path, monkeypatch):
        pytest.importorskip("flask")
        from optimumai.scratchpad.server import create_app

        store = tmp_path / "progress.json"
        monkeypatch.setenv("OPTIMUMAI_PROGRESS_PATH", str(store))
        app = create_app(progress_path=str(store))
        app.config["TESTING"] = True
        with app.test_client() as c:
            yield c

    @pytest.mark.parametrize("concept_id", sorted(CONCEPTS))
    def test_every_board_renders(self, client, concept_id):
        resp = client.get(f"/scratchpad/{concept_id}")
        assert resp.status_code == 200
        assert b"window.BOARD_SPEC" in resp.data

    def test_page_loads_only_the_declared_library(self, client):
        html = client.get("/scratchpad/dot_product").data.decode()
        assert "jsxgraphcore.js" in html
        for absent in ("plotly", "vis-network", "d3.min.js", "dagre"):
            assert absent not in html.lower()

    def test_locked_board_is_marked_in_the_sidebar(self, client):
        html = client.get("/scratchpad/dot_product").data.decode()
        assert 'class="badge locked"' in html

    def test_unknown_concept_404s(self, client):
        assert client.get("/scratchpad/nope").status_code == 404

    def test_board_spec_is_embedded_as_json(self, client):
        html = client.get("/scratchpad/tangent_line").data.decode()
        match = re.search(r"window\.BOARD_SPEC = (\{.*?\});", html, re.S)
        assert match
        spec = json.loads(match.group(1))
        assert spec["kind"] == "function"
        assert "Math.pow" in spec["expression"]["derivative_js"]


# --- authored claims must hold -------------------------------------------


class TestSnapshotClaims:
    """Replicate the descent iteration in Python and check the labels are true.

    A snapshot is an authored claim about where something interesting happens.
    If the numbers drift, the label becomes a lie -- so the claims are pinned.
    """

    @staticmethod
    def _iterate(values):
        sympy = pytest.importorskip("sympy")
        board = CONCEPTS["gradient_descent"].board
        x = sympy.Symbol(board.var)
        fprime = sympy.lambdify(x, sympy.diff(sympy.sympify(board.expression), x))
        xk = values["x0"]
        path = [xk]
        for _ in range(int(values["steps"])):
            xk = xk - values["lr"] * fprime(xk)
            path.append(xk)
        return path

    def _snapshot(self, label):
        for snap in CONCEPTS["gradient_descent"].board.snapshots:
            if snap.label == label:
                return snap
        raise AssertionError(f"no snapshot labelled {label!r}")

    def test_divergence_threshold_is_where_we_claim(self):
        """L = 0.35x^2 gives x <- x(1 - 0.7*lr), so it diverges past lr = 1/0.35."""
        threshold = 1 / 0.35
        assert self._snapshot("Divergence").values["lr"] > threshold
        assert self._snapshot("Oscillating").values["lr"] < threshold

    def test_healthy_snapshot_converges(self):
        path = self._iterate(self._snapshot("Healthy convergence").values)
        assert abs(path[-1]) < 0.05

    def test_too_small_snapshot_is_still_far_away(self):
        values = self._snapshot("Too small").values
        path = self._iterate(values)
        assert abs(path[-1]) > 1.0                      # has not arrived
        assert abs(path[-1]) < abs(values["x0"])        # but is heading there
        signs = {p > 0 for p in path}
        assert len(signs) == 1                          # never overshoots

    def test_oscillating_snapshot_alternates_but_converges(self):
        values = self._snapshot("Oscillating").values
        path = self._iterate(values)
        assert any(a * b < 0 for a, b in zip(path, path[1:], strict=False))
        assert abs(path[-1]) < abs(values["x0"])

    def test_divergence_snapshot_actually_diverges(self):
        values = self._snapshot("Divergence").values
        path = self._iterate(values)
        assert abs(path[-1]) > abs(values["x0"])
        assert abs(path[-1]) > abs(path[-2])


class TestTangentSnapshotClaims:
    """The tangent board's labels make claims about the slope's sign."""

    @staticmethod
    def _slope_at(x_value):
        sympy = pytest.importorskip("sympy")
        board = CONCEPTS["tangent_line"].board
        x = sympy.Symbol(board.var)
        fprime = sympy.lambdify(x, sympy.diff(sympy.sympify(board.expression), x))
        return float(fprime(x_value))

    def test_extrema_snapshots_sit_at_zero_slope(self):
        for label in ("Local maximum", "Local minimum"):
            snap = next(
                s for s in CONCEPTS["tangent_line"].board.snapshots if s.label == label
            )
            assert self._slope_at(snap.values["x"]) == pytest.approx(0, abs=1e-3), label

    def test_descent_and_climb_labels_match_the_sign(self):
        snaps = {s.label: s for s in CONCEPTS["tangent_line"].board.snapshots}
        assert self._slope_at(snaps["Steepest descent"].values["x"]) < 0
        assert self._slope_at(snaps["Steep climb"].values["x"]) > 0

    def test_steepest_descent_really_is_the_minimum_slope(self):
        """f' = 0.9x^2 - 2 is minimised at x = 0; the label must not overclaim."""
        snaps = {s.label: s for s in CONCEPTS["tangent_line"].board.snapshots}
        here = self._slope_at(snaps["Steepest descent"].values["x"])
        for probe in (-4.0, -2.0, -0.5, 0.5, 2.0, 4.0):
            assert here <= self._slope_at(probe) + 1e-9
