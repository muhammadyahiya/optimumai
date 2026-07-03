import os

import pytest
from click.testing import CliRunner

from optimumai.autograd import Value
from optimumai.circuit.graph import FlowGraph, build_from_expression, build_from_value
from optimumai.circuit.render import render, to_dot, to_html, to_terminal
from optimumai.cli.main import cli


def test_build_from_expression_computes_and_backprops():
    root, fg = build_from_expression("(a*b + c) * f", {"a": 2, "b": -3, "c": 10, "f": -2})
    assert root.data == pytest.approx(-8.0)  # (2·-3+10)·-2 = -8
    assert isinstance(fg, FlowGraph)
    # 4 leaves + 2 intermediates + result = 7 value nodes
    assert len(fg.value_nodes()) == 7
    assert len(fg.edges) > 0


def test_leaf_gradients_are_set():
    root, fg = build_from_expression("(a*b + c) * f", {"a": 2, "b": -3, "c": 10, "f": -2})
    leaves = {n.label: n.grad for n in fg.value_nodes() if n.label in {"a", "b", "c", "f"}}
    assert leaves["a"] == pytest.approx(6.0)   # dL/da = b*f = -3*-2
    assert leaves["c"] == pytest.approx(-2.0)  # dL/dc = f


def test_unset_variables_default_to_one():
    root, _ = build_from_expression("a + b")  # a=b=1 → 2
    assert root.data == pytest.approx(2.0)


def test_expression_rejects_unsafe_input():
    with pytest.raises(ValueError):
        build_from_expression("__import__('os').system('echo hi')")


def test_expression_rejects_empty():
    with pytest.raises(ValueError):
        build_from_expression("   ")


def test_build_from_value():
    a = Value(3.0, label="a")
    out = a * a
    out.label = "y"
    out.backward()
    fg = build_from_value(out)
    assert any(n.is_op for n in fg.nodes)  # an operator pseudo-node exists


def test_to_dot_is_graphviz():
    _, fg = build_from_expression("a*b + c")
    dot = to_dot(fg)
    assert "digraph" in dot and "->" in dot


def test_to_terminal_runs(capsys):
    _, fg = build_from_expression("a*b + c")
    to_terminal(fg)
    assert capsys.readouterr().out  # printed something


def test_to_html_writes_file(tmp_path):
    _, fg = build_from_expression("a*b + c")
    path = to_html(fg, str(tmp_path / "c.html"))
    assert os.path.getsize(path) > 500
    assert "vis-network" in open(path).read()


def test_render_dispatch(tmp_path):
    assert "digraph" in render("a*b", fmt="dot")
    path = render("a*b", fmt="html", out=str(tmp_path / "r.html"))
    assert os.path.exists(path)
    with pytest.raises(ValueError):
        render("a*b", fmt="hologram")


def test_cli_circuit_terminal():
    result = CliRunner().invoke(cli, ["circuit", "(a*b+c)*f", "--vars", "a=2,b=-3,c=10,f=-2"])
    assert result.exit_code == 0


def test_cli_circuit_dot():
    result = CliRunner().invoke(cli, ["circuit", "a*b+c", "--fmt", "dot"])
    assert result.exit_code == 0
    assert "digraph" in result.output


def test_cli_circuit_html(tmp_path):
    out = str(tmp_path / "cli.html")
    result = CliRunner().invoke(cli, ["circuit", "a*b", "--fmt", "html", "--out", out])
    assert result.exit_code == 0
    assert os.path.exists(out)
