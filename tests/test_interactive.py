import numpy as np
import pytest
from click.testing import CliRunner

from optimumai.analysis.compare import compare, compare_trace, sweep_trace
from optimumai.cli.main import cli
from optimumai.interactive.prompts import parse_matrix, parse_vector
from optimumai.interactive.repl import _apply_vars, _dispatch
from optimumai.symbolic.differentiate import differentiate, differentiate_trace
from optimumai.transformers.text_pipeline import TextPipeline


# --- text pipeline ------------------------------------------------------------
def test_text_pipeline_distribution_sums_to_one():
    t = TextPipeline.demo()
    assert np.sum(t.result) == pytest.approx(1.0, abs=1e-5)
    assert len(t) >= 5


def test_text_pipeline_rejects_empty():
    with pytest.raises(ValueError):
        TextPipeline("   ").trace()


# --- symbolic differentiation -------------------------------------------------
def test_differentiate_at_point():
    # d/dx (x³ + 2x) = 3x² + 2 → at 3 = 29
    assert differentiate("x**3 + 2*x", at=3.0) == pytest.approx(29.0)


def test_differentiate_symbolic_string_without_point():
    result = differentiate_trace("x**2").result
    assert isinstance(result, str) and "x" in result


# --- compare / sweep ----------------------------------------------------------
def test_compare_two_activations():
    t = compare_trace("relu", "gelu", [-2, -1, 0, 1, 2])
    out = np.asarray(t.result)
    assert out.shape[0] == 2  # two rows: relu, gelu
    assert np.allclose(out[0], [0, 0, 0, 1, 2])  # relu is exact


def test_compare_returns_value():
    result = compare("tanh", "sigmoid", input=[0.0])
    assert np.asarray(result).shape[0] == 2


def test_sweep_shape():
    t = sweep_trace("softmax", "temperature", [0.5, 1.0, 2.0])
    assert np.asarray(t.result).shape == (3, 3)


# --- prompt parsing -----------------------------------------------------------
def test_parse_vector_forms():
    assert parse_vector("[1, 2, 3]") == [1.0, 2.0, 3.0]
    assert parse_vector("1 2 3") == [1.0, 2.0, 3.0]
    assert parse_vector("1,2,3") == [1.0, 2.0, 3.0]


def test_parse_matrix():
    assert parse_matrix("[[1,2],[3,4]]") == [[1.0, 2.0], [3.0, 4.0]]


# --- repl helpers -------------------------------------------------------------
def test_apply_vars_injects_level():
    out = _apply_vars(["softmax", "[1,2,3]"], {"level": "engineer"})
    assert "--level" in out and "engineer" in out


def test_apply_vars_temperature_only_for_softmax():
    assert "--temperature" not in _apply_vars(["learn", "dot"], {"temperature": "0.5"})
    assert "--temperature" in _apply_vars(["softmax", "[1,2]"], {"temperature": "0.5"})


def test_dispatch_set_and_exit():
    variables: dict = {}
    assert _dispatch("set level engineer", variables, []) is True
    assert variables["level"] == "engineer"
    assert _dispatch("exit", variables, []) is False


# --- CLI ----------------------------------------------------------------------
def _run(*args):
    return CliRunner().invoke(cli, list(args))


def test_cli_trace_text():
    result = _run("trace-text", "hello world", "--layers", "1")
    assert result.exit_code == 0
    assert "TEXT" in result.output.upper()


def test_cli_compare():
    result = _run("compare", "relu", "gelu", "--input", "[-1,0,1]")
    assert result.exit_code == 0


def test_cli_sweep():
    result = _run("sweep", "softmax", "--values", "[0.5,1.0]")
    assert result.exit_code == 0


def test_cli_diff():
    result = _run("diff", "x**3 + 2*x", "--at", "3")
    assert result.exit_code == 0
    assert "29" in result.output


def test_cli_dot_interactive_via_stdin():
    # omitting args triggers a prompt; feed values on stdin
    result = CliRunner().invoke(cli, ["algebra", "dot"], input="[1,2,3]\n[4,5,6]\n")
    assert result.exit_code == 0
    assert "32" in result.output
