from click.testing import CliRunner

from optimumai import __version__
from optimumai.cli.main import cli


def _run(*args):
    return CliRunner().invoke(cli, list(args))


def test_version():
    result = _run("--version")
    assert result.exit_code == 0
    assert __version__ in result.output


def test_algebra_dot():
    result = _run("algebra", "dot", "[1,2,3]", "[4,5,6]")
    assert result.exit_code == 0
    assert "32" in result.output


def test_algebra_matmul():
    result = _run("algebra", "matmul", "[[1,2],[3,4]]", "[[5,6],[7,8]]")
    assert result.exit_code == 0
    assert "MATMUL" in result.output


def test_softmax_command():
    result = _run("softmax", "[2,1,0.1]", "--temperature", "0.5")
    assert result.exit_code == 0
    assert "SOFTMAX" in result.output


def test_attention_demo():
    result = _run("attention", "--demo")
    assert result.exit_code == 0
    assert "ATTENTION" in result.output


def test_attention_without_demo_errors():
    result = _run("attention")
    assert result.exit_code != 0


def test_learn_lists_topics():
    result = _run("learn")
    assert result.exit_code == 0
    assert "attention" in result.output


def test_learn_unknown_topic_errors():
    result = _run("learn", "quantum-gravity")
    assert result.exit_code != 0


def test_bad_vector_input_errors():
    result = _run("algebra", "dot", "not-a-vector", "[1,2,3]")
    assert result.exit_code != 0
