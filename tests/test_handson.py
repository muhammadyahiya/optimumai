import os

import pytest
from click.testing import CliRunner

from optimumai.cli.main import cli
from optimumai.exercises.engine import Workbook, all_exercises, available_exercises
from optimumai.visualization.interactive import editable_plot


# --- editable equation ↔ graph (standalone HTML) ------------------------------
def test_editable_plot_html_structure():
    html = editable_plot(out=None)
    assert "plotly" in html and "math.min.js" in html
    assert "a*x^2" in html  # the default equation
    assert "range" in html  # parameter sliders


def test_editable_plot_writes_file(tmp_path):
    path = editable_plot("m*x + b", out=str(tmp_path / "e.html"))
    assert os.path.getsize(path) > 500
    assert "m*x + b" in open(path).read()


# --- compute-the-value exercises ----------------------------------------------
def test_exercise_bank_populated():
    assert len(all_exercises()) >= 10
    assert "dot" in available_exercises()


def test_exercise_grading():
    wb = Workbook("dot")
    ex = wb.exercises[0]
    assert wb.grade(ex.id, ex.answer).correct is True
    assert wb.grade(ex.id, ex.answer + 5).correct is False


def test_exercise_answers_are_selfconsistent():
    for ex in all_exercises():
        wb = Workbook(ex.lesson_id)
        assert wb.grade(ex.id, ex.answer).correct is True


def test_exercise_unknown_lesson_raises():
    with pytest.raises(KeyError):
        Workbook("no-such-lesson")


# --- GIF export (needs matplotlib) --------------------------------------------
def test_gif_export(tmp_path):
    pytest.importorskip("matplotlib")
    from optimumai.visualization.animate import (
        animate_diffusion,
        animate_gradient_descent,
        animate_softmax_temperature,
    )

    for fn in (animate_gradient_descent, animate_diffusion, animate_softmax_temperature):
        path = fn(out=str(tmp_path / f"{fn.__name__}.gif"))
        assert os.path.getsize(path) > 5000


# --- CLI ----------------------------------------------------------------------
def _run(*args, **kw):
    return CliRunner().invoke(cli, list(args), **kw)


def test_cli_kernel():
    result = _run("kernel", "vector_add")
    assert result.exit_code == 0
    assert "VECTOR" in result.output.upper()


def test_cli_kernel_list_and_backends():
    assert _run("kernel").exit_code == 0
    assert "simulator" in _run("kernel", "--backends").output


def test_cli_editor(tmp_path):
    out = str(tmp_path / "cli.html")
    result = _run("editor", "a*x + b", "--out", out)
    assert result.exit_code == 0 and os.path.exists(out)


def test_cli_exercise_interactive():
    wb = Workbook("dot")
    answers = "\n".join(str(ex.answer) for ex in wb.exercises) + "\n"
    result = _run("exercise", "dot", input=answers)
    assert result.exit_code == 0
    assert "Score:" in result.output
