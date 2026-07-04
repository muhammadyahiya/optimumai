"""Tests for the NumPy and matplotlib tutorials: build, compile, run, export."""

from __future__ import annotations

import pytest

from optimumai.tutorials import get_tutorial

# --------------------------------------------------------------------------
# numpy
# --------------------------------------------------------------------------


def test_numpy_tutorial_builds_with_enough_cells():
    tut = get_tutorial("numpy")
    assert tut.name == "numpy"
    assert len(tut.code_cells()) >= 16


def test_numpy_tutorial_code_cells_compile():
    tut = get_tutorial("numpy")
    for i, cell in enumerate(tut.code_cells()):
        compile(cell.text, f"<numpy:cell{i}>", "exec")


def test_numpy_tutorial_runs_and_returns_known_values():
    tut = get_tutorial("numpy")
    outputs = tut.run(execute=True)
    assert outputs
    joined = "\n".join(outputs.values())
    # broadcasting section prints this exact line
    assert "trailing dimensions must match" not in joined  # sanity: prose not leaked into code
    assert "same seed -> identical sequence: True" in joined
    assert "probabilities sum to 1: True" in joined
    assert "vectorized was ~" in joined


def test_numpy_tutorial_all_cells_run_on_base_install():
    tut = get_tutorial("numpy")
    for cell in tut.code_cells():
        assert cell.requires == ()
        assert cell.runnable()


# --------------------------------------------------------------------------
# matplotlib
# --------------------------------------------------------------------------

pytest.importorskip("matplotlib")


def test_matplotlib_tutorial_builds_with_enough_cells():
    tut = get_tutorial("matplotlib")
    assert tut.name == "matplotlib"
    assert len(tut.code_cells()) >= 8


def test_matplotlib_tutorial_code_cells_compile():
    tut = get_tutorial("matplotlib")
    for i, cell in enumerate(tut.code_cells()):
        compile(cell.text, f"<matplotlib:cell{i}>", "exec")


def test_matplotlib_tutorial_runs_headless_without_error():
    tut = get_tutorial("matplotlib")
    outputs = tut.run(execute=True)
    assert outputs
    joined = "\n".join(outputs.values())
    assert "saved" in joined.lower()
    assert "show()" not in joined


def test_matplotlib_tutorial_cells_require_matplotlib_and_never_call_show():
    tut = get_tutorial("matplotlib")
    for cell in tut.code_cells():
        assert cell.requires == ("matplotlib",)
        assert "plt.show()" not in cell.text
        assert "savefig" in cell.text or "matplotlib.use" in cell.text


# --------------------------------------------------------------------------
# notebook export
# --------------------------------------------------------------------------


def test_numpy_tutorial_exports_notebook(tmp_path):
    tut = get_tutorial("numpy")
    path = tut.to_notebook(str(tmp_path / "numpy.ipynb"))
    assert (tmp_path / "numpy.ipynb").exists()
    assert path == str(tmp_path / "numpy.ipynb")


def test_matplotlib_tutorial_exports_notebook(tmp_path):
    tut = get_tutorial("matplotlib")
    path = tut.to_notebook(str(tmp_path / "matplotlib.ipynb"))
    assert (tmp_path / "matplotlib.ipynb").exists()
    assert path == str(tmp_path / "matplotlib.ipynb")
