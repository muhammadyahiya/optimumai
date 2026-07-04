"""Tests for Plot Studio: numpy stats, reproducible code, charts, playground."""

from __future__ import annotations

import os

import numpy as np
import pytest

from optimumai.visualization.plotstudio import (
    describe,
    plot_code,
    plot_studio_playground,
    plot_studio_trace,
)

_DATA = [4, 8, 15, 16, 23, 42, 8, 15, 16, 4, 9, 30]
_PAIRS = [(1, 2), (2, 4), (3, 5), (4, 8), (5, 7)]

_KIND_MARKERS = {
    "bar": "plt.bar",
    "hist": "plt.hist",
    "scatter": "plt.scatter",
    "box": "plt.boxplot",
    "line": "plt.plot",
    "pie": "plt.pie",
    "violin": "plt.violinplot",
}


def _png_ok(path: str) -> bool:
    return os.path.exists(path) and os.path.getsize(path) > 500


# --------------------------------------------------------------------------
# describe
# --------------------------------------------------------------------------


def test_describe_matches_numpy():
    stats = describe(_DATA)
    values = np.asarray(_DATA, dtype=float)
    assert stats["n"] == 12
    assert stats["mean"] == pytest.approx(float(np.mean(values)))
    assert stats["std"] == pytest.approx(float(np.std(values)))
    assert stats["min"] == pytest.approx(float(np.min(values)))
    assert stats["max"] == pytest.approx(float(np.max(values)))
    q1, median, q3 = np.percentile(values, [25, 50, 75])
    assert stats["q1"] == pytest.approx(float(q1))
    assert stats["median"] == pytest.approx(float(median))
    assert stats["q3"] == pytest.approx(float(q3))


def test_describe_empty_raises():
    with pytest.raises(ValueError):
        describe([])


# --------------------------------------------------------------------------
# plot_code
# --------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["bar", "hist", "box", "line", "pie", "violin"])
def test_plot_code_scalar_kinds(kind):
    code = plot_code(_DATA, kind=kind)
    assert code.startswith("import numpy as np\nimport matplotlib.pyplot as plt")
    assert "import matplotlib.pyplot as plt" in code
    assert "import numpy as np" in code
    assert _KIND_MARKERS[kind] in code
    assert "plt.show()" in code
    assert "np.array(" in code


def test_plot_code_scatter_pairs():
    code = plot_code(_PAIRS, kind="scatter")
    assert "import numpy as np" in code
    assert "import matplotlib.pyplot as plt" in code
    assert "plt.scatter" in code
    assert "x = np.array(" in code
    assert "y = np.array(" in code
    assert "plt.show()" in code


def test_plot_code_scatter_two_lists():
    xs = [1, 2, 3, 4]
    ys = [10, 20, 30, 40]
    code = plot_code([xs, ys], kind="scatter")
    assert "plt.scatter" in code
    assert "x = np.array([1" in code
    assert "y = np.array([10" in code


def test_plot_code_hist_uses_bins_option():
    code = plot_code(_DATA, kind="hist", bins=5)
    assert "bins = 5" in code
    assert "plt.hist" in code


def test_plot_code_is_runnable_source(tmp_path):
    """The generated source should actually execute against real matplotlib."""
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    code = plot_code(_DATA, kind="bar").replace("plt.show()", "plt.close('all')")
    exec(compile(code, "<plot_code>", "exec"), {})


def test_plot_code_empty_raises():
    with pytest.raises(ValueError):
        plot_code([], kind="bar")


def test_plot_code_unknown_kind_raises():
    with pytest.raises(ValueError):
        plot_code(_DATA, kind="not-a-kind")


def test_plot_code_scatter_bad_shape_raises():
    with pytest.raises(ValueError):
        plot_code([1, 2, 3], kind="scatter")


def test_plot_code_scatter_mismatched_lengths_raises():
    with pytest.raises(ValueError):
        plot_code([[1, 2, 3], [1, 2]], kind="scatter")


# --------------------------------------------------------------------------
# plot_data (needs matplotlib)
# --------------------------------------------------------------------------

pytest.importorskip("matplotlib")

from optimumai.visualization.plotstudio import plot_data  # noqa: E402


@pytest.mark.parametrize("kind", ["bar", "hist", "box", "line", "pie", "violin"])
def test_plot_data_saves_file_per_kind(tmp_path, kind):
    out = str(tmp_path / f"{kind}.png")
    path = plot_data(_DATA, kind=kind, out=out)
    assert path == out
    assert _png_ok(out)


def test_plot_data_scatter(tmp_path):
    out = str(tmp_path / "scatter.png")
    path = plot_data(_PAIRS, kind="scatter", out=out)
    assert path == out
    assert _png_ok(out)


def test_plot_data_default_out(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = plot_data(_DATA)
    assert path == "plot.png"
    assert _png_ok(path)


def test_plot_data_empty_raises():
    with pytest.raises(ValueError):
        plot_data([], kind="bar")


def test_plot_data_unknown_kind_raises():
    with pytest.raises(ValueError):
        plot_data(_DATA, kind="nope")


# --------------------------------------------------------------------------
# plot_studio_trace
# --------------------------------------------------------------------------


def test_plot_studio_trace_structure():
    trace = plot_studio_trace(_DATA, kind="bar")
    assert trace.op == "plot_studio"
    assert len(trace.steps) >= 3
    assert trace.why_ai
    assert trace.formula
    assert isinstance(trace.result, str)
    assert "import matplotlib.pyplot as plt" in trace.result
    assert "plt.bar" in trace.result


def test_plot_studio_trace_scatter():
    trace = plot_studio_trace(_PAIRS, kind="scatter")
    assert "plt.scatter" in trace.result


def test_plot_studio_trace_unknown_kind_raises():
    with pytest.raises(ValueError):
        plot_studio_trace(_DATA, kind="nope")


# --------------------------------------------------------------------------
# plot_studio_playground
# --------------------------------------------------------------------------


def test_plot_studio_playground_writes_expected_html(tmp_path):
    out = str(tmp_path / "studio.html")
    path = plot_studio_playground(out=out)
    assert path == out

    with open(out, encoding="utf-8") as fh:
        html = fh.read()

    assert len(html) > 1000
    assert "<textarea" in html
    assert "<canvas" in html or "<svg" in html
    assert "Copy" in html
    assert "plt." in html
    assert "matplotlib" in html.lower()
    assert "numpy" in html.lower() or "np.array" in html


def test_plot_studio_playground_default_out_name(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = plot_studio_playground()
    assert path == "plot_studio_playground.html"
    assert (tmp_path / path).exists()


def test_plot_studio_playground_no_cdn(tmp_path):
    out = str(tmp_path / "studio.html")
    plot_studio_playground(out=out)
    with open(out, encoding="utf-8") as fh:
        html = fh.read()
    assert "cdn" not in html.lower()
    assert "http://" not in html
    assert "https://" not in html
