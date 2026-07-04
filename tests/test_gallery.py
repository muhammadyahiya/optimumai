import os

import pytest

pytest.importorskip("matplotlib")  # viz needs the [viz] extra

from optimumai.visualization.gallery import (  # noqa: E402
    animate_astar,
    animate_kmeans,
    animate_value_iteration,
    plot_astar_grid,
    plot_calibration,
    plot_conv_feature_map,
    plot_decision_boundary,
    plot_kmeans,
    plot_ppo_clip,
    plot_value_function,
)


def _file_ok(path: str) -> bool:
    return os.path.exists(path) and os.path.getsize(path) > 0


def test_plot_kmeans(tmp_path):
    assert _file_ok(plot_kmeans(out=str(tmp_path / "kmeans.png")))


def test_plot_decision_boundary(tmp_path):
    assert _file_ok(plot_decision_boundary(out=str(tmp_path / "boundary.png")))


def test_plot_astar_grid(tmp_path):
    assert _file_ok(plot_astar_grid(out=str(tmp_path / "astar.png")))


def test_plot_value_function(tmp_path):
    assert _file_ok(plot_value_function(out=str(tmp_path / "value_fn.png")))


def test_plot_conv_feature_map(tmp_path):
    assert _file_ok(plot_conv_feature_map(out=str(tmp_path / "conv.png")))


def test_plot_calibration(tmp_path):
    assert _file_ok(plot_calibration(out=str(tmp_path / "calibration.png")))


def test_plot_ppo_clip(tmp_path):
    assert _file_ok(plot_ppo_clip(out=str(tmp_path / "ppo.png")))


def test_animate_kmeans(tmp_path):
    assert _file_ok(animate_kmeans(out=str(tmp_path / "kmeans.gif"), fps=2))


def test_animate_astar(tmp_path):
    assert _file_ok(animate_astar(out=str(tmp_path / "astar.gif"), fps=3))


def test_animate_value_iteration(tmp_path):
    assert _file_ok(animate_value_iteration(out=str(tmp_path / "value_iter.gif"), fps=2))
