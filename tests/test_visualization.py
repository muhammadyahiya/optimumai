import os

import numpy as np
import pytest
from click.testing import CliRunner

pytest.importorskip("matplotlib")  # viz needs the [viz] extra

from optimumai.cli.main import cli  # noqa: E402
from optimumai.visualization.landscape import landscape_demo, plot_loss_landscape  # noqa: E402
from optimumai.visualization.plots import (  # noqa: E402
    plot_activation,
    plot_attention,
    plot_embeddings,
    plot_heatmap,
    plot_softmax_temperature,
    plot_training_curve,
)


def _png_ok(path: str) -> bool:
    return os.path.exists(path) and os.path.getsize(path) > 1000


def test_plot_activation(tmp_path):
    assert _png_ok(plot_activation("gelu", out=str(tmp_path / "a.png")))


def test_plot_activation_unknown_raises():
    with pytest.raises(ValueError):
        plot_activation("banana")


def test_plot_softmax_temperature(tmp_path):
    assert _png_ok(plot_softmax_temperature(out=str(tmp_path / "s.png")))


def test_plot_heatmap(tmp_path):
    assert _png_ok(plot_heatmap(np.eye(3), title="I", out=str(tmp_path / "h.png")))


def test_plot_attention(tmp_path):
    assert _png_ok(plot_attention(text="the cat sat", out=str(tmp_path / "att.png")))


def test_plot_embeddings(tmp_path):
    assert _png_ok(plot_embeddings(out=str(tmp_path / "emb.png")))


def test_plot_training_curve(tmp_path):
    assert _png_ok(plot_training_curve(out=str(tmp_path / "loss.png")))


def test_returns_figure_when_no_out():
    fig = plot_activation("relu")
    assert fig is not None


@pytest.mark.parametrize("func", ["bowl", "saddle", "rosenbrock"])
def test_landscape_presets(tmp_path, func):
    assert _png_ok(plot_loss_landscape(func, out=str(tmp_path / f"{func}.png")))


@pytest.mark.parametrize("kind", ["contour", "surface", "both"])
def test_landscape_kinds(tmp_path, kind):
    assert _png_ok(plot_loss_landscape("bowl", kind=kind, out=str(tmp_path / f"{kind}.png")))


def test_landscape_custom_expression(tmp_path):
    assert _png_ok(plot_loss_landscape("sin(x) + cos(y)", out=str(tmp_path / "c.png")))


def test_landscape_rejects_unsafe_expression():
    with pytest.raises(ValueError):
        plot_loss_landscape("__import__('os').system('echo hi')")


def test_landscape_demo(tmp_path):
    assert _png_ok(landscape_demo(out=str(tmp_path / "demo.png")))


def test_cli_plot(tmp_path):
    out = str(tmp_path / "cli.png")
    result = CliRunner().invoke(cli, ["plot", "activation", "--out", out])
    assert result.exit_code == 0
    assert _png_ok(out)


def test_cli_landscape(tmp_path):
    out = str(tmp_path / "cli_land.png")
    result = CliRunner().invoke(cli, ["landscape", "rosenbrock", "--out", out, "--kind", "contour"])
    assert result.exit_code == 0
    assert _png_ok(out)
