import importlib.resources as resources
import os

import pytest
from click.testing import CliRunner

from optimumai.circuit.interactive import interactive, interactive_backprop, interactive_softmax
from optimumai.cli.main import cli
from optimumai.llm.generate import available_providers, generate, generate_trace
from optimumai.visualization.concepts import concept_formats, list_concepts, render_concept


# --- token generation (use the offline 'toy' provider — no network) ----------
def test_providers_always_include_toy():
    assert "toy" in available_providers()


def test_generate_toy_produces_tokens():
    text = generate("the model learns to", provider="toy", max_tokens=10)
    assert isinstance(text, str) and len(text.split()) >= 1


def test_generate_trace_metadata():
    t = generate_trace("attention is", provider="toy", max_tokens=8)
    assert t.result and t.meta["provider"] == "toy"
    assert t.meta["output_tokens"] >= 1


# --- concept visualization registry -------------------------------------------
def test_concept_registry_populated():
    assert len(list_concepts()) >= 12
    assert "attention" in list_concepts()
    assert "png" in concept_formats("softmax") and "gif" in concept_formats("softmax")


def test_render_concept_unknown_raises():
    with pytest.raises(ValueError):
        render_concept("teleportation")


def test_render_concept_png(tmp_path):
    pytest.importorskip("matplotlib")
    path = render_concept("matmul", fmt="png", out=str(tmp_path / "m.png"))
    assert os.path.getsize(path) > 1000


def test_render_concept_gif(tmp_path):
    pytest.importorskip("matplotlib")
    path = render_concept("softmax", fmt="gif", out=str(tmp_path / "s.gif"))
    assert os.path.getsize(path) > 5000


# --- interactive circuits (standalone HTML) -----------------------------------
def test_interactive_softmax_html():
    html = interactive_softmax(out=None)
    assert "Math.exp" in html and "render()" in html


def test_interactive_backprop_html():
    html = interactive_backprop(out=None)
    assert "grad" in html and "da=de*b" in html


def test_interactive_dispatch_and_unknown():
    assert "<html" in interactive("softmax", out=None)
    with pytest.raises(ValueError):
        interactive("nope")


# --- notebooks are packaged ---------------------------------------------------
def test_notebooks_are_bundled():
    nb_dir = resources.files("optimumai") / "_notebooks"
    names = [p.name for p in nb_dir.iterdir() if p.name.endswith(".ipynb")]
    assert len(names) == 3


# --- CLI ----------------------------------------------------------------------
def _run(*args, **kw):
    return CliRunner().invoke(cli, list(args), **kw)


def test_cli_generate_toy():
    result = _run("generate", "hello world", "--provider", "toy", "--max-tokens", "8")
    assert result.exit_code == 0
    assert "GENERATE" in result.output.upper()


def test_cli_providers():
    assert "toy" in _run("providers").output


def test_cli_visualize_list():
    result = _run("visualize")
    assert result.exit_code == 0 and "attention" in result.output


def test_cli_playground(tmp_path):
    out = str(tmp_path / "sm.html")
    result = _run("playground", "softmax", "--out", out)
    assert result.exit_code == 0 and os.path.exists(out)


def test_cli_notebooks_copy(tmp_path):
    dest = str(tmp_path / "nb")
    result = _run("notebooks", "--dir", dest, "--no-launch")
    assert result.exit_code == 0
    assert len([f for f in os.listdir(dest) if f.endswith(".ipynb")]) == 3
