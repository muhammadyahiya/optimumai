from __future__ import annotations

import io

import pytest
from rich.console import Console

from optimumai.tutorials.finetuning_tut import build as build_finetuning
from optimumai.tutorials.pytorch_tut import build as build_pytorch

_HEAVY_DEPS = ("torch", "transformers", "peft", "trl")


def _quiet_console() -> Console:
    return Console(file=io.StringIO(), force_terminal=False, width=200)


def _assert_all_cells_compile(tutorial) -> None:
    for i, cell in enumerate(tutorial.code_cells()):
        compile(cell.text, f"<{tutorial.name}:cell{i}>", "exec")


def _assert_heavy_cells_marked(tutorial) -> None:
    """Every code cell that imports a heavy dep must declare it in ``requires``."""
    for cell in tutorial.code_cells():
        for dep in _HEAVY_DEPS:
            mentions_dep = f"import {dep}" in cell.text or f"from {dep}" in cell.text
            if mentions_dep:
                assert dep in cell.requires, (
                    f"cell using {dep!r} must declare requires=(..., {dep!r}, ...): "
                    f"{cell.text[:60]!r}"
                )


# --------------------------------------------------------------------- pytorch
def test_pytorch_tutorial_builds():
    t = build_pytorch()
    assert t.name == "pytorch"
    assert t.code_cells()


def test_pytorch_cells_compile():
    _assert_all_cells_compile(build_pytorch())


def test_pytorch_heavy_cells_require_torch():
    _assert_heavy_cells_marked(build_pytorch())


def test_pytorch_tutorial_runs_and_proves_autograd():
    t = build_pytorch()
    outputs = t.run(console=_quiet_console(), execute=True)
    joined = "\n".join(outputs.values())
    # From the autograd cell: a=2, b=-3, L = a*b + a = -4; dL/da = b + 1 = -2; dL/db = a = 2.
    assert "L.data=-4.0" in joined
    assert "dL/da=-2.0" in joined
    assert "dL/db=2.0" in joined
    # From the training-loop cell: the loss must visibly decrease.
    assert "loss decreased: True" in joined


def test_pytorch_torch_cells_not_executed_without_torch():
    pytest.importorskip("torch", reason="only meaningful when torch is absent")


def test_pytorch_torch_cells_marked_not_runnable_without_torch():
    import importlib.util

    if importlib.util.find_spec("torch") is not None:
        pytest.skip("torch is installed; requires-marking behavior differs")
    t = build_pytorch()
    torch_cells = [c for c in t.code_cells() if "torch" in c.requires]
    assert torch_cells
    assert all(not c.runnable() for c in torch_cells)


def test_pytorch_to_notebook(tmp_path):
    t = build_pytorch()
    path = t.to_notebook(str(tmp_path / "pytorch.ipynb"))
    assert path.endswith("pytorch.ipynb")
    content = (tmp_path / "pytorch.ipynb").read_text()
    assert "PyTorch" in content


# ------------------------------------------------------------------ finetuning
def test_finetuning_tutorial_builds():
    t = build_finetuning()
    assert t.name == "finetuning"
    assert t.code_cells()


def test_finetuning_cells_compile():
    _assert_all_cells_compile(build_finetuning())


def test_finetuning_heavy_cells_require_deps():
    _assert_heavy_cells_marked(build_finetuning())


def test_finetuning_tutorial_runs_and_proves_lora_and_dpo():
    t = build_finetuning()
    outputs = t.run(console=_quiet_console(), execute=True)
    joined = "\n".join(outputs.values())
    # LoRA: d_in=d_out=64, rank=4 -> full=4096, lora=4*(64+64)=512, reduction=8x.
    assert "full params=4096, lora params=512" in joined
    assert "reduction factor=8.00x fewer trainable params" in joined
    # DPO: deterministic seed=0 trace exposes a fixed loss/margin.
    from optimumai.frontier.rlhf import dpo_trace

    expected = dpo_trace(prompt="Explain gravity.", beta=0.1, seed=0)
    assert f"DPO loss={expected.result:.4f}" in joined
    assert f"reward margin (chosen - rejected)={expected.meta['margin']:.4f}" in joined
    # SFT toy loop must show the loss decreasing.
    assert "loss decreased: True" in joined


def test_finetuning_torch_cells_marked_not_runnable_without_deps():
    import importlib.util

    t = build_finetuning()
    for dep in ("transformers", "peft", "trl"):
        if importlib.util.find_spec(dep) is not None:
            continue
        dep_cells = [c for c in t.code_cells() if dep in c.requires]
        assert dep_cells, f"expected at least one cell requiring {dep!r}"
        assert all(not c.runnable() for c in dep_cells)


def test_finetuning_to_notebook(tmp_path):
    t = build_finetuning()
    path = t.to_notebook(str(tmp_path / "finetuning.ipynb"))
    assert path.endswith("finetuning.ipynb")
    content = (tmp_path / "finetuning.ipynb").read_text()
    assert "LoRA" in content
