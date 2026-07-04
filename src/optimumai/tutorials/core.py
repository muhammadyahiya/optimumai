"""The tutorial engine — learn a tool by watching real code run, one cell at a time.

A :class:`Tutorial` is an ordered list of :class:`Cell` objects: prose that
explains an idea, then a short code cell that *demonstrates* it. Running the
tutorial executes each runnable code cell in a shared namespace and shows its
output right under the code — the same "explanation + live result" loop as the
rest of OptimumAI, but for the tools you actually type every day (NumPy,
matplotlib, PyTorch, and the LLM fine-tuning stack).

    from optimumai.tutorials import get_tutorial
    get_tutorial("numpy").run()                 # walk it in the terminal
    get_tutorial("numpy").to_notebook("np.ipynb")

Code cells may declare ``requires=("torch",)``: if the dependency is importable
the cell runs for real, otherwise the code + explanation are still shown (so the
lesson is fully readable with only the base install) with a note that it was not
executed.
"""

from __future__ import annotations

import contextlib
import importlib
import importlib.util
import io
from dataclasses import dataclass, field
from typing import Any

# name -> dotted module that exposes ``build() -> Tutorial`` (resolved lazily so
# the base import never pulls matplotlib/torch and a missing tutorial can't break
# the others).
_REGISTRY: dict[str, str] = {
    "numpy": "optimumai.tutorials.numpy_tut",
    "matplotlib": "optimumai.tutorials.matplotlib_tut",
    "pytorch": "optimumai.tutorials.pytorch_tut",
    "finetuning": "optimumai.tutorials.finetuning_tut",
}


def _installed(dep: str) -> bool:
    """True if ``dep`` (top-level import name) is importable, without importing it."""
    return importlib.util.find_spec(dep) is not None


@dataclass
class Cell:
    """One step: prose (``kind="md"``) or a runnable snippet (``kind="code"``)."""

    kind: str  # "md" | "code"
    text: str
    note: str = ""
    requires: tuple[str, ...] = ()

    def runnable(self) -> bool:
        """A code cell whose required dependencies are all importable."""
        return self.kind == "code" and all(_installed(d) for d in self.requires)


@dataclass
class Tutorial:
    """An ordered, runnable lesson for one tool."""

    name: str
    title: str
    summary: str
    cells: list[Cell] = field(default_factory=list)

    def md(self, text: str, note: str = "") -> Tutorial:
        self.cells.append(Cell("md", text, note=note))
        return self

    def code(self, text: str, note: str = "", requires: tuple[str, ...] = ()) -> Tutorial:
        self.cells.append(Cell("code", text.strip("\n"), note=note, requires=requires))
        return self

    def code_cells(self) -> list[Cell]:
        return [c for c in self.cells if c.kind == "code"]

    def run(self, console: Any = None, execute: bool = True) -> dict[str, str]:
        """Render the tutorial to the terminal, executing runnable code cells.

        Returns ``{cell_index: captured_output}`` for the cells that ran, so tests
        can assert on real results. Code runs in one shared namespace, so later
        cells can use variables from earlier ones.
        """
        from rich.console import Console
        from rich.panel import Panel
        from rich.syntax import Syntax

        console = console or Console()
        ns: dict[str, Any] = {}
        outputs: dict[str, str] = {}
        console.print(Panel(f"[bold cyan]{self.title}[/bold cyan]\n{self.summary}",
                            border_style="cyan"))
        for i, cell in enumerate(self.cells):
            if cell.kind == "md":
                console.print(cell.text)
                continue
            console.print(Syntax(cell.text, "python", theme="ansi_dark", word_wrap=True))
            if not execute:
                continue
            if not cell.runnable():
                miss = ", ".join(d for d in cell.requires if not _installed(d))
                console.print(f"[dim yellow]· not run (needs: {miss}) — code shown above[/]")
                continue
            buf = io.StringIO()
            try:
                with contextlib.redirect_stdout(buf):
                    exec(compile(cell.text, f"<{self.name}:cell{i}>", "exec"), ns)  # noqa: S102
            except Exception as exc:  # pragma: no cover - tutorial code is curated
                console.print(f"[red]error: {exc}[/]")
                raise
            out = buf.getvalue()
            outputs[str(i)] = out
            if out.strip():
                console.print(Panel(out.rstrip(), title="output", border_style="green",
                                    title_align="left"))
        return outputs

    def to_notebook(self, path: str) -> str:
        """Export to a runnable Jupyter notebook (needs nbformat, in the base dev set)."""
        import nbformat as nbf
        from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

        nb = new_notebook()
        nb.cells = [new_markdown_cell(f"# {self.title}\n\n{self.summary}")]
        for cell in self.cells:
            if cell.kind == "md":
                nb.cells.append(new_markdown_cell(cell.text))
            else:
                src = (f"# {cell.note}\n{cell.text}" if cell.note else cell.text)
                nb.cells.append(new_code_cell(src))
        nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python",
                                      "name": "python3"}, "language_info": {"name": "python"}}
        with open(path, "w") as fh:
            nbf.write(nb, fh)
        return path

    def to_markdown(self) -> str:
        """Render to Markdown (fenced code blocks) for the docs site."""
        parts = [f"# {self.title}\n", f"{self.summary}\n"]
        for cell in self.cells:
            if cell.kind == "md":
                parts.append(cell.text + "\n")
            else:
                note = f"  # {cell.note}" if cell.note else ""
                parts.append(f"```python{note}\n{cell.text}\n```\n")
        return "\n".join(parts)


def list_tutorials() -> list[str]:
    """The available tutorial names."""
    return list(_REGISTRY)


def get_tutorial(name: str) -> Tutorial:
    """Load and build the tutorial called ``name`` (lazy import)."""
    key = name.lower()
    if key not in _REGISTRY:
        raise ValueError(f"unknown tutorial {name!r}; choose from {list(_REGISTRY)}")
    module = importlib.import_module(_REGISTRY[key])
    return module.build()
