"""The ``optimumai`` command-line interface.

Examples:
    optimumai course                     # the full learning path
    optimumai learn dot                  # run a lesson (marks it complete)
    optimumai progress                   # how far you've come
    optimumai dashboard                  # launch the Streamlit dashboard
    optimumai ask "why softmax?"         # the (optional) LLM tutor
    optimumai algebra dot "[1,2,3]" "[4,5,6]"
    optimumai train --steps 150
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import click

from optimumai import __version__
from optimumai.algebra.matrix import Matrix
from optimumai.algebra.vector import Vector
from optimumai.core.explain import ExplainLevel
from optimumai.curriculum import COURSE
from optimumai.interpretability.superposition import superposition_trace
from optimumai.neural_networks.backprop import train_demo
from optimumai.probability.softmax import softmax_trace
from optimumai.progress import ProgressTracker
from optimumai.transformers.attention import Attention
from optimumai.tutor import Tutor
from optimumai.world_models.jepa import JEPA

_LEVEL_CHOICE = click.Choice([lvl.value for lvl in ExplainLevel], case_sensitive=False)


def _parse_literal(text: str, what: str):
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError) as exc:
        raise click.BadParameter(
            f'could not parse {what}: {text!r} (expected e.g. "[1, 2, 3]")'
        ) from exc


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__, prog_name="optimumai")
def cli() -> None:
    """OptimumAI — unlock the math behind AI, one explained step at a time."""


# ------------------------------------------------------------------ the course
@cli.command("course")
def course_cmd() -> None:
    """Show the full learning path, grouped by track, with your progress."""
    done = ProgressTracker().completed_ids()
    click.echo(click.style("OptimumAI — the AI learning path\n", bold=True))
    for track, lessons in COURSE.tracks().items():
        click.echo(click.style(track, fg="cyan", bold=True))
        for lesson in lessons:
            mark = click.style("✓", fg="green") if lesson.id in done else "○"
            click.echo(f"  {mark} {lesson.id:<14} {lesson.summary}")
        click.echo("")
    total = len(COURSE)
    rate = len(done) / total * 100 if total else 0
    click.echo(f"Progress: {len(done)}/{total} lessons ({rate:.0f}%)")
    nxt = COURSE.next_incomplete(done)
    if nxt:
        click.echo(f"Next up:  optimumai learn {nxt.id}")


@cli.command("learn")
@click.argument("topic", required=False)
@click.option("--level", type=_LEVEL_CHOICE, default="intermediate", help="Detail level.")
@click.option("--no-track", is_flag=True, help="Don't record this lesson as complete.")
def learn_cmd(topic: str | None, level: str, no_track: bool) -> None:
    """Run a lesson and record it as complete. Omit TOPIC to list the course."""
    if topic is None:
        for track, lessons in COURSE.tracks().items():
            click.echo(click.style(track, fg="cyan", bold=True))
            for lesson in lessons:
                click.echo(f"  {lesson.id:<14} {lesson.summary}")
        click.echo("\nRun:  optimumai learn <topic>   (or 'optimumai course')")
        return
    key = topic.lower()
    try:
        lesson = COURSE.get(key)
    except KeyError as exc:
        raise click.BadParameter(
            f"unknown topic {topic!r}. Run 'optimumai course' to see all lessons."
        ) from exc
    lesson.run(level)
    if not no_track:
        ProgressTracker().mark_complete(lesson.id)
        click.echo(click.style(f"\n✓ '{lesson.id}' complete.", fg="green"), nl=False)
        click.echo("  See 'optimumai progress' for your path.")


@cli.command("progress")
@click.option("--reset", is_flag=True, help="Clear all recorded progress.")
def progress_cmd(reset: bool) -> None:
    """Show how far through the course you are (and what's next)."""
    tracker = ProgressTracker()
    if reset:
        tracker.reset()
        click.echo("Progress reset.")
        return
    total = len(COURSE)
    done = tracker.completed_count()
    filled = int(tracker.completion_rate(total) * 20)
    bar = "█" * filled + "░" * (20 - filled)
    click.echo(f"[{bar}] {done}/{total} ({done / total * 100 if total else 0:.0f}%)")
    nxt = COURSE.next_incomplete(tracker.completed_ids())
    if nxt:
        click.echo(f"\nNext: optimumai learn {nxt.id}   — {nxt.title}")
    else:
        click.echo("\n🎉 Course complete — you've unlocked the frontier!")


@cli.command("dashboard")
@click.option("--port", type=int, default=8501, help="Port to serve on.")
def dashboard_cmd(port: int) -> None:
    """Launch the Streamlit progress dashboard (needs the [dashboard] extra)."""
    app = Path(__file__).resolve().parent.parent / "dashboard" / "app.py"
    try:
        import streamlit  # noqa: F401
    except ImportError as exc:
        raise click.ClickException(
            'Streamlit is not installed. Install it with:  pip install "optimumai[dashboard]"'
        ) from exc
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(app), "--server.port", str(port)],
        check=False,
    )


@cli.command("ask")
@click.argument("question")
@click.option("--level", type=_LEVEL_CHOICE, default="intermediate", help="Explanation depth.")
def ask_cmd(question: str, level: str) -> None:
    """Ask the (optional) LLM tutor a question (needs the [llm] extra + API key)."""
    click.echo(Tutor().ask(question))


# --------------------------------------------------------------------- algebra
@cli.group()
def algebra() -> None:
    """Vector and matrix operations."""


@algebra.command("dot")
@click.argument("a")
@click.argument("b")
@click.option("--level", type=_LEVEL_CHOICE, default="intermediate", help="Detail level.")
def algebra_dot(a: str, b: str, level: str) -> None:
    """Dot product of two vectors, e.g. optimumai algebra dot "[1,2,3]" "[4,5,6]"."""
    Vector(_parse_literal(a, "vector A")).dot(
        Vector(_parse_literal(b, "vector B")), explain=True, level=level
    )


@algebra.command("cosine")
@click.argument("a")
@click.argument("b")
@click.option("--level", type=_LEVEL_CHOICE, default="intermediate", help="Detail level.")
def algebra_cosine(a: str, b: str, level: str) -> None:
    """Cosine similarity of two vectors."""
    Vector(_parse_literal(a, "vector A")).cosine_similarity(
        Vector(_parse_literal(b, "vector B")), explain=True, level=level
    )


@algebra.command("matmul")
@click.argument("a")
@click.argument("b")
@click.option("--level", type=_LEVEL_CHOICE, default="intermediate", help="Detail level.")
def algebra_matmul(a: str, b: str, level: str) -> None:
    """Matrix product, e.g. optimumai algebra matmul "[[1,2],[3,4]]" "[[5,6],[7,8]]"."""
    Matrix(_parse_literal(a, "matrix A")).matmul(
        Matrix(_parse_literal(b, "matrix B")), explain=True, level=level
    )


# --------------------------------------------------------------------- softmax
@cli.command("softmax")
@click.argument("logits")
@click.option("--temperature", "-t", type=float, default=1.0, help="Sampling temperature (>0).")
@click.option("--level", type=_LEVEL_CHOICE, default="intermediate", help="Detail level.")
def softmax_cmd(logits: str, temperature: float, level: str) -> None:
    """Softmax of a logit vector, e.g. optimumai softmax "[2,1,0.1]"."""
    softmax_trace(_parse_literal(logits, "logits"), temperature=temperature).render(level)


# ------------------------------------------------------------------- attention
@cli.command("attention")
@click.option("--demo", "use_demo", is_flag=True, help="Run a built-in 3-token example.")
@click.option("--seed", type=int, default=0, help="Random seed for --demo.")
@click.option("--level", type=_LEVEL_CHOICE, default="engineer", help="Detail level.")
def attention_cmd(use_demo: bool, seed: int, level: str) -> None:
    """Scaled dot-product attention (currently via the built-in demo)."""
    if not use_demo:
        raise click.UsageError("pass --demo to run the built-in attention example")
    Attention.demo(seed=seed).render(level)


# -------------------------------------------------------------- autograd / train
@cli.command("backprop")
@click.option("--level", type=_LEVEL_CHOICE, default="engineer", help="Detail level.")
def backprop_cmd(level: str) -> None:
    """Backpropagate through a scalar autograd graph, chain rule step by step."""
    COURSE.get("backprop").run(level)


@cli.command("train")
@click.option("--steps", type=int, default=120, help="Training iterations.")
@click.option("--lr", type=float, default=0.05, help="Learning rate.")
@click.option("--level", type=_LEVEL_CHOICE, default="intermediate", help="Detail level.")
def train_cmd(steps: int, lr: float, level: str) -> None:
    """Train a tiny MLP on a toy set and watch the loss fall."""
    train_demo(steps=steps, lr=lr).render(level)


@cli.command("jepa")
@click.option("--demo", "use_demo", is_flag=True, help="Run the built-in JEPA example.")
@click.option("--seed", type=int, default=0, help="Random seed for --demo.")
@click.option("--level", type=_LEVEL_CHOICE, default="engineer", help="Detail level.")
def jepa_cmd(use_demo: bool, seed: int, level: str) -> None:
    """JEPA world model — predict in representation space, not pixels (LeCun)."""
    if not use_demo:
        raise click.UsageError("pass --demo to run the built-in JEPA example")
    JEPA.demo(seed=seed).render(level)


@cli.command("superposition")
@click.option("--features", type=int, default=5, help="Number of features (> neurons).")
@click.option("--neurons", type=int, default=2, help="Number of neurons / dimensions.")
@click.option("--level", type=_LEVEL_CHOICE, default="engineer", help="Detail level.")
def superposition_cmd(features: int, neurons: int, level: str) -> None:
    """Toy model of superposition — why neurons are polysemantic (Anthropic)."""
    superposition_trace(features, neurons).render(level)


if __name__ == "__main__":  # pragma: no cover
    cli()
