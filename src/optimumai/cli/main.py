"""The ``optimumai`` command-line interface.

Examples:
    optimumai course                     # the full learning path
    optimumai learn dot                  # run a lesson (marks it complete)
    optimumai progress                   # how far you've come
    optimumai dashboard                  # launch the Streamlit dashboard
    optimumai ask "why softmax?"         # the (optional) LLM tutor
    optimumai algebra dot "[1,2,3]" "[4,5,6]"
    optimumai train --steps 150
    optimumai kvcache --seq-len 8192      # KV cache VRAM for a config
    optimumai vram --params 70            # VRAM to train a 70B model
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
from optimumai.analysis.compare import compare_trace, sweep_trace
from optimumai.core.explain import ExplainLevel
from optimumai.curriculum import COURSE
from optimumai.foundations.kv_cache import kv_cache_trace
from optimumai.foundations.vram import vram_trace
from optimumai.interactive.prompts import prompt_matrix, prompt_vector
from optimumai.interactive.repl import run_repl
from optimumai.interpretability.superposition import superposition_trace
from optimumai.neural_networks.backprop import train_demo
from optimumai.probability.softmax import softmax_trace
from optimumai.progress import ProgressTracker
from optimumai.symbolic.differentiate import differentiate_trace
from optimumai.transformers.attention import Attention
from optimumai.transformers.text_pipeline import TextPipeline
from optimumai.tutor import Tutor
from optimumai.visualization.landscape import plot_loss_landscape
from optimumai.visualization.plots import (
    plot_activation,
    plot_attention,
    plot_embeddings,
    plot_softmax_temperature,
    plot_training_curve,
)
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


@cli.command("repl")
def repl_cmd() -> None:
    """Start an interactive REPL (arrow keys + tab-complete with the [repl] extra)."""
    run_repl()


@cli.command("trace-text")
@click.argument("text")
@click.option("--layers", type=int, default=2, help="Number of transformer blocks.")
@click.option("--level", type=_LEVEL_CHOICE, default="engineer", help="Detail level.")
def trace_text_cmd(text: str, layers: int, level: str) -> None:
    """Watch your own text flow through a toy transformer to a next-token distribution."""
    TextPipeline(text, layers=layers).trace().render(level)


# --------------------------------------------------------------------- algebra
@cli.group()
def algebra() -> None:
    """Vector and matrix operations."""


@algebra.command("dot")
@click.argument("a", required=False)
@click.argument("b", required=False)
@click.option("-i", "--interactive", is_flag=True, help="Enter the vectors interactively.")
@click.option("--level", type=_LEVEL_CHOICE, default="intermediate", help="Detail level.")
def algebra_dot(a: str | None, b: str | None, interactive: bool, level: str) -> None:
    """Dot product of two vectors. Omit args (or pass -i) to type them at the prompt."""
    va = prompt_vector("vector A") if interactive or a is None else _parse_literal(a, "vector A")
    vb = prompt_vector("vector B") if interactive or b is None else _parse_literal(b, "vector B")
    Vector(va).dot(Vector(vb), explain=True, level=level)


@algebra.command("cosine")
@click.argument("a", required=False)
@click.argument("b", required=False)
@click.option("-i", "--interactive", is_flag=True, help="Enter the vectors interactively.")
@click.option("--level", type=_LEVEL_CHOICE, default="intermediate", help="Detail level.")
def algebra_cosine(a: str | None, b: str | None, interactive: bool, level: str) -> None:
    """Cosine similarity of two vectors (omit args or pass -i to type them)."""
    va = prompt_vector("vector A") if interactive or a is None else _parse_literal(a, "vector A")
    vb = prompt_vector("vector B") if interactive or b is None else _parse_literal(b, "vector B")
    Vector(va).cosine_similarity(Vector(vb), explain=True, level=level)


@algebra.command("matmul")
@click.argument("a", required=False)
@click.argument("b", required=False)
@click.option("-i", "--interactive", is_flag=True, help="Enter the matrices interactively.")
@click.option("--level", type=_LEVEL_CHOICE, default="intermediate", help="Detail level.")
def algebra_matmul(a: str | None, b: str | None, interactive: bool, level: str) -> None:
    """Matrix product (omit args or pass -i to type them at the prompt)."""
    ma = prompt_matrix("matrix A") if interactive or a is None else _parse_literal(a, "matrix A")
    mb = prompt_matrix("matrix B") if interactive or b is None else _parse_literal(b, "matrix B")
    Matrix(ma).matmul(Matrix(mb), explain=True, level=level)


# --------------------------------------------------------------------- softmax
@cli.command("softmax")
@click.argument("logits", required=False)
@click.option("-i", "--interactive", is_flag=True, help="Enter the logits interactively.")
@click.option("--temperature", "-t", type=float, default=1.0, help="Sampling temperature (>0).")
@click.option("--level", type=_LEVEL_CHOICE, default="intermediate", help="Detail level.")
def softmax_cmd(logits: str | None, interactive: bool, temperature: float, level: str) -> None:
    """Softmax of a logit vector (omit the arg or pass -i to type it)."""
    values = prompt_vector("logits") if interactive or logits is None else _parse_literal(
        logits, "logits"
    )
    softmax_trace(values, temperature=temperature).render(level)


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


# --------------------------------------------------------- systems / foundations
@cli.command("kvcache")
@click.option("--layers", type=int, default=32, help="Number of transformer layers.")
@click.option("--heads", type=int, default=32, help="Number of attention heads.")
@click.option("--head-dim", type=int, default=128, help="Dimension per head.")
@click.option("--seq-len", type=int, default=4096, help="Context length in tokens.")
@click.option("--batch", type=int, default=1, help="Batch size.")
@click.option("--kv-heads", type=int, default=None, help="Fewer than heads = GQA; 1 = MQA.")
@click.option("--level", type=_LEVEL_CHOICE, default="engineer", help="Detail level.")
def kvcache_cmd(
    layers: int, heads: int, head_dim: int, seq_len: int, batch: int,
    kv_heads: int | None, level: str,
) -> None:
    """KV cache memory for a transformer config (why context length eats VRAM)."""
    kv_cache_trace(layers, heads, head_dim, seq_len, batch, kv_heads=kv_heads).render(level)


@cli.command("vram")
@click.option("--params", "params_billions", type=float, default=7.0, help="Model size (billions).")
@click.option("--precision", type=int, default=2, help="Bytes per parameter (2=fp16, 4=fp32).")
@click.option("--inference", is_flag=True, help="Estimate inference instead of training.")
@click.option("--level", type=_LEVEL_CHOICE, default="engineer", help="Detail level.")
def vram_cmd(params_billions: float, precision: int, inference: bool, level: str) -> None:
    """Estimate GPU VRAM for training or inference of an LLM."""
    vram_trace(params_billions, precision_bytes=precision, training=not inference).render(level)


# ----------------------------------------------------------- interactive analysis
@cli.command("diff")
@click.argument("expression")
@click.option("--var", default="x", help="Variable to differentiate with respect to.")
@click.option("--at", type=float, default=None, help="Evaluate f and f' at this point.")
@click.option("--level", type=_LEVEL_CHOICE, default="intermediate", help="Detail level.")
def diff_cmd(expression: str, var: str, at: float | None, level: str) -> None:
    """Symbolically differentiate YOUR equation, e.g. optimumai diff "x**3 + 2*x" --at 3."""
    try:
        differentiate_trace(expression, var=var, at=at).render(level)
    except ImportError as exc:
        raise click.ClickException(str(exc)) from exc


@cli.command("compare")
@click.argument("op_a")
@click.argument("op_b")
@click.option("--input", "input_", default="[-2,-1,0,1,2]", help="Input values.")
@click.option("--level", type=_LEVEL_CHOICE, default="engineer", help="Detail level.")
def compare_cmd(op_a: str, op_b: str, input_: str, level: str) -> None:
    """Compare two activations on your input, e.g. optimumai compare relu gelu."""
    compare_trace(op_a, op_b, _parse_literal(input_, "input")).render(level)


@cli.command("sweep")
@click.argument("op")
@click.option("--param", default="temperature", help="Parameter to sweep.")
@click.option("--values", default="[0.25,0.5,1.0,2.0]", help="Values to sweep over.")
@click.option("--level", type=_LEVEL_CHOICE, default="engineer", help="Detail level.")
def sweep_cmd(op: str, param: str, values: str, level: str) -> None:
    """Sweep a parameter and watch the output evolve, e.g. optimumai sweep softmax."""
    sweep_trace(op, param, _parse_literal(values, "values")).render(level)


# ---------------------------------------------------------------- visualization
@cli.command("plot")
@click.argument("kind", type=click.Choice(
    ["activation", "softmax", "attention", "embeddings", "training"]))
@click.option("--out", default="optimumai_plot.png", help="Output PNG path.")
@click.option("--name", default="gelu", help="Activation name (kind=activation).")
@click.option("--text", default=None, help="Text to attend over (kind=attention).")
def plot_cmd(kind: str, out: str, name: str, text: str | None) -> None:
    """Save a matplotlib graph to PNG (needs the [viz] extra)."""
    plotters = {
        "activation": lambda: plot_activation(name=name, out=out),
        "softmax": lambda: plot_softmax_temperature(out=out),
        "attention": lambda: plot_attention(text=text, out=out),
        "embeddings": lambda: plot_embeddings(out=out),
        "training": lambda: plot_training_curve(out=out),
    }
    try:
        path = plotters[kind]()
    except ImportError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"saved → {path}")


@cli.command("landscape")
@click.argument("func", default="rosenbrock")
@click.option("--out", default="optimumai_landscape.png", help="Output PNG path.")
@click.option("--kind", type=click.Choice(["contour", "surface", "both"]), default="both")
def landscape_cmd(func: str, out: str, kind: str) -> None:
    """Plot a 2D/3D loss landscape + gradient-descent trajectory (needs [viz]).

    FUNC is a preset (bowl, saddle, rosenbrock) or an expression in x and y.
    """
    try:
        path = plot_loss_landscape(func, out=out, kind=kind)
    except (ImportError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"saved → {path}")


if __name__ == "__main__":  # pragma: no cover
    cli()
