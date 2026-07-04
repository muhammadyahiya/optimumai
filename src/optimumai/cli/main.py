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

# --- v1.2: prompting · augmented RNNs · interactive playgrounds ---
from optimumai.augmented_rnns.act import demo as augrnn_act_demo
from optimumai.augmented_rnns.attention import demo as augrnn_attn_demo
from optimumai.augmented_rnns.ntm import demo as augrnn_ntm_demo
from optimumai.circuit.graph import build_from_expression
from optimumai.circuit.interactive import interactive as interactive_circuit
from optimumai.circuit.render import to_dot, to_html, to_terminal
from optimumai.core.explain import ExplainLevel
from optimumai.curriculum import COURSE

# --- v1.1 groups: classical ML · search · RL · NLP · vision · evaluation ---
from optimumai.evaluation.calibration import demo as eval_cal_demo
from optimumai.evaluation.hallucination import demo as eval_faith_demo
from optimumai.evaluation.perplexity import demo as eval_ppl_demo
from optimumai.evaluation.perplexity import perplexity_trace
from optimumai.evaluation.text_metrics import bleu_trace, rouge_n_trace
from optimumai.evaluation.text_metrics import demo as eval_text_demo
from optimumai.exercises.engine import Workbook, available_exercises

# --- v1.3: Plot Studio + concept-flow diagrams ---
from optimumai.flows import flow as build_flow
from optimumai.foundations.kv_cache import kv_cache_trace
from optimumai.foundations.vram import vram_trace
from optimumai.frontier.quantization import quantize_trace
from optimumai.interactive.prompts import prompt_matrix, prompt_vector
from optimumai.interactive.repl import run_repl
from optimumai.interpretability.superposition import superposition_trace
from optimumai.kernels.backends import backend_report
from optimumai.kernels.kernels import list_kernels, run_kernel
from optimumai.llm.generate import available_providers, generate_trace
from optimumai.ml.decision_tree import demo as ml_tree_demo
from optimumai.ml.kmeans import demo as ml_kmeans_demo
from optimumai.ml.kmeans import kmeans_trace
from optimumai.ml.knn import demo as ml_knn_demo
from optimumai.ml.linear_regression import demo as ml_linreg_demo
from optimumai.ml.linear_regression import linear_regression_trace
from optimumai.ml.logistic_regression import demo as ml_logreg_demo
from optimumai.ml.metrics import demo as ml_metrics_demo
from optimumai.ml.naive_bayes import demo as ml_nb_demo
from optimumai.ml.pca import demo as ml_pca_demo
from optimumai.neural_networks.backprop import train_demo
from optimumai.nlp.bpe import bpe_trace
from optimumai.nlp.bpe import demo as nlp_bpe_demo
from optimumai.nlp.edit_distance import demo as nlp_edit_demo
from optimumai.nlp.edit_distance import edit_distance_trace
from optimumai.nlp.ngram import demo as nlp_ngram_demo
from optimumai.nlp.tfidf import demo as nlp_tfidf_demo
from optimumai.nlp.tfidf import tfidf_trace
from optimumai.nlp.word2vec import demo as nlp_word2vec_demo
from optimumai.probability.softmax import softmax_trace
from optimumai.progress import ProgressTracker
from optimumai.prompting.chain_of_thought import demo as prompt_cot_demo
from optimumai.prompting.few_shot import demo as prompt_fewshot_demo
from optimumai.prompting.react import demo as prompt_react_demo
from optimumai.prompting.self_consistency import demo as prompt_selfcons_demo
from optimumai.prompting.structured_output import demo as prompt_structured_demo
from optimumai.prompting.zero_shot import demo as prompt_zeroshot_demo
from optimumai.quiz.engine import Quiz, available_quizzes
from optimumai.review.scheduler import ReviewScheduler
from optimumai.rl.mdp import demo as rl_mdp_demo
from optimumai.rl.policy_gradient import demo as rl_reinforce_demo
from optimumai.rl.ppo import demo as rl_ppo_demo
from optimumai.rl.q_learning import demo as rl_q_demo
from optimumai.search.adversarial import demo as algo_adversarial_demo
from optimumai.search.informed import demo as algo_informed_demo
from optimumai.search.uninformed import demo as algo_uninformed_demo
from optimumai.symbolic.differentiate import differentiate_trace
from optimumai.transformers.attention import Attention
from optimumai.transformers.text_pipeline import TextPipeline
from optimumai.tutor import Tutor
from optimumai.vision.cnn import demo as vision_cnn_demo
from optimumai.vision.convolution import conv2d_trace
from optimumai.vision.convolution import demo as vision_conv_demo
from optimumai.vision.edges import demo as vision_sobel_demo
from optimumai.vision.pooling import demo as vision_pool_demo
from optimumai.visualization.animate import (
    animate_diffusion,
    animate_gradient_descent,
    animate_softmax_temperature,
)
from optimumai.visualization.concepts import concept_formats, list_concepts, render_concept
from optimumai.visualization.interactive import editable_plot
from optimumai.visualization.landscape import plot_loss_landscape
from optimumai.visualization.playgrounds import playground as viz_playground
from optimumai.visualization.plots import (
    plot_activation,
    plot_attention,
    plot_embeddings,
    plot_softmax_temperature,
    plot_training_curve,
)
from optimumai.visualization.plotstudio import describe as ps_describe
from optimumai.visualization.plotstudio import plot_code as ps_plot_code
from optimumai.visualization.plotstudio import plot_data as ps_plot_data
from optimumai.visualization.plotstudio import plot_studio_playground
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


def _run_quiz(topic: str) -> float:
    """Run a quiz interactively (active recall) and return the score (0–1)."""
    quiz = Quiz(topic)
    answers: list[int] = []
    for i, q in enumerate(quiz.questions, 1):
        click.echo(f"\n{click.style(f'Q{i}.', bold=True)} {q.prompt}")
        for j, choice in enumerate(q.choices, 1):
            click.echo(f"  {j}) {choice}")
        pick = click.prompt("Your answer", type=click.IntRange(1, len(q.choices))) - 1
        answers.append(pick)
        if pick == q.answer:
            click.echo(click.style("  ✓ correct", fg="green"))
        else:
            click.echo(click.style(f"  ✗ correct answer: {q.choices[q.answer]}", fg="red"))
        click.echo(click.style(f"    {q.explanation}", dim=True))
    result = quiz.grade(answers)
    click.echo(f"\nScore: {result.correct}/{result.total} ({result.score * 100:.0f}%)")
    return result.score


@cli.command("quiz")
@click.argument("topic", required=False)
def quiz_cmd(topic: str | None) -> None:
    """Test yourself on a topic (active recall). Omit TOPIC to list quizzes."""
    if topic is None:
        click.echo("Quizzes: " + ", ".join(available_quizzes()))
        return
    key = topic.lower()
    try:
        score = _run_quiz(key)
    except (KeyError, ValueError) as exc:
        raise click.BadParameter(
            f"no quiz for {topic!r}. Available: {', '.join(available_quizzes())}"
        ) from exc
    # Grade feeds the spaced-repetition scheduler (quality 0–5 from the score).
    ReviewScheduler(ProgressTracker()).record(key, round(score * 5))
    click.echo("Scheduled for spaced review — run 'optimumai review' later.")


@cli.command("review")
def review_cmd() -> None:
    """Spaced repetition: quiz yourself on whatever's due for review."""
    scheduler = ReviewScheduler(ProgressTracker())
    topic = scheduler.next_due(available_quizzes())
    if topic is None:
        click.echo("Nothing due 🎉  Take a quiz first: optimumai quiz softmax")
        return
    click.echo(click.style(f"Reviewing: {topic}", bold=True))
    score = _run_quiz(topic)
    state = scheduler.record(topic, round(score * 5))
    click.echo(f"Next review in ~{state.interval_days:g} day(s).")


@cli.command("search")
@click.argument("query")
def search_cmd(query: str) -> None:
    """Search the course by keyword (id, title, summary, track)."""
    q = query.lower()
    hits = [
        lesson
        for lesson in COURSE
        if q in lesson.id.lower()
        or q in lesson.title.lower()
        or q in lesson.summary.lower()
        or q in lesson.track.lower()
    ]
    if not hits:
        click.echo(f"No lessons match {query!r}.")
        return
    for lesson in hits:
        click.echo(f"  {lesson.id:<16} {lesson.title} — {lesson.summary}")


@cli.command("start")
def start_cmd() -> None:
    """New here? The 30-second guided tour."""
    click.echo(click.style("\n  Welcome to OptimumAI — unlock the math behind AI 🧮\n", bold=True))
    click.echo("Here's a dot product, explained step by step:\n")
    COURSE.get("dot").run("beginner")
    click.echo(click.style("\nNext steps:", bold=True))
    click.echo("  optimumai course           # the full learning path (35 lessons)")
    click.echo("  optimumai learn attention  # run any lesson")
    click.echo("  optimumai quiz softmax     # test yourself (active recall)")
    click.echo("  optimumai circuit \"a*b+c\"  # see the computation as a circuit")
    click.echo("  optimumai dashboard        # the visual dashboard\n")


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


# --------------------------------------------------------------------- frontier
@cli.command("quantize")
@click.argument("values")
@click.option("--bits", type=click.Choice(["4", "8"]), default="8", help="Bit width.")
@click.option("--scheme", type=click.Choice(["symmetric", "asymmetric"]), default="symmetric")
@click.option("--level", type=_LEVEL_CHOICE, default="engineer", help="Detail level.")
def quantize_cmd(values: str, bits: str, scheme: str, level: str) -> None:
    """Quantize YOUR values to int8/int4 and see the error, e.g. quantize "[0.1,-2.3,4.5]"."""
    parsed = _parse_literal(values, "values")
    quantize_trace(parsed, bits=int(bits), scheme=scheme).render(level)


# --------------------------------------------------------------------- circuit
@cli.command("circuit")
@click.argument("expression")
@click.option("--vars", "vars_", default=None, help='Values, e.g. "a=2,b=-3,c=10" (unset → 1).')
@click.option("--fmt", type=click.Choice(["terminal", "html", "dot"]), default="terminal")
@click.option("--out", default=None, help="Output file for html/dot.")
def circuit_cmd(expression: str, vars_: str | None, fmt: str, out: str | None) -> None:
    """Render YOUR expression as a computation-graph circuit (data + grad on every wire).

    Example: optimumai circuit "(a*b + c) * f" --vars "a=2,b=-3,c=10,f=-2" --fmt html
    """
    variables: dict[str, float] = {}
    if vars_:
        for pair in vars_.split(","):
            key, _, val = pair.partition("=")
            try:
                variables[key.strip()] = float(val)
            except ValueError as exc:
                msg = f"bad --vars entry {pair!r}: expected name=number"
                raise click.BadParameter(msg) from exc
    try:
        _, graph = build_from_expression(expression, variables or None)
    except ValueError as exc:
        raise click.BadParameter(str(exc)) from exc

    if fmt == "terminal":
        to_terminal(graph)
    elif fmt == "dot":
        dot = to_dot(graph)
        if out:
            Path(out).write_text(dot)
            click.echo(f"saved → {out}")
        else:
            click.echo(dot)
    else:  # html
        click.echo(f"saved → {to_html(graph, out or 'circuit.html')}")


# ------------------------------------------------------------------ gpu kernels
@cli.command("kernel")
@click.argument("name", required=False)
@click.option("--backends", "show_backends", is_flag=True, help="Show available GPU backends.")
@click.option("--level", type=_LEVEL_CHOICE, default="engineer", help="Detail level.")
def kernel_cmd(name: str | None, show_backends: bool, level: str) -> None:
    """Run a GPU kernel on the simulator (scalar_add → flash attention)."""
    if show_backends:
        click.echo(backend_report())
        return
    if name is None:
        click.echo("Kernels: " + ", ".join(list_kernels()))
        click.echo("Run one:  optimumai kernel matmul   ·   backends:  optimumai kernel --backends")
        return
    try:
        run_kernel(name, explain=True, level=level)
    except ValueError as exc:
        raise click.BadParameter(str(exc)) from exc


@cli.command("animate")
@click.argument("what", type=click.Choice(["descent", "diffusion", "softmax"]))
@click.option("--out", default="optimumai.gif", help="Output GIF path.")
def animate_cmd(what: str, out: str) -> None:
    """Export an animated GIF (needs [viz]): descent | diffusion | softmax."""
    makers = {
        "descent": lambda: animate_gradient_descent(out=out),
        "diffusion": lambda: animate_diffusion(out=out),
        "softmax": lambda: animate_softmax_temperature(out=out),
    }
    try:
        path = makers[what]()
    except ImportError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"saved → {path}")


@cli.command("editor")
@click.argument("expression", default="a*x^2 + b*x + c")
@click.option("--out", default="editable_plot.html", help="Output HTML path.")
def editor_cmd(expression: str, out: str) -> None:
    """Generate an editable equation↔graph HTML (open it in a browser)."""
    path = editable_plot(expression, out=out)
    click.echo(f"saved → {path}  (open it in a browser — edit the equation & drag the sliders)")


@cli.command("exercise")
@click.argument("lesson", required=False)
def exercise_cmd(lesson: str | None) -> None:
    """Compute-the-answer exercises (active recall). Omit LESSON to list them."""
    if lesson is None:
        click.echo("Exercises for: " + ", ".join(available_exercises()))
        return
    try:
        workbook = Workbook(lesson.lower())
    except KeyError as exc:
        raise click.BadParameter(
            f"no exercises for {lesson!r}. Try: {', '.join(available_exercises())}"
        ) from exc
    correct = 0
    for ex in workbook.exercises:
        click.echo(f"\n{ex.prompt}")
        result = workbook.grade(ex.id, click.prompt("Your answer", type=float))
        if result.correct:
            click.echo(click.style("  ✓ correct", fg="green"))
            correct += 1
        else:
            click.echo(click.style(f"  ✗ expected {result.expected}", fg="red"))
        click.echo(click.style(f"    {ex.explanation}", dim=True))
    click.echo(f"\nScore: {correct}/{len(workbook.exercises)}")


# ------------------------------------------------------------- notebooks + LLM
@cli.command("notebooks")
@click.option("--dir", "dest", default="optimumai-notebooks", help="Where to copy the notebooks.")
@click.option("--launch/--no-launch", default=True, help="Launch Jupyter after copying.")
def notebooks_cmd(dest: str, launch: bool) -> None:
    """Copy the bundled notebooks locally and (optionally) launch Jupyter."""
    import importlib.resources as resources
    import importlib.util as util

    dest_path = Path(dest)
    dest_path.mkdir(parents=True, exist_ok=True)
    src = resources.files("optimumai") / "_notebooks"
    copied = []
    for nb in src.iterdir():
        if nb.name.endswith(".ipynb"):
            (dest_path / nb.name).write_bytes(nb.read_bytes())
            copied.append(nb.name)
    click.echo(f"Copied {len(copied)} notebooks → {dest_path}/")
    if not launch:
        return
    lab = util.find_spec("jupyterlab") is not None
    if lab or util.find_spec("notebook") is not None:
        subprocess.run(
            [sys.executable, "-m", "jupyter", "lab" if lab else "notebook", str(dest_path)],
            check=False,
        )
    else:
        click.echo('Jupyter isn\'t installed. Install it:  pip install "optimumai[notebooks]"')
        click.echo(f"Then run:  jupyter lab {dest_path}")


@cli.command("generate")
@click.argument("prompt")
@click.option("--provider", default="auto", help="ollama | huggingface | anthropic | toy | auto.")
@click.option("--model", default=None, help="Model name (provider-specific).")
@click.option("--max-tokens", type=int, default=64, help="Max tokens to generate.")
@click.option("--temperature", "-t", type=float, default=0.7, help="Sampling temperature.")
@click.option("--level", type=_LEVEL_CHOICE, default="intermediate", help="Detail level.")
def generate_cmd(
    prompt: str, provider: str, model: str | None, max_tokens: int, temperature: float, level: str
) -> None:
    """Generate tokens from a prompt via a local/remote model (auto-detects Ollama)."""
    generate_trace(
        prompt, provider=provider, model=model, max_tokens=max_tokens, temperature=temperature
    ).render(level)


@cli.command("providers")
def providers_cmd() -> None:
    """List the generation providers available on this machine."""
    click.echo("Generation providers: " + ", ".join(available_providers()))


# --------------------------------------------------------- visualize / playground
@cli.command("visualize")
@click.argument("concept", required=False)
@click.option("--fmt", type=click.Choice(["png", "gif"]), default="png", help="Output format.")
@click.option("--out", default=None, help="Output path.")
def visualize_cmd(concept: str | None, fmt: str, out: str | None) -> None:
    """Render any concept to a PNG or GIF (needs [viz]). Omit CONCEPT to list them."""
    if concept is None:
        for name in list_concepts():
            click.echo(f"  {name:<18} {', '.join(concept_formats(name))}")
        return
    try:
        path = render_concept(concept.lower(), fmt=fmt, out=out)
    except (ValueError, ImportError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"saved → {path}")


@cli.command("playground")
@click.argument("concept", default="softmax")
@click.option("--out", default=None, help="Output HTML path.")
def playground_cmd(concept: str, out: str | None) -> None:
    """Interactive HTML playground: softmax, backprop, attention, kmeans, astar, plots."""
    key = concept.lower()
    target = out or f"interactive_{key}.html"
    if key in ("plots", "plot", "plot-studio", "dataviz"):
        path = plot_studio_playground(out=target)
    else:
        try:
            path = interactive_circuit(key, out=target)
        except ValueError:
            # Fall through to the vanilla-JS playgrounds (attention/kmeans/astar).
            try:
                path = viz_playground(key, out=target)
            except ValueError as exc:
                raise click.BadParameter(str(exc)) from exc
    click.echo(f"saved → {path}  (open it in a browser)")


# ============================ v1.1 command groups ============================


# --------------------------------------------------------------------- ml
@cli.group()
def ml() -> None:
    """Classical machine learning — fit a small model and see every step."""


@ml.command("linreg")
@click.argument("x", required=False)
@click.argument("y", required=False)
@click.option("--level", type=_LEVEL_CHOICE, default="intermediate", help="Detail level.")
def ml_linreg(x: str | None, y: str | None, level: str) -> None:
    """OLS linear regression via the normal equation. Omit args for the demo."""
    if x and y:
        linear_regression_trace(_parse_literal(x, "X"), _parse_literal(y, "y")).render(level)
    else:
        ml_linreg_demo().render(level)


@ml.command("kmeans")
@click.argument("points", required=False)
@click.option("--k", type=int, default=2, help="Number of clusters.")
@click.option("--level", type=_LEVEL_CHOICE, default="intermediate", help="Detail level.")
def ml_kmeans(points: str | None, k: int, level: str) -> None:
    """k-means (Lloyd's algorithm). Omit POINTS for the demo."""
    if points:
        kmeans_trace(_parse_literal(points, "points"), k=k).render(level)
    else:
        ml_kmeans_demo().render(level)


@ml.command("logreg")
@click.option("--level", type=_LEVEL_CHOICE, default="intermediate", help="Detail level.")
def ml_logreg(level: str) -> None:
    """Logistic regression: sigmoid + cross-entropy + gradient descent."""
    ml_logreg_demo().render(level)


@ml.command("knn")
@click.option("--level", type=_LEVEL_CHOICE, default="intermediate", help="Detail level.")
def ml_knn(level: str) -> None:
    """k-nearest-neighbors classification by majority vote."""
    ml_knn_demo().render(level)


@ml.command("tree")
@click.option("--level", type=_LEVEL_CHOICE, default="intermediate", help="Detail level.")
def ml_tree(level: str) -> None:
    """Decision tree: best split by Gini / entropy information gain."""
    ml_tree_demo().render(level)


@ml.command("nb")
@click.option("--level", type=_LEVEL_CHOICE, default="intermediate", help="Detail level.")
def ml_nb(level: str) -> None:
    """Gaussian Naive Bayes via Bayes' rule."""
    ml_nb_demo().render(level)


@ml.command("pca")
@click.option("--level", type=_LEVEL_CHOICE, default="intermediate", help="Detail level.")
def ml_pca(level: str) -> None:
    """PCA via covariance eigendecomposition."""
    ml_pca_demo().render(level)


@ml.command("metrics")
@click.option("--level", type=_LEVEL_CHOICE, default="intermediate", help="Detail level.")
def ml_metrics(level: str) -> None:
    """Classification & regression metrics (accuracy, F1, MSE, R², ROC-AUC)."""
    ml_metrics_demo().render(level)


# ------------------------------------------------------------------- algo
@cli.group()
def algo() -> None:
    """Classical AI search — BFS/DFS/UCS, A*, minimax (distinct from `search`)."""


@algo.command("bfs")
@click.option("--level", type=_LEVEL_CHOICE, default="intermediate", help="Detail level.")
def algo_bfs(level: str) -> None:
    """Uninformed search (BFS / DFS / UCS) on a demo graph."""
    algo_uninformed_demo().render(level)


@algo.command("astar")
@click.option("--level", type=_LEVEL_CHOICE, default="intermediate", help="Detail level.")
def algo_astar(level: str) -> None:
    """Informed search — greedy best-first & A* — on a demo grid."""
    algo_informed_demo().render(level)


@algo.command("minimax")
@click.option("--level", type=_LEVEL_CHOICE, default="intermediate", help="Detail level.")
def algo_minimax(level: str) -> None:
    """Adversarial search — minimax + alpha-beta pruning — on a demo tree."""
    algo_adversarial_demo().render(level)


# --------------------------------------------------------------------- rl
@cli.group()
def rl() -> None:
    """Reinforcement learning — MDPs, Q-learning/SARSA, REINFORCE, PPO."""


@rl.command("mdp")
@click.option("--level", type=_LEVEL_CHOICE, default="intermediate", help="Detail level.")
def rl_mdp(level: str) -> None:
    """Value iteration on a demo MDP (the Bellman equation in action)."""
    rl_mdp_demo().render(level)


@rl.command("q-learning")
@click.option("--level", type=_LEVEL_CHOICE, default="intermediate", help="Detail level.")
def rl_qlearning(level: str) -> None:
    """Tabular Q-learning / SARSA on a demo gridworld."""
    rl_q_demo().render(level)


@rl.command("reinforce")
@click.option("--level", type=_LEVEL_CHOICE, default="intermediate", help="Detail level.")
def rl_reinforce(level: str) -> None:
    """Policy-gradient REINFORCE on a demo bandit."""
    rl_reinforce_demo().render(level)


@rl.command("ppo")
@click.option("--level", type=_LEVEL_CHOICE, default="intermediate", help="Detail level.")
def rl_ppo(level: str) -> None:
    """The PPO clipped surrogate objective on a demo batch."""
    rl_ppo_demo().render(level)


# -------------------------------------------------------------------- nlp
@cli.group()
def nlp() -> None:
    """Classical NLP — BPE, TF-IDF, n-grams, edit distance, word2vec."""


@nlp.command("bpe")
@click.argument("word", required=False)
@click.option("--merges", type=int, default=8, help="Number of merge rules to learn.")
@click.option("--level", type=_LEVEL_CHOICE, default="intermediate", help="Detail level.")
def nlp_bpe(word: str | None, merges: int, level: str) -> None:
    """Learn BPE merges on a toy corpus, then tokenize WORD. Omit WORD for the demo."""
    if word:
        corpus = ["low", "lower", "lowest", "newer", "newest", "wider", "widest"]
        bpe_trace(corpus, num_merges=merges, encode_word=word).render(level)
    else:
        nlp_bpe_demo().render(level)


@nlp.command("tfidf")
@click.argument("docs", nargs=-1)
@click.option("--level", type=_LEVEL_CHOICE, default="intermediate", help="Detail level.")
def nlp_tfidf(docs: tuple[str, ...], level: str) -> None:
    """TF-IDF over the given DOCS (quoted strings). Omit for the demo."""
    if docs:
        tfidf_trace(list(docs)).render(level)
    else:
        nlp_tfidf_demo().render(level)


@nlp.command("ngram")
@click.option("--level", type=_LEVEL_CHOICE, default="intermediate", help="Detail level.")
def nlp_ngram(level: str) -> None:
    """N-gram language model with add-k smoothing + perplexity."""
    nlp_ngram_demo().render(level)


@nlp.command("edit-distance")
@click.argument("a", required=False)
@click.argument("b", required=False)
@click.option("--level", type=_LEVEL_CHOICE, default="intermediate", help="Detail level.")
def nlp_edit_distance(a: str | None, b: str | None, level: str) -> None:
    """Levenshtein edit distance between A and B. Omit args for the demo."""
    if a and b:
        edit_distance_trace(a, b).render(level)
    else:
        nlp_edit_demo().render(level)


@nlp.command("word2vec")
@click.option("--level", type=_LEVEL_CHOICE, default="engineer", help="Detail level.")
def nlp_word2vec(level: str) -> None:
    """Skip-gram word2vec — one SGD step on a tiny corpus."""
    nlp_word2vec_demo().render(level)


# ----------------------------------------------------------------- vision
@cli.group()
def vision() -> None:
    """Computer vision — convolution, pooling, Sobel edges, a tiny CNN."""


@vision.command("conv")
@click.argument("image", required=False)
@click.argument("kernel", required=False)
@click.option("--stride", type=int, default=1)
@click.option("--padding", type=int, default=0)
@click.option("--level", type=_LEVEL_CHOICE, default="intermediate", help="Detail level.")
def vision_conv(
    image: str | None, kernel: str | None, stride: int, padding: int, level: str
) -> None:
    """2D convolution of IMAGE with KERNEL (nested lists). Omit args for the demo."""
    if image and kernel:
        conv2d_trace(
            _parse_literal(image, "image"),
            _parse_literal(kernel, "kernel"),
            stride=stride,
            padding=padding,
        ).render(level)
    else:
        vision_conv_demo().render(level)


@vision.command("pool")
@click.option("--level", type=_LEVEL_CHOICE, default="intermediate", help="Detail level.")
def vision_pool(level: str) -> None:
    """Max & average pooling on a demo feature map."""
    vision_pool_demo().render(level)


@vision.command("sobel")
@click.option("--level", type=_LEVEL_CHOICE, default="intermediate", help="Detail level.")
def vision_sobel(level: str) -> None:
    """Sobel edge detection on a demo image."""
    vision_sobel_demo().render(level)


@vision.command("cnn")
@click.option("--level", type=_LEVEL_CHOICE, default="engineer", help="Detail level.")
def vision_cnn(level: str) -> None:
    """A tiny CNN forward pass — watch the tensor shapes flow."""
    vision_cnn_demo().render(level)


# ------------------------------------------------------------------- eval
@cli.group("eval")
def eval_group() -> None:
    """LLM evaluation — BLEU/ROUGE, perplexity, calibration, faithfulness."""


@eval_group.command("bleu")
@click.argument("candidate", required=False)
@click.argument("reference", required=False)
@click.option("--max-n", type=int, default=4, help="Max n-gram order.")
@click.option("--level", type=_LEVEL_CHOICE, default="intermediate", help="Detail level.")
def eval_bleu(candidate: str | None, reference: str | None, max_n: int, level: str) -> None:
    """BLEU of CANDIDATE against REFERENCE. Omit args for the demo."""
    if candidate and reference:
        bleu_trace(candidate, reference, max_n=max_n).render(level)
    else:
        eval_text_demo().render(level)


@eval_group.command("rouge")
@click.argument("candidate", required=False)
@click.argument("reference", required=False)
@click.option("-n", type=int, default=1, help="ROUGE-N order.")
@click.option("--level", type=_LEVEL_CHOICE, default="intermediate", help="Detail level.")
def eval_rouge(candidate: str | None, reference: str | None, n: int, level: str) -> None:
    """ROUGE-N of CANDIDATE against REFERENCE. Omit args for the demo."""
    if candidate and reference:
        rouge_n_trace(candidate, reference, n=n).render(level)
    else:
        eval_text_demo().render(level)


@eval_group.command("perplexity")
@click.argument("probs", required=False)
@click.option("--level", type=_LEVEL_CHOICE, default="intermediate", help="Detail level.")
def eval_perplexity(probs: str | None, level: str) -> None:
    """Perplexity from token probabilities, e.g. "[0.5,0.25,0.8]". Omit for the demo."""
    if probs:
        perplexity_trace(_parse_literal(probs, "probs")).render(level)
    else:
        eval_ppl_demo().render(level)


@eval_group.command("calibration")
@click.option("--level", type=_LEVEL_CHOICE, default="intermediate", help="Detail level.")
def eval_calibration(level: str) -> None:
    """Expected Calibration Error (ECE) with reliability bins."""
    eval_cal_demo().render(level)


@eval_group.command("faithfulness")
@click.option("--level", type=_LEVEL_CHOICE, default="intermediate", help="Detail level.")
def eval_faithfulness(level: str) -> None:
    """A grounding/faithfulness heuristic (an educational hallucination proxy)."""
    eval_faith_demo().render(level)


_PROMPT_DEMOS = {
    "zero-shot": prompt_zeroshot_demo,
    "few-shot": prompt_fewshot_demo,
    "chain-of-thought": prompt_cot_demo,
    "react": prompt_react_demo,
    "self-consistency": prompt_selfcons_demo,
    "structured-output": prompt_structured_demo,
}
_AUGRNN_DEMOS = {
    "attention": augrnn_attn_demo,
    "ntm": augrnn_ntm_demo,
    "act": augrnn_act_demo,
}


def _list_choices(group: str, keys: list[str]) -> None:
    """Print ready-to-copy commands (one per line) — never a shell-piping `a|b|c`."""
    click.echo("Pick one — run any of these:\n")
    for k in keys:
        click.echo(f"  optimumai {group} {k}")


@cli.command("prompt")
@click.argument("pattern", required=False, type=click.Choice(list(_PROMPT_DEMOS)))
@click.option("--level", type=_LEVEL_CHOICE, default="intermediate", help="Detail level.")
def prompt_cmd(pattern: str | None, level: str) -> None:
    """Prompt-engineering patterns. Run 'optimumai prompt' to list them."""
    if pattern is None:
        _list_choices("prompt", list(_PROMPT_DEMOS))
        return
    _PROMPT_DEMOS[pattern]().render(level)


@cli.command("augrnn")
@click.argument("concept", required=False, type=click.Choice(list(_AUGRNN_DEMOS)))
@click.option("--level", type=_LEVEL_CHOICE, default="intermediate", help="Detail level.")
def augrnn_cmd(concept: str | None, level: str) -> None:
    """Augmented RNNs (distill.pub). Run 'optimumai augrnn' to list them."""
    if concept is None:
        _list_choices("augrnn", list(_AUGRNN_DEMOS))
        return
    _AUGRNN_DEMOS[concept]().render(level)


_FLOW_CONCEPTS = ["transformer", "attention", "tfidf", "word2vec"]
_PLOT_KINDS = ["bar", "hist", "scatter", "box", "line", "pie", "violin"]


@cli.command("flow")
@click.argument("concept", required=False, type=click.Choice(_FLOW_CONCEPTS))
@click.option("--out", default=None, help="Output HTML path.")
def flow_cmd(concept: str | None, out: str | None) -> None:
    """Interactive concept-flow diagram (distill-style). Run 'optimumai flow' to list."""
    if concept is None:
        click.echo("Interactive flow diagrams — run one of:\n")
        for c in _FLOW_CONCEPTS:
            click.echo(f"  optimumai flow {c}")
        return
    path = build_flow(concept, out=out)
    click.echo(f"saved → {path}  (open it in a browser)")


@cli.command("plot-studio")
@click.argument("numbers", required=False)
@click.option("--kind", type=click.Choice(_PLOT_KINDS), default="bar", help="Chart type.")
@click.option("--out", default="plot.png", help="Chart image path.")
def plot_studio_cmd(numbers: str | None, kind: str, out: str) -> None:
    """Chart your numbers AND print the matplotlib+numpy code, e.g.
    plot-studio "[3,1,4,1,5,9,2,6]" --kind hist.
    """
    data = _parse_literal(numbers, "numbers") if numbers else [3, 1, 4, 1, 5, 9, 2, 6]
    stats = ps_describe(data)
    click.echo(click.style("numpy summary:", bold=True))
    click.echo("  " + "  ".join(f"{k}={v:.4g}" for k, v in stats.items()))
    click.echo(click.style("\nmatplotlib + numpy code:", bold=True))
    click.echo(ps_plot_code(data, kind=kind))
    try:
        path = ps_plot_data(data, kind=kind, out=out)
        click.echo(f"\nsaved chart → {path}")
    except ImportError:
        click.echo('\n(install rendering deps to save the chart: pip install "optimumai[viz]")')


if __name__ == "__main__":  # pragma: no cover
    cli()
