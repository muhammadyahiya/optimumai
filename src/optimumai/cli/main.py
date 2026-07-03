"""The ``optimumai`` command-line interface.

Examples:
    optimumai algebra dot "[1,2,3]" "[4,5,6]"
    optimumai softmax "[2,1,0.1]" --temperature 0.5
    optimumai attention --demo
    optimumai backprop
    optimumai train --steps 150
    optimumai jepa --demo
    optimumai superposition --features 5 --neurons 2
    optimumai learn            # list every topic
"""

from __future__ import annotations

import ast

import click

from optimumai import __version__
from optimumai.algebra.matrix import Matrix
from optimumai.algebra.vector import Vector
from optimumai.autograd.value import Value
from optimumai.calculus.derivative import chain_rule_trace, derivative_trace, gradient_trace
from optimumai.core.explain import ExplainLevel
from optimumai.interpretability.superposition import superposition_trace
from optimumai.neural_networks.backprop import train_demo
from optimumai.optimization.optimizers import descent_demo
from optimumai.probability.softmax import softmax_trace
from optimumai.transformers.attention import Attention
from optimumai.transformers.block import TransformerBlock
from optimumai.transformers.multihead import MultiHeadAttention
from optimumai.transformers.positional import positional_encoding_trace
from optimumai.world_models.jepa import JEPA

_LEVEL_CHOICE = click.Choice([lvl.value for lvl in ExplainLevel], case_sensitive=False)


def _backprop_demo():
    """A small labelled scalar graph, differentiated end to end (micrograd-style)."""
    a = Value(2.0, label="a")
    b = Value(-3.0, label="b")
    c = Value(10.0, label="c")
    e = a * b
    e.label = "e"
    d = e + c
    d.label = "d"
    f = Value(-2.0, label="f")
    loss = d * f
    loss.label = "L"
    return loss.backward_trace()


# topic -> (one-line description, zero-arg callable returning a Trace)
_TOPICS: dict[str, tuple[str, callable]] = {
    "dot": (
        "Dot product — similarity and the atom of matrix multiply",
        lambda: Vector([1, 2, 3]).dot_trace(Vector([4, 5, 6])),
    ),
    "cosine": (
        "Cosine similarity — the RAG / semantic-search ranking score",
        lambda: Vector([1, 2, 3]).cosine_similarity_trace(Vector([2, 4, 6])),
    ),
    "matmul": (
        "Matrix multiplication — every dense layer, cell by cell",
        lambda: Matrix([[1, 2], [3, 4]]).matmul_trace(Matrix([[5, 6], [7, 8]])),
    ),
    "softmax": (
        "Softmax — logits into a probability distribution",
        lambda: softmax_trace([2.0, 1.0, 0.1]),
    ),
    "attention": (
        "Scaled dot-product attention — the transformer core",
        Attention.demo,
    ),
    "derivative": (
        "Derivative — the finite-difference slope, checked against autograd",
        lambda: derivative_trace(lambda x: x**3, 2.0, label="x³"),
    ),
    "gradient": (
        "Gradient — the vector of partial derivatives",
        lambda: gradient_trace(lambda p: p[0] ** 2 + p[1] ** 2, [3.0, 4.0]),
    ),
    "chain_rule": (
        "Chain rule — the single idea behind all backpropagation",
        chain_rule_trace,
    ),
    "backprop": (
        "Backpropagation — reverse-mode autodiff on a scalar graph (micrograd)",
        _backprop_demo,
    ),
    "descent": (
        "Gradient descent — Adam walking a loss bowl to its minimum",
        lambda: descent_demo("adam", steps=60),
    ),
    "train": (
        "Train an MLP — the full loop: predict → loss → backprop → step",
        lambda: train_demo(steps=120),
    ),
    "positional": (
        "Positional encoding — injecting word order into attention",
        lambda: positional_encoding_trace(6, 8),
    ),
    "multihead": (
        "Multi-head attention — parallel heads + causal mask (nanoGPT)",
        MultiHeadAttention.demo,
    ),
    "transformer": (
        "Transformer block — LayerNorm → attention → FFN with residuals",
        TransformerBlock.demo,
    ),
    "jepa": (
        "JEPA — LeCun's predict-in-latent-space world model",
        JEPA.demo,
    ),
    "superposition": (
        "Superposition — why neurons are polysemantic (Anthropic)",
        lambda: superposition_trace(5, 2),
    ),
}


def _parse_literal(text: str, what: str):
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError) as exc:
        raise click.BadParameter(
            f"could not parse {what}: {text!r} (expected e.g. \"[1, 2, 3]\")"
        ) from exc


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__, prog_name="optimumai")
def cli() -> None:
    """OptimumAI — unlock the math behind AI, one explained step at a time."""


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
    va = Vector(_parse_literal(a, "vector A"))
    vb = Vector(_parse_literal(b, "vector B"))
    va.dot(vb, explain=True, level=level)


@algebra.command("cosine")
@click.argument("a")
@click.argument("b")
@click.option("--level", type=_LEVEL_CHOICE, default="intermediate", help="Detail level.")
def algebra_cosine(a: str, b: str, level: str) -> None:
    """Cosine similarity of two vectors."""
    va = Vector(_parse_literal(a, "vector A"))
    vb = Vector(_parse_literal(b, "vector B"))
    va.cosine_similarity(vb, explain=True, level=level)


@algebra.command("matmul")
@click.argument("a")
@click.argument("b")
@click.option("--level", type=_LEVEL_CHOICE, default="intermediate", help="Detail level.")
def algebra_matmul(a: str, b: str, level: str) -> None:
    """Matrix product, e.g. optimumai algebra matmul "[[1,2],[3,4]]" "[[5,6],[7,8]]"."""
    ma = Matrix(_parse_literal(a, "matrix A"))
    mb = Matrix(_parse_literal(b, "matrix B"))
    ma.matmul(mb, explain=True, level=level)


# --------------------------------------------------------------------- softmax
@cli.command("softmax")
@click.argument("logits")
@click.option("--temperature", "-t", type=float, default=1.0, help="Sampling temperature (>0).")
@click.option("--level", type=_LEVEL_CHOICE, default="intermediate", help="Detail level.")
def softmax_cmd(logits: str, temperature: float, level: str) -> None:
    """Softmax of a logit vector, e.g. optimumai softmax "[2,1,0.1]"."""
    values = _parse_literal(logits, "logits")
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
    _backprop_demo().render(level)


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


# ----------------------------------------------------------------------- learn
@cli.command("learn")
@click.argument("topic", required=False)
@click.option("--level", type=_LEVEL_CHOICE, default="intermediate", help="Detail level.")
def learn_cmd(topic: str | None, level: str) -> None:
    """Walk through a topic's math. Run without a topic to list them all."""
    if topic is None:
        click.echo("Available topics:\n")
        for name, (desc, _) in _TOPICS.items():
            click.echo(f"  {name:<14} {desc}")
        click.echo("\nTry:  optimumai learn backprop --level engineer")
        return
    key = topic.lower()
    if key not in _TOPICS:
        raise click.BadParameter(
            f"unknown topic {topic!r}. Choose from: {', '.join(_TOPICS)}"
        )
    _, build = _TOPICS[key]
    build().render(level)


if __name__ == "__main__":  # pragma: no cover
    cli()
