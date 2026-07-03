"""Prompt helpers for the ``--interactive`` flag and the REPL.

These let a user type their own vectors/matrices at runtime and then watch their
numbers flow through the full step trace — the "give it your own input" idea.
"""

from __future__ import annotations

import ast

import click


def parse_vector(text: str) -> list[float]:
    """Parse ``"[1, 2, 3]"`` or ``"1 2 3"`` or ``"1,2,3"`` into a list of floats."""
    text = text.strip()
    try:
        value = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        parts = text.replace(",", " ").split()
        try:
            return [float(p) for p in parts]
        except ValueError as exc:
            raise click.BadParameter(f"could not parse a vector from {text!r}") from exc
    if isinstance(value, (int, float)):
        return [float(value)]
    return [float(v) for v in value]


def parse_matrix(text: str) -> list[list[float]]:
    """Parse ``"[[1,2],[3,4]]"`` into a list of rows of floats."""
    try:
        value = ast.literal_eval(text.strip())
    except (ValueError, SyntaxError) as exc:
        raise click.BadParameter(f"could not parse a matrix from {text!r}") from exc
    return [[float(x) for x in row] for row in value]


def prompt_vector(label: str = "vector") -> list[float]:
    """Interactively ask the user for a vector."""
    return parse_vector(click.prompt(f"Enter {label} (e.g. [1, 2, 3])", type=str))


def prompt_matrix(label: str = "matrix") -> list[list[float]]:
    """Interactively ask the user for a matrix."""
    return parse_matrix(click.prompt(f"Enter {label} (e.g. [[1,2],[3,4]])", type=str))
