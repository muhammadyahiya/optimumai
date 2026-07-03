"""Interactive input: prompt for your own values, or drive OptimumAI from a REPL."""

from optimumai.interactive.prompts import (
    parse_matrix,
    parse_vector,
    prompt_matrix,
    prompt_vector,
)
from optimumai.interactive.repl import run_repl

__all__ = [
    "parse_matrix",
    "parse_vector",
    "prompt_matrix",
    "prompt_vector",
    "run_repl",
]
