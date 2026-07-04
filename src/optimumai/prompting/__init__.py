"""Prompt-engineering patterns — constructed and explained offline.

Every pattern in this package builds a prompt *deterministically*, step by
step, with no live LLM calls: the point is to see exactly how the prompt
string is assembled and why the pattern helps (or where it fails), not to
call out to a model. Each submodule exposes a ``<name>_trace`` function
(returns a :class:`~optimumai.core.trace.Trace`), a thin ``<name>`` wrapper
(returns the assembled prompt string, or renders the trace if
``explain=True``), and a ``demo`` function for the curriculum/CLI.
"""

from optimumai.prompting.chain_of_thought import (
    chain_of_thought,
    chain_of_thought_trace,
)
from optimumai.prompting.few_shot import few_shot, few_shot_trace
from optimumai.prompting.react import react, react_trace
from optimumai.prompting.self_consistency import (
    self_consistency,
    self_consistency_trace,
)
from optimumai.prompting.structured_output import (
    structured_output,
    structured_output_trace,
)
from optimumai.prompting.zero_shot import zero_shot, zero_shot_trace

__all__ = [
    "chain_of_thought",
    "chain_of_thought_trace",
    "few_shot",
    "few_shot_trace",
    "react",
    "react_trace",
    "self_consistency",
    "self_consistency_trace",
    "structured_output",
    "structured_output_trace",
    "zero_shot",
    "zero_shot_trace",
]
