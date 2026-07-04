"""Zero-shot prompting — asking for a task with no worked examples.

**What it is.** The plainest prompt shape: a role/persona line, an instruction
that states the task and any constraints, and the input to act on. No
demonstrations of input→output pairs are given — the model must rely
entirely on what it learned during pretraining/instruction-tuning to infer
the mapping.

**How you build it.** Three slots, concatenated in order:

1. ``role`` — who the model should act as (sets tone/vocabulary/register).
2. ``instruction`` — the task statement plus constraints (format, length,
   tone). This is the load-bearing part of a zero-shot prompt, since there
   are no examples to lean on.
3. ``task`` — the actual input the instruction should be applied to.

**Why it works.** Large instruction-tuned models have already seen millions
of (instruction, response) pairs during fine-tuning/RLHF, so a clear enough
instruction alone often triggers the right behavior — no in-context
demonstration needed. It is the cheapest prompt in tokens and latency.

**Failure modes.**

* Ambiguous or underspecified instructions ("summarize this") let the model
  guess the format, length, and tone — variance goes up.
* Tasks that require a specific output *shape* (a rare label taxonomy, a
  house style, a niche notation) are exactly where zero-shot struggles —
  the model has no local pattern to match, only its prior.
* Multi-step reasoning tasks (arithmetic, multi-hop QA) tend to fail more
  often zero-shot than with a reasoning scaffold (see :mod:`chain_of_thought`)
  or worked exemplars (see :mod:`few_shot`).

Use zero-shot when the task is common, the instruction can be fully
specified in words, and you want the lowest latency/cost. Reach for
few-shot or CoT once instructions alone stop reliably producing the shape
of answer you need.
"""

from __future__ import annotations

from optimumai.core._fmt import arr
from optimumai.core.explain import ExplainLevel
from optimumai.core.trace import Trace

_DEFAULT_ROLE = "You are a helpful, precise assistant."


def zero_shot_trace(
    task: str,
    instruction: str = "Complete the task below.",
    role: str = _DEFAULT_ROLE,
) -> Trace:
    """Build a zero-shot prompt: role + instruction + task, step by step."""
    if not task.strip():
        raise ValueError("task must be a non-empty string")
    if not instruction.strip():
        raise ValueError("instruction must be a non-empty string")

    t = Trace(
        op="zero_shot",
        formula="prompt = role ⊕ instruction ⊕ task",
        complexity="O(1) prompt assembly; no demonstrations",
        why_ai=[
            "The default prompt shape for any well-specified, common task",
            "Cheapest option: no exemplar tokens, lowest latency",
            "Relies entirely on instruction-tuning/RLHF prior — no local pattern to copy",
            "Struggles on tasks needing a specific/rare output shape or multi-step reasoning",
        ],
        meta={"role": role, "instruction": instruction},
    )

    t.add(
        "Set the role",
        f"role = {arr(role)}",
        role,
        detail="Anchors tone, vocabulary, and implicit expertise before the task is stated.",
    )
    t.add(
        "State the instruction",
        f"instruction = {arr(instruction)}",
        instruction,
        detail=(
            "This is the load-bearing part of a zero-shot prompt: with no examples to "
            "imitate, ambiguity here becomes variance in the output."
        ),
    )
    t.add(
        "Append the task input",
        f"task = {arr(task)}",
        task,
        detail="The concrete input the instruction should be applied to.",
    )

    prompt = f"{role}\n\n{instruction}\n\nTask: {task}"
    t.add(
        "Assemble the final prompt",
        "prompt = role + '\\n\\n' + instruction + '\\n\\n' + 'Task: ' + task",
        prompt,
        detail="No demonstrations are included — the model must generalize from the "
        "instruction alone.",
    )

    t.result = prompt
    return t


def zero_shot(
    task: str,
    instruction: str = "Complete the task below.",
    role: str = _DEFAULT_ROLE,
    explain: bool = False,
    level: str | ExplainLevel = ExplainLevel.INTERMEDIATE,
) -> str:
    """Return the assembled zero-shot prompt. Set ``explain=True`` to print the trace."""
    t = zero_shot_trace(task, instruction=instruction, role=role)
    return t.render(level) if explain else t.result


def demo(seed: int = 0) -> Trace:
    """Zero-shot prompt for a sentiment-classification task."""
    return zero_shot_trace(
        task="The battery life on this laptop is disappointing.",
        instruction="Classify the sentiment of the review as positive, negative, or neutral. "
        "Respond with a single word.",
        role="You are a precise sentiment-classification assistant.",
    )
