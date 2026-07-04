"""Chain-of-thought (CoT) prompting — eliciting intermediate reasoning.

**What it is.** Rather than asking for the final answer directly, the prompt
asks the model to produce (or is shown examples of) the intermediate
reasoning steps that lead to it. Two common variants, both supported here:

* *Zero-shot CoT*: append a trigger phrase like "Let's think step by step"
  after the task — no worked examples needed.
* *Few-shot CoT*: prepend exemplars whose ``output`` field is itself a
  reasoning chain ending in a final answer, so the model imitates showing
  its work, not just the answer.

**How you build it.**

1. (Few-shot CoT only) add worked ``(question, reasoning → answer)``
   exemplars, each rendered with an explicit reasoning chain.
2. State the task/question.
3. Append the reasoning scaffold — the trigger phrase for zero-shot CoT, or
   just the question if exemplars already demonstrate the pattern.
4. Leave an explicit "Answer:" slot after the reasoning so the final answer
   can be parsed out separately from the scratch work.

**Why it helps.** Decomposing a problem into steps lets the model allocate
more computation (more forward passes' worth of tokens) to hard
sub-problems instead of jumping straight to a token-level guess. Empirically
this is most useful on tasks with multiple dependent steps — arithmetic,
multi-hop logic, symbolic manipulation — and offers little to no benefit on
single-step lookups.

**Failure modes.**

* CoT reasoning is not guaranteed to be *faithful*: a model can emit a
  plausible-looking chain that does not reflect how it actually arrived at
  the answer, or worse, **rationalize a wrong answer after the fact** with
  a chain that reads as if it justifies it.
* Longer chains cost more tokens/latency for tasks that did not need them.
* A flawed worked exemplar (few-shot CoT) teaches the wrong reasoning
  *pattern*, not just the wrong answer — errors compound more than in
  plain few-shot.
"""

from __future__ import annotations

from optimumai.core._fmt import arr
from optimumai.core.explain import ExplainLevel
from optimumai.core.trace import Trace

DEFAULT_TRIGGER = "Let's think step by step."


def chain_of_thought_trace(
    task: str,
    examples: list[tuple[str, str]] | None = None,
    trigger: str = DEFAULT_TRIGGER,
) -> Trace:
    """Build a CoT prompt: optional worked exemplars, task, reasoning scaffold, answer slot."""
    if not task.strip():
        raise ValueError("task must be a non-empty string")
    examples = examples or []

    mode = "few-shot CoT" if examples else "zero-shot CoT"
    t = Trace(
        op="chain_of_thought",
        formula="prompt = {(qᵢ, reasoningᵢ → answerᵢ)}* ⊕ task ⊕ scaffold ⊕ 'Answer:'",
        complexity=f"O(K) exemplars + O(reasoning length); mode = {mode}",
        why_ai=[
            "Spends more generated tokens (~compute) on hard sub-steps before committing "
            "to an answer",
            "Large, consistent gains on multi-step arithmetic/logic; little benefit on "
            "single-step lookups",
            "Reasoning is not guaranteed faithful — a chain can rationalize a wrong "
            "answer rather than derive it",
            "Zero-shot CoT ('Let's think step by step') needs no exemplars; few-shot CoT "
            "demonstrates the reasoning *style* directly",
        ],
        meta={"mode": mode, "trigger": trigger, "k": len(examples)},
    )

    blocks: list[str] = []
    for i, (question, reasoning) in enumerate(examples, start=1):
        block = f"Q: {question}\nA: {reasoning}"
        blocks.append(block)
        t.add(
            f"Add worked exemplar {i}/{len(examples)}",
            block,
            block,
            detail="The exemplar's answer *shows the reasoning*, not just the final value — "
            "the model imitates the process, not only the outcome.",
        )

    t.add(
        "State the task",
        f"task = {arr(task)}",
        task,
    )

    if examples:
        scaffold = f"Q: {task}\nA:"
        detail = "No trigger phrase needed — the exemplars already demonstrate showing work."
    else:
        scaffold = f"Q: {task}\nA: {trigger}"
        detail = (
            f"'{trigger}' is a zero-shot trigger: it costs one line and reliably elicits "
            "a reasoning chain from instruction-tuned models with no exemplars at all."
        )
    t.add("Append the reasoning scaffold", scaffold, scaffold, detail=detail)

    parts = blocks + [scaffold]
    prompt = "\n\n".join(parts)
    t.add(
        "Assemble the final prompt",
        "prompt = exemplar blocks + task/scaffold block, joined by blank lines",
        prompt,
        detail="The model is expected to continue with reasoning steps and a final answer "
        "after this prompt — e.g. terminated with a line like 'Answer: <value>'.",
    )

    t.result = prompt
    return t


def chain_of_thought(
    task: str,
    examples: list[tuple[str, str]] | None = None,
    trigger: str = DEFAULT_TRIGGER,
    explain: bool = False,
    level: str | ExplainLevel = ExplainLevel.INTERMEDIATE,
) -> str:
    """Return the assembled CoT prompt. Set ``explain=True`` to print the trace."""
    t = chain_of_thought_trace(task, examples=examples, trigger=trigger)
    return t.render(level) if explain else t.result


def demo(seed: int = 0) -> Trace:
    """Zero-shot CoT prompt for a simple multi-step arithmetic word problem."""
    return chain_of_thought_trace(
        task="A store had 23 apples, sold 8, then received a shipment of 15 more. "
        "How many apples does it have now?",
    )
