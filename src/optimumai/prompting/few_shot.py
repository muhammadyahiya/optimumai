"""Few-shot prompting — in-context learning from labeled examples.

**What it is.** Instead of (or in addition to) an instruction, the prompt
prepends *K* worked ``(input, output)`` exemplars before the actual query.
The model conditions on the pattern in those exemplars — the task, the
output format, the vocabulary, the level of detail — and continues it for
the new input. This is "in-context learning": no gradient update happens,
the demonstrations live entirely in the prompt.

**How you build it.**

1. An optional short instruction framing the task.
2. Each exemplar rendered in a consistent template, e.g.
   ``Input: ...\\nOutput: ...`` — added to the trace one at a time so the
   effect of each additional example is visible.
3. The real query, rendered in the *same* template but with the output left
   blank for the model to fill in.

**Why it works.** Exemplars pin down exactly what zero-shot instructions
leave ambiguous: output format, label vocabulary, granularity, tone. The
model performs implicit pattern completion — it does not need the task named
in words at all if the exemplars make the mapping unambiguous.

**Example selection + ordering effects (the two knobs that matter most).**

* *Selection*: exemplars close to the query (same domain, similar
  difficulty) transfer better than generic or unrelated ones; diverse
  exemplars covering edge cases reduce the chance the model latches onto a
  spurious shortcut (e.g. "reviews are always negative").
* *Ordering*: LLMs are sensitive to exemplar order — recency-biased models
  weight the last exemplar more heavily, and an unlucky order can shift
  accuracy by double digits on the same example set. Randomizing or
  putting the strongest/most-representative exemplar last are common
  mitigations.
* *Label balance*: an unbalanced set of exemplars (e.g. 4 "positive" and 1
  "negative") biases the model toward the majority label regardless of the
  true answer — a well-documented majority-label bias.

**Failure modes.** Too few exemplars under-specify the pattern; too many
burn context budget for diminishing returns; a bad or mislabeled exemplar
teaches the wrong mapping just as confidently as a correct one.
"""

from __future__ import annotations

from optimumai.core._fmt import arr
from optimumai.core.explain import ExplainLevel
from optimumai.core.trace import Trace


def few_shot_trace(
    task: str,
    examples: list[tuple[str, str]],
    instruction: str = "",
) -> Trace:
    """Build a few-shot prompt: instruction + K exemplars + query, one exemplar at a time."""
    if not task.strip():
        raise ValueError("task must be a non-empty string")
    if not examples:
        raise ValueError("few_shot requires at least one example, got 0")

    k = len(examples)
    t = Trace(
        op="few_shot",
        formula="prompt = instruction ⊕ {(inputᵢ, outputᵢ)}₁..ₖ ⊕ query",
        complexity=f"O(K) exemplar tokens, K = {k}",
        why_ai=[
            "In-context learning: the model conditions on demonstrations, no gradient step",
            "Exemplars pin down output format/vocabulary that zero-shot leaves ambiguous",
            "Order and label balance of exemplars measurably shift accuracy (recency + "
            "majority-label bias)",
            "Diminishing returns and rising latency/cost as K grows — pick quality over quantity",
        ],
        meta={"k": k, "instruction": instruction},
    )

    if instruction.strip():
        t.add(
            "State the instruction",
            f"instruction = {arr(instruction)}",
            instruction,
            detail="Optional in few-shot: exemplars alone can convey the task.",
        )

    blocks: list[str] = []
    for i, (ex_input, ex_output) in enumerate(examples, start=1):
        block = f"Input: {ex_input}\nOutput: {ex_output}"
        blocks.append(block)
        t.add(
            f"Add exemplar {i}/{k}",
            block,
            block,
            detail=(
                "Each exemplar narrows the space of mappings consistent with the prompt so far. "
                "Order matters: models are recency-sensitive, so a strong exemplar placed last "
                "carries more weight than the same exemplar placed first."
            ),
        )

    query_block = f"Input: {task}\nOutput:"
    t.add(
        "Append the query (output left blank)",
        query_block,
        query_block,
        detail="Rendered in the same template as the exemplars so the pattern continues.",
    )

    parts = ([instruction] if instruction.strip() else []) + blocks + [query_block]
    prompt = "\n\n".join(parts)
    t.add(
        "Assemble the final prompt",
        "prompt = instruction? + exemplar blocks + query block, joined by blank lines",
        prompt,
    )

    t.result = prompt
    return t


def few_shot(
    task: str,
    examples: list[tuple[str, str]],
    instruction: str = "",
    explain: bool = False,
    level: str | ExplainLevel = ExplainLevel.INTERMEDIATE,
) -> str:
    """Return the assembled few-shot prompt. Set ``explain=True`` to print the trace."""
    t = few_shot_trace(task, examples, instruction=instruction)
    return t.render(level) if explain else t.result


def demo(seed: int = 0) -> Trace:
    """Few-shot prompt for classifying review sentiment from 3 exemplars."""
    return few_shot_trace(
        task="The battery life on this laptop is disappointing.",
        examples=[
            ("The screen is bright and crisp.", "positive"),
            ("Shipping took three weeks and the box arrived damaged.", "negative"),
            ("It's a laptop. It turns on.", "neutral"),
        ],
        instruction="Classify the sentiment of each review as positive, negative, or neutral.",
    )
