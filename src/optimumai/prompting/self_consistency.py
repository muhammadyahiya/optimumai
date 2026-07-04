"""Self-consistency — sample many reasoning paths, vote on the answer.

**What it is.** A single chain-of-thought completion is one sample from the
model's reasoning distribution — it can go down an unlucky path and land on
a wrong answer even though the model "knows" the right one on other
samples. Self-consistency draws *N* independent CoT completions (via
temperature > 0 sampling) for the *same* prompt, extracts the final answer
from each, and returns the majority vote instead of trusting any single
chain.

**How you build it.** This module builds the underlying CoT *prompt* (via
:mod:`chain_of_thought`) exactly once — the same prompt is reused for every
sample, since it is the sampling temperature, not the prompt text, that
produces different chains. It then demonstrates the vote mechanics on fixed,
deterministic toy reasoning paths/answers (no live model calls, per this
SDK's offline design) to show:

1. The shared CoT prompt all *N* samples are drawn from.
2. Each of the *N* sampled (toy) reasoning chains and its extracted answer,
   added to the trace one at a time.
3. The vote tally over extracted answers.
4. The majority-vote winner as the final result.

**Why it helps (variance-reduction intuition).** Think of each CoT sample as
a noisy estimator of the "true" answer the model's knowledge supports.
Independent errors in different chains tend not to agree with each other,
while chains that reach the *correct* answer tend to agree with each other
even via different reasoning routes — so the mode of many samples is a
lower-variance estimator than any single sample, the same logic behind
bagging/ensembling in classical ML.

**Failure modes.**

* Cost scales linearly with *N* — this trades compute directly for
  reliability, with diminishing returns past roughly 5-20 samples.
* If a *systematic* bias makes the wrong answer more likely than the right
  one on every sample (not an independent error), majority voting reinforces
  the wrong answer instead of correcting it.
* Answer extraction from free-form chains is itself lossy — chains that
  reach the right answer via a slightly different final phrasing can be
  mis-tallied as a different "answer" unless extraction is normalized.
"""

from __future__ import annotations

from collections import Counter

from optimumai.core._fmt import arr
from optimumai.core.explain import ExplainLevel
from optimumai.core.trace import Trace
from optimumai.prompting.chain_of_thought import chain_of_thought_trace


def self_consistency_trace(
    task: str,
    sampled_answers: list[str] | None = None,
) -> Trace:
    """Build the shared CoT prompt, then tally N deterministic toy sampled answers."""
    if not task.strip():
        raise ValueError("task must be a non-empty string")
    if sampled_answers is None:
        # Deterministic toy chains: 3 agree, 1 diverges — mimics a typical run where
        # independent sampling noise occasionally derails a single chain.
        sampled_answers = ["30", "30", "42", "30"]
    if not sampled_answers:
        raise ValueError("self_consistency requires at least one sampled answer, got 0")

    n = len(sampled_answers)
    t = Trace(
        op="self_consistency",
        formula="answer = mode({extract(CoT_sample_i(prompt))}, i = 1..N)",
        complexity=f"O(N) independent model calls, N = {n}",
        why_ai=[
            "Variance reduction: independent reasoning errors disagree with each other, "
            "correct chains tend to agree even via different routes",
            "Same intuition as bagging/ensembling — the mode of many noisy samples beats "
            "any single sample",
            "Cost scales linearly with N; diminishing returns past ~5-20 samples",
            "Cannot fix a systematic bias shared by every sample, only independent noise",
        ],
        meta={"n": n},
    )

    cot = chain_of_thought_trace(task)
    shared_prompt = cot.result
    t.add(
        "Build the shared CoT prompt (sampled N times)",
        "prompt = chain_of_thought_trace(task).result",
        shared_prompt,
        detail="All N samples are drawn from this *same* prompt text — it is sampling "
        "temperature, not prompt variation, that produces different chains.",
    )

    for i, answer in enumerate(sampled_answers, start=1):
        t.add(
            f"Sample reasoning path {i}/{n}",
            f"chain {i} → extracted answer = {arr(answer)}",
            answer,
            detail="In a live system each chain would be a fresh temperature>0 completion "
            "of the shared prompt; here the chains are fixed toy answers for determinism.",
        )

    tally = Counter(sampled_answers)
    tally_str = ", ".join(f"{ans!r}: {count}" for ans, count in tally.most_common())
    t.add(
        "Tally the extracted answers",
        f"tally = {{{tally_str}}}",
        dict(tally),
    )

    winner, winner_count = tally.most_common(1)[0]
    t.add(
        "Majority vote",
        f"mode = {winner!r} ({winner_count}/{n} votes)",
        winner,
        detail=f"'{winner}' outvotes every other sampled answer "
        f"{winner_count} to {n - winner_count}.",
    )

    t.result = winner
    return t


def self_consistency(
    task: str,
    sampled_answers: list[str] | None = None,
    explain: bool = False,
    level: str | ExplainLevel = ExplainLevel.INTERMEDIATE,
) -> str:
    """Return the majority-vote answer. Set ``explain=True`` to print the trace."""
    t = self_consistency_trace(task, sampled_answers=sampled_answers)
    return t.render(level) if explain else t.result


def demo(seed: int = 0) -> Trace:
    """Self-consistency over 4 toy reasoning chains, 3 of which agree."""
    return self_consistency_trace(
        task="A store had 23 apples, sold 8, then received a shipment of 15 more. "
        "How many apples does it have now?",
        sampled_answers=["30", "30", "42", "30"],
    )
