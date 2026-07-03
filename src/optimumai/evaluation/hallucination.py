"""A grounding/faithfulness HEURISTIC — not a hallucination detector.

Hallucination detection is an unsolved research problem. "Did the model make
something up?" is really a question about the model's internal (unobservable)
belief state versus external truth, and no cheap lexical check can answer it
reliably. This module is deliberately named around "grounding" and
"faithfulness" rather than "hallucination detection" for that reason: it
implements a *proxy* — claim-token overlap between an answer and the source
context it was supposedly grounded in — that is transparent, fast, and
useful for teaching the *shape* of the problem, but should not be mistaken
for a solved one.

The heuristic
--------------
Given an ``answer`` and the ``context`` it was supposed to be grounded in:

1. Tokenize both, drop stopwords (function words carry no factual claim).
2. A content token in the answer is **supported** if it also appears
   somewhere in the context; otherwise it is **unsupported**.
3. ``faithfulness = |supported tokens| / |answer content tokens|`` — the
   fraction of the answer's content that can be traced back to the source.
4. The unsupported tokens are surfaced directly as the flagged spans, so you
   can see *what* looks unsupported, not just a single opaque score.

This score sits in ``[0, 1]``: 1.0 means every content word in the answer
also appears in the context (fully "grounded" by this proxy); lower scores
flag more novel vocabulary that the context does not obviously supply.

Why this is only a proxy, and can be badly wrong
---------------------------------------------------
* **False alarms (flags correct content).** Valid inference, arithmetic,
  world knowledge, paraphrase, or synonyms ("automobile" answering from a
  context that says "car") introduce tokens that are true and grounded in
  meaning but absent from the context's literal vocabulary. Genuinely
  correct reasoning gets penalized just as much as fabrication.
* **Misses (lets hallucination through).** A model can reuse every single
  context word while completely inverting the relationship between them
  ("the drug reduces side effects" vs. context "the drug causes side
  effects") — 100% token overlap, 0% faithfulness in reality. Word-level
  overlap cannot see negation, causality, or numerical claims being wrong.
* **No notion of entailment.** The heuristic checks *presence*, not whether
  the context actually *supports the claim being made* with those words.

What real hallucination/faithfulness evaluation looks like
---------------------------------------------------------------
* **NLI-based (Natural Language Inference)** methods run an entailment model
  over (context, claim) pairs and check whether the context entails,
  contradicts, or is neutral toward each claim in the answer.
* **QA-based / QAG (Question-Answer Generation)** methods generate questions
  from the answer's claims, ask them against the context, and check whether
  the answers match (e.g. QAGS, FActScore-style pipelines).
* **LLM-as-judge** methods prompt a strong model to assess faithfulness
  directly, often decomposing the answer into atomic claims first.
* **Retrieval-grounded verification** cross-checks each claim against an
  external knowledge base rather than only the provided context.

All of these require a model (or an external knowledge source) in the loop;
none is a solved problem, and all can themselves be wrong or gamed. The
lexical heuristic here is a zero-dependency starting point for the
*concept* of faithfulness, not a production hallucination detector.
"""

from __future__ import annotations

import re

from optimumai.core._fmt import num
from optimumai.core.trace import Trace

# Function words carry no factual claim on their own; dropping them focuses the
# overlap check on content that could actually be "made up."
_STOPWORDS = frozenset(
    """
    a an the is are was were be been being of in on at to for and or but with
    as by that this it its from has have had he she they them his her their
    which who what when where why how do does did not no so than then there
    here also into over under up down out about i you we my your our
    """.split()
)


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _content_tokens(text: str) -> list[str]:
    return [w for w in _tokenize(text) if w not in _STOPWORDS]


def _validate(answer: str, context: str) -> None:
    if not answer or not answer.strip():
        raise ValueError("answer must be a non-empty string")
    if not context or not context.strip():
        raise ValueError("context must be a non-empty string")


def faithfulness_trace(answer: str, context: str) -> Trace:
    """Build the full trace: which answer tokens are supported by the context, and the score.

    Args:
        answer: The model's generated answer to score.
        context: The source text the answer was supposed to be grounded in
            (e.g. retrieved documents, a system prompt's reference material).
    """
    _validate(answer, context)

    context_vocab = set(_content_tokens(context))
    answer_tokens = _content_tokens(answer)

    t = Trace(
        op="faithfulness",
        formula="faithfulness = |supported content tokens| / |answer content tokens|  "
        "(HEURISTIC — see module docstring)",
        complexity="O(|answer| + |context|)",
        why_ai=[
            "RAG systems are only as trustworthy as their grounding — this is the "
            "cheapest possible check that an answer isn't inventing content absent "
            "from its retrieved context",
            "Real hallucination detection uses NLI entailment models, QA-generation "
            "consistency checks, or LLM-as-judge — this token-overlap version is an "
            "educational proxy, not a production detector (see docstring for why)",
            "The flagged 'unsupported' spans are exactly the kind of thing a human "
            "reviewer or a stronger verifier model would want to double-check first",
        ],
        meta={"answer": answer, "context": context},
    )
    t.add(
        "Extract content vocabulary from context",
        f"{len(context_vocab)} unique content words (stopwords removed)",
        detail=f"context content words: {sorted(context_vocab)}",
    )
    t.add(
        "Extract content tokens from answer",
        f"{len(answer_tokens)} content words: {answer_tokens}",
        detail="Stopwords are dropped because function words don't carry a factual claim.",
    )

    if not answer_tokens:
        t.add(
            "No content tokens in answer",
            "faithfulness defined as 1.0 (nothing to be unfaithful about)",
            1.0,
        )
        t.result = 1.0
        t.meta["supported"] = []
        t.meta["unsupported"] = []
        return t

    supported = [w for w in answer_tokens if w in context_vocab]
    unsupported = [w for w in answer_tokens if w not in context_vocab]
    score = len(supported) / len(answer_tokens)

    t.add(
        "Supported tokens (present in context)",
        f"{len(supported)}/{len(answer_tokens)}: {supported}",
        supported,
        detail="These content words in the answer also appear in the context's vocabulary.",
    )
    t.add(
        "Unsupported tokens (flagged spans)",
        f"{len(unsupported)}/{len(answer_tokens)}: {unsupported}",
        unsupported,
        detail="Absent from the context's vocabulary by this heuristic — could be a "
        "hallucination, or could be valid inference/paraphrase/synonym (see docstring). "
        "Worth a closer look either way.",
    )
    t.add(
        "Faithfulness score",
        f"{len(supported)} / {len(answer_tokens)} = {num(score)}",
        score,
        detail="1.0 = every content word traces back to the context by this proxy. "
        "This is NOT the same as 'the answer is true.'",
    )
    t.result = score
    t.meta["supported"] = supported
    t.meta["unsupported"] = unsupported
    return t


def faithfulness_score(answer: str, context: str) -> float:
    """Return the claim-overlap faithfulness heuristic in ``[0, 1]`` (see module docstring)."""
    return faithfulness_trace(answer, context).result


def unsupported_spans(answer: str, context: str) -> list[str]:
    """Return the answer's content tokens absent from the context's vocabulary.

    These are the heuristic's flagged candidates for hallucinated content — treat
    them as "worth checking," not as confirmed fabrications (see module docstring).
    """
    return faithfulness_trace(answer, context).meta["unsupported"]


def demo(seed: int = 0) -> Trace:
    """Contrast a fully grounded answer with a half-hallucinated one on the same context."""
    context = "The Eiffel Tower was completed in 1889 and stands 330 meters tall in Paris."
    hallucinated_answer = (
        "The Eiffel Tower was completed in 1889 and was designed by Leonardo da Vinci."
    )
    return faithfulness_trace(hallucinated_answer, context)
