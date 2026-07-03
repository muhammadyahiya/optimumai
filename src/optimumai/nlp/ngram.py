"""N-gram language models — counting your way to "what word comes next."

**Intuition.** The simplest possible language model: to predict the next
word, just look at the last ``n-1`` words and ask "in my training corpus, what
usually followed this?" A *bigram* model (n=2) conditions on 1 previous word;
a *trigram* model (n=3) conditions on 2. No neural network, no gradient
descent — just counting how often each ``n``-word window occurred and turning
those counts into conditional probabilities.

**Math.** The chain rule factors a sentence's probability into a product of
next-word-given-history terms:

    P(w_1, ..., w_T) = Π_i P(w_i | w_1, ..., w_{i-1})

An n-gram model makes this tractable with a *Markov assumption*: truncate the
history to the last ``n-1`` words,

    P(w_i | w_1, ..., w_{i-1}) ≈ P(w_i | w_{i-n+1}, ..., w_{i-1})
                                = count(w_{i-n+1..i}) / count(w_{i-n+1..i-1})

The problem: any n-gram never seen in training gets probability exactly 0,
which would make the *whole sentence* probability 0 the moment one unseen
n-gram appears. **Add-k (Laplace) smoothing** fixes this by pretending every
possible n-gram occurred ``k`` extra times, borrowing a little probability
mass from seen events and giving it to unseen ones:

    P(w_i | history) = (count(w_{i-n+1..i}) + k) / (count(w_{i-n+1..i-1}) + k*V)

where ``V`` is the vocabulary size (``k=1`` is classic Laplace smoothing;
``0 < k < 1`` is gentler). **Perplexity** then measures how "surprised" the
model is by held-out text — the exponentiated average negative log-likelihood
per token:

    PPL = exp( -(1/N) * Σ_i log P(w_i | history) )

Lower perplexity means the model assigned higher probability to what actually
happened; a perfect model has PPL=1, and a uniform-random guess over a
vocabulary of size V has PPL=V.

**Why modern LLMs still rest on this.** GPT-style models are still, at their
core, "predict the next token given history" — they just replace n-gram
counting with a neural network's learned conditional distribution, and
replace a hard n-1-word cutoff with a soft-attention window that can reach
arbitrarily far back. Perplexity remains *the* standard metric for comparing
language models, from n-grams to GPT-4, precisely because the underlying
math — exponentiated average negative log-likelihood — never changed.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from optimumai.core._fmt import num
from optimumai.core.trace import Trace

_BOS = "<s>"
_EOS = "</s>"


def _tokenize(sentence: str) -> list[str]:
    return sentence.lower().split()


def _pad(tokens: list[str], n: int) -> list[str]:
    """Pad a token sequence with ``n-1`` start markers and one end marker."""
    return [_BOS] * (n - 1) + tokens + [_EOS]


def _ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


class NGramModel:
    """A counting-based n-gram language model with add-k smoothing.

    Args:
        n: Order of the model (2 = bigram, 3 = trigram, ...). Must be >= 1.
        k: Add-k smoothing constant (``k=1`` is classic Laplace smoothing).
    """

    def __init__(self, n: int = 2, k: float = 1.0) -> None:
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        if k < 0:
            raise ValueError(f"k must be >= 0, got {k}")
        self.n = n
        self.k = k
        self.ngram_counts: Counter[tuple[str, ...]] = Counter()
        self.context_counts: Counter[tuple[str, ...]] = Counter()
        self.vocab: set[str] = set()
        self._fitted = False

    def fit(self, corpus: list[str]) -> NGramModel:
        """Count n-grams (and their contexts) over a list of sentences."""
        if not corpus:
            raise ValueError("corpus must be a non-empty list of sentences")
        for sentence in corpus:
            tokens = _pad(_tokenize(sentence), self.n)
            self.vocab.update(tokens)
            for gram in _ngrams(tokens, self.n):
                self.ngram_counts[gram] += 1
                self.context_counts[gram[:-1]] += 1
        self._fitted = True
        return self

    @property
    def vocab_size(self) -> int:
        """Number of distinct tokens seen during :meth:`fit` (including BOS/EOS)."""
        return len(self.vocab)

    def prob(self, context: tuple[str, ...], word: str) -> float:
        """Add-k smoothed ``P(word | context)`` for a ``(n-1)``-length context."""
        if not self._fitted:
            raise RuntimeError("call fit(...) before prob(...)")
        if len(context) != self.n - 1:
            raise ValueError(f"context must have length {self.n - 1}, got {len(context)}")
        gram = (*context, word)
        num_count = self.ngram_counts.get(gram, 0) + self.k
        denom_count = self.context_counts.get(context, 0) + self.k * self.vocab_size
        return num_count / denom_count

    def sentence_log_prob(self, sentence: str) -> float:
        """Sum of log P(w_i | context) over every n-gram in a padded sentence."""
        tokens = _pad(_tokenize(sentence), self.n)
        return float(
            sum(np.log(self.prob(gram[:-1], gram[-1])) for gram in _ngrams(tokens, self.n))
        )

    def perplexity(self, corpus: list[str]) -> float:
        """PPL = exp(-(1/N) * sum of log-probs) over every n-gram in ``corpus``."""
        if not corpus:
            raise ValueError("corpus must be a non-empty list of sentences")
        total_log_prob = 0.0
        total_grams = 0
        for sentence in corpus:
            tokens = _pad(_tokenize(sentence), self.n)
            grams = _ngrams(tokens, self.n)
            total_log_prob += sum(np.log(self.prob(g[:-1], g[-1])) for g in grams)
            total_grams += len(grams)
        return float(np.exp(-total_log_prob / total_grams))


def ngram_trace(train_corpus: list[str], test_sentence: str, n: int = 2, k: float = 1.0) -> Trace:
    """Fit an n-gram model on ``train_corpus`` and trace its PPL on ``test_sentence``."""
    if not train_corpus:
        raise ValueError("train_corpus must be a non-empty list of sentences")
    if not isinstance(test_sentence, str) or not test_sentence.strip():
        raise ValueError("test_sentence must be a non-empty string")

    model = NGramModel(n=n, k=k).fit(train_corpus)

    t = Trace(
        op="ngram",
        formula=(
            "P(w|ctx) = (count(ctx,w)+k) / (count(ctx)+k*V);  "
            "PPL = exp(-(1/N) * sum(log P(w_i|ctx_i)))"
        ),
        complexity=f"O(corpus length) to count; O({n}-gram lookups) per query",
        why_ai=[
            "Chain-rule factorization P(sentence) = product of P(next word | history) is "
            "exactly what GPT-style decoders compute, just with a learned distribution",
            "Add-k smoothing is the ancestor of every 'never let the model assign zero "
            "probability' trick, from Laplace smoothing to label smoothing in cross-entropy",
            "Perplexity is still the standard cross-model comparison metric, from n-grams "
            "to GPT-4",
        ],
        meta={"n": n, "k": k, "vocab_size": model.vocab_size},
    )

    t.add(
        f"Tokenize + pad {len(train_corpus)} training sentence(s) with {n - 1}x <s> and </s>",
        "  ".join(str(_pad(_tokenize(s), n)) for s in train_corpus),
        [_pad(_tokenize(s), n) for s in train_corpus],
    )

    top_ngrams = model.ngram_counts.most_common(5)
    t.add(
        f"Count {n}-grams (top {len(top_ngrams)} by frequency)",
        ", ".join(f"{gram}={count}" for gram, count in top_ngrams),
        dict(model.ngram_counts),
        detail=f"Vocabulary size V={model.vocab_size} (including <s>/</s>) feeds the "
        "smoothing denominator below.",
    )

    test_tokens = _pad(_tokenize(test_sentence), n)
    test_grams = _ngrams(test_tokens, n)
    prob_lines = []
    log_probs = []
    for gram in test_grams:
        context, word = gram[:-1], gram[-1]
        p = model.prob(context, word)
        log_probs.append(float(np.log(p)))
        prob_lines.append(f"P({word}|{context})={num(p)}")
    t.add(
        f"Add-k (k={k}) smoothed conditional probs for the test sentence",
        "\n".join(prob_lines),
        [model.prob(g[:-1], g[-1]) for g in test_grams],
        detail="P(w|ctx) = (count(ctx,w)+k) / (count(ctx)+k*V); k extra 'pretend' counts "
        "keep unseen n-grams above 0.",
    )

    total_log_prob = float(sum(log_probs))
    n_grams = len(test_grams)
    ppl = float(np.exp(-total_log_prob / n_grams))
    t.add(
        f"Perplexity over {n_grams} test {n}-grams",
        f"PPL = exp(-(1/{n_grams}) * {num(total_log_prob)}) = {num(ppl)}",
        ppl,
        detail="Lower is better: PPL=1 is a perfect model, PPL=V is a uniform random guess "
        "over the vocabulary.",
    )

    t.result = ppl
    t.meta["model"] = model
    t.meta["log_probs"] = log_probs
    return t


def perplexity(model: NGramModel, corpus: list[str]) -> float:
    """Convenience wrapper: ``model.perplexity(corpus)``.

    See :func:`optimumai.evaluation.perplexity.perplexity` for the general
    formula applied to a bare list of per-token probabilities; this variant is
    specific to a fitted :class:`NGramModel` scoring a corpus of sentences.
    """
    return model.perplexity(corpus)


def demo(seed: int = 0) -> Trace:
    """Train a bigram model on a tiny corpus and score a held-out sentence."""
    train = ["the cat sat on the mat", "the dog sat on the log", "the cat chased the dog"]
    return ngram_trace(train, "the cat sat on the log", n=2, k=1.0)
