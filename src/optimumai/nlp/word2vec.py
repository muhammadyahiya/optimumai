"""Skip-gram word2vec — the seed idea behind every learned embedding table.

**Intuition.** "You shall know a word by the company it keeps" (J.R. Firth).
Skip-gram operationalizes this distributional hypothesis directly: train a
tiny network to predict a word's *neighbors* from the word itself. The
network never succeeds perfectly — "the" appears next to almost everything —
but the *by-product* of trying is what we actually want: to predict context
well, words that appear in similar contexts must end up with similar vectors.
The prediction task is thrown away after training; the embedding table is the
point.

**Math.** From a corpus, slide a window of radius ``w`` over each sentence and
emit every ``(center, context)`` pair within it — e.g. window=1 over
"the cat sat" gives (cat, the) and (cat, sat) as training pairs. Two
embedding matrices are learned: ``W_in`` (vocab_size x d), one row per word as
a *center*, and ``W_out`` (vocab_size x d), one row per word as a *context*.
For a pair (center ``c``, context ``o``), the model scores every vocabulary
word as a candidate context via a dot product, then softmaxes:

    v_c = W_in[c]                              (center embedding, shape d)
    scores = v_c @ W_out.T                     (shape V — one score per vocab word)
    P(o | c) = softmax(scores)[o] = exp(v_c . u_o) / sum_j exp(v_c . u_j)

Training maximizes ``P(o | c)`` for observed pairs via gradient descent on the
cross-entropy loss ``L = -log P(o | c)``. The gradient has a clean form
(softmax's classic "predicted minus actual" derivative):

    dL/d(scores) = P(*|c) - one_hot(o)          (shape V)
    dL/dW_out    = outer(dL/d(scores), v_c)     (shape V x d)
    dL/dv_c      = W_out^T @ dL/d(scores)       (shape d)

This module runs a handful of *full-batch softmax* SGD steps (no negative
sampling) on a deliberately tiny embedding matrix, so every number is
inspectable — real word2vec uses **negative sampling** instead of full
softmax purely as a speed hack (approximating "push away from everything"
with "push away from a few random negative words"), not a different idea.

**Why modern LLMs still rest on this.** This is the origin story of every
embedding table used today, including the one at the very first layer of a
transformer (:mod:`optimumai.embeddings.lookup`). Word2vec (2013) was the
first widely-used proof that "learn to predict context, keep the weights, use
them as features" produces vectors with real geometric structure (the famous
king - man + woman ≈ queen). Modern LLM embeddings are learned end-to-end
inside a much bigger model instead of via a standalone side task, but the
underlying bet — that predicting neighbors forces semantically similar words
into nearby vectors — is unchanged.
"""

from __future__ import annotations

import numpy as np

from optimumai.core._fmt import arr, num
from optimumai.core.trace import Trace


def _tokenize(corpus: list[str]) -> list[list[str]]:
    return [sentence.lower().split() for sentence in corpus]


def build_vocab(corpus: list[str]) -> list[str]:
    """Return the sorted, deduplicated vocabulary of a tokenized corpus."""
    tokens = {tok for sentence in _tokenize(corpus) for tok in sentence}
    return sorted(tokens)


def skipgram_pairs(corpus: list[str], window: int = 1) -> list[tuple[str, str]]:
    """Build every ``(center, context)`` pair within ``window`` of each center word."""
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    pairs: list[tuple[str, str]] = []
    for sentence in _tokenize(corpus):
        for i, center in enumerate(sentence):
            lo, hi = max(0, i - window), min(len(sentence), i + window + 1)
            for j in range(lo, hi):
                if j != i:
                    pairs.append((center, sentence[j]))
    return pairs


def _softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - np.max(scores)
    exps = np.exp(shifted)
    return exps / np.sum(exps)


class SkipGramModel:
    """A tiny skip-gram word2vec model trained with full-softmax SGD.

    Args:
        vocab: The (deduplicated) vocabulary; row ``i`` of both embedding
            matrices belongs to ``vocab[i]``.
        dim: Embedding dimensionality.
        seed: Seed for the initial embedding matrices.
    """

    def __init__(self, vocab: list[str], dim: int = 4, seed: int = 0) -> None:
        if not vocab:
            raise ValueError("vocab must be non-empty")
        if dim < 1:
            raise ValueError(f"dim must be >= 1, got {dim}")
        self.vocab = vocab
        self.dim = dim
        self.index_of = {word: i for i, word in enumerate(vocab)}
        rng = np.random.default_rng(seed)
        scale = 1.0 / np.sqrt(dim)
        self.w_in = rng.normal(scale=scale, size=(len(vocab), dim))
        self.w_out = rng.normal(scale=scale, size=(len(vocab), dim))

    def predict(self, center: str) -> np.ndarray:
        """Return ``P(context | center)`` over the whole vocabulary."""
        if center not in self.index_of:
            raise KeyError(f"{center!r} is not in the vocabulary")
        v_c = self.w_in[self.index_of[center]]
        scores = self.w_out @ v_c
        return _softmax(scores)

    def step(self, center: str, context: str, lr: float = 0.1) -> float:
        """One full-softmax SGD step on a single ``(center, context)`` pair.

        Returns the cross-entropy loss *before* the update.
        """
        if center not in self.index_of or context not in self.index_of:
            raise KeyError("center and context must both be in the vocabulary")
        c_idx, o_idx = self.index_of[center], self.index_of[context]
        v_c = self.w_in[c_idx]
        probs = self.predict(center)
        loss = float(-np.log(probs[o_idx] + 1e-12))

        d_scores = probs.copy()
        d_scores[o_idx] -= 1.0  # softmax + cross-entropy gradient: predicted - actual
        grad_w_out = np.outer(d_scores, v_c)
        grad_v_c = self.w_out.T @ d_scores

        self.w_out -= lr * grad_w_out
        self.w_in[c_idx] -= lr * grad_v_c
        return loss

    def most_similar(self, word: str, k: int = 3) -> list[tuple[str, float]]:
        """Rank the vocabulary by cosine similarity to ``word`` in ``w_in`` space."""
        if word not in self.index_of:
            raise KeyError(f"{word!r} is not in the vocabulary")
        v = self.w_in[self.index_of[word]]
        v_norm = np.linalg.norm(v)
        scored = []
        for other in self.vocab:
            if other == word:
                continue
            u = self.w_in[self.index_of[other]]
            denom = v_norm * np.linalg.norm(u)
            score = float(np.dot(v, u) / denom) if denom else 0.0
            scored.append((other, score))
        return sorted(scored, key=lambda pair: pair[1], reverse=True)[:k]


def word2vec_trace(
    corpus: list[str],
    window: int = 1,
    dim: int = 4,
    steps: int = 20,
    lr: float = 0.1,
    seed: int = 0,
) -> Trace:
    """Build (center, context) pairs, train a few SGD steps, and trace the whole thing."""
    if not corpus:
        raise ValueError("corpus must be a non-empty list of sentences")
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")

    vocab = build_vocab(corpus)
    pairs = skipgram_pairs(corpus, window=window)
    if not pairs:
        raise ValueError("corpus produced no (center, context) pairs — need >= 2 word sentences")

    t = Trace(
        op="word2vec",
        formula=(
            "P(context|center) = softmax(W_out @ W_in[center]);  "
            "grad(scores) = P(*|center) - one_hot(context)"
        ),
        complexity=f"O(V*d) per SGD step for vocab size V, dim d (V={len(vocab)}, d={dim})",
        why_ai=[
            "This is the origin story of every learned embedding table, including the "
            "first layer of a transformer",
            "Proved that 'predict context, keep the weights' forces semantically similar "
            "words into nearby vectors (king - man + woman ~ queen)",
            "Negative sampling (used in production word2vec) is a speed optimization of "
            "this exact full-softmax gradient, not a different algorithm",
        ],
        meta={"window": window, "dim": dim, "steps": steps, "lr": lr, "vocab_size": len(vocab)},
    )

    t.add(
        f"Vocabulary ({len(vocab)} words)",
        str(vocab),
        vocab,
    )
    t.add(
        f"Build (center, context) pairs, window={window} ({len(pairs)} pairs)",
        ", ".join(f"({c},{o})" for c, o in pairs[:10]) + (" ..." if len(pairs) > 10 else ""),
        pairs,
        detail="Every word within `window` positions of a center word becomes one training "
        "pair — this is the entire 'labels' for the task.",
    )

    model = SkipGramModel(vocab, dim=dim, seed=seed)
    t.add(
        f"Initialize W_in, W_out (each {len(vocab)}x{dim}, seed={seed})",
        f"W_in[0]={arr(model.w_in[0])}",
        {"w_in": model.w_in.copy(), "w_out": model.w_out.copy()},
        detail="Two separate matrices: a word's 'center' vector and its 'context' vector "
        "start independent and only W_in is kept as the final embedding.",
    )

    rng = np.random.default_rng(seed)
    losses: list[float] = []
    first_center, first_context = pairs[0]
    for i in range(steps):
        center, context = pairs[rng.integers(len(pairs))]
        loss = model.step(center, context, lr=lr)
        losses.append(loss)
        if i == 0:
            probs_before = model.predict(first_center)
            p_context = num(probs_before[model.index_of[first_context]])
            t.add(
                f"Step 1: forward pass for ({first_center}, {first_context})",
                f"P({first_context}|{first_center}) = {p_context}  (loss = -log P = {num(loss)})",
                probs_before,
                detail="scores = W_out @ v_center, then softmax turns them into a "
                "distribution over which word is 'the' context.",
            )

    t.add(
        f"Run {steps} SGD steps (lr={lr}), sampling a random pair each time",
        f"loss: {num(losses[0])} -> {num(losses[-1])} "
        f"(mean first 3: {num(float(np.mean(losses[:3])))}, "
        f"mean last 3: {num(float(np.mean(losses[-3:])))})",
        losses,
        detail="grad(scores) = predicted_probs - one_hot(actual_context); this 'push down "
        "what you predicted, push up what was true' signal updates both matrices.",
    )

    neighbor_word = first_center
    neighbors = model.most_similar(neighbor_word, k=min(3, len(vocab) - 1))
    t.add(
        f"Nearest neighbors of {neighbor_word!r} after training (cosine in W_in space)",
        ", ".join(f"{w}={num(s)}" for w, s in neighbors),
        neighbors,
        detail="Even a handful of steps nudges words that share contexts toward each "
        "other — this is the whole mechanism, just scaled up in real training.",
    )

    t.result = model
    t.meta["losses"] = losses
    t.meta["pairs"] = pairs
    return t


def demo(seed: int = 0) -> Trace:
    """Train skip-gram on a tiny corpus where 'cat'/'dog' share contexts with 'the ... sat'."""
    corpus = [
        "the cat sat on the mat",
        "the dog sat on the mat",
        "the cat chased the mouse",
        "the dog chased the mouse",
    ]
    return word2vec_trace(corpus, window=1, dim=4, steps=30, lr=0.2, seed=seed)
