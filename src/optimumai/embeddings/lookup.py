"""Embeddings — turning discrete tokens into dense, learnable vectors.

A tokenizer hands the model integer ids; those integers carry no notion of
"cat is closer to dog than to plane". The embedding table fixes that: it is a
``vocab_size × dim`` matrix whose rows are learned during training, and the very
first thing every LLM does is *look up* one row per token. This module builds a
tiny deterministic table so you can watch the lookup happen and see how cosine
similarity in that space powers semantic search.
"""

from __future__ import annotations

import numpy as np

from optimumai.core._fmt import arr, num
from optimumai.core.explain import ExplainLevel
from optimumai.core.trace import Trace


def _embedding_table(vocab: list[str], dim: int, seed: int) -> tuple[list[str], np.ndarray]:
    """Build a deterministic ``len(unique) × dim`` table for a vocabulary.

    The vocabulary is de-duplicated (order preserved) and each row is drawn from
    a seeded normal so the same inputs always yield the same table.
    """
    ordered: list[str] = list(dict.fromkeys(vocab))
    rng = np.random.default_rng(seed)
    table = rng.normal(size=(len(ordered), dim)).round(3)
    return ordered, table


def embedding_lookup_trace(tokens: list[str], dim: int = 4, seed: int = 0) -> Trace:
    """Build the full trace of looking up embedding rows for ``tokens``."""
    if not tokens:
        raise ValueError("embedding_lookup_trace needs at least one token")
    if dim < 1:
        raise ValueError(f"dim must be >= 1, got {dim}")

    vocab, table = _embedding_table(tokens, dim, seed)
    index_of = {word: i for i, word in enumerate(vocab)}

    t = Trace(
        op="embedding_lookup",
        formula="E[token] = W_emb[id(token)]   (W_emb ∈ ℝ^{V×d})",
        complexity="O(1) per token (a row index into the table)",
        why_ai=[
            "Tokens are discrete ids; a network cannot do math on symbols directly",
            "Embeddings map each id to a learnable dense vector that encodes meaning",
            "This lookup is the first layer of every LLM (GPT, BERT, ...)",
        ],
        meta={"vocab_size": len(vocab), "dim": dim, "seed": seed},
    )
    t.add(
        "Build the embedding table",
        f"W_emb has shape ({len(vocab)}, {dim}) — one row per unique token\n{arr(table)}",
        table,
        detail="Rows are seeded random here; in a real model they are learned by gradient descent.",
    )

    rows = []
    for token in tokens:
        idx = index_of[token]
        row = table[idx]
        rows.append(row)
        t.add(
            f"Look up {token!r}",
            f"id({token!r}) = {idx}  →  W_emb[{idx}] = {arr(row)}",
            row,
            detail="The string is never used again — only its row of numbers flows onward.",
        )

    result = np.stack(rows)
    t.add(
        "Stack the rows",
        f"embeddings shape ({len(tokens)}, {dim})\n{arr(result)}",
        result,
        detail="This matrix (sequence × dim) is what the next layer actually consumes.",
    )
    t.result = result
    return t


def nearest_neighbors_trace(
    query: str,
    vocab: list[str],
    dim: int = 4,
    seed: int = 0,
    k: int = 3,
) -> Trace:
    """Rank ``vocab`` by cosine similarity to ``query`` in embedding space."""
    if not vocab:
        raise ValueError("nearest_neighbors_trace needs a non-empty vocab")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")

    # Embed the query alongside the vocab so every word shares one table.
    words, table = _embedding_table([query, *vocab], dim, seed)
    index_of = {word: i for i, word in enumerate(words)}
    q_vec = table[index_of[query]]

    t = Trace(
        op="nearest_neighbors",
        formula="score(w) = cos(E[query], E[w]) = (q·w) / (‖q‖‖w‖)",
        complexity="O(V·d) — one cosine per vocabulary entry",
        why_ai=[
            "Semantic search and RAG rank documents by cosine to the query vector",
            "Analogy arithmetic lives here too: king − man + woman ≈ queen",
            "A vector database is just this lookup made fast over millions of rows",
        ],
        meta={"query": query, "vocab_size": len(vocab), "dim": dim, "seed": seed, "k": k},
    )
    t.add(
        "Embed the query",
        f"E[{query!r}] = {arr(q_vec)}",
        q_vec,
        detail="Same table as the vocabulary, so the vectors are directly comparable.",
    )

    q_norm = float(np.linalg.norm(q_vec))
    scores: list[tuple[str, float]] = []
    for word in dict.fromkeys(vocab):
        w_vec = table[index_of[word]]
        w_norm = float(np.linalg.norm(w_vec))
        denom = q_norm * w_norm
        score = float(np.dot(q_vec, w_vec) / denom) if denom else 0.0
        scores.append((word, score))
        t.add(
            f"Cosine vs {word!r}",
            f"(q · {word!r}) / (‖q‖·‖{word!r}‖) = {num(score)}",
            score,
        )

    ranked = sorted(scores, key=lambda pair: pair[1], reverse=True)
    top = ranked[:k]
    t.add(
        f"Take the top {k}",
        "  ".join(f"{word}={num(score)}" for word, score in top),
        top,
        detail="Higher cosine = closer direction = more semantically similar.",
    )
    t.result = top
    return t


def embedding_lookup(
    tokens: list[str],
    dim: int = 4,
    seed: int = 0,
    explain: bool = False,
    level: str | ExplainLevel = ExplainLevel.INTERMEDIATE,
) -> np.ndarray:
    """Look up embedding rows for ``tokens``. Set ``explain=True`` to print it."""
    t = embedding_lookup_trace(tokens, dim=dim, seed=seed)
    return t.render(level) if explain else t.result


def nearest_neighbors(
    query: str,
    vocab: list[str],
    dim: int = 4,
    seed: int = 0,
    k: int = 3,
    explain: bool = False,
    level: str | ExplainLevel = ExplainLevel.INTERMEDIATE,
) -> list[tuple[str, float]]:
    """Return the top-``k`` cosine neighbours of ``query`` within ``vocab``."""
    t = nearest_neighbors_trace(query, vocab, dim=dim, seed=seed, k=k)
    return t.render(level) if explain else t.result


def demo() -> Trace:
    """A tiny, reproducible embedding-lookup example for docs and the CLI."""
    return embedding_lookup_trace(["cat", "sat", "mat"])
