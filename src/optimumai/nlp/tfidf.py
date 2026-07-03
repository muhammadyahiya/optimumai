"""TF-IDF — weighting words by how much they actually distinguish a document.

**Intuition.** Raw word counts are a bad relevance signal: "the" appears
everywhere and says nothing about a document's *topic*, while a rare word like
"mitochondria" appearing even once is a strong signal. TF-IDF fixes this by
multiplying two opposing forces — how often a term shows up *in this
document* (term frequency, rewarding repetition) against how rare it is
*across all documents* (inverse document frequency, punishing ubiquity).

**Math.** For term ``t`` in document ``d`` drawn from a corpus of ``N``
documents:

    tf(t, d)  = count(t, d) / |d|                     (fraction of d's words that are t)
    df(t)     = number of documents containing t
    idf(t)    = log(N / df(t)) + 1                     (smoothed: add-1 so idf never hits 0)
    tfidf(t, d) = tf(t, d) * idf(t)

This module uses the common "smooth idf" variant, ``log(N/df) + 1`` (as in
scikit-learn's default), so that a term appearing in *every* document still
gets a small positive weight instead of being zeroed out entirely by a bare
``log(N/N) = 0``. A term that is frequent locally *and* rare globally gets the
highest score; a term that is common everywhere (like "the", where
``df(t) = N``) gets pulled toward the floor of ``idf = 1``.

**Why modern LLMs still rest on this.** TF-IDF predates neural embeddings by
decades but never left production: it is still the default scorer behind
BM25-based search (a refinement of the same tf/idf intuition), sparse
retrieval baselines in RAG pipelines, and keyword extraction. Conceptually it
is also the direct ancestor of *attention* — both are ways of asking "which
tokens matter most here relative to the rest," just computed one with counts
and logs, the other with learned dot products and softmax.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from optimumai.core._fmt import arr, num
from optimumai.core.trace import Trace


def _tokenize(doc: str) -> list[str]:
    """Lowercase and split on whitespace — a deliberately simple word splitter."""
    return doc.lower().split()


def term_frequency(doc: str) -> dict[str, float]:
    """Return ``{term: count(term) / len(doc_tokens)}`` for one document."""
    tokens = _tokenize(doc)
    if not tokens:
        raise ValueError("document must contain at least one token")
    counts = Counter(tokens)
    return {term: count / len(tokens) for term, count in counts.items()}


def document_frequency(docs: list[str]) -> dict[str, int]:
    """Return ``{term: number of docs containing term}`` across a corpus."""
    if not docs:
        raise ValueError("docs must be a non-empty list")
    df: Counter[str] = Counter()
    for doc in docs:
        df.update(set(_tokenize(doc)))
    return dict(df)


def inverse_document_frequency(docs: list[str]) -> dict[str, float]:
    """Return smoothed idf: ``log(N / df(t)) + 1`` for every term in the corpus."""
    df = document_frequency(docs)
    n = len(docs)
    return {term: float(np.log(n / count) + 1.0) for term, count in df.items()}


def tfidf_trace(docs: list[str]) -> Trace:
    """Build the full TF, IDF, and weighted-matrix trace for a document set."""
    if not docs:
        raise ValueError("docs must be a non-empty list of strings")
    if any(not isinstance(d, str) for d in docs):
        raise TypeError("every document must be a string")

    n = len(docs)
    tokenized = [_tokenize(d) for d in docs]
    if any(not toks for toks in tokenized):
        raise ValueError("every document must contain at least one token")

    vocab = sorted({tok for toks in tokenized for tok in toks})

    t = Trace(
        op="tfidf",
        formula="tfidf(t,d) = tf(t,d) * idf(t),  idf(t) = log(N/df(t)) + 1  (smooth idf)",
        complexity="O(sum of document lengths) to count; O(|vocab|·N) to build the matrix",
        why_ai=[
            "Sparse retrieval (BM25, classic search engines) still ranks documents this way",
            "RAG pipelines often blend a TF-IDF/BM25 score with dense embedding similarity",
            "Same goal as attention: decide which tokens matter most, just via counts+logs "
            "instead of learned dot products",
        ],
        meta={"n_docs": n, "vocab_size": len(vocab), "vocab": vocab},
    )

    t.add(
        "Tokenize (lowercase, whitespace split)",
        "  ".join(f"d{i}={toks}" for i, toks in enumerate(tokenized)),
        tokenized,
    )

    tf_rows = [term_frequency(d) for d in docs]
    tf_lines = "\n".join(
        f"d{i}: " + ", ".join(f"tf({term})={num(tf_rows[i][term])}" for term in sorted(tf_rows[i]))
        for i in range(n)
    )
    t.add(
        "Term frequency per document: tf(t,d) = count(t,d) / |d|",
        tf_lines,
        tf_rows,
        detail="Longer documents don't automatically dominate — counts are normalized by "
        "document length.",
    )

    df = document_frequency(docs)
    t.add(
        "Document frequency: df(t) = #docs containing t",
        ", ".join(f"df({term})={df[term]}" for term in vocab),
        df,
    )

    idf = inverse_document_frequency(docs)
    t.add(
        f"Smoothed IDF: idf(t) = log(N/df(t)) + 1, N={n}",
        ", ".join(f"idf({term})={num(idf[term])}" for term in vocab),
        idf,
        detail="+1 smoothing means a term in every document still gets idf=1 instead of "
        "being zeroed out by log(N/N)=0.",
    )

    matrix = np.zeros((n, len(vocab)))
    for i, tf_row in enumerate(tf_rows):
        for j, term in enumerate(vocab):
            matrix[i, j] = tf_row.get(term, 0.0) * idf[term]
    t.add(
        f"Weighted matrix ({n} docs x {len(vocab)} terms): tfidf = tf * idf",
        f"columns = {vocab}\n{arr(matrix)}",
        matrix,
        detail="Each row is a document's vector; a term scores high only if it's frequent "
        "here AND rare across the corpus.",
    )

    t.result = matrix
    t.meta["idf"] = idf
    t.meta["tf"] = tf_rows
    return t


class TfidfVectorizer:
    """Fit a vocabulary + idf table on a corpus, then transform documents to vectors.

    Mirrors the fit/transform shape of familiar vectorizer APIs, but stays
    numpy + stdlib only, matching the "tf * smoothed-idf" formula documented
    in :func:`tfidf_trace`.
    """

    def __init__(self) -> None:
        self.vocab_: list[str] = []
        self.idf_: dict[str, float] = {}

    def fit(self, docs: list[str]) -> TfidfVectorizer:
        """Learn the vocabulary and idf weights from ``docs``."""
        self.idf_ = inverse_document_frequency(docs)
        self.vocab_ = sorted(self.idf_)
        return self

    def transform(self, docs: list[str]) -> np.ndarray:
        """Map each document in ``docs`` to a tf-idf row using the fitted idf."""
        if not self.vocab_:
            raise RuntimeError("call fit(...) before transform(...)")
        matrix = np.zeros((len(docs), len(self.vocab_)))
        for i, doc in enumerate(docs):
            tf_row = term_frequency(doc)
            for j, term in enumerate(self.vocab_):
                matrix[i, j] = tf_row.get(term, 0.0) * self.idf_.get(term, 0.0)
        return matrix

    def fit_transform(self, docs: list[str]) -> np.ndarray:
        """Fit on ``docs`` then transform them in one call."""
        return self.fit(docs).transform(docs)


def tfidf(docs: list[str]) -> np.ndarray:
    """Return the ``(n_docs, n_vocab)`` tf-idf matrix for ``docs``."""
    return tfidf_trace(docs).result


def demo(seed: int = 0) -> Trace:
    """TF-IDF over a tiny three-document corpus where 'the' is common but uninformative."""
    docs = [
        "the cat sat on the mat",
        "the dog sat on the log",
        "cats and dogs are friends",
    ]
    return tfidf_trace(docs)
