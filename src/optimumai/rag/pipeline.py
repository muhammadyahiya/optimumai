"""Retrieval-Augmented Generation — grounding an LLM in your own documents.

A language model only knows what was in its training data; RAG lets it answer
from a private corpus by (1) embedding the query, (2) embedding each document,
(3) ranking documents by cosine similarity, (4) keeping the top-k, and (5)
stuffing those chunks into the prompt as context. The retrieval step is nothing
but cosine search in embedding space — no external model needed to see how it
works. This pipeline uses a deterministic bag-of-hashed-words embedding so the
whole flow is reproducible and fully traceable.
"""

from __future__ import annotations

import numpy as np

from optimumai.core._fmt import arr, num
from optimumai.core.base_op import BaseOp
from optimumai.core.trace import Trace

_DEFAULT_CORPUS = [
    "Neural networks learn by adjusting weights with gradient descent.",
    "Backpropagation computes gradients using the chain rule of calculus.",
    "Attention lets a transformer weigh which tokens matter most.",
    "Embeddings map words to dense vectors that capture meaning.",
    "Overfitting happens when a model memorizes the training data.",
]


class RAGPipeline(BaseOp):
    """A minimal, deterministic retrieval-augmented-generation pipeline.

    Args:
        corpus: The documents to retrieve from. Defaults to five ML sentences.
        dim: Embedding dimension for the bag-of-hashed-words vectors.
        seed: Base seed; each word gets its own vector from ``seed ⊕ hash(word)``.
    """

    name = "rag"

    def __init__(
        self,
        corpus: list[str] | None = None,
        dim: int = 16,
        seed: int = 0,
    ):
        self.corpus = list(corpus) if corpus is not None else list(_DEFAULT_CORPUS)
        if not self.corpus:
            raise ValueError("RAGPipeline needs a non-empty corpus")
        if dim < 1:
            raise ValueError(f"dim must be >= 1, got {dim}")
        self.dim = dim
        self.seed = seed
        # Precompute document embeddings once; retrieval reuses them per query.
        self._doc_embeddings = np.stack([self._embed(doc) for doc in self.corpus])

    # ------------------------------------------------------------- embedding
    def _word_vector(self, word: str) -> np.ndarray:
        """A deterministic vector for a single lowercased word."""
        # Stable across processes: derive the rng seed from the base seed and a
        # hash of the word (Python's built-in hash is salted, so avoid it here).
        word_hash = abs(int.from_bytes(word.encode("utf-8"), "little")) if word else 0
        rng = np.random.default_rng(self.seed + word_hash)
        return rng.normal(size=self.dim)

    def _embed(self, text: str) -> np.ndarray:
        """Sum per-word vectors, then L2-normalize (bag of hashed words)."""
        words = [w for w in text.lower().split() if w]
        if not words:
            return np.zeros(self.dim)
        vec = np.sum([self._word_vector(w) for w in words], axis=0)
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm else vec

    # ------------------------------------------------------------------ trace
    def trace(self, query: str, k: int = 2) -> Trace:
        """Build the full five-stage RAG trace for ``query``."""
        if not query.strip():
            raise ValueError("query must be a non-empty string")
        k = max(1, min(k, len(self.corpus)))

        t = Trace(
            op="rag",
            formula="prompt = stuff( top_k( cos(E[q], E[dᵢ]) ) ) ⊕ q",
            complexity="O(n_docs · d) for n documents of embedding dim d",
            why_ai=[
                "RAG grounds an LLM in external documents it never saw in training",
                "Retrieval is cosine search in embedding space — the same math as attention scores",
                "The top-k chunks become the context the model reads before answering",
            ],
            meta={"n_docs": len(self.corpus), "dim": self.dim, "seed": self.seed, "k": k},
        )

        q_vec = self._embed(query)
        t.add(
            "Embed the query",
            f"E[{query!r}] → vector of dim {self.dim}\n{arr(q_vec)}",
            q_vec,
            detail="Bag-of-hashed-words: sum a fixed vector per word, then L2-normalize.",
        )

        t.add(
            "Embed the documents",
            f"corpus already encoded into a ({len(self.corpus)}, {self.dim}) matrix",
            self._doc_embeddings,
            detail="Precomputed at construction time — real systems store these in a vector DB.",
        )

        scores = self._doc_embeddings @ q_vec
        for i, (doc, score) in enumerate(zip(self.corpus, scores, strict=True)):
            t.add(
                f"Cosine score doc {i}",
                f"cos(q, dᵢ) = {num(float(score))}   ⟨{doc}⟩",
                float(score),
            )

        top_idx = np.argsort(scores)[::-1][:k]
        retrieved = [self.corpus[i] for i in top_idx]
        t.add(
            f"Select top-{k}",
            "  ".join(f"doc {i}={num(float(scores[i]))}" for i in top_idx),
            retrieved,
            detail="Highest cosine similarity = most relevant context to retrieve.",
        )

        context = "\n".join(f"[{n}] {doc}" for n, doc in enumerate(retrieved, start=1))
        prompt = (
            "Answer the question using ONLY the context below.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n"
            "Answer:"
        )
        t.add(
            "Assemble the prompt",
            f"stuff {k} retrieved chunk(s) + the question into one prompt\n\n{prompt}",
            prompt,
            detail="This grounded prompt is what the LLM actually receives.",
        )
        t.result = prompt
        return t

    def forward(
        self,
        query: str,
        k: int = 2,
        explain: bool = False,
        level: str = "engineer",
    ) -> str:
        """Run the pipeline. Set ``explain=True`` to print the five-stage trace."""
        t = self.trace(query, k=k)
        return t.render(level) if explain else t.result

    @classmethod
    def demo(cls) -> Trace:
        """A reproducible RAG example over the default ML corpus."""
        return cls().trace("how do neural networks learn?")
