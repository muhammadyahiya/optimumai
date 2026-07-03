"""Vectors that can explain themselves.

The dot product is the atom of modern AI: it is the similarity between two
embeddings, the raw attention score ``q · k``, and the inner loop of every
matrix multiply. Teaching it well pays off everywhere downstream.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from optimumai.core._fmt import num
from optimumai.core.explain import ExplainLevel
from optimumai.core.trace import Trace


class Vector:
    """A 1-D array with step-by-step explanations of its operations."""

    def __init__(self, data: Iterable[float]):
        raw = data if isinstance(data, np.ndarray) else list(data)
        self.data = np.asarray(raw, dtype=float)
        if self.data.ndim != 1:
            raise ValueError(f"Vector expects 1-D data, got shape {self.data.shape}")

    # ------------------------------------------------------------------ repr
    def __len__(self) -> int:
        return int(self.data.shape[0])

    def __repr__(self) -> str:
        return f"Vector({self.data.tolist()})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Vector) and np.array_equal(self.data, other.data)

    # ------------------------------------------------------------- dot product
    def dot_trace(self, other: Vector) -> Trace:
        """Build the full trace of ``self · other``."""
        a, b = self.data, self._as_vector(other).data
        if a.shape != b.shape:
            raise ValueError(
                f"dot product needs equal-length vectors, got {a.shape} and {b.shape}"
            )
        t = Trace(
            op="dot",
            formula="a · b = Σᵢ aᵢ·bᵢ",
            complexity="O(n)",
            why_ai=[
                "Similarity between two embedding vectors",
                "The raw attention score q · k before scaling",
                "The inner loop of every matrix multiply and linear layer",
            ],
        )
        products = []
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            p = float(x * y)
            products.append(p)
            t.add(f"Multiply component {i}", f"{num(x)} × {num(y)} = {num(p)}", p)
        total = float(np.sum(products))
        summation = " + ".join(num(p) for p in products) if products else "0"
        t.add("Sum the products", f"{summation} = {num(total)}", total)
        t.result = total
        return t

    def dot(
        self,
        other: Vector,
        explain: bool = False,
        level: str | ExplainLevel = ExplainLevel.INTERMEDIATE,
    ) -> float:
        """Inner product. Set ``explain=True`` to print the full trace."""
        t = self.dot_trace(other)
        return t.render(level) if explain else t.result

    # -------------------------------------------------------------------- norm
    def norm_trace(self) -> Trace:
        """Euclidean (L2) norm, ``||a|| = √Σ aᵢ²``."""
        a = self.data
        t = Trace(
            op="norm",
            formula="||a|| = √(Σᵢ aᵢ²)",
            complexity="O(n)",
            why_ai=[
                "The length of an embedding vector",
                "Denominator of cosine similarity",
                "Weight/gradient magnitudes for clipping and regularization",
            ],
        )
        squares = [float(x * x) for x in a]
        for i, (x, sq) in enumerate(zip(a, squares, strict=True)):
            t.add(f"Square component {i}", f"{num(x)}² = {num(sq)}", sq)
        ssum = float(np.sum(squares))
        t.add("Sum of squares", f"{' + '.join(num(s) for s in squares) or '0'} = {num(ssum)}", ssum)
        result = float(np.sqrt(ssum))
        t.add("Square root", f"√{num(ssum)} = {num(result)}", result)
        t.result = result
        return t

    def norm(
        self,
        explain: bool = False,
        level: str | ExplainLevel = ExplainLevel.INTERMEDIATE,
    ) -> float:
        t = self.norm_trace()
        return t.render(level) if explain else t.result

    # ---------------------------------------------------------- cosine similarity
    def cosine_similarity_trace(self, other: Vector) -> Trace:
        """Cosine similarity ``(a · b) / (||a|| ||b||)`` — the RAG workhorse."""
        b = self._as_vector(other)
        if self.data.shape != b.data.shape:
            raise ValueError(
                f"cosine similarity needs equal-length vectors, "
                f"got {self.data.shape} and {b.data.shape}"
            )
        t = Trace(
            op="cosine_similarity",
            formula="cos(θ) = (a · b) / (||a|| · ||b||)",
            complexity="O(n)",
            why_ai=[
                "Ranking documents against a query in RAG / semantic search",
                "Measuring how aligned two embeddings are, ignoring magnitude",
                "Nearest-neighbour lookup in a vector database",
            ],
        )
        dot = float(np.dot(self.data, b.data))
        na = float(np.linalg.norm(self.data))
        nb = float(np.linalg.norm(b.data))
        t.add("Dot product a · b", f"a · b = {num(dot)}", dot)
        t.add("Norm of a", f"||a|| = {num(na)}", na)
        t.add("Norm of b", f"||b|| = {num(nb)}", nb)
        denom = na * nb
        if denom == 0.0:
            raise ValueError("cosine similarity is undefined for a zero vector")
        result = dot / denom
        t.add(
            "Divide",
            f"{num(dot)} / ({num(na)} × {num(nb)}) = {num(result)}",
            result,
        )
        t.result = result
        return t

    def cosine_similarity(
        self,
        other: Vector,
        explain: bool = False,
        level: str | ExplainLevel = ExplainLevel.INTERMEDIATE,
    ) -> float:
        t = self.cosine_similarity_trace(other)
        return t.render(level) if explain else t.result

    # ---------------------------------------------------------------- internals
    @staticmethod
    def _as_vector(other: Vector | Iterable[float]) -> Vector:
        return other if isinstance(other, Vector) else Vector(other)
