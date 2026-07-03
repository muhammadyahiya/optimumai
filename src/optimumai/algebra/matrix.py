"""Matrices that narrate their own multiplication.

``y = W·x + b`` is every dense layer. ``Q·Kᵀ`` is the attention score matrix.
Watching a matrix multiply fill in cell by cell is the fastest way to make the
shape rules (``(m,k) @ (k,n) → (m,n)``) click.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

from optimumai.core._fmt import num
from optimumai.core.explain import ExplainLevel
from optimumai.core.trace import Trace


class Matrix:
    """A 2-D array with an explainable matrix product."""

    def __init__(self, data: Sequence[Sequence[float]] | np.ndarray):
        self.data = np.asarray(data, dtype=float)
        if self.data.ndim != 2:
            raise ValueError(f"Matrix expects 2-D data, got shape {self.data.shape}")

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self.data.shape[0]), int(self.data.shape[1]))

    def __repr__(self) -> str:
        return f"Matrix(shape={self.shape})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Matrix) and np.array_equal(self.data, other.data)

    # ------------------------------------------------------------ matrix product
    def matmul_trace(self, other: Matrix | Iterable) -> Trace:
        """Trace ``C = A @ B``, one output cell at a time."""
        b = other if isinstance(other, Matrix) else Matrix(np.atleast_2d(other))
        A, B = self.data, b.data
        if A.shape[1] != B.shape[0]:
            raise ValueError(
                f"cannot multiply {A.shape} by {B.shape}: "
                f"inner dimensions {A.shape[1]} and {B.shape[0]} must match"
            )
        m, k = A.shape
        _, n = B.shape
        t = Trace(
            op="matmul",
            formula="C[i,j] = Σₖ A[i,k]·B[k,j]",
            complexity=f"O(m·k·n) = O({m}·{k}·{n})",
            why_ai=[
                "Every dense/linear layer: y = W·x + b",
                "Projecting inputs into Q, K, V inside attention",
                "Composing linear transformations across network layers",
            ],
            meta={"a_shape": A.shape, "b_shape": B.shape},
        )
        C = np.zeros((m, n))
        for i in range(m):
            for j in range(n):
                terms = [f"{num(A[i, p])}×{num(B[p, j])}" for p in range(k)]
                val = float(np.dot(A[i, :], B[:, j]))
                C[i, j] = val
                t.add(
                    f"Cell C[{i},{j}]",
                    f"{' + '.join(terms)} = {num(val)}",
                    val,
                    detail=f"row {i} of A · column {j} of B",
                )
        t.result = C
        return t

    def matmul(
        self,
        other: Matrix | Iterable,
        explain: bool = False,
        level: str | ExplainLevel = ExplainLevel.INTERMEDIATE,
    ) -> np.ndarray:
        """Matrix product. Set ``explain=True`` to print the cell-by-cell trace."""
        t = self.matmul_trace(other)
        return t.render(level) if explain else t.result

    def __matmul__(self, other: Matrix) -> np.ndarray:
        return self.matmul(other)

    # ------------------------------------------------------------------ transpose
    def transpose(self) -> Matrix:
        return Matrix(self.data.T)

    @property
    def T(self) -> Matrix:  # noqa: N802 - mirror numpy's attribute name
        return self.transpose()
