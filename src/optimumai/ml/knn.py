"""k-nearest neighbors — classify a point by asking its closest neighbors.

k-NN is "memorize the training set, do the work at prediction time." There is
no fitting step beyond storing ``(X, y)``. To classify a new point ``x``:

1. Compute the distance from ``x`` to every stored training point (usually
   Euclidean: ``d(x, xᵢ) = ‖x − xᵢ‖``).
2. Take the ``k`` closest training points.
3. Predict the **majority label** among those ``k`` neighbors (ties broken by
   whichever label appears first among the nearest).

There is no loss function to minimize and no gradient — the "model" is
literally the dataset plus a distance function, so k-NN is often called a
**non-parametric**, or **instance-based**, learner. Larger ``k`` smooths the
decision boundary (more votes, less sensitive to a single noisy neighbor) but
can blur genuinely separate small clusters; ``k=1`` traces the training data
exactly and overfits.

Why AI cares
------------
k-NN is the simplest possible use of a learned **embedding space**: once
inputs are turned into vectors (word embeddings, image embeddings, user
embeddings), "find the k nearest" is exactly the retrieval step in
retrieval-augmented generation, recommendation ("users like you also liked"),
and duplicate/anomaly detection. Production systems replace the brute-force
``O(n)`` scan here with approximate nearest-neighbor indexes (FAISS, HNSW),
but the underlying question — "what's close to this point in vector space?" —
is unchanged.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from optimumai.core._fmt import arr, num
from optimumai.core.explain import ExplainLevel
from optimumai.core.trace import Trace


def _as_2d(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    return X.reshape(-1, 1) if X.ndim == 1 else X


def _majority_vote(labels: np.ndarray) -> int:
    """Most common label; ties go to whichever label appears first (stable)."""
    counts = Counter(labels.tolist())
    best_count = max(counts.values())
    for label in labels:
        if counts[label] == best_count:
            return int(label)
    raise AssertionError("unreachable: labels is non-empty")  # pragma: no cover


def knn_trace(X_train, y_train, x_query, k: int = 3) -> Trace:
    """Build the full distance/vote trace classifying one query point.

    Args:
        X_train: Training features, shape ``(n, d)`` (or ``(n,)``).
        y_train: Training labels, shape ``(n,)``.
        x_query: A single query point, shape ``(d,)`` (or a scalar for 1-D).
        k: Number of neighbors to vote.
    """
    Xmat = _as_2d(X_train)
    y = np.asarray(y_train).reshape(-1)
    xq = np.asarray(x_query, dtype=float).reshape(-1)
    if Xmat.shape[0] != y.shape[0]:
        raise ValueError(f"X_train has {Xmat.shape[0]} rows but y_train has {y.shape[0]} entries")
    if xq.shape[0] != Xmat.shape[1]:
        raise ValueError(f"x_query has {xq.shape[0]} features, expected {Xmat.shape[1]}")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if k > Xmat.shape[0]:
        raise ValueError(f"k ({k}) cannot exceed the number of training points ({Xmat.shape[0]})")

    n, d = Xmat.shape
    t = Trace(
        op="knn",
        formula="label(x) = majority_vote({yᵢ : xᵢ among the k closest to x})",
        complexity=f"O(n·d) to score, O(n log n) to sort, n={n}, d={d}, k={k}",
        why_ai=[
            "The simplest possible use of an embedding space: 'find what's "
            "nearby' is the retrieval step behind RAG, recommendations, and "
            "duplicate detection",
            "No training phase — the model *is* the dataset, which is why it's "
            "called instance-based / non-parametric learning",
            "Production systems swap this O(n) scan for approximate "
            "nearest-neighbor indexes (FAISS, HNSW) but ask the exact same "
            "'who is close to me?' question",
        ],
        meta={"k": k, "n_train": n, "n_features": d},
    )

    distances = np.sqrt(np.sum((Xmat - xq) ** 2, axis=1))
    t.add(
        "Distance to every training point",
        f"d(x, xᵢ) = ‖x − xᵢ‖ = {arr(distances)}",
        distances,
        detail=f"Query point x = {arr(xq)}; training labels y = {arr(y)}.",
    )

    order = np.argsort(distances, kind="stable")
    nearest_idx = order[:k]
    nearest_distances = distances[nearest_idx]
    nearest_labels = y[nearest_idx]
    t.add(
        f"Take the {k} nearest neighbors",
        f"indices = {arr(nearest_idx)}, distances = {arr(nearest_distances)}, "
        f"labels = {arr(nearest_labels)}",
        nearest_labels.copy(),
        detail="Sorted by distance, closest first.",
    )

    vote_counts = Counter(nearest_labels.tolist())
    prediction = _majority_vote(nearest_labels)
    t.add(
        "Majority vote",
        f"vote counts = {dict(vote_counts)}  →  predicted label = {num(prediction)}",
        prediction,
        detail="Ties are broken by whichever tied label is closest to the query.",
    )

    t.result = prediction
    t.meta["nearest_indices"] = nearest_idx
    t.meta["nearest_distances"] = nearest_distances
    t.meta["nearest_labels"] = nearest_labels
    return t


class KNN:
    """k-nearest-neighbors classifier: store the data, vote at prediction time.

    Example:
        >>> model = KNN(k=1).fit([[0], [1], [10], [11]], [0, 0, 1, 1])
        >>> model.predict([[0.5], [10.5]])
        array([0, 1])
    """

    def __init__(self, k: int = 3) -> None:
        self.k = k
        self.X_train: np.ndarray | None = None
        self.y_train: np.ndarray | None = None

    def fit(self, X, y) -> KNN:
        """Store the training data (k-NN has no learned parameters)."""
        self.X_train = _as_2d(X)
        self.y_train = np.asarray(y).reshape(-1)
        if self.X_train.shape[0] != self.y_train.shape[0]:
            raise ValueError(
                f"X has {self.X_train.shape[0]} rows but y has {self.y_train.shape[0]} entries"
            )
        return self

    def predict(self, X) -> np.ndarray:
        """Classify each row of ``X`` by majority vote of its k nearest neighbors."""
        if self.X_train is None or self.y_train is None:
            raise ValueError("model is not fitted yet — call fit(X, y) first")
        Xq = _as_2d(X)
        return np.array(
            [
                knn_trace(self.X_train, self.y_train, row, k=self.k).result
                for row in Xq
            ]
        )

    def explain(
        self, x_query, level: str | ExplainLevel = ExplainLevel.INTERMEDIATE
    ) -> int:
        """Print the full distance/vote trace for one query point and return its label."""
        if self.X_train is None or self.y_train is None:
            raise ValueError("model is not fitted yet — call fit(X, y) first")
        t = knn_trace(self.X_train, self.y_train, x_query, k=self.k)
        return t.render(level)


def demo(seed: int = 0) -> Trace:
    """Classify the point x=4.5 among two well-separated 1-D clusters (k=3)."""
    X_train = np.array([0.0, 1.0, 2.0, 8.0, 9.0, 10.0])
    y_train = np.array([0, 0, 0, 1, 1, 1])
    return knn_trace(X_train, y_train, x_query=4.5, k=3)
