"""k-means — grouping points by repeatedly asking "who is my nearest centroid?"

k-means (Lloyd's algorithm) partitions ``n`` points into ``k`` clusters by
alternating two simple steps until nothing changes:

1. **Assign** — put every point in the cluster of its nearest centroid
   (Euclidean distance).
2. **Update** — move each centroid to the mean of the points now assigned
   to it.

Both steps only ever *decrease* (or hold steady) the objective, the **inertia**
— the total squared distance from each point to its assigned centroid:

    ``inertia = Σᵢ ‖xᵢ − c_{assign(i)}‖²``

so the loop is guaranteed to converge (though only to a *local* optimum — the
result depends on where the centroids started, which is why real
implementations restart from several random initializations and keep the best
run). There is no gradient here and no closed form; it is coordinate-descent
style hard assignment, alternating between "best label given centroids" and
"best centroids given labels."

Why AI cares
------------
k-means is the default way to explore unlabeled data: bucketing embeddings
into topics, building a codebook for vector quantization, initializing
mixture models, or finding representative anchor points before a more
expensive supervised model is trained. The "assign to nearest, then
re-center" pattern reappears in vector-quantized VAEs (VQ-VAE) and in
approximate nearest-neighbor search indexes.
"""

from __future__ import annotations

import numpy as np

from optimumai.core._fmt import arr, num
from optimumai.core.explain import ExplainLevel
from optimumai.core.trace import Trace


def _pairwise_distances(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Euclidean distance from every point in ``X`` to every centroid, shape (n, k)."""
    diffs = X[:, None, :] - centroids[None, :, :]
    return np.sqrt(np.sum(diffs**2, axis=2))


def _inertia(X: np.ndarray, centroids: np.ndarray, labels: np.ndarray) -> float:
    return float(np.sum((X - centroids[labels]) ** 2))


def kmeans_trace(
    X, k: int = 2, max_iters: int = 10, init: np.ndarray | None = None
) -> Trace:
    """Build the full Lloyd's-algorithm trace, clustering ``X`` into ``k`` groups.

    Args:
        X: Data points, shape ``(n, d)`` (or ``(n,)`` for 1-D data).
        k: Number of clusters.
        max_iters: Safety cap on assign/update rounds.
        init: Optional starting centroids, shape ``(k, d)``. Defaults to the
            first ``k`` points, which keeps the demo deterministic.
    """
    Xmat = np.asarray(X, dtype=float)
    if Xmat.ndim == 1:
        Xmat = Xmat.reshape(-1, 1)
    if Xmat.ndim != 2:
        raise ValueError(f"X must be 1-D or 2-D, got shape {np.asarray(X).shape}")
    n, d = Xmat.shape
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if k > n:
        raise ValueError(f"k ({k}) cannot exceed the number of points ({n})")

    centroids = (
        Xmat[:k].copy() if init is None else np.asarray(init, dtype=float).reshape(k, d).copy()
    )

    t = Trace(
        op="kmeans",
        formula="assign: label(i) = argminⱼ ‖xᵢ − cⱼ‖²;  update: cⱼ = mean({xᵢ : label(i)=j})",
        complexity=f"O(n·k·d) per iteration, n={n}, k={k}, d={d}",
        why_ai=[
            "The default first move on unlabeled data — topic buckets for text "
            "embeddings, codebooks for vector quantization, cluster-based "
            "anchor points before supervised training",
            "Converges to a local optimum of the inertia objective, never worse — "
            "which is why real systems restart from several initializations",
            "The 'assign to nearest, re-center' loop reappears in VQ-VAE codebooks "
            "and approximate nearest-neighbor index construction",
        ],
        meta={"k": k, "n_samples": n, "n_features": d},
    )

    t.add(
        "Initialize centroids",
        f"c = {arr(centroids)}",
        centroids.copy(),
        detail="Started at the first k points for a reproducible trace; real "
        "runs use random restarts (k-means++ is a smarter random start).",
    )

    labels = np.full(n, -1)
    for iteration in range(1, max_iters + 1):
        distances = _pairwise_distances(Xmat, centroids)
        new_labels = np.argmin(distances, axis=1)
        t.add(
            f"Assign step (iteration {iteration})",
            f"distances to centroids:\n{arr(distances)}\nlabels = {arr(new_labels)}",
            new_labels.copy(),
            detail="Each point joins the cluster of its closest centroid.",
        )

        if np.array_equal(new_labels, labels):
            t.add(
                f"Converged at iteration {iteration}",
                "labels unchanged from the previous iteration — stopping",
                detail="No point switched clusters, so the update step would not move "
                "any centroid either.",
            )
            labels = new_labels
            break
        labels = new_labels

        new_centroids = np.array(
            [
                Xmat[labels == j].mean(axis=0) if np.any(labels == j) else centroids[j]
                for j in range(k)
            ]
        )
        t.add(
            f"Update step (iteration {iteration})",
            f"c = mean of each cluster's points = {arr(new_centroids)}",
            new_centroids.copy(),
        )
        centroids = new_centroids
    else:
        t.add(
            f"Stopped at max_iters = {max_iters}",
            "reached the iteration cap without labels stabilizing",
            detail="Increase max_iters, or accept this as a good-enough approximation.",
        )

    inertia = _inertia(Xmat, centroids, labels)
    t.add(
        "Final inertia",
        f"inertia = Σ‖xᵢ − c_label(i)‖² = {num(inertia)}",
        inertia,
        detail="Lower is tighter clusters; k-means always converges to a local "
        "minimum of this quantity, never a worse one.",
    )

    t.result = labels
    t.meta["centroids"] = centroids
    t.meta["inertia"] = inertia
    return t


class KMeans:
    """Lloyd's-algorithm k-means clustering.

    Example:
        >>> model = KMeans(k=2).fit([[0], [1], [10], [11]])
        >>> model.predict([[0.5], [10.5]])
        array([0, 1])
    """

    def __init__(self, k: int = 2, max_iters: int = 10) -> None:
        self.k = k
        self.max_iters = max_iters
        self.centroids: np.ndarray | None = None
        self.labels_: np.ndarray | None = None
        self.inertia_: float | None = None

    def fit(self, X) -> KMeans:
        """Run Lloyd's algorithm to convergence (or :attr:`max_iters`)."""
        t = kmeans_trace(X, k=self.k, max_iters=self.max_iters)
        self.centroids = t.meta["centroids"]
        self.labels_ = t.result
        self.inertia_ = t.meta["inertia"]
        return self

    def predict(self, X) -> np.ndarray:
        """Assign new points to the nearest fitted centroid."""
        if self.centroids is None:
            raise ValueError("model is not fitted yet — call fit(X) first")
        Xmat = np.asarray(X, dtype=float)
        if Xmat.ndim == 1:
            Xmat = Xmat.reshape(-1, 1)
        distances = _pairwise_distances(Xmat, self.centroids)
        return np.argmin(distances, axis=1)

    def explain(self, X, level: str | ExplainLevel = ExplainLevel.INTERMEDIATE) -> np.ndarray:
        """Fit, print the full assign/update trace, and return cluster labels."""
        t = kmeans_trace(X, k=self.k, max_iters=self.max_iters)
        self.centroids = t.meta["centroids"]
        self.labels_ = t.result
        self.inertia_ = t.meta["inertia"]
        return t.render(level)


def demo(seed: int = 0) -> Trace:
    """Cluster two well-separated 1-D blobs — {0,1,2} and {10,11,12} — into k=2."""
    X = np.array([0.0, 1.0, 2.0, 10.0, 11.0, 12.0])
    return kmeans_trace(X, k=2)
