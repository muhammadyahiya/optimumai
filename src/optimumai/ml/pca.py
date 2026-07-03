"""PCA — finding the few directions that explain most of the spread in the data.

High-dimensional data (embeddings, pixels, sensor readings) often has far
fewer *effective* dimensions than raw dimensions: many features move together.
Principal Component Analysis (PCA) finds a new set of axes — ordered by how
much variance they capture — so that keeping only the first few axes still
preserves most of the information.

The recipe:

1. **Center** the data so each feature has mean 0 (PCA is about spread
   *around the mean*, not the mean itself): ``X_c = X − mean(X)``.
2. Compute the **covariance matrix** ``Σ = (1/(n−1)) X_cᵀX_c`` — entry
   ``(i, j)`` is how much features ``i`` and ``j`` vary together.
3. **Eigendecompose** ``Σ``: eigenvectors are directions in feature space
   that ``Σ`` only stretches (does not rotate); the paired eigenvalue is how
   much variance lies along that direction: ``Σv = λv``.
4. **Sort** eigenvectors by eigenvalue, descending. The top eigenvector is
   the single direction of maximum variance — the **first principal
   component**; the second is the max-variance direction *orthogonal* to the
   first, and so on.
5. **Project** the centered data onto the top ``k`` eigenvectors:
   ``X_proj = X_c · V_k``. This is the dimensionality-reduced representation.

The **explained-variance ratio** of a component is its eigenvalue divided by
the sum of all eigenvalues — "what fraction of the total spread does this
axis account for" — and is the standard way to decide how many components
``k`` are worth keeping.

Why AI cares
------------
PCA is the classic way to compress, denoise, or visualize high-dimensional
embeddings (projecting a 768-D sentence embedding down to 2D for a plot),
and whitening/decorrelating features before a simpler model. The
eigendecomposition here is the same linear-algebra operation used to analyze
weight-matrix structure and the loss landscape's curvature (the Hessian's
eigenvectors) elsewhere in deep learning.
"""

from __future__ import annotations

import numpy as np

from optimumai.core._fmt import arr, num
from optimumai.core.explain import ExplainLevel
from optimumai.core.trace import Trace


def pca_trace(X, n_components: int = 1) -> Trace:
    """Build the full center/covariance/eigendecomposition/project trace.

    Args:
        X: Data, shape ``(n, d)``.
        n_components: How many top principal components to keep (``k <= d``).
    """
    Xmat = np.asarray(X, dtype=float)
    if Xmat.ndim != 2:
        raise ValueError(f"X must be 2-D, got shape {Xmat.shape}")
    n, d = Xmat.shape
    if n < 2:
        raise ValueError("need at least 2 samples to compute a covariance matrix")
    if not (1 <= n_components <= d):
        raise ValueError(f"n_components must be between 1 and {d}, got {n_components}")

    t = Trace(
        op="pca",
        formula="X_c = X − mean(X);  Σ = X_cᵀX_c/(n−1);  Σv = λv;  X_proj = X_c·V_k",
        complexity=f"O(nd²) for covariance + O(d³) to eigendecompose, n={n}, d={d}",
        why_ai=[
            "The standard way to compress, denoise, or visualize "
            "high-dimensional embeddings — e.g. projecting a 768-D sentence "
            "embedding down to 2D for a plot",
            "Explained-variance ratio is the principled way to choose how "
            "many dimensions k are 'enough' before throwing the rest away",
            "The eigendecomposition here is the same operation used to study "
            "weight-matrix structure and loss-landscape curvature elsewhere "
            "in deep learning",
        ],
        meta={"n_samples": n, "n_features": d, "n_components": n_components},
    )

    mean = Xmat.mean(axis=0)
    X_centered = Xmat - mean
    t.add(
        "Center the data",
        f"mean = {arr(mean)}\nX_c = X − mean =\n{arr(X_centered)}",
        X_centered,
        detail="PCA finds directions of spread around the mean, so the mean itself "
        "must be removed first.",
    )

    cov = (X_centered.T @ X_centered) / (n - 1)
    t.add(
        "Covariance matrix",
        f"Σ = X_cᵀX_c / (n−1) =\n{arr(cov)}",
        cov,
        detail="Σ[i,j] is how features i and j vary together; the diagonal is each "
        "feature's own variance.",
    )

    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    t.add(
        "Eigendecomposition, sorted by eigenvalue (descending)",
        f"eigenvalues = {arr(eigenvalues)}\neigenvectors (columns) =\n{arr(eigenvectors)}",
        {"eigenvalues": eigenvalues, "eigenvectors": eigenvectors},
        detail="Σv = λv: each eigenvector is a direction Σ only stretches by λ, "
        "never rotates. np.linalg.eigh exploits Σ being symmetric.",
    )

    total_variance = float(np.sum(eigenvalues))
    explained_ratio = eigenvalues / total_variance
    t.add(
        "Explained-variance ratio",
        f"λᵢ / Σλ = {arr(explained_ratio)}",
        explained_ratio,
        detail=f"The top {n_components} component(s) explain "
        f"{num(float(np.sum(explained_ratio[:n_components])) * 100)}% of the total variance.",
    )

    components = eigenvectors[:, :n_components]
    projected = X_centered @ components
    t.add(
        f"Project onto the top {n_components} component(s)",
        f"X_proj = X_c · V_k =\n{arr(projected)}",
        projected,
        detail=f"Reduced from {d} dimension(s) to {n_components}.",
    )

    t.result = projected
    t.meta["mean"] = mean
    t.meta["components"] = components
    t.meta["eigenvalues"] = eigenvalues
    t.meta["explained_variance_ratio"] = explained_ratio
    return t


def pca(X, n_components: int = 1) -> np.ndarray:
    """Return the data projected onto its top ``n_components`` principal components."""
    return pca_trace(X, n_components=n_components).result


class PCA:
    """Principal Component Analysis via covariance eigendecomposition.

    Example:
        >>> model = PCA(n_components=1).fit([[0, 0], [1, 1], [2, 2], [3, 3]])
        >>> model.transform([[4, 4]]).shape
        (1, 1)
    """

    def __init__(self, n_components: int = 1) -> None:
        self.n_components = n_components
        self.mean: np.ndarray | None = None
        self.components: np.ndarray | None = None
        self.explained_variance_ratio: np.ndarray | None = None

    def fit(self, X) -> PCA:
        """Compute the mean and top principal components from ``X``."""
        t = pca_trace(X, n_components=self.n_components)
        self.mean = t.meta["mean"]
        self.components = t.meta["components"]
        self.explained_variance_ratio = t.meta["explained_variance_ratio"]
        return self

    def transform(self, X) -> np.ndarray:
        """Project new data onto the fitted principal components."""
        if self.mean is None or self.components is None:
            raise ValueError("model is not fitted yet — call fit(X) first")
        Xmat = np.asarray(X, dtype=float)
        return (Xmat - self.mean) @ self.components

    def fit_transform(self, X) -> np.ndarray:
        """Fit on ``X`` and return the projected training data in one call."""
        return self.fit(X).transform(X)

    def explain(self, X, level: str | ExplainLevel = ExplainLevel.INTERMEDIATE) -> np.ndarray:
        """Fit, print the full center/covariance/eigen trace, and return the projection."""
        t = pca_trace(X, n_components=self.n_components)
        self.mean = t.meta["mean"]
        self.components = t.meta["components"]
        self.explained_variance_ratio = t.meta["explained_variance_ratio"]
        return t.render(level)


def demo(seed: int = 0) -> Trace:
    """Project a stretched-along-one-axis 2-D cloud down to its top 1 component."""
    X = np.array([[0.0, 0.1], [1.0, 0.9], [2.0, 2.1], [3.0, 2.9], [4.0, 4.1]])
    return pca_trace(X, n_components=1)
