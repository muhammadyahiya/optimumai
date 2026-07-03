"""Gaussian Naive Bayes — Bayes' rule plus one convenient (and "naive") lie.

Bayes' rule says the probability of a class given features is proportional to
how likely those features are under that class, times how common the class
is:

    ``P(y | x) ∝ P(x | y) · P(y)``

The trouble is ``P(x | y)`` for a whole feature vector is hard to estimate
directly. **Naive** Bayes makes the simplifying (usually false, often good
enough) assumption that features are conditionally independent given the
class, which factorizes the joint likelihood into a simple product:

    ``P(x | y) = Πⱼ P(xⱼ | y)``

**Gaussian** Naive Bayes further assumes each feature, within each class,
follows a normal distribution — so "training" is just computing a mean ``μ``
and variance ``σ²`` per (feature, class) pair from the training data:

    ``P(xⱼ | y) = (1 / √(2πσ²)) · exp(−(xⱼ − μ)² / (2σ²))``

To classify a new point, multiply the prior by every feature's likelihood for
each class and pick the largest. In practice this is done in **log-space**
(summing log-likelihoods instead of multiplying tiny probabilities) to avoid
numerical underflow — the same max-subtraction-style stability trick that
:mod:`softmax <optimumai.probability.softmax>` uses:

    ``ŷ = argmaxᵧ [ log P(y) + Σⱼ log P(xⱼ | y) ]``

Why AI cares
------------
Naive Bayes is the classic baseline for text classification (spam filters,
sentiment) because "independent given the class" is a tolerable lie for bags
of words, and it needs almost no data to get reasonable estimates — a good
first thing to try before reaching for anything heavier. The log-sum-of-
likelihoods pattern here is the same shape as computing a sequence
probability in a language model.
"""

from __future__ import annotations

import numpy as np

from optimumai.core._fmt import arr, num
from optimumai.core.explain import ExplainLevel
from optimumai.core.trace import Trace

_EPS = 1e-9  # variance floor so a feature that's constant within a class doesn't blow up log-space


def _gaussian_log_likelihood(x: np.ndarray, mean: np.ndarray, var: np.ndarray) -> np.ndarray:
    var = np.maximum(var, _EPS)
    return -0.5 * np.log(2 * np.pi * var) - ((x - mean) ** 2) / (2 * var)


def _as_2d(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    return X.reshape(-1, 1) if X.ndim == 1 else X


def naive_bayes_trace(X_train, y_train, x_query) -> Trace:
    """Build the full prior/likelihood/posterior trace classifying one query point.

    Args:
        X_train: Training features, shape ``(n, d)`` (or ``(n,)``).
        y_train: Training labels, shape ``(n,)``.
        x_query: A single query point, shape ``(d,)`` (or a scalar for 1-D).
    """
    Xmat = _as_2d(X_train)
    y = np.asarray(y_train).reshape(-1)
    xq = np.asarray(x_query, dtype=float).reshape(-1)
    if Xmat.shape[0] != y.shape[0]:
        raise ValueError(f"X_train has {Xmat.shape[0]} rows but y_train has {y.shape[0]} entries")
    if xq.shape[0] != Xmat.shape[1]:
        raise ValueError(f"x_query has {xq.shape[0]} features, expected {Xmat.shape[1]}")

    classes = np.unique(y)
    if classes.size < 2:
        raise ValueError("need at least 2 classes to classify")
    n, d = Xmat.shape

    t = Trace(
        op="naive_bayes",
        formula="ŷ = argmaxᵧ [ log P(y) + Σⱼ log P(xⱼ|y) ],  P(xⱼ|y) ~ Normal(μ_{y,j}, σ²_{y,j})",
        complexity=f"O(n·d) to fit means/variances, O(k·d) to classify, k={classes.size} classes",
        why_ai=[
            "The classic text-classification baseline (spam, sentiment) — "
            "'independent given the class' is a tolerable lie for bags of words",
            "Needs very little data to get usable estimates, so it's a good "
            "first model to try before reaching for anything heavier",
            "Classifying in log-space to avoid multiplying tiny probabilities "
            "is the same numerical-stability idea softmax uses",
        ],
        meta={"n_train": n, "n_features": d, "classes": classes.tolist()},
    )

    priors: dict[int, float] = {}
    means: dict[int, np.ndarray] = {}
    variances: dict[int, np.ndarray] = {}
    for c in classes:
        mask = y == c
        priors[int(c)] = float(np.sum(mask)) / n
        means[int(c)] = Xmat[mask].mean(axis=0)
        variances[int(c)] = Xmat[mask].var(axis=0)

    t.add(
        "Class priors P(y)",
        ", ".join(f"P(y={int(c)}) = {num(priors[int(c)])}" for c in classes),
        dict(priors),
        detail="The fraction of training points in each class.",
    )
    for c in classes:
        t.add(
            f"Per-feature Gaussian for class {int(c)}",
            f"μ = {arr(means[int(c)])}, σ² = {arr(variances[int(c)])}",
            {"mean": means[int(c)], "var": variances[int(c)]},
            detail="Mean and (biased) variance of each feature, restricted to this class's rows.",
        )

    log_posteriors: dict[int, float] = {}
    for c in classes:
        log_prior = np.log(priors[int(c)])
        log_likelihoods = _gaussian_log_likelihood(xq, means[int(c)], variances[int(c)])
        log_posterior = float(log_prior + np.sum(log_likelihoods))
        log_posteriors[int(c)] = log_posterior
        t.add(
            f"Log-posterior for class {int(c)}",
            f"log P(y={int(c)}) + Σ log P(xⱼ|y={int(c)}) "
            f"= {num(log_prior)} + {num(float(np.sum(log_likelihoods)))} = {num(log_posterior)}",
            log_posterior,
            detail=f"Per-feature log-likelihoods: {arr(log_likelihoods)}",
        )

    prediction = max(log_posteriors, key=lambda c: log_posteriors[c])
    t.add(
        "Argmax over classes",
        f"log-posteriors = {log_posteriors}  →  predicted label = {num(prediction)}",
        prediction,
        detail="Log is monotonic, so comparing log-posteriors gives the same "
        "ranking as comparing the (much tinier) raw probabilities.",
    )

    t.result = prediction
    t.meta["priors"] = priors
    t.meta["means"] = means
    t.meta["variances"] = variances
    t.meta["log_posteriors"] = log_posteriors
    return t


class GaussianNB:
    """Gaussian Naive Bayes classifier.

    Example:
        >>> model = GaussianNB().fit([[0], [1], [10], [11]], [0, 0, 1, 1])
        >>> model.predict([[0.5], [10.5]])
        array([0, 1])
    """

    def __init__(self) -> None:
        self.classes: np.ndarray | None = None
        self.priors: dict[int, float] | None = None
        self.means: dict[int, np.ndarray] | None = None
        self.variances: dict[int, np.ndarray] | None = None
        self._X_train: np.ndarray | None = None
        self._y_train: np.ndarray | None = None

    def fit(self, X, y) -> GaussianNB:
        """Estimate class priors and per-feature Gaussian parameters."""
        Xmat = _as_2d(X)
        yvec = np.asarray(y).reshape(-1)
        if Xmat.shape[0] != yvec.shape[0]:
            raise ValueError(f"X has {Xmat.shape[0]} rows but y has {yvec.shape[0]} entries")
        self.classes = np.unique(yvec)
        if self.classes.size < 2:
            raise ValueError("need at least 2 classes to fit")
        n = Xmat.shape[0]
        self.priors = {}
        self.means = {}
        self.variances = {}
        for c in self.classes:
            mask = yvec == c
            self.priors[int(c)] = float(np.sum(mask)) / n
            self.means[int(c)] = Xmat[mask].mean(axis=0)
            self.variances[int(c)] = Xmat[mask].var(axis=0)
        self._X_train, self._y_train = Xmat, yvec
        return self

    def predict(self, X) -> np.ndarray:
        """Classify each row of ``X`` by argmax log-posterior."""
        if self.classes is None or self.priors is None or self.means is None:
            raise ValueError("model is not fitted yet — call fit(X, y) first")
        assert self.variances is not None
        Xq = _as_2d(X)
        preds = []
        for row in Xq:
            log_posteriors = {
                int(c): np.log(self.priors[int(c)])
                + np.sum(
                    _gaussian_log_likelihood(row, self.means[int(c)], self.variances[int(c)])
                )
                for c in self.classes
            }
            preds.append(max(log_posteriors, key=lambda c: log_posteriors[c]))
        return np.array(preds)

    def explain(
        self, x_query, level: str | ExplainLevel = ExplainLevel.INTERMEDIATE
    ) -> int:
        """Print the full prior/likelihood/posterior trace for one query point."""
        if self._X_train is None or self._y_train is None:
            raise ValueError("model is not fitted yet — call fit(X, y) first")
        t = naive_bayes_trace(self._X_train, self._y_train, x_query)
        return t.render(level)


def demo(seed: int = 0) -> Trace:
    """Classify x=4.5 between two well-separated 1-D Gaussian-ish clusters."""
    X_train = np.array([0.0, 1.0, 2.0, 8.0, 9.0, 10.0])
    y_train = np.array([0, 0, 0, 1, 1, 1])
    return naive_bayes_trace(X_train, y_train, x_query=4.5)
