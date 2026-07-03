"""Logistic regression — linear regression pushed through a sigmoid.

Linear regression predicts an unbounded number; classification needs a
*probability* in ``[0, 1]``. Logistic regression fixes this with one extra
step: run the linear score through the **sigmoid** function

    ``σ(z) = 1 / (1 + e^(−z))``,   ``z = Xθ``

which squashes any real number onto ``(0, 1)``. The model predicts
``P(y=1|x) = σ(Xθ)``.

Unlike OLS, there is no closed form for the weights that minimize the natural
loss for probabilities — **binary cross-entropy**:

    ``L(θ) = −(1/n) Σᵢ [yᵢ log(ŷᵢ) + (1−yᵢ) log(1−ŷᵢ)]``

This loss is still convex in ``θ``, so gradient descent reliably finds the
global minimum, it just takes iterative steps instead of one matrix solve. The
gradient has a strikingly clean form — the same shape as OLS's:

    ``∂L/∂θ = (1/n) Xᵀ(ŷ − y)``

("predicted minus actual, weighted by the inputs" — this identity is why
logistic regression and linear regression share so much training code.) Each
step nudges the weights downhill: ``θ ← θ − η · ∂L/∂θ``.

Why AI cares
------------
Logistic regression is the atomic binary classifier and the direct ancestor of
the neural network output layer: a single-neuron network with a sigmoid
activation trained on cross-entropy loss *is* logistic regression. The same
sigmoid, extended to many classes, generalizes to :mod:`softmax
<optimumai.probability.softmax>`.
"""

from __future__ import annotations

import numpy as np

from optimumai.core._fmt import arr, num
from optimumai.core.explain import ExplainLevel
from optimumai.core.trace import Trace

_EPS = 1e-12


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def _as_2d(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    return X.reshape(-1, 1) if X.ndim == 1 else X


def _design_matrix(X: np.ndarray) -> np.ndarray:
    ones = np.ones((X.shape[0], 1))
    return np.hstack([ones, X])


def _bce(y: np.ndarray, p: np.ndarray) -> float:
    p_clipped = np.clip(p, _EPS, 1 - _EPS)
    return float(-np.mean(y * np.log(p_clipped) + (1 - y) * np.log(1 - p_clipped)))


def logistic_regression_trace(
    X, y, lr: float = 0.5, steps: int = 3, theta0: np.ndarray | None = None
) -> Trace:
    """Build a trace of a few gradient-descent steps fitting logistic regression.

    Args:
        X: Feature matrix, shape ``(n, d)`` (or ``(n,)`` for one feature).
        y: Binary labels in ``{0, 1}``, shape ``(n,)``.
        lr: Learning rate for each gradient step.
        steps: Number of gradient-descent steps to trace (kept small — a real
            fit would run until the loss plateaus).
        theta0: Optional starting weights (zeros by default).
    """
    Xmat = _as_2d(X)
    yvec = np.asarray(y, dtype=float).reshape(-1)
    if Xmat.shape[0] != yvec.shape[0]:
        raise ValueError(f"X has {Xmat.shape[0]} rows but y has {yvec.shape[0]} entries")
    if not np.all(np.isin(yvec, [0.0, 1.0])):
        raise ValueError("y must contain only 0/1 labels for binary logistic regression")
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")
    if lr <= 0:
        raise ValueError(f"lr must be > 0, got {lr}")

    Xb = _design_matrix(Xmat)
    n, d = Xb.shape
    theta = np.zeros(d) if theta0 is None else np.asarray(theta0, dtype=float).copy()

    t = Trace(
        op="logistic_regression",
        formula="ŷ = σ(Xθ);  L = BCE(y, ŷ);  θ ← θ − η·(1/n)Xᵀ(ŷ − y)",
        complexity=f"O(nd) per gradient step, {steps} steps traced, n={n} samples",
        why_ai=[
            "The atomic binary classifier: spam/not-spam, click/no-click, "
            "fraud/not-fraud all start here",
            "A single sigmoid output neuron trained with cross-entropy loss "
            "*is* logistic regression — it's the last layer of countless "
            "classification networks",
            "The gradient (ŷ − y)ᵀX has the same clean form as linear regression's, "
            "which is why the two share so much training machinery",
        ],
        meta={"n_samples": Xmat.shape[0], "n_features": Xmat.shape[1], "lr": lr},
    )

    z0 = Xb @ theta
    p0 = _sigmoid(z0)
    t.add(
        "Forward pass: initial probabilities",
        f"z = Xθ = {arr(z0)}\nŷ = σ(z) = {arr(p0)}",
        p0,
        detail=f"θ starts at zeros, so σ(0) = 0.5 for every sample: {arr(p0)}.",
    )
    loss0 = _bce(yvec, p0)
    t.add(
        "Binary cross-entropy loss",
        f"L = −mean(y·log ŷ + (1−y)·log(1−ŷ)) = {num(loss0)}",
        loss0,
        detail=f"Labels y = {arr(yvec)}",
    )

    losses = [loss0]
    for step in range(1, steps + 1):
        z = Xb @ theta
        p = _sigmoid(z)
        grad = Xb.T @ (p - yvec) / n
        t.add(
            f"Gradient (step {step})",
            f"∂L/∂θ = (1/n)Xᵀ(ŷ − y) = {arr(grad)}",
            grad,
            detail="Points uphill; θ moves the opposite way.",
        )
        theta = theta - lr * grad
        z_new = Xb @ theta
        p_new = _sigmoid(z_new)
        loss = _bce(yvec, p_new)
        losses.append(loss)
        t.add(
            f"Weight update (step {step})",
            f"θ ← θ − {num(lr)}·grad = {arr(theta)}   (loss {num(losses[step - 1])} → {num(loss)})",
            theta.copy(),
        )

    final_p = _sigmoid(Xb @ theta)
    preds = (final_p >= 0.5).astype(int)
    accuracy = float(np.mean(preds == yvec))
    t.add(
        "Final predictions + accuracy",
        f"ŷ = σ(Xθ) = {arr(final_p)}\npredicted labels = {arr(preds)}, accuracy = {num(accuracy)}",
        preds,
        detail=f"Loss fell {num(losses[0])} → {num(losses[-1])} over {steps} step(s).",
    )

    t.result = preds
    t.meta["theta"] = theta
    t.meta["probabilities"] = final_p
    t.meta["accuracy"] = accuracy
    t.meta["losses"] = losses
    return t


class LogisticRegression:
    """Binary logistic regression trained by batch gradient descent.

    Example:
        >>> model = LogisticRegression(lr=0.5, steps=200).fit([[0], [1], [2], [3]], [0, 0, 1, 1])
        >>> model.predict([[0.1], [2.9]])
        array([0, 1])
    """

    def __init__(self, lr: float = 0.5, steps: int = 200) -> None:
        self.lr = lr
        self.steps = steps
        self.theta: np.ndarray | None = None

    def fit(self, X, y) -> LogisticRegression:
        """Run :attr:`steps` gradient-descent updates and store the final weights."""
        t = logistic_regression_trace(X, y, lr=self.lr, steps=self.steps)
        self.theta = t.meta["theta"]
        return self

    def predict_proba(self, X) -> np.ndarray:
        """Return ``P(y=1|x)`` for new samples (must call :meth:`fit` first)."""
        if self.theta is None:
            raise ValueError("model is not fitted yet — call fit(X, y) first")
        Xb = _design_matrix(_as_2d(X))
        if Xb.shape[1] != self.theta.shape[0]:
            raise ValueError(
                f"expected {self.theta.shape[0] - 1} features, got {Xb.shape[1] - 1}"
            )
        return _sigmoid(Xb @ self.theta)

    def predict(self, X) -> np.ndarray:
        """Return the hard 0/1 label (threshold at 0.5)."""
        return (self.predict_proba(X) >= 0.5).astype(int)

    def explain(
        self, X, y, level: str | ExplainLevel = ExplainLevel.INTERMEDIATE
    ) -> np.ndarray:
        """Fit, print the full trace (with :attr:`steps` gradient updates shown), return labels."""
        t = logistic_regression_trace(X, y, lr=self.lr, steps=min(self.steps, 3))
        self.theta = t.meta["theta"]
        return t.render(level)


def demo(seed: int = 0) -> Trace:
    """Classify a trivially separable 1-D set: 0s below 2, 1s at/above 2, three GD steps."""
    X = np.array([0.0, 0.5, 1.0, 2.0, 2.5, 3.0])
    y = np.array([0, 0, 0, 1, 1, 1])
    return logistic_regression_trace(X, y, lr=0.5, steps=3)
