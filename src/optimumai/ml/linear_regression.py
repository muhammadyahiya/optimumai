"""Linear regression — fitting a straight line (or hyperplane) by least squares.

The model predicts ``ŷ = Xθ``: a weighted sum of features plus a bias. "Fitting"
means choosing the weights ``θ`` that make the squared prediction error as small
as possible. Because the squared-error loss is a smooth bowl (convex quadratic)
in ``θ``, calculus gives a closed-form answer instead of an iterative search:
set the gradient of the loss to zero and solve.

Loss:  ``L(θ) = (1/n) ‖Xθ − y‖²``
Gradient:  ``∂L/∂θ = (2/n) Xᵀ(Xθ − y)``

Setting the gradient to zero and solving for ``θ`` gives the **normal
equation**:

    ``XᵀXθ = Xᵀy   ⟹   θ = (XᵀX)⁻¹ Xᵀy``

A column of ones is appended to ``X`` so the first entry of ``θ`` is the
intercept (bias) — geometrically, it lets the fitted line/plane sit anywhere,
not just through the origin.

Normal equation vs. gradient descent
-------------------------------------
The normal equation is exact and needs no learning rate, but it inverts a
``(d+1) × (d+1)`` matrix — ``O(d³)`` — and needs the whole dataset in memory at
once. When ``d`` is large (millions of features, as in some text/embedding
models) or the data streams in, models instead take small steps down the same
loss surface with **gradient descent**: ``θ ← θ − η · ∂L/∂θ``, repeated until
convergence. Both roads reach the same bottom of the same bowl; the normal
equation just jumps straight there. See
:mod:`optimumai.optimization.optimizers` for the iterative version and
:mod:`optimumai.ml.logistic_regression` for a model where no closed form
exists and gradient descent is the only option.

Why AI cares
------------
Linear regression is the simplest supervised learner and the scaffolding for
much of the rest: logistic regression is linear regression pushed through a
sigmoid, a single-layer neural net with no activation *is* linear regression,
and ``R²`` / MSE reappear as evaluation metrics everywhere.
"""

from __future__ import annotations

import numpy as np

from optimumai.core._fmt import arr, num
from optimumai.core.explain import ExplainLevel
from optimumai.core.trace import Trace


def _design_matrix(X: np.ndarray) -> np.ndarray:
    """Prepend a column of ones to ``X`` so the bias is just another weight."""
    ones = np.ones((X.shape[0], 1))
    return np.hstack([ones, X])


def _as_2d(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    return X.reshape(-1, 1) if X.ndim == 1 else X


def linear_regression_trace(X, y) -> Trace:
    """Build the full normal-equation trace fitting ``ŷ = Xθ`` to ``(X, y)``."""
    Xmat = _as_2d(X)
    yvec = np.asarray(y, dtype=float).reshape(-1)
    if Xmat.ndim != 2:
        raise ValueError(f"X must be 1-D or 2-D, got shape {np.asarray(X).shape}")
    if Xmat.shape[0] != yvec.shape[0]:
        raise ValueError(f"X has {Xmat.shape[0]} rows but y has {yvec.shape[0]} entries")
    if Xmat.shape[0] < Xmat.shape[1] + 1:
        raise ValueError(
            f"need at least {Xmat.shape[1] + 1} samples for {Xmat.shape[1]} features "
            f"+ bias, got {Xmat.shape[0]}"
        )

    n, d = Xmat.shape
    t = Trace(
        op="linear_regression",
        formula="θ = (XᵀX)⁻¹Xᵀy;  ŷ = Xθ",
        complexity=f"O(nd² + d³) for n={n} samples, d={d} features (dominated by inverting XᵀX)",
        why_ai=[
            "The baseline supervised learner — everything from logistic regression to "
            "a linear layer is a variation on ŷ = Xθ",
            "The normal equation is calculus in closed form: it is the exact point "
            "gradient descent would eventually reach on this convex loss",
            "R² and MSE, computed here, are the standard regression metrics used to "
            "evaluate any model, not just this one",
        ],
        meta={"n_samples": n, "n_features": d},
    )

    Xb = _design_matrix(Xmat)
    t.add(
        "Design matrix (prepend bias column)",
        f"X → [1 | X], shape {Xmat.shape} → {Xb.shape}\n{arr(Xb)}",
        Xb,
        detail="The leading column of 1s lets θ₀ act as the intercept: ŷ = θ₀·1 + θ₁x₁ + ...",
    )

    XtX = Xb.T @ Xb
    t.add(
        "XᵀX (the Gram matrix)",
        f"XᵀX =\n{arr(XtX)}",
        XtX,
        detail="Symmetric (d+1)×(d+1) matrix; must be invertible for a unique solution "
        "(no redundant/collinear features).",
    )

    Xty = Xb.T @ yvec
    t.add("Xᵀy", f"Xᵀy = {arr(Xty)}", Xty)

    try:
        XtX_inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "XᵀX is singular — features are perfectly collinear; cannot solve the "
            "normal equation"
        ) from exc
    theta = XtX_inv @ Xty
    t.add(
        "Solve the normal equation",
        f"θ = (XᵀX)⁻¹Xᵀy = {arr(theta)}",
        theta,
        detail=f"θ₀ = {num(theta[0])} is the intercept; θ₁.. are the feature weights.",
    )

    preds = Xb @ theta
    t.add(
        "Predictions",
        f"ŷ = Xθ = {arr(preds)}",
        preds,
        detail=f"Actual y = {arr(yvec)}",
    )

    residuals = yvec - preds
    mse = float(np.mean(residuals**2))
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((yvec - np.mean(yvec)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    t.add(
        "Mean squared error + R²",
        f"MSE = mean((y − ŷ)²) = {num(mse)};  R² = 1 − SSres/SStot = {num(r2)}",
        {"mse": mse, "r2": r2},
        detail="R² = 1 means a perfect fit; R² = 0 means no better than predicting the mean.",
    )

    t.result = preds
    t.meta["theta"] = theta
    t.meta["mse"] = mse
    t.meta["r2"] = r2
    return t


class LinearRegression:
    """Ordinary least squares via the normal equation.

    Example:
        >>> model = LinearRegression().fit([[1], [2], [3]], [2, 4, 6])
        >>> model.predict([[4]])
        array([8.])
    """

    def __init__(self) -> None:
        self.theta: np.ndarray | None = None

    def fit(self, X, y) -> LinearRegression:
        """Fit ``θ`` by the normal equation and store it on the model."""
        t = linear_regression_trace(X, y)
        self.theta = t.meta["theta"]
        return self

    def predict(self, X) -> np.ndarray:
        """Predict ``ŷ = Xθ`` for new samples (must call :meth:`fit` first)."""
        if self.theta is None:
            raise ValueError("model is not fitted yet — call fit(X, y) first")
        Xb = _design_matrix(_as_2d(X))
        if Xb.shape[1] != self.theta.shape[0]:
            raise ValueError(
                f"expected {self.theta.shape[0] - 1} features, got {Xb.shape[1] - 1}"
            )
        return Xb @ self.theta

    def explain(
        self, X, y, level: str | ExplainLevel = ExplainLevel.INTERMEDIATE
    ) -> np.ndarray:
        """Fit, print the full trace, and return the training predictions."""
        t = linear_regression_trace(X, y)
        self.theta = t.meta["theta"]
        return t.render(level)


def demo(seed: int = 0) -> Trace:
    """Fit ``y = 2x + 1`` (plus tiny noise) — the coefficients should come back ~[1, 2]."""
    rng = np.random.default_rng(seed)
    X = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    noise = rng.normal(0, 0.05, size=X.shape)
    y = 2.0 * X + 1.0 + noise
    return linear_regression_trace(X, y)
