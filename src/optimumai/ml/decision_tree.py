"""Decision trees — repeatedly asking the single yes/no question that helps most.

A decision tree classifies by asking a sequence of threshold questions
("is feature 0 ≤ 1.5?") and following the branch that matches, until it
reaches a leaf holding a class prediction. Building the tree means choosing,
at every node, *which* question to ask — the one that splits the remaining
points into the purest possible children.

"Purity" is measured with an **impurity** score computed from the class
proportions ``pᵢ`` at a node. Two standard choices:

    Gini:     ``G = 1 − Σᵢ pᵢ²``           (0 = pure, higher = more mixed)
    Entropy:  ``H = −Σᵢ pᵢ log₂(pᵢ)``      (0 = pure, higher = more mixed)

Both are 0 exactly when a node holds only one class and largest when classes
are evenly mixed; they usually pick the same splits in practice, and Gini is
the common default because it avoids the log.

To find the best split at a node, the algorithm tries every feature and, for
each feature, every midpoint between consecutive sorted values as a candidate
threshold. For each candidate it computes the impurity of the two resulting
children, weighted by how many points land in each, and keeps the split with
the largest **information gain**:

    ``gain = impurity(parent) − [ (n_left/n)·impurity(left) + (n_right/n)·impurity(right) ]``

The tree recurses on each child, stopping at ``max_depth``, a pure node, or a
node too small to split — this implementation stays shallow on purpose: an
unpruned tree memorizes the training set (impurity always hits 0 by isolating
single points) and generalizes badly.

Why AI cares
------------
A single decision tree is rarely the final model, but it is the building
block of the strongest tabular-data methods in production: **random forests**
average many trees grown on random subsets, and **gradient boosting**
(XGBoost, LightGBM) grows trees sequentially to correct the previous
ensemble's errors. Trees are also one of the few models whose reasoning is
directly human-readable — "if income > $50k and age > 30, predict approve" —
which matters wherever a prediction needs to be explained.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from optimumai.core._fmt import arr, num
from optimumai.core.explain import ExplainLevel
from optimumai.core.trace import Trace


def _class_counts(y: np.ndarray) -> dict[int, int]:
    labels, counts = np.unique(y, return_counts=True)
    return {int(label): int(count) for label, count in zip(labels, counts, strict=True)}


def gini_impurity(y: np.ndarray) -> float:
    """Gini impurity ``1 − Σ pᵢ²`` of the labels in ``y``."""
    y = np.asarray(y)
    if y.size == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    probs = counts / counts.sum()
    return float(1.0 - np.sum(probs**2))


def entropy_impurity(y: np.ndarray) -> float:
    """Shannon entropy ``−Σ pᵢ log₂(pᵢ)`` of the labels in ``y``."""
    y = np.asarray(y)
    if y.size == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


_CRITERIA = {"gini": gini_impurity, "entropy": entropy_impurity}


def _majority_label(y: np.ndarray) -> int:
    labels, counts = np.unique(y, return_counts=True)
    return int(labels[np.argmax(counts)])


@dataclass
class _Node:
    """One node of the fitted tree: either a leaf or an internal split."""

    is_leaf: bool
    prediction: int | None = None
    feature: int | None = None
    threshold: float | None = None
    left: _Node | None = None
    right: _Node | None = None


def _best_split(
    X: np.ndarray, y: np.ndarray, criterion: str
) -> tuple[int, float, float] | None:
    """Search every feature/threshold candidate; return ``(feature, threshold, gain)``."""
    impurity_fn = _CRITERIA[criterion]
    parent_impurity = impurity_fn(y)
    n = len(y)
    best: tuple[int, float, float] | None = None

    for feature in range(X.shape[1]):
        values = np.unique(X[:, feature])
        thresholds = (values[:-1] + values[1:]) / 2.0
        for threshold in thresholds:
            left_mask = X[:, feature] <= threshold
            n_left, n_right = int(left_mask.sum()), int((~left_mask).sum())
            if n_left == 0 or n_right == 0:
                continue
            weighted = (
                n_left / n * impurity_fn(y[left_mask])
                + n_right / n * impurity_fn(y[~left_mask])
            )
            gain = parent_impurity - weighted
            if best is None or gain > best[2]:
                best = (feature, float(threshold), float(gain))
    return best


def decision_tree_trace(
    X, y, max_depth: int = 2, criterion: str = "gini"
) -> Trace:
    """Build a trace of the best-split search at the root, then grow a shallow tree.

    Args:
        X: Features, shape ``(n, d)`` (or ``(n,)``).
        y: Integer class labels, shape ``(n,)``.
        max_depth: Maximum depth of the fitted tree.
        criterion: ``"gini"`` or ``"entropy"``.
    """
    Xmat = np.asarray(X, dtype=float)
    if Xmat.ndim == 1:
        Xmat = Xmat.reshape(-1, 1)
    yvec = np.asarray(y).reshape(-1)
    if Xmat.shape[0] != yvec.shape[0]:
        raise ValueError(f"X has {Xmat.shape[0]} rows but y has {yvec.shape[0]} entries")
    if criterion not in _CRITERIA:
        raise ValueError(f"criterion must be one of {list(_CRITERIA)}, got {criterion!r}")
    if max_depth < 1:
        raise ValueError(f"max_depth must be >= 1, got {max_depth}")
    if Xmat.shape[0] < 2:
        raise ValueError("need at least 2 samples to consider a split")

    impurity_fn = _CRITERIA[criterion]
    n, d = Xmat.shape

    t = Trace(
        op="decision_tree",
        formula=(
            "gini = 1 − Σpᵢ²  (or entropy = −Σpᵢlog₂pᵢ);  "
            "gain = impurity(parent) − Σ (nₖ/n)·impurity(child_k)"
        ),
        complexity=f"O(n·d·log n) per split search, max_depth={max_depth}, n={n}, d={d}",
        why_ai=[
            "The building block of random forests (bag many trees) and "
            "gradient-boosted trees (XGBoost, LightGBM) — the strongest "
            "methods on tabular data",
            "One of the few models whose reasoning is directly human-readable: "
            "'if income > $50k, predict approve'",
            "Unpruned trees hit zero training impurity by isolating single "
            "points, which is exactly why they overfit and are kept shallow "
            "or ensembled",
        ],
        meta={"n_samples": n, "n_features": d, "max_depth": max_depth, "criterion": criterion},
    )

    root_impurity = impurity_fn(yvec)
    t.add(
        f"Root {criterion} impurity",
        f"{criterion}(y) = {num(root_impurity)}  from class counts {_class_counts(yvec)}",
        root_impurity,
        detail="0 would mean the node is already pure (single class).",
    )

    # Show the candidate-threshold search explicitly for the root split.
    for feature in range(d):
        values = np.unique(Xmat[:, feature])
        thresholds = (values[:-1] + values[1:]) / 2.0
        if thresholds.size == 0:
            t.add(
                f"Feature {feature}: no candidate thresholds",
                "all values identical — this feature cannot split the data",
            )
            continue
        rows = []
        for threshold in thresholds:
            left_mask = Xmat[:, feature] <= threshold
            n_left, n_right = int(left_mask.sum()), int((~left_mask).sum())
            weighted = (
                n_left / n * impurity_fn(yvec[left_mask])
                + n_right / n * impurity_fn(yvec[~left_mask])
            )
            gain = root_impurity - weighted
            rows.append(
                f"thr={num(threshold)}: weighted_impurity={num(weighted)}, gain={num(gain)}"
            )
        t.add(
            f"Feature {feature}: candidate thresholds",
            "; ".join(rows),
            detail=f"Midpoints between consecutive sorted values of feature {feature}: "
            f"{arr(thresholds)}.",
        )

    best = _best_split(Xmat, yvec, criterion)
    if best is None:
        raise ValueError("no valid split found — all points may be identical")
    best_feature, best_threshold, best_gain = best
    t.add(
        "Chosen root split (max information gain)",
        f"feature {best_feature} <= {num(best_threshold)}  →  gain = {num(best_gain)}",
        {"feature": best_feature, "threshold": best_threshold, "gain": best_gain},
        detail="The split with the largest reduction in weighted child impurity wins.",
    )

    root = _grow(Xmat, yvec, depth=1, max_depth=max_depth, criterion=criterion)
    preds = np.array([_predict_one(root, row) for row in Xmat])
    accuracy = float(np.mean(preds == yvec))
    t.add(
        "Fitted tree predictions (training set)",
        f"ŷ = {arr(preds)}, accuracy = {num(accuracy)}",
        preds,
        detail=f"True labels y = {arr(yvec)}",
    )

    t.result = preds
    t.meta["tree"] = root
    t.meta["accuracy"] = accuracy
    return t


def _grow(X: np.ndarray, y: np.ndarray, depth: int, max_depth: int, criterion: str) -> _Node:
    """Recursively build the tree (used internally; the trace only shows the root)."""
    if len(np.unique(y)) == 1 or depth >= max_depth or len(y) < 2:
        return _Node(is_leaf=True, prediction=_majority_label(y))

    best = _best_split(X, y, criterion)
    if best is None:
        return _Node(is_leaf=True, prediction=_majority_label(y))

    feature, threshold, _gain = best
    left_mask = X[:, feature] <= threshold
    left = _grow(X[left_mask], y[left_mask], depth + 1, max_depth, criterion)
    right = _grow(X[~left_mask], y[~left_mask], depth + 1, max_depth, criterion)
    return _Node(is_leaf=False, feature=feature, threshold=threshold, left=left, right=right)


def _predict_one(node: _Node, x: np.ndarray) -> int:
    while not node.is_leaf:
        assert node.feature is not None and node.threshold is not None and node.left and node.right
        node = node.left if x[node.feature] <= node.threshold else node.right
    assert node.prediction is not None
    return node.prediction


class DecisionTree:
    """A shallow classification tree grown by greedy impurity-reduction splits.

    Example:
        >>> model = DecisionTree(max_depth=1).fit([[0], [1], [10], [11]], [0, 0, 1, 1])
        >>> model.predict([[0.5], [10.5]])
        array([0, 1])
    """

    def __init__(self, max_depth: int = 2, criterion: str = "gini") -> None:
        self.max_depth = max_depth
        self.criterion = criterion
        self.root: _Node | None = None

    def fit(self, X, y) -> DecisionTree:
        """Grow the tree greedily, splitting on maximum information gain."""
        t = decision_tree_trace(X, y, max_depth=self.max_depth, criterion=self.criterion)
        self.root = t.meta["tree"]
        return self

    def predict(self, X) -> np.ndarray:
        """Route each row of ``X`` through the fitted tree to a leaf prediction."""
        if self.root is None:
            raise ValueError("model is not fitted yet — call fit(X, y) first")
        Xmat = np.asarray(X, dtype=float)
        if Xmat.ndim == 1:
            Xmat = Xmat.reshape(-1, 1)
        return np.array([_predict_one(self.root, row) for row in Xmat])

    def explain(self, X, y, level: str | ExplainLevel = ExplainLevel.INTERMEDIATE) -> np.ndarray:
        """Fit, print the root-split search trace, and return training predictions."""
        t = decision_tree_trace(X, y, max_depth=self.max_depth, criterion=self.criterion)
        self.root = t.meta["tree"]
        return t.render(level)


def demo(seed: int = 0) -> Trace:
    """Split a small 2-class, 2-feature set by the single best root threshold."""
    X = np.array([[0.0, 5.0], [1.0, 4.0], [1.5, 6.0], [5.0, 1.0], [6.0, 0.5], [5.5, 2.0]])
    y = np.array([0, 0, 0, 1, 1, 1])
    return decision_tree_trace(X, y, max_depth=2)
