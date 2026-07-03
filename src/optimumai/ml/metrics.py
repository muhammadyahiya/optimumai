"""Evaluation metrics — turning predictions into a single trustworthy number.

A model's raw predictions are not, by themselves, an answer to "is this
model good?" — that requires comparing predictions to ground truth with a
metric chosen for what actually matters in the task. This module builds a
:class:`~optimumai.core.trace.Trace` for each of the standard ones so the
arithmetic behind a reported number is never a mystery.

Classification
--------------
* **Confusion matrix** — counts of (true label, predicted label) pairs; every
  other classification metric here is read off of it.
* **Accuracy** — ``(TP + TN) / total``: fraction of predictions that were
  correct. Misleading on imbalanced data (predicting "not fraud" every time
  can be 99% accurate).
* **Precision** — ``TP / (TP + FP)``: of everything flagged positive, how
  much really was. High precision means few false alarms.
* **Recall** — ``TP / (TP + FN)``: of everything that really was positive,
  how much got flagged. High recall means few misses.
* **F1** — ``2 · precision · recall / (precision + recall)``, the harmonic
  mean of the two; harmonic (not arithmetic) mean so F1 stays low unless
  *both* precision and recall are reasonably high.
* **ROC-AUC** — sweep every possible decision threshold on a model's scores;
  at each threshold measure the true-positive rate vs. false-positive rate.
  AUC is the probability a random positive example scores higher than a
  random negative one — a threshold-free view of ranking quality (computed
  here via the rank-sum formula, equivalent to but faster than integrating
  the ROC curve).

Regression
----------
* **MSE** — ``mean((y − ŷ)²)``: average squared error, the same loss
  :mod:`linear regression <optimumai.ml.linear_regression>` minimizes.
* **R²** — ``1 − SSres/SStot``: fraction of the target's variance the model
  explains, where 1.0 is a perfect fit and 0.0 matches "always predict the
  mean."

Why AI cares
------------
Every model, however it was trained, gets judged by one of these numbers
before it ships. Precision/recall/F1 and ROC-AUC in particular are what
separate a model that merely "looks accurate" from one that actually does
its job on the class you care about (fraud, disease, spam).
"""

from __future__ import annotations

import numpy as np

from optimumai.core._fmt import arr, num
from optimumai.core.trace import Trace


def _as_arrays(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    yt = np.asarray(y_true).reshape(-1)
    yp = np.asarray(y_pred).reshape(-1)
    if yt.shape[0] != yp.shape[0]:
        raise ValueError(f"y_true has {yt.shape[0]} entries but y_pred has {yp.shape[0]}")
    if yt.shape[0] == 0:
        raise ValueError("y_true/y_pred must not be empty")
    return yt, yp


# --------------------------------------------------------------------- accuracy
def accuracy_trace(y_true, y_pred) -> Trace:
    """Build the trace of accuracy = fraction of exactly-correct predictions."""
    yt, yp = _as_arrays(y_true, y_pred)
    t = Trace(
        op="accuracy",
        formula="accuracy = (# correct) / (# total)",
        complexity="O(n)",
        why_ai=[
            "The simplest classification metric, and the most easily misleading "
            "one on imbalanced classes",
            "Always report it alongside precision/recall on skewed data",
        ],
        meta={"n": yt.shape[0]},
    )
    correct = yt == yp
    n_correct = int(np.sum(correct))
    acc = n_correct / yt.shape[0]
    t.add(
        "Compare predictions to truth",
        f"correct = {arr(correct.astype(int))}",
        correct,
        detail=f"y_true = {arr(yt)}, y_pred = {arr(yp)}",
    )
    t.add(
        "Accuracy",
        f"{n_correct} / {yt.shape[0]} = {num(acc)}",
        acc,
    )
    t.result = acc
    return t


def accuracy(y_true, y_pred) -> float:
    """Fraction of predictions that exactly match the true labels."""
    return accuracy_trace(y_true, y_pred).result


# ------------------------------------------------------------- confusion matrix
def confusion_matrix_trace(y_true, y_pred, labels: list[int] | None = None) -> Trace:
    """Build the trace of the confusion matrix (rows = true, columns = predicted)."""
    yt, yp = _as_arrays(y_true, y_pred)
    label_list = sorted(set(yt.tolist()) | set(yp.tolist())) if labels is None else list(labels)
    k = len(label_list)
    index = {label: i for i, label in enumerate(label_list)}

    t = Trace(
        op="confusion_matrix",
        formula="M[i, j] = # examples with true label i and predicted label j",
        complexity="O(n)",
        why_ai=[
            "Every classification metric here (precision, recall, F1) is "
            "read directly off this table",
            "The diagonal is correct predictions; off-diagonal cells show "
            "exactly which classes get confused for which",
        ],
        meta={"labels": label_list},
    )

    matrix = np.zeros((k, k), dtype=int)
    for true_label, pred_label in zip(yt.tolist(), yp.tolist(), strict=True):
        matrix[index[true_label], index[pred_label]] += 1

    t.add(
        "Tally (true label, predicted label) pairs",
        f"labels order = {label_list}\nM =\n{arr(matrix)}",
        matrix,
        detail="Row i = true label; column j = predicted label; M[i, j] is the count.",
    )
    t.result = matrix
    return t


def confusion_matrix(y_true, y_pred, labels: list[int] | None = None) -> np.ndarray:
    """Confusion matrix (rows = true label, columns = predicted label)."""
    return confusion_matrix_trace(y_true, y_pred, labels=labels).result


# ---------------------------------------------------------- precision/recall/f1
def precision_recall_f1_trace(y_true, y_pred, positive_label: int = 1) -> Trace:
    """Build the trace of binary precision, recall, and F1 for ``positive_label``."""
    yt, yp = _as_arrays(y_true, y_pred)
    t = Trace(
        op="precision_recall_f1",
        formula="precision = TP/(TP+FP);  recall = TP/(TP+FN);  "
        "F1 = 2·precision·recall/(precision+recall)",
        complexity="O(n)",
        why_ai=[
            "Precision answers 'of what I flagged, how much was real?'; "
            "recall answers 'of what was real, how much did I catch?'",
            "F1's harmonic mean punishes models that trade one off entirely "
            "for the other (e.g. flagging everything to get 100% recall)",
        ],
        meta={"positive_label": positive_label, "n": yt.shape[0]},
    )

    tp = int(np.sum((yt == positive_label) & (yp == positive_label)))
    fp = int(np.sum((yt != positive_label) & (yp == positive_label)))
    fn = int(np.sum((yt == positive_label) & (yp != positive_label)))
    tn = int(np.sum((yt != positive_label) & (yp != positive_label)))
    t.add(
        "Count TP / FP / FN / TN",
        f"TP={tp}, FP={fp}, FN={fn}, TN={tn}",
        {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        detail=f"Positive class = {positive_label}.",
    )

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    t.add("Precision", f"TP/(TP+FP) = {tp}/{tp + fp} = {num(precision)}", precision)

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    t.add("Recall", f"TP/(TP+FN) = {tp}/{tp + fn} = {num(recall)}", recall)

    f1 = (
        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    )
    t.add(
        "F1 (harmonic mean of precision and recall)",
        f"2·{num(precision)}·{num(recall)} / ({num(precision)}+{num(recall)}) = {num(f1)}",
        f1,
    )

    t.result = {"precision": precision, "recall": recall, "f1": f1}
    return t


def precision_recall_f1(y_true, y_pred, positive_label: int = 1) -> dict[str, float]:
    """Return ``{"precision": ..., "recall": ..., "f1": ...}`` for ``positive_label``."""
    return precision_recall_f1_trace(y_true, y_pred, positive_label=positive_label).result


# -------------------------------------------------------------------------- mse
def mse_trace(y_true, y_pred) -> Trace:
    """Build the trace of mean squared error."""
    yt, yp = _as_arrays(y_true, y_pred)
    t = Trace(
        op="mse",
        formula="MSE = mean((y − ŷ)²)",
        complexity="O(n)",
        why_ai=[
            "The loss linear regression (and many regressors) directly minimizes",
            "Squaring penalizes large errors far more than small ones — a single "
            "bad miss dominates the score",
        ],
        meta={"n": yt.shape[0]},
    )
    residuals = yt - yp
    squared = residuals**2
    t.add(
        "Squared residuals",
        f"(y − ŷ)² = {arr(squared)}",
        squared,
        detail=f"residuals = {arr(residuals)}",
    )
    mse_value = float(np.mean(squared))
    t.add("Mean", f"mean = {num(mse_value)}", mse_value)
    t.result = mse_value
    return t


def mse(y_true, y_pred) -> float:
    """Mean squared error between ``y_true`` and ``y_pred``."""
    return mse_trace(y_true, y_pred).result


# --------------------------------------------------------------------------- r2
def r2_score_trace(y_true, y_pred) -> Trace:
    """Build the trace of the R² (coefficient of determination) score."""
    yt, yp = _as_arrays(y_true, y_pred)
    t = Trace(
        op="r2_score",
        formula="R² = 1 − SSres/SStot,  SSres = Σ(y−ŷ)²,  SStot = Σ(y−mean(y))²",
        complexity="O(n)",
        why_ai=[
            "Answers 'how much better than just predicting the mean is this "
            "model?' — 1.0 is perfect, 0.0 ties the mean-only baseline",
            "Unit-free, so it is comparable across targets with different scales",
        ],
        meta={"n": yt.shape[0]},
    )
    ss_res = float(np.sum((yt - yp) ** 2))
    t.add("Residual sum of squares", f"SSres = Σ(y−ŷ)² = {num(ss_res)}", ss_res)
    mean_y = float(np.mean(yt))
    ss_tot = float(np.sum((yt - mean_y) ** 2))
    t.add(
        "Total sum of squares",
        f"SStot = Σ(y−mean(y))² = {num(ss_tot)}",
        ss_tot,
        detail=f"mean(y) = {num(mean_y)}",
    )
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    t.add("R²", f"1 − {num(ss_res)}/{num(ss_tot)} = {num(r2)}", r2)
    t.result = r2
    return t


def r2_score(y_true, y_pred) -> float:
    """R² (coefficient of determination) between ``y_true`` and ``y_pred``."""
    return r2_score_trace(y_true, y_pred).result


# ---------------------------------------------------------------------- roc_auc
def roc_auc_trace(y_true, y_scores) -> Trace:
    """Build the trace of ROC-AUC via the Mann-Whitney rank-sum formula.

    ``y_true`` must be binary (0/1); ``y_scores`` are the model's continuous
    scores or probabilities for the positive class.
    """
    yt = np.asarray(y_true).reshape(-1)
    scores = np.asarray(y_scores, dtype=float).reshape(-1)
    if yt.shape[0] != scores.shape[0]:
        raise ValueError(f"y_true has {yt.shape[0]} entries but y_scores has {scores.shape[0]}")
    if not np.all(np.isin(yt, [0, 1])):
        raise ValueError("y_true must be binary (0/1) for ROC-AUC")
    n_pos = int(np.sum(yt == 1))
    n_neg = int(np.sum(yt == 0))
    if n_pos == 0 or n_neg == 0:
        raise ValueError("need at least one positive and one negative example for ROC-AUC")

    t = Trace(
        op="roc_auc",
        formula="AUC = P(score(random positive) > score(random negative)),  "
        "computed via rank-sum: AUC = (Σranks(positives) − n_pos(n_pos+1)/2) / (n_pos·n_neg)",
        complexity=f"O(n log n) to rank {yt.shape[0]} scores",
        why_ai=[
            "A threshold-free view of ranking quality — unlike accuracy, it "
            "doesn't depend on where you draw the 0.5 cutoff",
            "0.5 means the model ranks positives no better than a coin flip; "
            "1.0 means every positive outranks every negative",
        ],
        meta={"n_pos": n_pos, "n_neg": n_neg},
    )

    order = np.argsort(scores, kind="stable")
    ranks = np.empty(scores.shape[0])
    ranks[order] = np.arange(1, scores.shape[0] + 1)
    # average tied ranks
    for value in np.unique(scores):
        tie_mask = scores == value
        if np.sum(tie_mask) > 1:
            ranks[tie_mask] = ranks[tie_mask].mean()
    t.add(
        "Rank every score (ties get the average rank)",
        f"scores = {arr(scores)}\nranks = {arr(ranks)}",
        ranks,
        detail=f"Labels y = {arr(yt)}.",
    )

    rank_sum_pos = float(np.sum(ranks[yt == 1]))
    t.add(
        "Sum of ranks among positive examples",
        f"Σ ranks(y=1) = {num(rank_sum_pos)}",
        rank_sum_pos,
    )

    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    t.add(
        "AUC via the rank-sum (Mann-Whitney U) formula",
        f"(Σranks_pos − n_pos(n_pos+1)/2) / (n_pos·n_neg) "
        f"= ({num(rank_sum_pos)} − {n_pos}·{n_pos + 1}/2) / ({n_pos}·{n_neg}) = {num(auc)}",
        auc,
        detail="Equivalent to (but much faster than) integrating the ROC curve "
        "over every possible threshold.",
    )

    t.result = auc
    return t


def roc_auc(y_true, y_scores) -> float:
    """ROC-AUC for binary labels ``y_true`` given continuous ``y_scores``."""
    return roc_auc_trace(y_true, y_scores).result


def demo(seed: int = 0) -> Trace:
    """Score a small binary classifier's predictions with accuracy + precision/recall/F1."""
    y_true = [0, 0, 1, 1, 1]
    y_pred = [0, 1, 1, 1, 0]
    return precision_recall_f1_trace(y_true, y_pred)
