"""Classical machine learning — the algorithms that predate (and still outlive) deep nets.

Every estimator here follows the same shape as the rest of OptimumAI: a
``<name>_trace(...)`` function builds a full step-by-step
:class:`~optimumai.core.trace.Trace` with real numbers, and a thin class
(``LinearRegression``, ``KMeans``, ...) wraps it in the familiar
``fit``/``predict`` interface.
"""

from optimumai.ml.decision_tree import (
    DecisionTree,
    decision_tree_trace,
    entropy_impurity,
    gini_impurity,
)
from optimumai.ml.kmeans import KMeans, kmeans_trace
from optimumai.ml.knn import KNN, knn_trace
from optimumai.ml.linear_regression import LinearRegression, linear_regression_trace
from optimumai.ml.logistic_regression import LogisticRegression, logistic_regression_trace
from optimumai.ml.metrics import (
    accuracy,
    accuracy_trace,
    confusion_matrix,
    confusion_matrix_trace,
    mse,
    mse_trace,
    precision_recall_f1,
    precision_recall_f1_trace,
    r2_score,
    r2_score_trace,
    roc_auc,
    roc_auc_trace,
)
from optimumai.ml.naive_bayes import GaussianNB, naive_bayes_trace
from optimumai.ml.pca import PCA, pca, pca_trace

__all__ = [
    "PCA",
    "DecisionTree",
    "GaussianNB",
    "KMeans",
    "KNN",
    "LinearRegression",
    "LogisticRegression",
    "accuracy",
    "accuracy_trace",
    "confusion_matrix",
    "confusion_matrix_trace",
    "decision_tree_trace",
    "entropy_impurity",
    "gini_impurity",
    "kmeans_trace",
    "knn_trace",
    "linear_regression_trace",
    "logistic_regression_trace",
    "mse",
    "mse_trace",
    "naive_bayes_trace",
    "pca",
    "pca_trace",
    "precision_recall_f1",
    "precision_recall_f1_trace",
    "r2_score",
    "r2_score_trace",
    "roc_auc",
    "roc_auc_trace",
]
