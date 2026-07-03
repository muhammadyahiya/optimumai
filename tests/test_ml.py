import numpy as np
import pytest

from optimumai.ml.decision_tree import (
    DecisionTree,
    decision_tree_trace,
    entropy_impurity,
    gini_impurity,
)
from optimumai.ml.decision_tree import demo as decision_tree_demo
from optimumai.ml.kmeans import KMeans, kmeans_trace
from optimumai.ml.kmeans import demo as kmeans_demo
from optimumai.ml.knn import KNN, knn_trace
from optimumai.ml.knn import demo as knn_demo
from optimumai.ml.linear_regression import LinearRegression, linear_regression_trace
from optimumai.ml.linear_regression import demo as linear_regression_demo
from optimumai.ml.logistic_regression import LogisticRegression, logistic_regression_trace
from optimumai.ml.logistic_regression import demo as logistic_regression_demo
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
from optimumai.ml.metrics import demo as metrics_demo
from optimumai.ml.naive_bayes import GaussianNB, naive_bayes_trace
from optimumai.ml.naive_bayes import demo as naive_bayes_demo
from optimumai.ml.pca import PCA, pca, pca_trace
from optimumai.ml.pca import demo as pca_demo

# --- Linear regression --------------------------------------------------------


def test_linear_regression_recovers_known_coefficients():
    # y = 2x + 1 exactly (no noise) -> normal equation must recover [1, 2] exactly.
    X = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    y = 2.0 * X + 1.0
    model = LinearRegression().fit(X, y)
    assert model.theta is not None
    assert np.allclose(model.theta, [1.0, 2.0], atol=1e-8)
    assert np.allclose(model.predict(np.array([5.0])), [11.0], atol=1e-8)


def test_linear_regression_matches_numpy_lstsq_reference():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 3))
    true_theta = np.array([0.5, -1.2, 2.0, 3.0])  # bias, w1, w2, w3
    Xb = np.hstack([np.ones((20, 1)), X])
    y = Xb @ true_theta
    theta_ref, *_ = np.linalg.lstsq(Xb, y, rcond=None)
    model = LinearRegression().fit(X, y)
    assert np.allclose(model.theta, theta_ref, atol=1e-6)


def test_linear_regression_trace_shape_and_metrics():
    t = linear_regression_demo()
    # design matrix, XtX, Xty, solve, predictions, mse+r2 = 6 steps
    assert len(t) == 6
    assert t.result is not None
    assert t.why_ai
    assert t.formula
    assert t.meta["r2"] > 0.99  # near-perfect fit to a near-linear signal


def test_linear_regression_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        linear_regression_trace([1.0, 2.0, 3.0], [1.0, 2.0])


def test_linear_regression_predict_before_fit_raises():
    with pytest.raises(ValueError):
        LinearRegression().predict([[1.0]])


# --- Logistic regression -------------------------------------------------------


def test_logistic_regression_classifies_separable_set():
    X = np.array([0.0, 0.5, 1.0, 5.0, 5.5, 6.0])
    y = np.array([0, 0, 0, 1, 1, 1])
    model = LogisticRegression(lr=0.5, steps=300).fit(X, y)
    preds = model.predict(X)
    assert np.array_equal(preds, y)


def test_logistic_regression_gradient_matches_independent_reference():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    Xb = np.hstack([np.ones((4, 1)), X])
    theta = np.zeros(2)
    p = 1.0 / (1.0 + np.exp(-(Xb @ theta)))
    grad_ref = Xb.T @ (p - y) / 4

    t = logistic_regression_trace(X, y, lr=0.5, steps=1)
    grad_step = t.steps[2]  # forward, loss, then first gradient step
    assert np.allclose(grad_step.value, grad_ref)


def test_logistic_regression_trace_has_why_ai_and_result():
    t = logistic_regression_demo()
    assert t.result is not None
    assert t.why_ai
    assert t.formula
    assert "accuracy" in t.meta


def test_logistic_regression_rejects_non_binary_labels():
    with pytest.raises(ValueError):
        logistic_regression_trace([0.0, 1.0, 2.0], [0, 1, 2])


def test_logistic_regression_rejects_bad_lr():
    with pytest.raises(ValueError):
        logistic_regression_trace([0.0, 1.0], [0, 1], lr=0.0)


# --- k-means --------------------------------------------------------------------


def test_kmeans_finds_separated_clusters():
    X = np.array([0.0, 0.5, 1.0, 20.0, 20.5, 21.0])
    model = KMeans(k=2).fit(X)
    labels = model.labels_
    assert labels is not None
    # the three low points share a label, the three high points share the other
    assert len(set(labels[:3].tolist())) == 1
    assert len(set(labels[3:].tolist())) == 1
    assert labels[0] != labels[3]


def test_kmeans_inertia_matches_independent_reference():
    X = np.array([[0.0], [1.0], [10.0], [11.0]])
    t = kmeans_trace(X, k=2)
    centroids = t.meta["centroids"]
    labels = t.result
    inertia_ref = float(np.sum((X - centroids[labels]) ** 2))
    assert t.meta["inertia"] == pytest.approx(inertia_ref)


def test_kmeans_predict_assigns_nearest_centroid():
    model = KMeans(k=2).fit([[0.0], [1.0], [10.0], [11.0]])
    preds = model.predict([[0.5], [10.5]])
    assert preds[0] != preds[1]


def test_kmeans_demo_trace_has_result_and_why_ai():
    t = kmeans_demo()
    assert t.result is not None
    assert t.why_ai
    assert t.formula


def test_kmeans_rejects_k_greater_than_n():
    with pytest.raises(ValueError):
        kmeans_trace([1.0, 2.0], k=5)


# --- k-NN -------------------------------------------------------------------


def test_knn_classifies_trivially_separable_set():
    X_train = np.array([0.0, 1.0, 2.0, 10.0, 11.0, 12.0])
    y_train = np.array([0, 0, 0, 1, 1, 1])
    model = KNN(k=3).fit(X_train, y_train)
    preds = model.predict([[1.0], [11.0]])
    assert preds[0] == 0
    assert preds[1] == 1


def test_knn_trace_matches_independent_reference():
    X_train = np.array([[0.0], [1.0], [5.0], [6.0]])
    y_train = np.array([0, 0, 1, 1])
    x_query = np.array([2.0])
    dists_ref = np.sqrt(np.sum((X_train - x_query) ** 2, axis=1))
    nearest_ref = np.argsort(dists_ref, kind="stable")[:2]
    labels_ref = y_train[nearest_ref]
    majority_ref = int(np.bincount(labels_ref).argmax())

    t = knn_trace(X_train, y_train, x_query, k=2)
    assert t.result == majority_ref


def test_knn_trace_has_why_ai_and_result():
    t = knn_demo()
    assert t.result is not None
    assert t.why_ai
    assert t.formula


def test_knn_rejects_k_greater_than_n_train():
    with pytest.raises(ValueError):
        knn_trace([[0.0], [1.0]], [0, 1], [0.5], k=5)


# --- Decision tree ------------------------------------------------------------


def test_gini_and_entropy_zero_for_pure_labels():
    pure = np.array([1, 1, 1, 1])
    assert gini_impurity(pure) == pytest.approx(0.0)
    assert entropy_impurity(pure) == pytest.approx(0.0)


def test_gini_matches_hand_computed_value():
    # 2 of class 0, 2 of class 1 -> p = [0.5, 0.5] -> gini = 1 - (0.25+0.25) = 0.5
    y = np.array([0, 0, 1, 1])
    assert gini_impurity(y) == pytest.approx(0.5)


def test_decision_tree_classifies_separable_2d_set():
    X = np.array([[0.0, 5.0], [1.0, 4.0], [1.5, 6.0], [5.0, 1.0], [6.0, 0.5], [5.5, 2.0]])
    y = np.array([0, 0, 0, 1, 1, 1])
    model = DecisionTree(max_depth=2).fit(X, y)
    preds = model.predict(X)
    assert np.array_equal(preds, y)


def test_decision_tree_trace_has_why_ai_and_result():
    t = decision_tree_demo()
    assert t.result is not None
    assert t.why_ai
    assert t.formula
    assert t.meta["accuracy"] == pytest.approx(1.0)


def test_decision_tree_rejects_bad_criterion():
    with pytest.raises(ValueError):
        decision_tree_trace([[0.0], [1.0]], [0, 1], criterion="banana")


def test_decision_tree_predict_before_fit_raises():
    with pytest.raises(ValueError):
        DecisionTree().predict([[0.0]])


# --- Naive Bayes ----------------------------------------------------------------


def test_naive_bayes_classifies_trivially_separable_set():
    X_train = np.array([0.0, 1.0, 2.0, 20.0, 21.0, 22.0])
    y_train = np.array([0, 0, 0, 1, 1, 1])
    model = GaussianNB().fit(X_train, y_train)
    preds = model.predict([[1.0], [21.0]])
    assert preds[0] == 0
    assert preds[1] == 1


def test_naive_bayes_posterior_matches_independent_reference():
    X_train = np.array([[0.0], [1.0], [2.0], [10.0], [11.0], [12.0]])
    y_train = np.array([0, 0, 0, 1, 1, 1])
    x_query = np.array([1.0])

    mean0, var0 = X_train[y_train == 0].mean(), X_train[y_train == 0].var()
    mean1, var1 = X_train[y_train == 1].mean(), X_train[y_train == 1].var()

    def log_gauss(x, mean, var):
        return -0.5 * np.log(2 * np.pi * var) - (x - mean) ** 2 / (2 * var)

    log_post0 = np.log(0.5) + log_gauss(x_query[0], mean0, var0)
    log_post1 = np.log(0.5) + log_gauss(x_query[0], mean1, var1)
    expected = 0 if log_post0 > log_post1 else 1

    t = naive_bayes_trace(X_train, y_train, x_query)
    assert t.result == expected
    assert t.meta["log_posteriors"][0] == pytest.approx(log_post0)
    assert t.meta["log_posteriors"][1] == pytest.approx(log_post1)


def test_naive_bayes_trace_has_why_ai_and_result():
    t = naive_bayes_demo()
    assert t.result is not None
    assert t.why_ai
    assert t.formula


def test_naive_bayes_rejects_single_class():
    with pytest.raises(ValueError):
        naive_bayes_trace([[0.0], [1.0]], [0, 0], [0.5])


# --- PCA --------------------------------------------------------------------


def test_pca_top_component_aligns_with_high_variance_axis():
    # Variance is entirely along the x-axis; y is constant (zero variance).
    X = np.array([[-2.0, 5.0], [-1.0, 5.0], [0.0, 5.0], [1.0, 5.0], [2.0, 5.0]])
    t = pca_trace(X, n_components=1)
    component = t.meta["components"][:, 0]
    # The top component should be (±1, 0): aligned with the x-axis, none of y.
    assert abs(component[0]) == pytest.approx(1.0, abs=1e-8)
    assert abs(component[1]) == pytest.approx(0.0, abs=1e-8)
    assert t.meta["explained_variance_ratio"][0] == pytest.approx(1.0, abs=1e-8)


def test_pca_projection_matches_independent_numpy_reference():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(10, 3))
    mean = X.mean(axis=0)
    Xc = X - mean
    cov = (Xc.T @ Xc) / (X.shape[0] - 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    top_component_ref = eigvecs[:, order][:, :2]
    expected_projection = Xc @ top_component_ref

    result = pca(X, n_components=2)
    assert np.allclose(np.abs(result), np.abs(expected_projection), atol=1e-8)


def test_pca_trace_has_why_ai_and_result():
    t = pca_demo()
    assert t.result is not None
    assert t.why_ai
    assert t.formula


def test_pca_rejects_n_components_out_of_range():
    with pytest.raises(ValueError):
        pca_trace([[1.0, 2.0], [3.0, 4.0]], n_components=5)


def test_pca_class_fit_transform():
    X = [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]
    model = PCA(n_components=1)
    projected = model.fit_transform(X)
    assert projected.shape == (4, 1)
    assert model.explained_variance_ratio[0] == pytest.approx(1.0, abs=1e-8)


# --- Metrics ------------------------------------------------------------------


def test_accuracy_matches_hand_computed_value():
    y_true = [0, 1, 1, 0, 1]
    y_pred = [0, 1, 0, 0, 1]
    assert accuracy(y_true, y_pred) == pytest.approx(4 / 5)


def test_confusion_matrix_matches_hand_computed_value():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 1]
    m = confusion_matrix(y_true, y_pred, labels=[0, 1])
    # true=0: one predicted 0, one predicted 1 -> row [1, 1]
    # true=1: both predicted 1 -> row [0, 2]
    assert np.array_equal(m, np.array([[1, 1], [0, 2]]))


def test_precision_recall_f1_matches_hand_computed_value():
    y_true = [1, 1, 1, 0, 0]
    y_pred = [1, 1, 0, 1, 0]
    # TP=2, FP=1, FN=1 -> precision=2/3, recall=2/3, f1=2/3
    result = precision_recall_f1(y_true, y_pred)
    assert result["precision"] == pytest.approx(2 / 3)
    assert result["recall"] == pytest.approx(2 / 3)
    assert result["f1"] == pytest.approx(2 / 3)


def test_mse_matches_independent_numpy_reference():
    y_true = np.array([3.0, -0.5, 2.0, 7.0])
    y_pred = np.array([2.5, 0.0, 2.0, 8.0])
    ref = float(np.mean((y_true - y_pred) ** 2))
    assert mse(y_true, y_pred) == pytest.approx(ref)


def test_r2_score_perfect_fit_is_one():
    y_true = [1.0, 2.0, 3.0, 4.0]
    assert r2_score(y_true, y_true) == pytest.approx(1.0)


def test_r2_score_matches_independent_numpy_reference():
    y_true = np.array([3.0, -0.5, 2.0, 7.0])
    y_pred = np.array([2.5, 0.0, 2.0, 8.0])
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    ref = 1.0 - ss_res / ss_tot
    assert r2_score(y_true, y_pred) == pytest.approx(ref)


def test_roc_auc_perfect_separation_is_one():
    y_true = [0, 0, 1, 1]
    y_scores = [0.1, 0.2, 0.8, 0.9]
    assert roc_auc(y_true, y_scores) == pytest.approx(1.0)


def test_roc_auc_matches_hand_computed_value():
    # One positive, one negative scored higher, one lower: 1/2 correctly ranked pairs.
    y_true = [1, 0, 0]
    y_scores = [0.5, 0.9, 0.1]
    # pairs: (pos=0.5 vs neg=0.9) -> wrong; (pos=0.5 vs neg=0.1) -> right => AUC 0.5
    assert roc_auc(y_true, y_scores) == pytest.approx(0.5)


def test_roc_auc_rejects_non_binary_labels():
    with pytest.raises(ValueError):
        roc_auc_trace([0, 1, 2], [0.1, 0.5, 0.9])


def test_metrics_trace_shapes_and_why_ai():
    for trace_fn, args in [
        (accuracy_trace, ([0, 1], [0, 1])),
        (confusion_matrix_trace, ([0, 1], [0, 1])),
        (precision_recall_f1_trace, ([0, 1, 1], [0, 1, 0])),
        (mse_trace, ([1.0, 2.0], [1.0, 2.5])),
        (r2_score_trace, ([1.0, 2.0], [1.0, 2.5])),
        (roc_auc_trace, ([0, 1], [0.1, 0.9])),
    ]:
        t = trace_fn(*args)
        assert len(t) > 0
        assert t.result is not None
        assert t.why_ai
        assert t.formula


def test_metrics_demo_returns_trace():
    t = metrics_demo()
    assert t.result is not None
    assert t.why_ai
    assert t.formula


def test_metrics_reject_mismatched_lengths():
    with pytest.raises(ValueError):
        accuracy_trace([0, 1, 1], [0, 1])
