"""
Automated test suite for Phase 3: Classical SVM Baseline.

Tests cover:
1. Model instantiation and hyperparameter validation
2. Metrics computation with positive class = Fraud (1)
3. Training-set cross-validation for class_weight evaluation
4. Training execution and valid prediction / score generation
5. Model artifact persistence, loading, and prediction equivalence
6. Evaluation plotting routines (confusion matrix, PR curve, ROC curve)
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.svm import LinearSVC, SVC

from src.models.classical_svm import (
    create_svm_classifier,
    evaluate_class_weight_strategies,
    predict_and_score,
    save_model_artifacts,
    train_svm_model,
)
from src.models.evaluate import (
    compute_classification_metrics,
    plot_confusion_matrix,
    plot_curves,
)


@pytest.fixture
def synthetic_train_data() -> tuple[pd.DataFrame, pd.Series]:
    """Generates a small synthetic dataset for fast, isolated unit testing."""
    np.random.seed(42)
    n_samples = 300
    n_features = 30
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=["Time", "Amount"] + [f"V{i}" for i in range(1, 29)],
    )
    y = pd.Series(np.zeros(n_samples, dtype=int), name="Class")
    # 5% fraud samples
    y.iloc[np.random.choice(n_samples, size=15, replace=False)] = 1
    return X, y


# ============================================================================
# 1. Model Instantiation Tests
# ============================================================================

def test_create_svm_classifier_linear():
    """Verifies creation of LinearSVC with correct arguments."""
    clf = create_svm_classifier(kernel="linear", class_weight="balanced", C=0.5, random_state=42)
    assert isinstance(clf, LinearSVC)
    assert clf.C == 0.5
    assert clf.class_weight == "balanced"
    assert clf.random_state == 42


def test_create_svm_classifier_rbf():
    """Verifies creation of SVC with RBF kernel."""
    clf = create_svm_classifier(kernel="rbf", class_weight=None, C=1.0, random_state=42)
    assert isinstance(clf, SVC)
    assert clf.kernel == "rbf"
    assert clf.class_weight is None
    assert clf.random_state == 42


def test_create_svm_classifier_invalid_kernel():
    """Verifies error raised on unsupported kernel."""
    with pytest.raises(ValueError, match="Unsupported kernel"):
        create_svm_classifier(kernel="polynomial")  # type: ignore


# ============================================================================
# 2. Metric Computation Tests (Positive Class = Fraud)
# ============================================================================

def test_compute_classification_metrics():
    """Verifies metrics computation with known confusion matrix."""
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_pred = np.array([0, 0, 0, 1, 0, 1, 1, 1])
    # 3 TN, 1 FP, 1 FN, 3 TP
    y_score = np.array([-2.0, -1.5, -0.8, 0.5, -0.2, 1.2, 1.8, 2.5])

    metrics = compute_classification_metrics(y_true, y_pred, y_score)

    cm = metrics["confusion_matrix"]
    assert cm["true_negatives"] == 3
    assert cm["false_positives"] == 1
    assert cm["false_negatives"] == 1
    assert cm["true_positives"] == 3

    # Precision = TP / (TP + FP) = 3 / 4 = 0.75
    assert metrics["precision"] == 0.75
    # Recall = TP / (TP + FN) = 3 / 4 = 0.75
    assert metrics["recall"] == 0.75
    # F1 = 0.75
    assert metrics["f1_score"] == 0.75
    # Specificity = TN / (TN + FP) = 3 / 4 = 0.75
    assert metrics["specificity"] == 0.75

    assert "pr_auc" in metrics
    assert "roc_auc" in metrics
    assert 0.0 <= metrics["pr_auc"] <= 1.0
    assert 0.0 <= metrics["roc_auc"] <= 1.0


# ============================================================================
# 3. Class Weight Evaluation Tests (Training CV Only)
# ============================================================================

def test_evaluate_class_weight_strategies(synthetic_train_data):
    """Verifies that both unweighted and balanced strategies are evaluated via CV."""
    X, y = synthetic_train_data
    comparison = evaluate_class_weight_strategies(
        X_train=X, y_train=y, kernel="linear", n_splits=3, random_state=42
    )

    assert "unweighted" in comparison
    assert "balanced" in comparison
    for strat in ["unweighted", "balanced"]:
        assert "mean_recall" in comparison[strat]
        assert "mean_precision" in comparison[strat]
        assert "mean_f1" in comparison[strat]
        assert "mean_pr_auc" in comparison[strat]
        assert "mean_fit_time_seconds" in comparison[strat]


# ============================================================================
# 4. Training and Prediction Execution Tests
# ============================================================================

def test_train_and_predict(synthetic_train_data):
    """Verifies end-to-end model fitting and inference."""
    X, y = synthetic_train_data
    model, fit_time = train_svm_model(
        X_train=X, y_train=y, kernel="linear", class_weight="balanced", random_state=42
    )

    assert fit_time > 0.0

    preds, scores, pred_time = predict_and_score(model, X)

    assert len(preds) == len(X)
    assert len(scores) == len(X)
    assert pred_time > 0.0
    assert set(np.unique(preds)).issubset({0, 1})
    assert scores.dtype in (np.float32, np.float64)


# ============================================================================
# 5. Model Artifact Persistence and Equivalence Tests
# ============================================================================

def test_save_and_load_model_artifacts(synthetic_train_data, tmp_path):
    """Verifies that saved models can be loaded and reproduce predictions exactly."""
    X, y = synthetic_train_data
    model, _ = train_svm_model(
        X_train=X, y_train=y, kernel="linear", class_weight="balanced", random_state=42
    )

    config = {"kernel": "linear", "class_weight": "balanced", "random_state": 42}
    model_path, config_path = save_model_artifacts(model, config, tmp_path)

    assert model_path.exists()
    assert config_path.exists()

    # Load back and verify equivalence
    loaded_model = joblib.load(model_path)
    loaded_config = json.loads(config_path.read_text(encoding="utf-8"))

    assert loaded_config["kernel"] == "linear"

    orig_preds = model.predict(X)
    loaded_preds = loaded_model.predict(X)
    np.testing.assert_array_equal(orig_preds, loaded_preds)


# ============================================================================
# 6. Evaluation Visualization Tests
# ============================================================================

def test_plot_generation(tmp_path):
    """Verifies that confusion matrix and curves are saved properly."""
    cm = np.array([[100, 5], [2, 10]])
    cm_path = tmp_path / "cm.png"
    plot_confusion_matrix(cm, cm_path, title="Test Confusion Matrix")
    assert cm_path.exists()
    assert cm_path.stat().st_size > 0

    y_true = np.array([0] * 50 + [1] * 10)
    y_score = np.linspace(-1, 1, 60)
    pr_path = tmp_path / "pr.png"
    roc_path = tmp_path / "roc.png"
    plot_curves(y_true, y_score, pr_path, roc_path, pr_auc=0.75, roc_auc=0.85)

    assert pr_path.exists()
    assert pr_path.stat().st_size > 0
    assert roc_path.exists()
    assert roc_path.stat().st_size > 0
