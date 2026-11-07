"""
Classical Support Vector Machine (SVM) baseline module.

Provides training, cross-validation for class weight configuration,
prediction, and model persistence for financial fraud detection.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC, SVC

from src.models.evaluate import compute_classification_metrics, plot_confusion_matrix, plot_curves

logger = logging.getLogger("quantum_fraud.classical_svm")


def create_svm_classifier(
    kernel: Literal["linear", "rbf"] = "linear",
    class_weight: Optional[Literal["balanced"]] = "balanced",
    C: float = 1.0,
    random_state: int = 42,
    max_iter: int = 2000,
) -> Any:
    """Instantiates a scikit-learn SVM classifier with specified hyperparameters.

    Args:
        kernel: SVM kernel type ('linear' or 'rbf').
        class_weight: Class weight mode (None or 'balanced').
        C: Regularization parameter.
        random_state: Seed for reproducibility.
        max_iter: Iteration limit (relevant for LinearSVC).

    Returns:
        Instantiated SVM estimator.
    """
    if kernel == "linear":
        # LinearSVC uses LIBLINEAR (dual=False is optimal for N >> d)
        return LinearSVC(
            C=C,
            class_weight=class_weight,
            dual=False,
            random_state=random_state,
            max_iter=max_iter,
        )
    elif kernel == "rbf":
        return SVC(
            C=C,
            kernel="rbf",
            class_weight=class_weight,
            random_state=random_state,
            cache_size=1000,
        )
    else:
        raise ValueError(f"Unsupported kernel '{kernel}'. Choose 'linear' or 'rbf'.")


def evaluate_class_weight_strategies(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    kernel: Literal["linear", "rbf"] = "linear",
    n_splits: int = 3,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Evaluates class_weight=None vs class_weight='balanced' strictly using training cross-validation.

    Guarantees no test set leakage during hyperparameter / strategy evaluation.

    Args:
        X_train: Training features.
        y_train: Training target.
        kernel: SVM kernel.
        n_splits: Number of cross-validation folds.
        random_state: Random state for K-Fold split.

    Returns:
        Dict[str, Any]: Performance comparison across class weighting strategies.
    """
    logger.info("Evaluating class_weight strategies via %d-fold CV on training set...", n_splits)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    strategies = [None, "balanced"]
    comparison: Dict[str, Any] = {}

    for cw in strategies:
        strategy_name = "unweighted" if cw is None else "balanced"
        f1_list, prec_list, rec_list, pr_auc_list, roc_auc_list = [], [], [], [], []
        fit_times = []

        for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

            clf = create_svm_classifier(kernel=kernel, class_weight=cw, random_state=random_state)
            t0 = time.time()
            clf.fit(X_tr, y_tr)
            fit_times.append(time.time() - t0)

            scores = clf.decision_function(X_val)
            preds = clf.predict(X_val)

            metrics = compute_classification_metrics(y_val, preds, scores)
            f1_list.append(metrics["f1_score"])
            prec_list.append(metrics["precision"])
            rec_list.append(metrics["recall"])
            pr_auc_list.append(metrics["pr_auc"])
            roc_auc_list.append(metrics["roc_auc"])

        comparison[strategy_name] = {
            "class_weight": cw,
            "mean_precision": round(float(np.mean(prec_list)), 4),
            "mean_recall": round(float(np.mean(rec_list)), 4),
            "mean_f1": round(float(np.mean(f1_list)), 4),
            "mean_pr_auc": round(float(np.mean(pr_auc_list)), 4),
            "mean_roc_auc": round(float(np.mean(roc_auc_list)), 4),
            "mean_fit_time_seconds": round(float(np.mean(fit_times)), 4),
        }
        logger.info(
            "Strategy '%s': Mean Recall=%.4f, Mean Precision=%.4f, Mean F1=%.4f, Mean PR-AUC=%.4f",
            strategy_name,
            comparison[strategy_name]["mean_recall"],
            comparison[strategy_name]["mean_precision"],
            comparison[strategy_name]["mean_f1"],
            comparison[strategy_name]["mean_pr_auc"],
        )

    return comparison


def train_svm_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    kernel: Literal["linear", "rbf"] = "linear",
    class_weight: Optional[Literal["balanced"]] = "balanced",
    C: float = 1.0,
    random_state: int = 42,
) -> Tuple[Any, float]:
    """Fits an SVM classifier on training data and records runtime.

    Args:
        X_train: Training feature matrix.
        y_train: Training label vector.
        kernel: Kernel type.
        class_weight: Class weighting configuration.
        C: Penalty parameter.
        random_state: Seed.

    Returns:
        Tuple[Any, float]: Fitted estimator and training runtime in seconds.
    """
    clf = create_svm_classifier(
        kernel=kernel,
        class_weight=class_weight,
        C=C,
        random_state=random_state,
    )
    logger.info(
        "Fitting %s on %d samples (kernel=%s, class_weight=%s)...",
        type(clf).__name__,
        len(X_train),
        kernel,
        class_weight,
    )
    t0 = time.time()
    clf.fit(X_train, y_train)
    fit_time = time.time() - t0
    logger.info("Training completed in %.2f seconds.", fit_time)
    return clf, fit_time


def predict_and_score(
    model: Any,
    X: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Generates predictions and continuous decision scores for evaluation.

    Args:
        model: Fitted SVM classifier.
        X: Feature matrix.

    Returns:
        Tuple[np.ndarray, np.ndarray, float]:
            Binary predictions, decision function scores, and inference time in seconds.
    """
    t0 = time.time()
    preds = model.predict(X)
    scores = model.decision_function(X)
    inference_time = time.time() - t0
    logger.info("Inference on %d samples completed in %.4f seconds.", len(X), inference_time)
    return preds, scores, inference_time


def save_model_artifacts(
    model: Any,
    config: Dict[str, Any],
    output_dir: Path,
) -> Tuple[Path, Path]:
    """Saves serialized model and configuration metadata.

    Args:
        model: Trained model object.
        config: Model parameters and metadata dictionary.
        output_dir: Target directory (models/).

    Returns:
        Tuple[Path, Path]: Saved model path and config path.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "classical_svm.joblib"
    config_path = output_dir / "classical_svm_config.json"

    joblib.dump(model, model_path)
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    logger.info("Model saved to %s", model_path)
    logger.info("Config saved to %s", config_path)
    return model_path, config_path
