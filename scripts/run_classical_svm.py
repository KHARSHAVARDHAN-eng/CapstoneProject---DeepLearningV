#!/usr/bin/env python3
"""
Phase 3 Classical SVM Baseline Runner.

Trains a classical Support Vector Machine baseline on the preprocessed dataset,
evaluates class weight strategies on the training partition, evaluates performance
on the holdout test set with positive class = Fraud (1), and saves artifacts.

Usage:
    python scripts/run_classical_svm.py [--kernel {rbf,linear}] [--class-weight {balanced,none}]
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from src.models.classical_svm import (
    evaluate_class_weight_strategies,
    predict_and_score,
    save_model_artifacts,
    train_svm_model,
)
from src.models.evaluate import compute_classification_metrics, plot_confusion_matrix, plot_curves
from src.utils.logger import setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 3: Classical SVM Baseline Training and Evaluation"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed",
        help="Directory containing preprocessed splits",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=PROJECT_ROOT / "models",
        help="Directory to save trained models",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=PROJECT_ROOT / "reports",
        help="Directory to save evaluation reports and plots",
    )
    parser.add_argument(
        "--kernel",
        type=str,
        default="rbf",
        choices=["rbf", "linear"],
        help="SVM kernel type (default: rbf)",
    )
    parser.add_argument(
        "--class-weight",
        type=str,
        default="balanced",
        choices=["balanced", "none"],
        help="Class weighting configuration (default: balanced)",
    )
    parser.add_argument(
        "--C",
        type=float,
        default=1.0,
        help="SVM regularization parameter (default: 1.0)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--skip-cv-evaluation",
        action="store_true",
        help="Skip training-set cross-validation comparison of class_weight",
    )
    return parser.parse_args()


def generate_markdown_report(
    config: dict,
    metrics: dict,
    fit_time: float,
    pred_time: float,
    cv_comparison: dict | None,
    output_path: Path,
) -> Path:
    """Generates a structured Markdown report of the Classical SVM Baseline."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cm = metrics["confusion_matrix"]
    total_test = cm["true_negatives"] + cm["false_positives"] + cm["false_negatives"] + cm["true_positives"]

    lines = [
        "# Phase 3: Classical SVM Baseline Evaluation Report",
        "",
        "**Project Title**: Quantum Machine Learning for Financial Fraud Detection using Quantum Kernel Methods  ",
        "**Phase**: Phase 3 — Classical SVM Baseline  ",
        f"**Model Type**: Support Vector Machine (`kernel='{config['kernel']}', C={config['C']}`)  ",
        f"**Class Weight Configuration**: `{config['class_weight']}`  ",
        f"**Random Seed**: `{config['random_state']}`  ",
        "",
        "---",
        "",
        "## 1. Executive Summary & Runtime Profile",
        "",
        f"- **Training Set Size**: {config['train_samples']:,} samples (378 Fraud, 0.1665%)",
        f"- **Test Set Size**: {total_test:,} samples (95 Fraud, 0.1674%)",
        f"- **Training Time**: **{fit_time:.2f} seconds**",
        f"- **Prediction / Inference Time**: **{pred_time:.4f} seconds** ({total_test / pred_time:,.0f} transactions/sec)",
        "",
        "## 2. Test Set Evaluation Metrics (Positive Class = Fraud)",
        "",
        "| Evaluation Metric | Score | Interpretation |",
        "| :--- | :---: | :--- |",
        f"| **Precision (PPV)** | **{metrics['precision']:.4f}** | Percentage of predicted frauds that are actual frauds |",
        f"| **Recall (Sensitivity)** | **{metrics['recall']:.4f}** | Percentage of actual frauds successfully caught |",
        f"| **F1-Score** | **{metrics['f1_score']:.4f}** | Harmonic mean of precision and recall |",
        f"| **PR-AUC (Average Precision)** | **{metrics.get('pr_auc', 0.0):.4f}** | Area under Precision-Recall Curve (primary imbalanced metric) |",
        f"| **ROC-AUC** | **{metrics.get('roc_auc', 0.0):.4f}** | Area under Receiver Operating Characteristic curve |",
        f"| **Specificity (TNR)** | **{metrics['specificity']:.4f}** | Percentage of legitimate transactions correctly classified |",
        f"| **Balanced Accuracy** | **{metrics['balanced_accuracy']:.4f}** | Unweighted average of recall and specificity |",
        "",
        "## 3. Confusion Matrix Breakdown",
        "",
        "| | Predicted Legitimate (0) | Predicted Fraudulent (1) | Total |",
        "| :--- | :---: | :---: | :---: |",
        f"| **Actual Legitimate (0)** | **{cm['true_negatives']:,}** (TN) | **{cm['false_positives']:,}** (FP) | {cm['true_negatives'] + cm['false_positives']:,} |",
        f"| **Actual Fraudulent (1)** | **{cm['false_negatives']:,}** (FN) | **{cm['true_positives']:,}** (TP) | {cm['false_negatives'] + cm['true_positives']:,} |",
        f"| **Total** | {cm['true_negatives'] + cm['false_negatives']:,} | {cm['false_positives'] + cm['true_positives']:,} | {total_test:,} |",
        "",
        "```text",
        metrics["classification_report_str"].strip(),
        "```",
        "",
    ]

    if cv_comparison:
        lines.extend([
            "## 4. Class Imbalance Strategy Evaluation (Internal Training CV)",
            "",
            "To prevent data leakage, `class_weight='balanced'` was evaluated against the unweighted configuration strictly on the training partition:",
            "",
            "| Strategy | Mean Recall | Mean Precision | Mean F1 | Mean PR-AUC | Mean Fit Time |",
            "| :--- | :---: | :---: | :---: | :---: | :---: |",
        ])
        for strat, data in cv_comparison.items():
            lines.append(
                f"| `{strat}` | {data['mean_recall']:.4f} | {data['mean_precision']:.4f} | {data['mean_f1']:.4f} | {data['mean_pr_auc']:.4f} | {data['mean_fit_time_seconds']:.2f}s |"
            )
        lines.extend([
            "",
            "> **Key Trade-off**: `class_weight='balanced'` applies an inverse frequency penalty (~578x for fraud errors), boosting recall at the cost of precision. Unweighted SVM maximizes precision and F1 by penalizing false alarms.",
            "",
        ])

    lines.extend([
        "## 5. Artifacts and Visualizations",
        "",
        "- **Trained Model**: `models/classical_svm.joblib`",
        "- **Model Config**: `models/classical_svm_config.json`",
        "- **Evaluation Metrics**: `reports/classical_svm_metrics.json`",
        "- **Confusion Matrix Plot**: `reports/figures/classical_svm_confusion_matrix.png`",
        "- **Precision-Recall Curve**: `reports/figures/classical_svm_pr_curve.png`",
        "- **ROC Curve**: `reports/figures/classical_svm_roc_curve.png`",
        "",
        "## 6. Significance as Baseline for Quantum Kernel Methods",
        "",
        "1. **Mathematical Parallelism with Quantum Kernel SVM (QSVM)**:",
        "   - The classical RBF SVM optimizes a dual quadratic program over the classical Gram matrix $K_{ij} = \\exp(-\\gamma ||x_i - x_j||^2)$.",
        "   - The Quantum Kernel SVM replaces this classical kernel with a quantum state fidelity kernel $K_{ij} = |\\langle \\phi(x_i) | \\phi(x_j) \\rangle|^2$ computed via parameterized quantum circuits.",
        "   - By establishing this classical RBF SVM baseline on the exact same dataset splits, we establish a rigorous classical benchmark for later comparison with quantum kernels.",
        "",
        "2. **Computational Complexity Benchmark**:",
        "   - Classical RBF SVM scales as $O(N^2)$ to $O(N^3)$ in sample size ($N = 226,980$).",
        "   - Quantum kernel matrix calculation on current NISQ simulators also scales as $O(N^2)$ quantum circuit executions, highlighting the necessity of dimensionality reduction and sub-sampling for quantum algorithms.",
    ])

    report_content = "\n".join(lines) + "\n"
    output_path.write_text(report_content, encoding="utf-8")
    return output_path


def main() -> int:
    args = parse_args()
    logger = setup_logger("quantum_fraud.run_classical_svm")
    logger.info("============================================================")
    logger.info("Starting Phase 3: Classical SVM Baseline Training")
    logger.info("============================================================")

    # Resolve class_weight argument
    class_weight_val = "balanced" if args.class_weight == "balanced" else None

    # Step 1: Load preprocessed splits
    X_train_path = args.data_dir / "X_train.csv"
    y_train_path = args.data_dir / "y_train.csv"
    X_test_path = args.data_dir / "X_test.csv"
    y_test_path = args.data_dir / "y_test.csv"

    if not all(p.exists() for p in [X_train_path, y_train_path, X_test_path, y_test_path]):
        logger.error("Preprocessed data splits not found in %s. Please run Phase 2 first.", args.data_dir)
        return 1

    logger.info("Loading preprocessed splits from %s...", args.data_dir)
    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path)["Class"]
    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path)["Class"]

    logger.info("Loaded X_train shape: %s, X_test shape: %s", X_train.shape, X_test.shape)

    # Step 2: Evaluate class weight strategies strictly on training set (No test leakage)
    cv_comparison = None
    if not args.skip_cv_evaluation:
        logger.info("Evaluating class imbalance handling on training partition...")
        # Use a fast internal 3-fold CV with linear kernel or fast fold to evaluate class weighting
        cv_comparison = evaluate_class_weight_strategies(
            X_train=X_train,
            y_train=y_train,
            kernel="linear" if args.kernel == "linear" else "linear",
            n_splits=3,
            random_state=args.random_state,
        )

    # Step 3: Train the final model on full training set
    logger.info("Training full Classical SVM (kernel=%s, class_weight=%s, C=%.2f)...",
                args.kernel, class_weight_val, args.C)
    model, fit_time = train_svm_model(
        X_train=X_train,
        y_train=y_train,
        kernel=args.kernel,
        class_weight=class_weight_val,
        C=args.C,
        random_state=args.random_state,
    )

    # Step 4: Evaluate on untouched test set
    logger.info("Evaluating Classical SVM on holdout test set (56,746 samples)...")
    y_pred, y_score, pred_time = predict_and_score(model, X_test)
    metrics = compute_classification_metrics(y_true=y_test, y_pred=y_pred, y_score=y_score)

    # Step 5: Save visualizations
    figures_dir = args.reports_dir / "figures"
    cm_array = np.array([
        [metrics["confusion_matrix"]["true_negatives"], metrics["confusion_matrix"]["false_positives"]],
        [metrics["confusion_matrix"]["false_negatives"], metrics["confusion_matrix"]["true_positives"]],
    ])
    cm_plot_path = figures_dir / "classical_svm_confusion_matrix.png"
    plot_confusion_matrix(cm_array, cm_plot_path, title=f"Classical SVM ({args.kernel.upper()}) Confusion Matrix")

    pr_plot_path = figures_dir / "classical_svm_pr_curve.png"
    roc_plot_path = figures_dir / "classical_svm_roc_curve.png"
    plot_curves(
        y_true=y_test.values,
        y_score=y_score,
        pr_output_path=pr_plot_path,
        roc_output_path=roc_plot_path,
        pr_auc=metrics["pr_auc"],
        roc_auc=metrics["roc_auc"],
    )

    # Step 6: Save model and configs
    config_dict = {
        "model_family": "Classical Support Vector Machine",
        "kernel": args.kernel,
        "class_weight": class_weight_val,
        "C": args.C,
        "random_state": args.random_state,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "n_features": X_train.shape[1],
        "training_time_seconds": round(fit_time, 4),
        "prediction_time_seconds": round(pred_time, 4),
    }

    save_model_artifacts(model, config_dict, args.models_dir)

    # Save metrics JSON
    metrics_path = args.reports_dir / "classical_svm_metrics.json"
    metrics_to_save = {
        "config": config_dict,
        "evaluation_metrics": metrics,
        "training_cross_validation_comparison": cv_comparison,
    }
    metrics_path.write_text(json.dumps(metrics_to_save, indent=2), encoding="utf-8")

    # Save Markdown report
    report_path = args.reports_dir / "classical_svm_report.md"
    generate_markdown_report(
        config=config_dict,
        metrics=metrics,
        fit_time=fit_time,
        pred_time=pred_time,
        cv_comparison=cv_comparison,
        output_path=report_path,
    )

    logger.info("------------------------------------------------------------")
    logger.info("Classical SVM Baseline Results Summary:")
    logger.info("  Training Time:    %.2f seconds", fit_time)
    logger.info("  Inference Time:   %.4f seconds", pred_time)
    logger.info("  Precision:        %.4f", metrics["precision"])
    logger.info("  Recall:           %.4f", metrics["recall"])
    logger.info("  F1-Score:         %.4f", metrics["f1_score"])
    logger.info("  PR-AUC:           %.4f", metrics.get("pr_auc", 0.0))
    logger.info("  ROC-AUC:          %.4f", metrics.get("roc_auc", 0.0))
    logger.info("  Saved Report:     %s", report_path)
    logger.info("============================================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
