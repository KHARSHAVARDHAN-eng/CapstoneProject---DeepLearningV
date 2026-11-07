"""
Model evaluation module for financial fraud detection.

Computes precision, recall, F1, PR-AUC, ROC-AUC, and confusion matrix metrics,
specifically designating Fraud (1) as the positive class, and renders publication-ready plots.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger("quantum_fraud.evaluate")


def compute_classification_metrics(
    y_true: np.ndarray | list,
    y_pred: np.ndarray | list,
    y_score: Optional[np.ndarray | list] = None,
) -> Dict[str, Any]:
    """Computes comprehensive binary classification metrics for fraud detection.

    Args:
        y_true: Ground truth binary labels (0 = Legitimate, 1 = Fraud).
        y_pred: Predicted binary labels (0 or 1).
        y_score: Raw decision scores or predicted probabilities for class 1.

    Returns:
        Dict[str, Any]: Dictionary of evaluation metrics.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    precision = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
    recall = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    balanced_acc = (recall + specificity) / 2.0

    metrics: Dict[str, Any] = {
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "specificity": round(specificity, 4),
        "balanced_accuracy": round(balanced_acc, 4),
    }

    if y_score is not None:
        y_score = np.asarray(y_score)
        pr_auc = float(average_precision_score(y_true, y_score, pos_label=1))
        roc_auc = float(roc_auc_score(y_true, y_score))
        metrics["pr_auc"] = round(pr_auc, 4)
        metrics["roc_auc"] = round(roc_auc, 4)

    metrics["classification_report_str"] = classification_report(
        y_true,
        y_pred,
        target_names=["Legitimate (0)", "Fraud (1)"],
        digits=4,
        zero_division=0,
    )

    logger.info(
        "Evaluation summary: Precision=%.4f, Recall=%.4f, F1=%.4f%s",
        metrics["precision"],
        metrics["recall"],
        metrics["f1_score"],
        f", PR-AUC={metrics['pr_auc']:.4f}, ROC-AUC={metrics['roc_auc']:.4f}" if y_score is not None else "",
    )
    return metrics


def plot_confusion_matrix(
    cm: np.ndarray,
    output_path: Path,
    title: str = "Classical SVM Confusion Matrix",
    labels: Tuple[str, str] = ("Legitimate (0)", "Fraud (1)"),
) -> Path:
    """Plots and saves an annotated confusion matrix heatmap.

    Args:
        cm: 2x2 confusion matrix array.
        output_path: Path to save figure.
        title: Plot title.
        labels: Class labels.

    Returns:
        Path: Output file path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))

    total = np.sum(cm)
    annot = np.array([
        [f"{cm[0, 0]:,}\n({cm[0, 0]/total*100:.2f}%)", f"{cm[0, 1]:,}\n({cm[0, 1]/total*100:.2f}%)"],
        [f"{cm[1, 0]:,}\n({cm[1, 0]/total*100:.2f}%)", f"{cm[1, 1]:,}\n({cm[1, 1]/total*100:.2f}%)"],
    ])

    sns.heatmap(
        cm,
        annot=annot,
        fmt="",
        cmap="Blues",
        cbar=False,
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        linewidths=1.5,
        linecolor="#cccccc",
        annot_kws={"fontsize": 11, "fontweight": "bold"},
    )
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Label", fontsize=11, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=11, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Confusion matrix plot saved to %s", output_path)
    return output_path


def plot_curves(
    y_true: np.ndarray,
    y_score: np.ndarray,
    pr_output_path: Path,
    roc_output_path: Path,
    pr_auc: float,
    roc_auc: float,
) -> Tuple[Path, Path]:
    """Plots Precision-Recall curve and ROC curve.

    Args:
        y_true: Ground truth binary labels.
        y_score: Decision function scores.
        pr_output_path: Path for PR curve figure.
        roc_output_path: Path for ROC curve figure.
        pr_auc: Average Precision score.
        roc_auc: ROC-AUC score.

    Returns:
        Tuple[Path, Path]: Paths to saved figures.
    """
    pr_output_path.parent.mkdir(parents=True, exist_ok=True)
    roc_output_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Precision-Recall Curve (Critical for Imbalance)
    precision, recall, _ = precision_recall_curve(y_true, y_score, pos_label=1)
    baseline_pr = float(np.sum(y_true == 1) / len(y_true))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall, precision, color="#2b5c8f", lw=2, label=f"Classical SVM (PR-AUC = {pr_auc:.4f})")
    ax.axhline(baseline_pr, color="#d9534f", linestyle="--", lw=1.5, label=f"No-skill Baseline ({baseline_pr*100:.3f}%)")
    ax.set_title("Precision-Recall Curve (Fraud Detection)", fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Recall (True Positive Rate)", fontsize=11)
    ax.set_ylabel("Precision (Positive Predictive Value)", fontsize=11)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlim(-0.02, 1.02)
    ax.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    plt.savefig(pr_output_path, dpi=300, bbox_inches="tight")
    plt.close()

    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=1)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, color="#2a9d8f", lw=2, label=f"Classical SVM (ROC-AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="#888888", linestyle="--", lw=1.5, label="Random Guess (AUC = 0.5000)")
    ax.set_title("Receiver Operating Characteristic (ROC) Curve", fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    ax.set_ylabel("True Positive Rate (Recall)", fontsize=11)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlim(-0.02, 1.02)
    ax.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(roc_output_path, dpi=300, bbox_inches="tight")
    plt.close()

    return pr_output_path, roc_output_path


def plot_comparative_curves(
    models_dict: Dict[str, Dict[str, Any]],
    y_true: np.ndarray,
    pr_output_path: Path,
    roc_output_path: Path,
) -> Tuple[Path, Path]:
    """Plots comparative Precision-Recall and ROC curves across multiple models.

    Args:
        models_dict: Dict mapping model_name -> {"scores": y_score, "pr_auc": float, "roc_auc": float, "color": str}.
        y_true: Ground truth binary labels.
        pr_output_path: Path for output PR curve plot.
        roc_output_path: Path for output ROC curve plot.

    Returns:
        Tuple of (pr_output_path, roc_output_path).
    """
    pr_output_path.parent.mkdir(parents=True, exist_ok=True)
    roc_output_path.parent.mkdir(parents=True, exist_ok=True)

    baseline_pr = float(np.sum(y_true == 1) / len(y_true))

    # 1. Comparative PR Curve
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, data in models_dict.items():
        precision, recall, _ = precision_recall_curve(y_true, data["scores"], pos_label=1)
        ax.plot(
            recall,
            precision,
            color=data.get("color", "#2b5c8f"),
            lw=2.2,
            label=f"{name} (PR-AUC = {data['pr_auc']:.4f})",
        )
    ax.axhline(
        baseline_pr,
        color="#d9534f",
        linestyle="--",
        lw=1.5,
        label=f"Natural Prevalence ({baseline_pr * 100:.3f}%)",
    )
    ax.set_title("Comparative Precision-Recall Curves (Fraud Class)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Recall (True Positive Rate)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Precision (Positive Predictive Value)", fontsize=11, fontweight="bold")
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlim(-0.02, 1.02)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", frameon=True, fontsize=10)
    plt.tight_layout()
    plt.savefig(pr_output_path, dpi=300, bbox_inches="tight")
    plt.close()

    # 2. Comparative ROC Curve
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, data in models_dict.items():
        fpr, tpr, _ = roc_curve(y_true, data["scores"], pos_label=1)
        ax.plot(
            fpr,
            tpr,
            color=data.get("color", "#2a9d8f"),
            lw=2.2,
            label=f"{name} (ROC-AUC = {data['roc_auc']:.4f})",
        )
    ax.plot([0, 1], [0, 1], color="#888888", linestyle="--", lw=1.5, label="Random Guess (AUC = 0.5000)")
    ax.set_title("Comparative ROC Curves (Fraud Detection)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight="bold")
    ax.set_ylabel("True Positive Rate (Recall)", fontsize=11, fontweight="bold")
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlim(-0.02, 1.02)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower right", frameon=True, fontsize=10)
    plt.tight_layout()
    plt.savefig(roc_output_path, dpi=300, bbox_inches="tight")
    plt.close()

    return pr_output_path, roc_output_path


def plot_kernel_matrix_heatmap(
    K: np.ndarray,
    output_path: Path,
    title: str = "Quantum Kernel Gram Matrix",
    cmap: str = "viridis",
) -> Path:
    """Plots and saves a heatmap of a quantum kernel matrix."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(K, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Fidelity |⟨ϕ(x)|ϕ(x')⟩|²", fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Sample Index", fontsize=11, fontweight="bold")
    ax.set_ylabel("Sample Index", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return output_path
