"""
Phase 5: Paired Non-Parametric Bootstrap Confidence Interval Module.

Quantifies validation evaluation uncertainty across 1,000 paired replicates,
evaluating metric distributions and paired differences (RBF vs QSVM, Linear vs QSVM).
"""

from typing import Any, Dict, List, Tuple
import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
)


def compute_paired_bootstrap_confidence_intervals(
    y_true: np.ndarray,
    model_predictions: Dict[str, Tuple[np.ndarray, np.ndarray]],
    n_bootstraps: int = 1000,
    random_state: int = 42,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Computes paired non-parametric bootstrap confidence intervals on validation predictions.

    Coupled resampling draws identical sample indices for every model in each replicate,
    permitting valid calculation of paired differences (e.g. delta_PR_AUC, delta_Precision).

    Args:
        y_true: Ground-truth binary labels (shape: [N]).
        model_predictions: Dict mapping model_key -> (y_pred, decision_scores).
        n_bootstraps: Number of valid bootstrap replicates (default: 1000).
        random_state: Random seed for reproducible resampling.
        alpha: Significance level for two-sided percentile interval (default: 0.05 for 95% CI).

    Returns:
        Dictionary containing individual model intervals, paired difference intervals,
        and degenerate replicate statistics.
    """
    n_samples = len(y_true)
    rng = np.random.RandomState(random_state)

    # Storage for bootstrap metrics
    model_keys = list(model_predictions.keys())
    boot_metrics = {
        m: {
            "recall": [],
            "precision": [],
            "f1_score": [],
            "pr_auc": [],
            "roc_auc": [],
        }
        for m in model_keys
    }

    # Paired differences storage (relative to QSVM if present)
    paired_diffs = {}
    if "qsvm" in model_keys:
        for m in model_keys:
            if m != "qsvm":
                paired_diffs[f"{m}_minus_qsvm"] = {
                    "delta_recall": [],
                    "delta_precision": [],
                    "delta_f1": [],
                    "delta_pr_auc": [],
                    "delta_roc_auc": [],
                }

    valid_replicates = 0
    discarded_degenerate = 0

    while valid_replicates < n_bootstraps:
        indices = rng.randint(0, n_samples, size=n_samples)
        y_b = y_true[indices]

        # Check degenerate sample: requires at least 1 positive and 1 negative
        n_pos = np.sum(y_b == 1)
        n_neg = np.sum(y_b == 0)
        if n_pos == 0 or n_neg == 0:
            discarded_degenerate += 1
            continue

        # Evaluate metrics for each model
        rep_evals = {}
        for m in model_keys:
            pred_m, score_m = model_predictions[m]
            pred_b = pred_m[indices]
            score_b = score_m[indices]

            rec = float(recall_score(y_b, pred_b, pos_label=1, zero_division=0))
            prec = float(precision_score(y_b, pred_b, pos_label=1, zero_division=0))
            f1 = float(f1_score(y_b, pred_b, pos_label=1, zero_division=0))
            pr_auc = float(average_precision_score(y_b, score_b))
            roc_auc = float(roc_auc_score(y_b, score_b))

            boot_metrics[m]["recall"].append(rec)
            boot_metrics[m]["precision"].append(prec)
            boot_metrics[m]["f1_score"].append(f1)
            boot_metrics[m]["pr_auc"].append(pr_auc)
            boot_metrics[m]["roc_auc"].append(roc_auc)

            rep_evals[m] = {
                "recall": rec,
                "precision": prec,
                "f1_score": f1,
                "pr_auc": pr_auc,
                "roc_auc": roc_auc,
            }

        # Calculate paired differences
        if "qsvm" in model_keys:
            q_eval = rep_evals["qsvm"]
            for m in model_keys:
                if m != "qsvm":
                    diff_dict = paired_diffs[f"{m}_minus_qsvm"]
                    diff_dict["delta_recall"].append(rep_evals[m]["recall"] - q_eval["recall"])
                    diff_dict["delta_precision"].append(rep_evals[m]["precision"] - q_eval["precision"])
                    diff_dict["delta_f1"].append(rep_evals[m]["f1_score"] - q_eval["f1_score"])
                    diff_dict["delta_pr_auc"].append(rep_evals[m]["pr_auc"] - q_eval["pr_auc"])
                    diff_dict["delta_roc_auc"].append(rep_evals[m]["roc_auc"] - q_eval["roc_auc"])

        valid_replicates += 1

    # Compute empirical percentile intervals
    lower_p = 100 * (alpha / 2.0)
    upper_p = 100 * (1.0 - alpha / 2.0)

    summary = {
        "n_bootstraps_requested": n_bootstraps,
        "n_bootstraps_completed": valid_replicates,
        "discarded_degenerate_replicates": discarded_degenerate,
        "alpha": alpha,
        "models": {},
        "paired_differences": {},
    }

    for m in model_keys:
        summary["models"][m] = {}
        for metric_name, values in boot_metrics[m].items():
            arr = np.array(values)
            summary["models"][m][metric_name] = {
                "mean": float(np.mean(arr)),
                "median": float(np.median(arr)),
                "std": float(np.std(arr)),
                "ci_lower": float(np.percentile(arr, lower_p)),
                "ci_upper": float(np.percentile(arr, upper_p)),
            }

    for diff_key, metrics_dict in paired_diffs.items():
        summary["paired_differences"][diff_key] = {}
        for metric_name, values in metrics_dict.items():
            arr = np.array(values)
            summary["paired_differences"][diff_key][metric_name] = {
                "mean": float(np.mean(arr)),
                "median": float(np.median(arr)),
                "std": float(np.std(arr)),
                "ci_lower": float(np.percentile(arr, lower_p)),
                "ci_upper": float(np.percentile(arr, upper_p)),
            }

    return summary
