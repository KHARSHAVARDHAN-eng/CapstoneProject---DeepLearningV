"""
Phase 5: Decision-Threshold Analysis and Scenario Financial Cost Module.

Provides:
1. Exact discrete operating-point calculation (closest achievable recall >= target).
2. Multi-scenario financial loss sensitivity analysis across varying cost ratios R = c_fn / c_fp.
"""

from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, confusion_matrix


def evaluate_closest_operating_points(
    y_true: np.ndarray,
    decision_scores: np.ndarray,
    target_recalls: List[float] = [0.75, 0.80, 0.85, 0.90],
) -> List[Dict[str, Any]]:
    """Evaluates the closest achievable operating points at or above target recall thresholds.

    Because discrete fraud counts (e.g., 33 frauds) restrict recall to discrete steps k/N_fraud,
    exact continuous target recalls are unachievable. This function locates the minimal recall
    such that recall >= target_recall.

    Args:
        y_true: Ground-truth binary labels (0 or 1).
        decision_scores: Continuous decision function scores.
        target_recalls: List of target recall thresholds.

    Returns:
        List of dictionaries with exact operating point metrics.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, decision_scores)
    total_positives = int(np.sum(y_true == 1))
    total_negatives = int(np.sum(y_true == 0))

    results = []
    # Note: precision_recall_curve returns thresholds with length len(recalls) - 1.
    # The last element of recalls is 0, precision is 1, with no threshold.
    # We map operating points by scanning thresholds.
    for target in target_recalls:
        # Find points where recall >= target
        valid_indices = np.where(recalls >= target)[0]
        if len(valid_indices) == 0:
            # If no point achieves target, pick max recall point
            best_idx = int(np.argmax(recalls))
        else:
            # Pick the point with recall >= target closest to target (minimal recall >= target)
            # Break ties by selecting the point that maximizes precision (minimizing false positives)
            min_recall = np.min(recalls[valid_indices])
            min_recall_indices = valid_indices[recalls[valid_indices] == min_recall]
            best_idx = int(min_recall_indices[np.argmax(precisions[min_recall_indices])])

        achieved_recall = float(recalls[best_idx])
        achieved_precision = float(precisions[best_idx])
        
        # If best_idx is within thresholds range, use it; otherwise use max score threshold
        if best_idx < len(thresholds):
            thresh = float(thresholds[best_idx])
        else:
            thresh = float(np.min(decision_scores))

        # Compute exact counts at this threshold
        y_pred = (decision_scores >= thresh).astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        results.append({
            "target_recall": target,
            "achieved_recall": achieved_recall,
            "achieved_recall_fraction": f"{tp}/{total_positives}",
            "precision": float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0,
            "decision_threshold": thresh,
            "true_positives": int(tp),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_negatives": int(tn),
            "positive_support": total_positives,
        })

    return results


def compute_scenario_cost_curve(
    y_true: np.ndarray,
    decision_scores: np.ndarray,
    cost_ratios: List[float] = [5.0, 20.0, 50.0, 100.0],
    c_fp: float = 10.0,
) -> Dict[str, Any]:
    """Computes illustrative scenario financial loss curves across varying cost ratios.

    Total Loss(R) = c_fp * FP + c_fn * FN = c_fp * (FP + R * FN)
    where R = c_fn / c_fp.

    Evaluates an exhaustive threshold sweep over all distinct decision score transition
    boundaries (all unique decision scores plus extreme boundaries) to locate the exact
    global minimum scenario loss. In case of ties with equal minimum loss, the threshold
    yielding the smallest false-positive count (highest precision / highest threshold)
    is selected.

    DISCLAIMER: These calculations represent scenario sensitivity analyses,
    not empirically audited commercial banking loss outcomes.

    Args:
        y_true: Ground-truth binary labels (0 or 1).
        decision_scores: Continuous decision function scores.
        cost_ratios: Ratios R = c_fn / c_fp to evaluate.
        c_fp: Illustrative unit cost of a false alarm (default: $10.0).

    Returns:
        Dictionary mapping cost ratio to optimal operating point and minimum loss.
    """
    total_positives = int(np.sum(y_true == 1))
    pos_scores = np.sort(decision_scores[y_true == 1])
    neg_scores = np.sort(decision_scores[y_true == 0])

    # Exhaustive threshold candidate vector: all unique scores + inf boundary
    unique_scores = np.sort(np.unique(decision_scores))
    cand_thresholds = np.concatenate([unique_scores, [np.inf]])

    # Fast vectorized searchsorted counts
    fn_arr = np.searchsorted(pos_scores, cand_thresholds, side="left")
    tp_arr = len(pos_scores) - fn_arr
    fp_arr = len(neg_scores) - np.searchsorted(neg_scores, cand_thresholds, side="left")

    scenario_results = {}

    for R in cost_ratios:
        c_fn = c_fp * R
        losses = c_fp * fp_arr + c_fn * fn_arr
        min_loss = np.min(losses)
        opt_indices = np.where(losses == min_loss)[0]

        # Break ties by choosing minimal false positives (i.e. highest threshold index)
        best_idx = int(opt_indices[-1])

        best_loss = float(losses[best_idx])
        best_thresh = float(cand_thresholds[best_idx])
        best_fp = int(fp_arr[best_idx])
        best_fn = int(fn_arr[best_idx])
        best_tp = int(tp_arr[best_idx])

        # Verify exact mathematical identity
        assert best_loss == c_fp * best_fp + c_fn * best_fn, "Scenario loss formula mismatch!"

        recall = float(best_tp / total_positives) if total_positives > 0 else 0.0
        precision = float(best_tp / (best_tp + best_fp)) if (best_tp + best_fp) > 0 else 0.0

        scenario_results[str(int(R))] = {
            "cost_ratio_R": R,
            "c_fp": c_fp,
            "c_fn": c_fn,
            "optimal_threshold": best_thresh,
            "minimized_scenario_loss": best_loss,
            "resulting_recall": recall,
            "resulting_recall_fraction": f"{best_tp}/{total_positives}",
            "resulting_precision": precision,
            "false_positives": best_fp,
            "false_negatives": best_fn,
            "true_positives": best_tp,
        }

    return scenario_results
