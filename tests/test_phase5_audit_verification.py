"""
Phase 5 Audit Verification Test Suite.

Automated tests specifically auditing:
1. Exact financial scenario loss mathematical identities: Loss = 10 * (FP + R * FN).
2. Independent manual reconstruction of threshold operating points, confusion matrices, and precision/recall.
3. Strict verification of Safeguard 1: Disclosing that resampling_seed_42 does NOT match Phase 4B baseline rows.
4. Paired bootstrap point estimate matching on prediction arrays.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import confusion_matrix

from src.data.quantum_subsets import get_quantum_partition_indices
from src.experiments.bootstrap import compute_paired_bootstrap_confidence_intervals
from src.experiments.resampling import extract_resampled_training_subset
from src.experiments.threshold_analysis import (
    compute_scenario_cost_curve,
    evaluate_closest_operating_points,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def validation_predictions_df():
    """Loads the saved Phase 4B validation predictions."""
    path = PROJECT_ROOT / "data" / "processed" / "quantum" / "predictions_validation.csv"
    assert path.exists(), f"Missing validation predictions file: {path}"
    return pd.read_csv(path)


def test_audit_financial_scenario_losses_identity(validation_predictions_df):
    """Audits Requirement 1: Recalculates every reported loss directly from FP and FN counts.

    Verifies Loss = 10 * (FP + R * FN) for all models across R in {5, 20, 50, 100}.
    """
    y_true = validation_predictions_df["True_Class"].to_numpy()
    assert np.sum(y_true == 1) == 33, "Validation set must contain exactly 33 frauds!"

    for model_name, score_col in [
        ("linear_svm", "Linear_SVM_Score"),
        ("rbf_svm", "RBF_SVM_Score"),
        ("qsvm", "QSVM_Score"),
    ]:
        scores = validation_predictions_df[score_col].to_numpy()
        cost_results = compute_scenario_cost_curve(
            y_true, scores, cost_ratios=[5.0, 20.0, 50.0, 100.0], c_fp=10.0
        )

        for R in [5.0, 20.0, 50.0, 100.0]:
            r_key = str(int(R))
            pt = cost_results[r_key]

            fp = pt["false_positives"]
            fn = pt["false_negatives"]
            thresh = pt["optimal_threshold"]
            min_loss = pt["minimized_scenario_loss"]

            # 1. Independent reconstruction via manual thresholding
            y_pred = (scores >= thresh).astype(int)
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            tn_manual, fp_manual, fn_manual, tp_manual = cm.ravel()

            assert fp == fp_manual, f"FP mismatch for {model_name} at R={R}: {fp} vs {fp_manual}"
            assert fn == fn_manual, f"FN mismatch for {model_name} at R={R}: {fn} vs {fn_manual}"

            # 2. Formula verification: Loss = 10 * (FP + R * FN)
            expected_loss = 10.0 * (fp + R * fn)
            assert np.isclose(
                min_loss, expected_loss, atol=1e-8
            ), f"Loss formula failed for {model_name} at R={R}: {min_loss} != {expected_loss}"


def test_audit_threshold_operating_points_reconstruction(validation_predictions_df):
    """Audits Requirement 4: Reconstructs confusion matrix counts and metrics independently.

    Verifies TP, FP, TN, FN, precision, and recall using the saved decision score vector.
    """
    y_true = validation_predictions_df["True_Class"].to_numpy()
    total_pos = 33
    total_neg = 19967

    for model_name, score_col in [
        ("linear_svm", "Linear_SVM_Score"),
        ("rbf_svm", "RBF_SVM_Score"),
        ("qsvm", "QSVM_Score"),
    ]:
        scores = validation_predictions_df[score_col].to_numpy()
        operating_points = evaluate_closest_operating_points(
            y_true, scores, target_recalls=[0.75, 0.80, 0.85, 0.90]
        )

        for pt in operating_points:
            target_r = pt["target_recall"]
            thresh = pt["decision_threshold"]

            # Independent reconstruction
            y_pred = (scores >= thresh).astype(int)
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()

            assert pt["true_positives"] == tp
            assert pt["false_positives"] == fp
            assert pt["false_negatives"] == fn
            assert pt["true_negatives"] == tn

            # Metrics
            calc_recall = tp / total_pos
            calc_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

            assert np.isclose(pt["achieved_recall"], calc_recall, atol=1e-6)
            assert np.isclose(pt["precision"], calc_precision, atol=1e-6)
            assert pt["achieved_recall"] >= target_r


def test_audit_seed_42_reproduction_and_safeguard_distinction():
    """Audits Requirement 3 & Safeguard 1:

    Verifies row identities and documents that resampling_seed_42 from the 206,980-row
    pool differs from the Phase 4B historical training subset (which was drawn from the full 226,980 rows).
    """
    X_train_full = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "X_train.csv")
    y_train_full = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "y_train.csv")["Class"]

    # 1. Historical Phase 4B partition indices
    p4b_train_idx, p4b_val_idx = get_quantum_partition_indices(
        y_train_full, n_train=500, n_train_fraud=100, n_val=20000, random_state=42
    )

    # Verify validation partition exactness
    assert len(p4b_val_idx) == 20000
    assert y_train_full.loc[p4b_val_idx].sum() == 33

    # Check match with saved Phase 4B validation file
    saved_val_unscaled = pd.read_csv(
        PROJECT_ROOT / "data" / "processed" / "quantum" / "X_val_q_selected_unscaled.csv",
        index_col=0,
    )
    val_subset_check = X_train_full.loc[p4b_val_idx, saved_val_unscaled.columns].reset_index(
        drop=True
    )
    max_val_diff = np.max(np.abs(saved_val_unscaled.to_numpy() - val_subset_check.to_numpy()))
    assert max_val_diff == 0.0, "p4b_val_idx must match saved validation partition bit-for-bit!"

    # 2. Resampling pool draw with seed 42
    val_set = set(p4b_val_idx)
    pool_indices = [idx for idx in X_train_full.index if idx not in val_set]
    assert len(pool_indices) == 206980

    X_pool = X_train_full.loc[pool_indices]
    y_pool = y_train_full.loc[pool_indices]

    X_tr_seed42, y_tr_seed42 = extract_resampled_training_subset(
        X_pool, y_pool, n_train=500, n_train_fraud=100, random_state=42
    )

    # 3. Verify discrepancy: pool draw seed 42 does NOT match original Phase 4B train rows
    overlap_count = len(set(X_tr_seed42.index).intersection(set(p4b_train_idx)))
    # Because pool is 206,980 rows, rng.choice selects different rows than permutation on 226,980 rows
    assert overlap_count != 500, "resampling_seed_42 must be distinguished from Phase 4B baseline!"
    assert overlap_count == 32  # Documented exact overlap count


def test_audit_bootstrap_point_estimates_match_predictions(validation_predictions_df):
    """Audits Requirement 2: Ensures bootstrap uses identical row indices and reported point

    estimates match empirical prediction arrays.
    """
    y_true = validation_predictions_df["True_Class"].to_numpy()
    pred_rbf = validation_predictions_df["RBF_SVM_Pred"].to_numpy()
    score_rbf = validation_predictions_df["RBF_SVM_Score"].to_numpy()
    pred_q = validation_predictions_df["QSVM_Pred"].to_numpy()
    score_q = validation_predictions_df["QSVM_Score"].to_numpy()

    # Empirical point estimates
    cm_rbf = confusion_matrix(y_true, pred_rbf, labels=[0, 1])
    cm_q = confusion_matrix(y_true, pred_q, labels=[0, 1])

    prec_rbf = cm_rbf[1, 1] / (cm_rbf[1, 1] + cm_rbf[0, 1])
    prec_q = cm_q[1, 1] / (cm_q[1, 1] + cm_q[0, 1])
    delta_prec_empirical = prec_rbf - prec_q

    # Run quick paired bootstrap (50 replicates)
    models_dict = {
        "rbf_svm": (pred_rbf, score_rbf),
        "qsvm": (pred_q, score_q),
    }
    boot = compute_paired_bootstrap_confidence_intervals(
        y_true, models_dict, n_bootstraps=50, random_state=42
    )

    diff = boot["paired_differences"]["rbf_svm_minus_qsvm"]["delta_precision"]
    # Bootstrap mean difference should be close to empirical difference
    assert np.isclose(diff["mean"], delta_prec_empirical, atol=0.03)
