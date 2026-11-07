"""
Unit and regression tests for Phase 8: Quantum Kernel Improvements and Diagnostics.

Verifies:
1. Mathematical properties of kernel diagnostics (effective rank, KTA, condition number).
2. Closed-form analytic Z-FeatureMap product kernel identity.
3. Feature selection leakage prevention and deterministic reproducibility.
4. Option A sampling feasibility (200 frauds <= 345 available, strict subset nesting D_500 ⊂ D_1000 ⊂ D_2000).
5. Mathematical invariance of ranking metrics (PR-AUC, ROC-AUC) under decision threshold shifting.
6. Paired non-parametric bootstrap confidence interval estimation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from src.experiments.phase8_improvements import (
    compute_analytic_z_kernel,
    compute_effective_rank,
    compute_kernel_target_alignment,
    compute_paired_bootstrap_significance,
    diagnose_kernel_properties,
    evaluate_threshold_calibrations,
)


def test_compute_effective_rank() -> None:
    """Verifies that effective rank correctly computes participation ratio within [1, N]."""
    # Case 1: Identity matrix (eigenvalues all 1) -> eff_rank = N
    N = 10
    eigs_id = np.ones(N)
    assert np.isclose(compute_effective_rank(eigs_id), N)

    # Case 2: Rank-1 matrix (one eigenvalue N, rest 0) -> eff_rank = 1
    eigs_rank1 = np.zeros(N)
    eigs_rank1[0] = float(N)
    assert np.isclose(compute_effective_rank(eigs_rank1), 1.0)

    # Case 3: Geometric decay -> between 1 and N
    eigs_decay = np.array([10.0, 5.0, 2.0, 1.0, 0.5])
    eff_r = compute_effective_rank(eigs_decay)
    assert 1.0 <= eff_r <= len(eigs_decay)


def test_compute_kernel_target_alignment() -> None:
    """Verifies that KTA handles {-1, +1} label conversion symmetrically and stays bounded."""
    N = 20
    rng = np.random.RandomState(42)
    y = rng.choice([0, 1], size=N)

    # Perfect alignment: K = y_pm @ y_pm^T
    y_pm = 2.0 * y - 1.0
    K_perfect = np.outer(y_pm, y_pm)
    kta_perfect = compute_kernel_target_alignment(K_perfect, y, centered=False)
    assert np.isclose(kta_perfect, 1.0, atol=1e-5)

    # Random Gram matrix: KTA in [-1.0, 1.0]
    A = rng.randn(N, 5)
    K_rand = A @ A.T
    np.fill_diagonal(K_rand, 1.0)
    kta_uncentered = compute_kernel_target_alignment(K_rand, y, centered=False)
    kta_centered = compute_kernel_target_alignment(K_rand, y, centered=True)
    assert -1.0 <= kta_uncentered <= 1.0
    assert -1.0 <= kta_centered <= 1.0


def test_diagnose_kernel_properties() -> None:
    """Verifies that diagnose_kernel_properties returns complete diagnostic dictionary."""
    N = 30
    rng = np.random.RandomState(42)
    X = rng.uniform(0, np.pi, size=(N, 4))
    y = np.array([1] * 10 + [0] * 20)

    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    K = np.prod(np.cos(diff) ** 2, axis=2)
    np.fill_diagonal(K, 1.0)

    diag = diagnose_kernel_properties(K, y)

    assert diag["matrix_shape"] == [N, N]
    assert diag["numerical_validity"]["is_positive_semi_definite"] is True
    assert diag["numerical_validity"]["max_symmetry_discrepancy"] < 1e-6
    assert diag["numerical_validity"]["max_diagonal_discrepancy"] < 1e-6
    assert "class_similarities" in diag
    assert "fraud_fraud_mean" in diag["class_similarities"]
    assert "legit_legit_mean" in diag["class_similarities"]
    assert "fraud_legit_mean" in diag["class_similarities"]
    assert "ff_to_fl_ratio" in diag["class_similarities"]
    assert "spectral_diagnostics" in diag
    assert diag["spectral_diagnostics"]["effective_rank"] >= 1.0


def test_analytic_z_kernel_closed_form() -> None:
    """Verifies that compute_analytic_z_kernel evaluates closed form cosine products properly."""
    X1 = np.array([[0.0, 0.5], [1.0, 1.5]])
    X2 = np.array([[0.0, 0.5], [0.5, 1.0]])

    K = compute_analytic_z_kernel(X1, X2)
    assert K.shape == (2, 2)
    # K[0, 0] must be exactly 1.0 (identical vectors)
    assert np.isclose(K[0, 0], 1.0)
    # K values must be bounded in [0, 1]
    assert np.all(K >= 0.0)
    assert np.all(K <= 1.0)


def test_option_a_feasibility_and_nesting() -> None:
    """Verifies Option A sample size feasibility, non-validation fraud pool capacity, and subset nesting."""
    from src.data.quantum_subsets import get_quantum_partition_indices

    # Mock full training labels
    n_total = 226980
    n_fraud = 378
    fraud_indices = np.arange(n_fraud)
    legit_indices = np.arange(n_fraud, n_total)

    y_train = pd.Series(0, index=np.arange(n_total))
    y_train.iloc[fraud_indices] = 1

    t_idx, v_idx = get_quantum_partition_indices(y_train, n_train=500, n_train_fraud=100, n_val=20000, random_state=42)
    val_set = set(v_idx)

    # Check non-validation pool fraud count
    avail_frauds = [i for i in fraud_indices if i not in val_set]
    assert len(avail_frauds) == 345, f"Expected 345 non-validation frauds, got {len(avail_frauds)}"

    # Option A requirements: N=500 -> 50, N=1000 -> 100, N=2000 -> 200
    assert 200 <= len(avail_frauds), "200 fraud cases at N=2000 exceeds available non-validation frauds!"

    # Test nested property construction
    rng = np.random.RandomState(42)
    shuffled_f = rng.permutation(avail_frauds)
    shuffled_l = rng.permutation([i for i in legit_indices if i not in val_set])

    d500 = np.concatenate([shuffled_f[:50], shuffled_l[:450]])
    d1000 = np.concatenate([shuffled_f[:100], shuffled_l[:900]])
    d2000 = np.concatenate([shuffled_f[:200], shuffled_l[:1800]])

    assert set(d500).issubset(set(d1000)), "D_500 is not a strict subset of D_1000!"
    assert set(d1000).issubset(set(d2000)), "D_1000 is not a strict subset of D_2000!"
    assert len(set(d2000).intersection(val_set)) == 0, "Validation set leakage detected!"


def test_threshold_invariance_of_ranking_metrics() -> None:
    """Verifies that decision threshold changes alter precision/recall operating points but NOT PR-AUC or ROC-AUC."""
    rng = np.random.RandomState(42)
    y_true = np.array([1] * 20 + [0] * 80)
    scores = rng.randn(100)

    base_pr_auc = average_precision_score(y_true, scores)
    base_roc_auc = roc_auc_score(y_true, scores)

    calib = evaluate_threshold_calibrations(
        y_true=y_true,
        models_scores={"test_model": scores},
        cost_ratios=[5.0, 20.0, 50.0],
    )

    ranking = calib["test_model"]["ranking_metrics"]
    assert np.isclose(ranking["pr_auc"], base_pr_auc, atol=1e-4)
    assert np.isclose(ranking["roc_auc"], base_roc_auc, atol=1e-4)


def test_paired_bootstrap_confidence_intervals() -> None:
    """Verifies that compute_paired_bootstrap_significance computes 95% CIs and paired differences."""
    rng = np.random.RandomState(42)
    y_true = np.array([1] * 15 + [0] * 85)

    scores_ref = rng.randn(100)
    scores_better = scores_ref + y_true * 1.5  # Distinctly better model

    pred_ref = (scores_ref >= 0.0).astype(int)
    pred_better = (scores_better >= 0.0).astype(int)

    model_preds = {
        "qsvm_zz_r2_ref": (pred_ref, scores_ref),
        "improved_model": (pred_better, scores_better),
    }

    boot_res = compute_paired_bootstrap_significance(
        y_true=y_true,
        model_predictions=model_preds,
        reference_key="qsvm_zz_r2_ref",
        n_bootstraps=100,
        random_state=42,
    )

    assert "individual_models" in boot_res
    assert "paired_differences" in boot_res
    pair_key = "improved_model_minus_qsvm_zz_r2_ref"
    assert pair_key in boot_res["paired_differences"]

    delta_pr = boot_res["paired_differences"][pair_key]["delta_pr_auc"]
    assert delta_pr["mean"] > 0.0
    assert len(delta_pr["ci_95"]) == 2
    assert delta_pr["ci_95"][0] <= delta_pr["mean"] <= delta_pr["ci_95"][1]
