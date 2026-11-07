"""
Unit and regression tests for Phase 11: Final Scientific Validation and Capstone Finalization.

Verifies:
1. Leak-free fold isolation and proper data provenance inside stratified CV.
2. Natural-prevalence preservation across validation folds.
3. Mathematical validity of analytical and quantum feature kernels.
4. Paired bootstrap hypothesis calculation, confidence intervals, and subtraction directions.
5. Monotonic behavior of financial cost optimization vs loss ratio R.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import f_classif

from src.experiments.phase11_validation import (
    calibrate_financial_cost_thresholds,
    compute_anisotropic_kernel_matrix,
    compute_bandwidth_separable_kernel_matrix,
    compute_evaluation_summary,
    compute_paired_bootstrap_comparison,
)


def test_cv_leakage_and_fold_isolation() -> None:
    """Verifies that fold train and fold val indices are disjoint and scalers/selectors are fitted on train only."""
    rng = np.random.RandomState(42)
    N = 1000
    d = 10
    X = pd.DataFrame(rng.randn(N, d), columns=[f"V{i}" for i in range(d)])
    y = np.zeros(N, dtype=int)
    y[rng.choice(N, size=20, replace=False)] = 1  # 2% fraud

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        # 1. Disjointness
        assert len(set(tr_idx).intersection(set(val_idx))) == 0

        # 2. Fit feature selector strictly on tr_idx
        f_scores, _ = f_classif(X.iloc[tr_idx], y[tr_idx])
        top_cols = X.columns[np.argsort(f_scores)[::-1][:4]].tolist()
        assert len(top_cols) == 4

        # 3. Fit scaler strictly on tr_idx
        scaler = MinMaxScaler(feature_range=(0.0, np.pi))
        X_tr_s = scaler.fit_transform(X.iloc[tr_idx][top_cols])
        X_va_s = scaler.transform(X.iloc[val_idx][top_cols])

        # Verify training bounds are strictly in [0, pi]
        assert np.min(X_tr_s) >= 0.0 - 1e-10
        assert np.max(X_tr_s) <= np.pi + 1e-10


def test_natural_prevalence_preservation() -> None:
    """Verifies that stratified splitting preserves the natural low prevalence across folds."""
    N = 10000
    n_fraud = 17  # 0.17% natural prevalence
    y = np.zeros(N, dtype=int)
    rng = np.random.RandomState(42)
    y[rng.choice(N, size=n_fraud, replace=False)] = 1

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    prevalences = []

    for _, val_idx in skf.split(np.zeros((N, 2)), y):
        prev = np.mean(y[val_idx])
        prevalences.append(prev)
        # Each fold should have either 3 or 4 frauds in 2000 samples (~0.15% - 0.20%)
        assert 0.0010 <= prev <= 0.0025

    # Mean prevalence across folds equals true prevalence
    assert np.isclose(np.mean(prevalences), n_fraud / N, atol=1e-5)


def test_bandwidth_separable_kernel_properties() -> None:
    """Verifies symmetry, diagonal unity, positive semi-definiteness of separable kernel."""
    rng = np.random.RandomState(42)
    X = rng.uniform(0.0, np.pi, size=(30, 6))

    for beta in [0.2, 0.5, 1.0]:
        K = compute_bandwidth_separable_kernel_matrix(X, X, beta=beta)
        assert K.shape == (30, 30)
        # Symmetry
        assert np.max(np.abs(K - K.T)) < 1e-12
        # Diagonal unity
        assert np.max(np.abs(np.diag(K) - 1.0)) < 1e-12
        # Unit interval
        assert np.all(K >= 0.0) and np.all(K <= 1.0 + 1e-12)
        # PSD
        eigs = np.linalg.eigvalsh(K)
        assert np.min(eigs) >= -1e-10


def test_anisotropic_kernel_properties() -> None:
    """Verifies anisotropic feature-weighted kernel satisfies Gram matrix validity."""
    rng = np.random.RandomState(42)
    X = rng.uniform(0.0, np.pi, size=(25, 6))
    weights = np.array([0.2, 0.5, 0.8, 1.0, 1.2, 1.5])

    K = compute_anisotropic_kernel_matrix(X, X, weights=weights)
    assert K.shape == (25, 25)
    assert np.max(np.abs(K - K.T)) < 1e-12
    assert np.max(np.abs(np.diag(K) - 1.0)) < 1e-12
    eigs = np.linalg.eigvalsh(K)
    assert np.min(eigs) >= -1e-10


def test_paired_bootstrap_comparison_logic() -> None:
    """Verifies that paired bootstrap correctly computes differences, CIs, and directions."""
    rng = np.random.RandomState(42)
    N = 500
    y_true = rng.choice([0, 1], size=N, p=[0.90, 0.10])
    
    # Model A is strictly better than Model B
    scores_a = y_true * 2.0 + rng.randn(N) * 0.5
    scores_b = y_true * 0.5 + rng.randn(N) * 1.5

    res = compute_paired_bootstrap_comparison(
        y_true=y_true,
        scores_a=scores_a,
        scores_b=scores_b,
        model_a_name="ModelA",
        model_b_name="ModelB",
        n_bootstraps=200,
        random_state=42,
    )

    # Subtraction direction: ModelA minus ModelB
    assert res["subtraction_direction"] == "Delta = Metric(ModelA) - Metric(ModelB)"
    
    # Delta PR-AUC must be positive and significant
    d_pr = res["delta_pr_auc"]
    assert d_pr["mean"] > 0.0
    assert d_pr["ci_95"][0] > 0.0
    assert d_pr["statistically_significant"] is True

    # When comparing identical models, Delta must be approximately 0
    res_ident = compute_paired_bootstrap_comparison(
        y_true=y_true,
        scores_a=scores_a,
        scores_b=scores_a,
        model_a_name="ModelA",
        model_b_name="ModelA",
        n_bootstraps=100,
        random_state=42,
    )
    assert np.isclose(res_ident["delta_pr_auc"]["mean"], 0.0, atol=1e-4)
    assert res_ident["delta_pr_auc"]["statistically_significant"] is False


def test_financial_cost_optimization_monotonicity() -> None:
    """Verifies that as cost ratio R increases, the calibrated threshold prioritizes recall."""
    rng = np.random.RandomState(42)
    N = 1000
    y_true = np.zeros(N, dtype=int)
    y_true[:20] = 1  # 20 frauds
    scores = rng.randn(N)
    scores[:20] += 2.0  # frauds have higher scores

    calib = calibrate_financial_cost_thresholds(
        y_true=y_true,
        scores=scores,
        cost_ratios=[5.0, 20.0, 50.0, 100.0],
        c_fp=10.0,
        n_threshold_candidates=100,
    )

    # Check that recall at R=100 is >= recall at R=5
    rec_r5 = calib["5"]["recall"]
    rec_r100 = calib["100"]["recall"]
    assert rec_r100 >= rec_r5, f"Expected higher or equal recall at R=100 ({rec_r100}) vs R=5 ({rec_r5})"

    # Check exact formula for loss: C = c_fp * FP + c_fn * FN
    for r_str, data in calib.items():
        fp = data["false_positives"]
        fn = data["false_negatives"]
        c_fp = data["c_fp"]
        c_fn = data["c_fn"]
        expected_cost = c_fp * fp + c_fn * fn
        assert np.isclose(data["financial_cost"], expected_cost, atol=1e-2)
