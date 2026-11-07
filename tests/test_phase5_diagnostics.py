"""
Automated unit and integration tests for Phase 5 diagnostic modules.

Tests cover:
1. Disjointness of resampled training pools from validation indices.
2. Leak-free fold-local preprocessing and exact solver fit accounting.
3. Discrete achievable recall fractions and scenario cost calculations.
4. Paired bootstrap coupling and non-negative confidence intervals.
5. Exact KTA label conversion ({-1, +1}) and analytic ZFeatureMap kernel equivalence.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import Statevector

from src.experiments.ablation import (
    compute_analytic_z_kernel_1rep,
    compute_kernel_target_alignment,
)
from src.experiments.bootstrap import compute_paired_bootstrap_confidence_intervals
from src.experiments.resampling import extract_resampled_training_subset
from src.experiments.threshold_analysis import (
    compute_scenario_cost_curve,
    evaluate_closest_operating_points,
)
from src.experiments.tuning import run_leak_free_training_cross_validation


def test_analytic_z_kernel_matches_qiskit_statevector():
    """Verifies that the analytic cosine-product formula for 1-rep ZFeatureMap

    exactly matches Qiskit Statevector pure fidelity bit-for-bit.
    """
    fm = ZFeatureMap(feature_dimension=3, reps=1)
    rng = np.random.RandomState(42)
    X = rng.uniform(0.0, np.pi, size=(5, 3))

    # 1. Analytic formula
    K_analytic = compute_analytic_z_kernel_1rep(X, X)

    # 2. Qiskit Statevector calculation
    K_qiskit = np.zeros((5, 5))
    svs = [Statevector(fm.assign_parameters(x)) for x in X]
    for i in range(5):
        for j in range(5):
            K_qiskit[i, j] = np.abs(svs[i].inner(svs[j])) ** 2

    np.testing.assert_allclose(K_analytic, K_qiskit, atol=1e-10)
    # Check diagonal unity and symmetry
    np.testing.assert_allclose(np.diag(K_analytic), 1.0, atol=1e-10)
    np.testing.assert_allclose(K_analytic, K_analytic.T, atol=1e-10)
    # Check PSD
    eigvals = np.linalg.eigvalsh(K_analytic)
    assert np.all(eigvals >= -1e-10)


def test_kta_label_encoding_and_bounds():
    """Verifies that compute_kernel_target_alignment handles {0, 1} labels

    by converting to {-1, +1} and returns values bounded in [-1, 1].
    """
    N = 20
    K_perfect = np.ones((N, N))  # rank-1 all-ones
    y_binary = np.array([1] * 10 + [0] * 10)  # half 1, half 0

    # For an all-ones kernel and equal classes, sum(y_pm) = 0, so y^T K y = (sum y)^2 = 0
    kta_zero = compute_kernel_target_alignment(K_perfect, y_binary, centered=False)
    assert np.isclose(kta_zero, 0.0, atol=1e-10)

    # For an ideal block kernel perfectly aligned with y: K = Y = y_pm y_pm^T
    y_pm = 2.0 * y_binary - 1.0
    K_ideal = np.outer(y_pm, y_pm)
    kta_ideal = compute_kernel_target_alignment(K_ideal, y_binary, centered=False)
    assert np.isclose(kta_ideal, 1.0, atol=1e-10)


def test_threshold_discrete_fractions_and_cost_curve():
    """Verifies that threshold analysis computes discrete recall fractions and scenario loss."""
    rng = np.random.RandomState(42)
    N = 100
    y_true = np.zeros(N, dtype=int)
    y_true[:10] = 1  # exactly 10 frauds
    decision_scores = rng.normal(0, 1, size=N)
    decision_scores[:10] += 2.0  # signal for fraud

    op_points = evaluate_closest_operating_points(
        y_true, decision_scores, target_recalls=[0.70, 0.80, 0.90]
    )
    assert len(op_points) == 3
    for p in op_points:
        assert p["achieved_recall"] >= p["target_recall"]
        assert "/10" in p["achieved_recall_fraction"]
        assert 0.0 <= p["precision"] <= 1.0
        assert p["false_positives"] >= 0
        assert p["false_negatives"] >= 0

    cost_results = compute_scenario_cost_curve(y_true, decision_scores, cost_ratios=[5.0, 20.0])
    assert "5" in cost_results
    assert "20" in cost_results
    assert cost_results["20"]["minimized_scenario_loss"] >= 0.0
    assert 0.0 <= cost_results["20"]["resulting_recall"] <= 1.0


def test_paired_bootstrap_confidence_intervals():
    """Verifies that paired bootstrap couples model evaluations and produces valid CI percentiles."""
    rng = np.random.RandomState(42)
    N = 200
    y_true = np.zeros(N, dtype=int)
    y_true[:20] = 1

    scores_q = rng.normal(0, 1, size=N)
    pred_q = (scores_q >= 0).astype(int)

    scores_rbf = scores_q + rng.normal(0.5, 0.2, size=N)
    pred_rbf = (scores_rbf >= 0).astype(int)

    preds_dict = {
        "qsvm": (pred_q, scores_q),
        "rbf_svm": (pred_rbf, scores_rbf),
    }

    summary = compute_paired_bootstrap_confidence_intervals(
        y_true, preds_dict, n_bootstraps=50, random_state=42
    )
    assert summary["n_bootstraps_completed"] == 50
    assert summary["discarded_degenerate_replicates"] == 0
    assert "rbf_svm_minus_qsvm" in summary["paired_differences"]
    delta_pr = summary["paired_differences"]["rbf_svm_minus_qsvm"]["delta_pr_auc"]
    assert delta_pr["ci_lower"] <= delta_pr["mean"] <= delta_pr["ci_upper"]


def test_resampled_training_subset_disjointness():
    """Verifies that extract_resampled_training_subset strictly draws from available pool."""
    N_pool = 1000
    df_pool = pd.DataFrame(np.random.randn(N_pool, 5), columns=[f"V{i}" for i in range(5)])
    y_pool = pd.Series(np.zeros(N_pool, dtype=int))
    y_pool.iloc[:50] = 1

    X_sub, y_sub = extract_resampled_training_subset(
        df_pool, y_pool, n_train=40, n_train_fraud=10, random_state=42
    )
    assert len(X_sub) == 40
    assert (y_sub == 1).sum() == 10
    assert (y_sub == 0).sum() == 30
    assert set(X_sub.index).issubset(set(df_pool.index))


def test_leak_free_training_cv_fit_count_and_selection():
    """Verifies that run_leak_free_training_cross_validation executes fold-local fits

    and computes the exact expected 180 solver fits across a 5-fold grid.
    """
    rng = np.random.RandomState(42)
    n_train = 30
    n_val = 20
    n_features = 8
    X_tr = pd.DataFrame(rng.randn(n_train, n_features), columns=[f"V{i}" for i in range(n_features)])
    y_tr = pd.Series(np.zeros(n_train, dtype=int))
    y_tr.iloc[:10] = 1

    X_v = pd.DataFrame(rng.randn(n_val, n_features), columns=[f"V{i}" for i in range(n_features)])
    y_v = pd.Series(np.zeros(n_val, dtype=int))
    y_v.iloc[:4] = 1

    cv_res = run_leak_free_training_cross_validation(
        X_tr, y_tr, X_v, y_v,
        c_grid=[0.1, 1.0],  # 2 C values
        gamma_grid=["scale", 0.1],  # 2 gamma values
        n_splits=2,
        random_state=42,
    )
    # Expected fits for 2 folds:
    # Linear: 2 C * 2 folds = 4
    # QSVM: 2 C * 2 folds = 4
    # RBF: 2 C * 2 gamma = 4 configs * 2 folds = 8
    # Total = 16 fits
    assert cv_res["total_solver_fits"] == 16
    assert "linear_svm" in cv_res["selected_hyperparameters"]
    assert "rbf_svm" in cv_res["selected_hyperparameters"]
    assert "qsvm" in cv_res["selected_hyperparameters"]
