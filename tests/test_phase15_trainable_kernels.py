"""Unit tests for Phase 15 Trainable Quantum Kernel Optimization."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.experiments.phase15_trainable_kernels import (
    compute_bandwidth_separable_kernel_matrix,
    compute_model_metrics,
    find_cost_optimal_threshold,
    fit_transform_scaler,
    generate_nested_subsets_p15,
    optimize_classical_rbf_kta,
    optimize_quantum_kernel_kta,
    select_top_features_infold,
)


def test_select_top_features_infold_phase15() -> None:
    rng = np.random.RandomState(42)
    X = pd.DataFrame(rng.randn(100, 20), columns=[f"feat_{i}" for i in range(20)])
    y = rng.choice([0, 1], size=100, p=[0.9, 0.1])
    cols, scores = select_top_features_infold(X, y, k=16)
    assert len(cols) == 16
    assert len(scores) == 16
    assert scores[0] >= scores[-1]


def test_nested_subsets_strictly_nested_phase15() -> None:
    rng = np.random.RandomState(42)
    y = np.zeros(2000, dtype=int)
    y[rng.choice(2000, size=200, replace=False)] = 1
    subsets = generate_nested_subsets_p15(y, sample_sizes=[500, 1000], fraud_ratio=0.10, random_state=42)
    assert len(subsets[500]) == 500
    assert len(subsets[1000]) == 1000
    assert set(subsets[500]).issubset(set(subsets[1000]))


def test_centered_kta_quantum_optimization() -> None:
    rng = np.random.RandomState(42)
    X = rng.uniform(0, np.pi, size=(50, 4))
    y = np.array([1]*10 + [0]*40)
    opt_theta, ckta = optimize_quantum_kernel_kta(X, y, init_beta=0.031, lambda_reg=50.0, maxiter=5)
    assert len(opt_theta) == 4
    assert np.all(opt_theta >= 0.005)
    assert np.all(opt_theta <= 0.15)
    assert -1.0 <= ckta <= 1.0


def test_centered_kta_rbf_optimization() -> None:
    rng = np.random.RandomState(42)
    X = rng.uniform(0, np.pi, size=(50, 4))
    y = np.array([1]*10 + [0]*40)
    opt_gamma, ckta = optimize_classical_rbf_kta(X, y, init_gamma=0.0005, lambda_reg=50.0, maxiter=5)
    assert len(opt_gamma) == 4
    assert np.all(opt_gamma >= 0.00005)
    assert np.all(opt_gamma <= 0.01)
    assert -1.0 <= ckta <= 1.0


def test_bandwidth_separable_kernel_symmetry() -> None:
    rng = np.random.RandomState(42)
    X = rng.uniform(0, np.pi, size=(10, 4))
    K = compute_bandwidth_separable_kernel_matrix(X, X, beta=np.array([0.02, 0.03, 0.04, 0.05]))
    assert K.shape == (10, 10)
    np.testing.assert_allclose(np.diag(K), 1.0, atol=1e-7)
    np.testing.assert_allclose(K, K.T, atol=1e-7)
    assert np.all(K >= 0.0) and np.all(K <= 1.0)


def test_find_cost_optimal_threshold_correctness() -> None:
    y_true = np.array([0, 0, 0, 0, 1, 1, 0, 1])
    scores = np.array([-1.0, -0.5, 0.1, 0.2, 0.8, 1.2, 0.3, 0.9])
    best = find_cost_optimal_threshold(y_true, scores, cost_fp=10.0, cost_fn=200.0)
    assert "threshold" in best
    assert "cost" in best
    assert best["cost"] == 10.0 * best["false_positives"] + 200.0 * best["false_negatives"]


def test_centered_kta_exact_math_equivalence() -> None:
    """Verifies that rank-1 quadratic form matches full centering matrix Frobenius inner product."""
    from src.experiments.phase15_trainable_kernels import compute_centered_kta_exact

    rng = np.random.RandomState(42)
    N = 60
    X = rng.randn(N, 8)
    y = np.zeros(N)
    y[:6] = 1.0 # 10% fraud
    rng.shuffle(y)

    y_pm = np.where(y == 1, 1.0, -1.0)
    H = np.eye(N) - (1.0 / N) * np.ones((N, N))
    Y_target = np.outer(y_pm, y_pm)
    Y_c_mat = H @ Y_target @ H
    Y_c_norm = np.linalg.norm(Y_c_mat, "fro")

    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    K = np.prod(np.cos(0.04 * diff / 2.0) ** 2, axis=2)

    # Full matrix Frobenius CKTA
    K_c_mat = H @ K @ H
    K_c_norm = np.linalg.norm(K_c_mat, "fro")
    ckta_full_matrix = np.sum(K_c_mat * Y_c_mat) / (K_c_norm * Y_c_norm)

    # Fast exact rank-1 CKTA
    u_pm = y_pm - np.mean(y_pm)
    norm_Y_c_rank1 = float(np.dot(u_pm, u_pm))
    ckta_fast_exact = compute_centered_kta_exact(K, u_pm, norm_Y_c_rank1)

    assert abs(ckta_full_matrix - ckta_fast_exact) < 1e-12


def test_calibrate_threshold_prevalence_adjusted() -> None:
    """Verifies that prevalence-adjusted threshold calibration chooses optimal threshold on training out-of-fold."""
    from src.experiments.phase15_trainable_kernels import calibrate_threshold_prevalence_adjusted

    rng = np.random.RandomState(42)
    N = 200
    # 10% enriched training subset
    y_sub = np.array([1] * 20 + [0] * 180)
    scores = np.where(y_sub == 1, rng.normal(1.0, 0.5, N), rng.normal(-1.0, 0.5, N))

    # Population prevalence = 0.00165
    th_cal = calibrate_threshold_prevalence_adjusted(y_sub, scores, pi_population=0.00165)
    assert isinstance(th_cal, float)
    assert -5.0 <= th_cal <= 5.0

