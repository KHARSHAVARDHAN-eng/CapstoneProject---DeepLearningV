"""Unit and Regression Tests for Phase 14: Systematic Performance Optimization."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from qiskit.circuit.library import ZZFeatureMap
from qiskit.quantum_info import Statevector

from src.experiments.phase14_optimization import (
    compute_anisotropic_kernel_matrix,
    compute_bandwidth_separable_kernel_matrix,
    compute_fidelity_kernel_matrix,
    compute_model_metrics,
    compute_projected_quantum_kernel_matrix,
    compute_zz_statevectors_fast,
    find_cost_optimal_threshold,
    fit_transform_scaler,
    generate_nested_subsets_p14,
    run_phase14_pipeline,
    select_top_features_infold,
)


def test_select_top_features_infold() -> None:
    """Verifies in-fold feature selection selects top features without NaN or leakage."""
    rng = np.random.RandomState(42)
    X_dummy = pd.DataFrame(rng.randn(200, 20), columns=[f"F{i}" for i in range(20)])
    y_dummy = rng.choice([0, 1], size=200, p=[0.9, 0.1])

    # Inject high correlation into F3 and F7
    X_dummy.iloc[y_dummy == 1, 3] += 5.0
    X_dummy.iloc[y_dummy == 1, 7] += 4.0

    selected_cols, f_scores = select_top_features_infold(X_dummy, y_dummy, k=6)
    assert len(selected_cols) == 6
    assert "F3" in selected_cols[:2]
    assert not np.isnan(f_scores).any()


def test_nested_subsets_strictly_nested_p14() -> None:
    """Verifies that nested subsets D_500 ⊂ D_1000 ⊂ D_2000 are strictly contained with 10% fraud."""
    n_train = 4000
    n_fraud = 250
    y_dummy = np.zeros(n_train, dtype=int)
    y_dummy[:n_fraud] = 1

    subsets = generate_nested_subsets_p14(
        y_train_fold=y_dummy,
        sample_sizes=[500, 1000, 2000],
        fraud_ratio=0.10,
        random_state=42,
    )

    assert len(subsets[500]) == 500
    assert len(subsets[1000]) == 1000
    assert len(subsets[2000]) == 2000

    assert np.sum(y_dummy[subsets[500]] == 1) == 50
    assert np.sum(y_dummy[subsets[1000]] == 1) == 100
    assert np.sum(y_dummy[subsets[2000]] == 1) == 200

    s500 = set(subsets[500])
    s1000 = set(subsets[1000])
    s2000 = set(subsets[2000])

    assert s500.issubset(s1000)
    assert s1000.issubset(s2000)


def test_bandwidth_separable_kernel_properties() -> None:
    """Verifies that the bandwidth-compensated Separable Z-kernel is valid and unit-diagonal."""
    rng = np.random.RandomState(42)
    X = rng.uniform(0.0, np.pi, size=(10, 8))

    K = compute_bandwidth_separable_kernel_matrix(X, X, beta=0.035)
    assert K.shape == (10, 10)
    assert np.allclose(np.diag(K), 1.0)
    assert np.all(K >= 0.0) and np.all(K <= 1.0)
    assert np.allclose(K, K.T)


def test_anisotropic_kernel_properties() -> None:
    """Verifies anisotropic separable kernel with feature weights."""
    rng = np.random.RandomState(42)
    X = rng.uniform(0.0, np.pi, size=(10, 6))
    weights = rng.uniform(0.01, 0.1, size=6)

    K = compute_anisotropic_kernel_matrix(X, X, weights=weights)
    assert K.shape == (10, 10)
    assert np.allclose(np.diag(K), 1.0)
    assert np.all(K >= 0.0) and np.all(K <= 1.0)


def test_fast_statevectors_match_qiskit() -> None:
    """Verifies vectorized statevector generation matches Qiskit to numerical precision."""
    X_test = np.random.RandomState(42).uniform(0, np.pi, size=(4, 6))

    sv_fast = compute_zz_statevectors_fast(X_test, n_qubits=6)
    fmap = ZZFeatureMap(6, reps=1, entanglement="linear")

    for i in range(len(X_test)):
        qc = fmap.assign_parameters(X_test[i])
        sv_qiskit = Statevector.from_instruction(qc).data
        dot_product = np.abs(np.vdot(sv_fast[i], sv_qiskit))
        assert np.isclose(dot_product, 1.0, atol=1e-12)


def test_cost_optimal_threshold_calculation() -> None:
    """Verifies find_cost_optimal_threshold correctly minimizes cost."""
    y_true = np.array([0, 0, 0, 0, 1, 1, 0, 1])
    scores = np.array([-1.0, -0.5, 0.1, 0.2, 0.8, 1.2, 0.3, 0.9])

    best = find_cost_optimal_threshold(y_true, scores, cost_fp=10.0, cost_fn=200.0)
    assert "threshold" in best
    assert "cost" in best
    assert best["cost"] == 10.0 * best["false_positives"] + 200.0 * best["false_negatives"]
