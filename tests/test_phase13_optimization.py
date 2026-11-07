"""Tests for Phase 13: Systematic Model Optimization Module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from qiskit.circuit.library import ZZFeatureMap
from qiskit.quantum_info import Statevector
from sklearn.svm import SVC

from src.experiments.phase13_optimization import (
    compute_anisotropic_kernel_matrix,
    compute_bandwidth_separable_kernel_matrix,
    compute_fidelity_kernel_matrix,
    compute_model_metrics,
    compute_zz_statevectors_fast,
    find_cost_optimal_threshold,
    fit_transform_scaler,
    generate_nested_subsets,
    score_kernel_chunked,
    select_top_features_infold,
)


def test_nested_subsets_strictly_nested_and_proportions() -> None:
    """Verifies that nested subsets are strictly contained D_500 ⊂ D_1000 ⊂ D_2000 with 10% fraud."""
    n_train = 5000
    n_fraud = 300
    y_dummy = np.zeros(n_train, dtype=int)
    y_dummy[:n_fraud] = 1

    subsets = generate_nested_subsets(
        y_train_fold=y_dummy,
        sample_sizes=[500, 1000, 2000],
        fraud_ratio=0.10,
        random_state=42,
    )

    # 1. Total counts and fraud proportions
    assert len(subsets[500]) == 500
    assert len(subsets[1000]) == 1000
    assert len(subsets[2000]) == 2000

    assert np.sum(y_dummy[subsets[500]] == 1) == 50
    assert np.sum(y_dummy[subsets[1000]] == 1) == 100
    assert np.sum(y_dummy[subsets[2000]] == 1) == 200

    assert np.sum(y_dummy[subsets[500]] == 0) == 450
    assert np.sum(y_dummy[subsets[1000]] == 0) == 900
    assert np.sum(y_dummy[subsets[2000]] == 0) == 1800

    # 2. Strict subset hierarchy
    s500 = set(subsets[500])
    s1000 = set(subsets[1000])
    s2000 = set(subsets[2000])

    assert s500.issubset(s1000)
    assert s1000.issubset(s2000)


def test_nested_subsets_no_duplicates() -> None:
    """Verifies zero duplicate rows exist in any generated subset."""
    n_train = 3000
    y_dummy = np.zeros(n_train, dtype=int)
    y_dummy[:250] = 1

    subsets = generate_nested_subsets(
        y_train_fold=y_dummy,
        sample_sizes=[500, 1000, 2000],
        fraud_ratio=0.10,
        random_state=101,
    )

    for N in [500, 1000, 2000]:
        idx = subsets[N]
        assert len(idx) == len(np.unique(idx)), f"Duplicates found in subset N={N}!"


def test_insufficient_fraud_raises_error() -> None:
    """Verifies that requesting more fraud than available raises a clear ValueError."""
    y_dummy = np.zeros(1000, dtype=int)
    y_dummy[:50] = 1  # only 50 frauds available

    # Requesting N=2,000 at 10% requires 200 frauds -> should fail!
    with pytest.raises(ValueError, match="Insufficient fraud samples"):
        generate_nested_subsets(
            y_train_fold=y_dummy,
            sample_sizes=[500, 1000, 2000],
            fraud_ratio=0.10,
            random_state=42,
        )


def test_fast_statevectors_exact_match_qiskit() -> None:
    """Verifies that vectorized statevector generation matches Qiskit to numerical precision."""
    X_test = np.random.RandomState(42).uniform(0, np.pi, size=(5, 6))

    # Vectorized fast method
    sv_fast = compute_zz_statevectors_fast(X_test, n_qubits=6)

    # Reference Qiskit method
    fmap = ZZFeatureMap(6, reps=1, entanglement="linear")
    sv_qiskit = np.array([
        Statevector.from_instruction(fmap.assign_parameters(x)).data
        for x in X_test
    ], dtype=np.complex128)

    max_diff = np.max(np.abs(sv_fast - sv_qiskit))
    assert max_diff < 1e-12, f"Discrepancy with Qiskit statevectors: {max_diff}"


def test_separable_kernel_mathematical_properties() -> None:
    """Verifies symmetry, diagonal unity, boundedness, and positive semi-definiteness."""
    X = np.random.RandomState(42).uniform(0, np.pi, size=(30, 6))
    K = compute_bandwidth_separable_kernel_matrix(X, X, beta=0.50)

    assert K.shape == (30, 30)
    assert np.allclose(K, K.T, atol=1e-12), "Gram matrix must be symmetric!"
    assert np.allclose(np.diag(K), 1.0, atol=1e-12), "Diagonal entries must be 1.0!"
    assert np.all(K >= 0.0) and np.all(K <= 1.0 + 1e-12), "Values must be in [0, 1]!"

    eigvals = np.linalg.eigvalsh(K)
    assert np.min(eigvals) >= -1e-10, "Gram matrix must be positive semi-definite!"


def test_infold_feature_selection_isolation() -> None:
    """Verifies that ANOVA F-statistic feature selection returns 6 features correctly."""
    rng = np.random.RandomState(42)
    df = pd.DataFrame(rng.randn(200, 10), columns=[f"feat_{i}" for i in range(10)])
    y = rng.choice([0, 1], size=200, p=[0.9, 0.1])

    feats, scores = select_top_features_infold(df, y, k=6)
    assert len(feats) == 6
    assert len(scores) == 6
    assert all(f in df.columns for f in feats)


def test_scaler_isolation() -> None:
    """Verifies MinMaxScaler maps training subset strictly to [0, pi]."""
    rng = np.random.RandomState(42)
    X_tr = rng.randn(100, 6) * 10
    X_va = rng.randn(50, 6) * 10

    X_tr_s, X_va_s, scaler = fit_transform_scaler(X_tr, X_va)
    assert np.isclose(X_tr_s.min(), 0.0)
    assert np.isclose(X_tr_s.max(), np.pi)


def test_chunked_scoring_matches_unvectorized() -> None:
    """Verifies that chunked decision function scoring matches full batch scoring."""
    rng = np.random.RandomState(42)
    X_tr = rng.uniform(0, np.pi, size=(50, 6))
    X_va = rng.uniform(0, np.pi, size=(120, 6))
    y_tr = rng.choice([0, 1], size=50, p=[0.8, 0.2])

    K_tr = compute_bandwidth_separable_kernel_matrix(X_tr, X_tr, beta=0.50)
    clf = SVC(kernel="precomputed", C=1.0)
    clf.fit(K_tr, y_tr)

    # Full batch scoring
    K_va_full = compute_bandwidth_separable_kernel_matrix(X_va, X_tr, beta=0.50)
    scores_full = clf.decision_function(K_va_full)

    # Chunked scoring with chunk_size = 25
    def get_chunk(s_idx: int, e_idx: int) -> np.ndarray:
        return compute_bandwidth_separable_kernel_matrix(X_va[s_idx:e_idx], X_tr, beta=0.50)

    scores_chunked = score_kernel_chunked(clf, get_chunk, len(X_va), chunk_size=25)

    assert np.allclose(scores_full, scores_chunked, atol=1e-12)


def test_metrics_and_cost_threshold() -> None:
    """Verifies metric computation and cost-optimal threshold selection."""
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 0, 0, 1])
    scores = np.array([-1.5, -0.8, -0.2, 0.1, 0.5, 0.8, 1.2, -0.5, -0.1, 0.9])

    metrics = compute_model_metrics(y_true, scores, threshold=0.0)
    assert 0.0 <= metrics["pr_auc"] <= 1.0
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert metrics["true_positives"] + metrics["false_negatives"] == 4

    thresh, cost, calib = find_cost_optimal_threshold(y_true, scores, cost_ratio=20.0, c_fp=10.0)
    assert cost >= 0.0
    assert "cost" in calib
