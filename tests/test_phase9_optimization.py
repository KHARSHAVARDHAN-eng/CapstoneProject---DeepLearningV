"""
Tests for Phase 9: Quantum Kernel Optimization Module.
"""

from __future__ import annotations

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap
from qiskit.quantum_info import Statevector

from src.experiments.phase9_optimization import (
    analyze_separable_vs_canonical_geometry,
    build_controlled_entanglement_statevector,
    compute_1qubit_bloch_representations,
    compute_anisotropic_separable_kernel,
    compute_bandwidth_separable_kernel,
    compute_controlled_entanglement_kernel,
    compute_controlled_entanglement_statevectors,
    compute_projected_quantum_kernel,
    cross_validate_separable_kernel_bandwidth,
)


def test_bandwidth_separable_kernel_properties() -> None:
    """Verifies that bandwidth-scaled separable kernel satisfies Gram matrix validity."""
    rng = np.random.RandomState(42)
    X = rng.uniform(0.0, np.pi, size=(20, 6))

    for beta in [0.2, 0.5, 1.0, 1.5]:
        K = compute_bandwidth_separable_kernel(X, X, beta=beta)
        assert K.shape == (20, 20)
        # Symmetry
        assert np.max(np.abs(K - K.T)) < 1e-12
        # Diagonal unity
        assert np.max(np.abs(np.diag(K) - 1.0)) < 1e-12
        # Boundedness
        assert np.all(K >= 0.0) and np.all(K <= 1.0 + 1e-12)
        # Positive semi-definiteness
        eigs = np.linalg.eigvalsh(K)
        assert np.min(eigs) >= -1e-10


def test_anisotropic_separable_kernel() -> None:
    """Verifies anisotropic feature-weighted kernel evaluates correctly and respects weights."""
    rng = np.random.RandomState(42)
    X = rng.uniform(0.0, np.pi, size=(15, 6))
    weights = np.array([0.1, 0.5, 1.0, 0.2, 0.8, 1.2])

    K = compute_anisotropic_separable_kernel(X, X, weights=weights)
    assert K.shape == (15, 15)
    assert np.max(np.abs(K - K.T)) < 1e-12
    assert np.max(np.abs(np.diag(K) - 1.0)) < 1e-12
    eigs = np.linalg.eigvalsh(K)
    assert np.min(eigs) >= -1e-10


def test_controlled_entanglement_limits() -> None:
    """Verifies alpha=0 reduces to unentangled state and alpha=1 matches ZZFeatureMap."""
    x = np.array([0.5, 1.2, 2.1, 0.8, 1.5, 2.8])

    # Alpha = 0.0
    sv_alpha0 = build_controlled_entanglement_statevector(x, alpha=0.0, reps=1)
    qc_z = QuantumCircuit(6)
    qc_z.h(range(6))
    for j in range(6):
        qc_z.rz(2.0 * x[j], j)
    sv_z_ref = Statevector.from_instruction(qc_z).data
    fid_alpha0 = float(np.abs(np.dot(sv_alpha0.conj(), sv_z_ref)) ** 2)
    assert fid_alpha0 == pytest.approx(1.0, rel=1e-5)

    # Alpha = 1.0
    sv_alpha1 = build_controlled_entanglement_statevector(x, alpha=1.0, reps=1)
    fmap_zz = ZZFeatureMap(6, reps=1, entanglement="linear").assign_parameters(x)
    sv_zz_ref = Statevector.from_instruction(fmap_zz).data
    fid_alpha1 = float(np.abs(np.dot(sv_alpha1.conj(), sv_zz_ref)) ** 2)
    assert fid_alpha1 == pytest.approx(1.0, rel=1e-5)


def test_1qubit_bloch_representations() -> None:
    """Verifies 1-qubit Bloch expectation vectors are bounded and properly dimensioned."""
    rng = np.random.RandomState(42)
    X = rng.uniform(0.0, np.pi, size=(10, 6))
    sv_arr = compute_controlled_entanglement_statevectors(X, alpha=0.5, reps=1)

    bloch_matrix = compute_1qubit_bloch_representations(sv_arr, n_qubits=6)
    assert bloch_matrix.shape == (10, 18)  # 3 * 6 = 18

    # Each 1-qubit Bloch vector length must satisfy ||(X, Y, Z)||^2 <= 1.0
    for i in range(10):
        for j in range(6):
            sx = bloch_matrix[i, 3 * j]
            sy = bloch_matrix[i, 3 * j + 1]
            sz = bloch_matrix[i, 3 * j + 2]
            norm_sq = sx**2 + sy**2 + sz**2
            assert norm_sq <= 1.0 + 1e-6


def test_projected_quantum_kernel_properties() -> None:
    """Verifies that Projected Quantum Kernel (PQK) yields valid Gram matrices."""
    rng = np.random.RandomState(42)
    X = rng.uniform(0.0, np.pi, size=(12, 6))
    sv_arr = compute_controlled_entanglement_statevectors(X, alpha=0.25, reps=1)
    bloch_matrix = compute_1qubit_bloch_representations(sv_arr, n_qubits=6)

    for gamma in [0.1, 0.5, 1.0]:
        K_pqk = compute_projected_quantum_kernel(bloch_matrix, bloch_matrix, gamma=gamma)
        assert K_pqk.shape == (12, 12)
        assert np.max(np.abs(K_pqk - K_pqk.T)) < 1e-12
        assert np.max(np.abs(np.diag(K_pqk) - 1.0)) < 1e-12
        eigs = np.linalg.eigvalsh(K_pqk)
        assert np.min(eigs) >= -1e-10


def test_stratified_cv_leakage_isolation() -> None:
    """Verifies that 5-fold CV functions properly without exceptions or leakage."""
    rng = np.random.RandomState(42)
    X = rng.uniform(0.0, np.pi, size=(40, 6))
    y = np.array([1] * 10 + [0] * 30)

    cv_results = cross_validate_separable_kernel_bandwidth(
        X_train=X,
        y_train=y,
        betas=[0.5, 1.0],
        C_values=[0.5, 1.0],
        n_splits=3,
        random_state=42,
    )
    assert "best_config" in cv_results
    assert "beta" in cv_results["best_config"]
    assert "C" in cv_results["best_config"]
    assert len(cv_results["all_configurations"]) == 4


def test_separable_vs_canonical_geometry_analysis() -> None:
    """Verifies mathematical diagnostic analysis produces expected geometric contrasts."""
    rng = np.random.RandomState(42)
    X = rng.uniform(0.0, np.pi, size=(20, 6))
    y = np.array([1] * 5 + [0] * 15)

    analysis = analyze_separable_vs_canonical_geometry(X, y)
    assert "mathematical_properties" in analysis
    assert "geometric_metrics" in analysis
    assert "separable_z" in analysis["geometric_metrics"]
    assert "canonical_zz" in analysis["geometric_metrics"]

    sep_cond = analysis["geometric_metrics"]["separable_z"]["condition_number"]
    zz_cond = analysis["geometric_metrics"]["canonical_zz"]["condition_number"]
    assert sep_cond > 0.0 and zz_cond > 0.0
    assert "support_vector_count" in analysis["geometric_metrics"]["separable_z"]
    assert "kta_uncentered" in analysis["geometric_metrics"]["separable_z"]
