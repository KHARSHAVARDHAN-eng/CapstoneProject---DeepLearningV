"""
Phase 5: Quantum Feature Map Mathematical Ablation Module.

Evaluates:
1. Case A: 1-Repetition ZFeatureMap (Separable, analytic cosine-product pure-state fidelity).
2. Case B: 2-Repetition ZFeatureMap (Separable, alternating Hadamard statevector simulation).
3. Case C: Canonical 2-Repetition ZZFeatureMap (Entangled linear CNOT-P-CNOT statevector simulation).

Implements:
- Numerically validated Kernel-Target Alignment (KTA) with explicit {-1, +1} label conversion.
- Gram matrix numerical diagnostics (Symmetry, Diagonal Unity, Positive Semidefiniteness, Condition Number).
"""

import logging
from typing import Any, Dict, Tuple
import numpy as np
from qiskit.circuit.library import ZFeatureMap
from sklearn.svm import SVC

from src.models.quantum_kernel import (
    create_zz_feature_map,
    compute_fidelity_kernel_matrix,
    validate_kernel_matrix,
)
from src.models.evaluate import compute_classification_metrics

logger = logging.getLogger("quantum_fraud.ablation")


def compute_kernel_target_alignment(
    K: np.ndarray,
    y: np.ndarray,
    centered: bool = False,
) -> float:
    """Computes mathematically rigorous Kernel-Target Alignment (KTA).

    Explicitly maps labels from {0, 1} to {-1, +1} via y_pm = 2y - 1,
    guaranteeing that ||Y||_F = ||y_pm y_pm^T||_F = N is exact and accounts
    symmetrically for both within-class clustering (+1) and between-class separation (-1).

    Args:
        K: Symmetric Gram matrix of shape (N, N).
        y: Binary class labels (0 or 1, or already -1 and +1).
        centered: If True, centers K and Y using projection matrix H = I - 1/N * J.

    Returns:
        Scalar alignment score in [-1.0, 1.0].
    """
    N = len(y)
    # Explicit conversion to {-1, +1}
    y_arr = np.asarray(y, dtype=float)
    if set(np.unique(y_arr)).issubset({0.0, 1.0}):
        y_pm = 2.0 * y_arr - 1.0
    else:
        y_pm = y_arr

    Y = np.outer(y_pm, y_pm)

    if centered:
        H = np.eye(N) - (1.0 / N) * np.ones((N, N))
        K_c = H @ K @ H
        Y_c = H @ Y @ H
        norm_K = np.linalg.norm(K_c, ord="fro")
        norm_Y = np.linalg.norm(Y_c, ord="fro")
        if norm_K == 0 or norm_Y == 0:
            return 0.0
        return float(np.sum(K_c * Y_c) / (norm_K * norm_Y))
    else:
        # Standard uncentered KTA: <K, Y>_F / (||K||_F * ||Y||_F)
        # For {-1, +1} labels, ||Y||_F = sqrt(sum (y_i y_j)^2) = sqrt(N^2) = N.
        norm_K = np.linalg.norm(K, ord="fro")
        if norm_K == 0:
            return 0.0
        numerator = float(y_pm @ K @ y_pm)
        denominator = float(norm_K * N)
        return float(numerator / denominator)


def compute_analytic_z_kernel_1rep(
    X1: np.ndarray,
    X2: np.ndarray,
) -> np.ndarray:
    """Computes exact closed-form pure-state fidelity for 1-repetition ZFeatureMap.

    For 1-rep ZFeatureMap with angles in [0, π]:
    |psi(x)> = Tensor_k [ 1/sqrt(2) |0> + e^{i 2 x_k}/sqrt(2) |1> ]
    Fidelity: |<psi(x')|psi(x)>|^2 = Prod_{k=1}^n cos^2(x_k - x_k').

    This is an exact classical cosine-product kernel requiring zero circuit compilation.

    Args:
        X1: Feature matrix of shape (N1, n_features).
        X2: Feature matrix of shape (N2, n_features).

    Returns:
        Exact Gram / cross-kernel matrix of shape (N1, N2) bounded in [0, 1].
    """
    # X1 shape: (N1, 1, n_features), X2 shape: (1, N2, n_features)
    diff = X1[:, np.newaxis, :] - X2[np.newaxis, :, :]  # (N1, N2, n_features)
    cos_sq = np.cos(diff) ** 2  # (N1, N2, n_features)
    kernel_matrix = np.prod(cos_sq, axis=2)  # (N1, N2)
    return np.clip(kernel_matrix, 0.0, 1.0)


def run_feature_map_ablation(
    X_train_scaled: np.ndarray,
    y_train: np.ndarray,
    X_val_scaled: np.ndarray,
    y_val: np.ndarray,
) -> Dict[str, Any]:
    """Runs the controlled feature map ablation across Cases A, B, and C.

    Evaluates:
    - Case A: 1-Repetition ZFeatureMap (Analytic separable cosine product)
    - Case B: 2-Repetition ZFeatureMap (Separable alternating Hadamard statevector simulation)
    - Case C: 2-Repetition ZZFeatureMap (Canonical entangled statevector simulation)

    Args:
        X_train_scaled: Scaled training features in [0, π] of shape (500, 6).
        y_train: Binary training labels (500,).
        X_val_scaled: Scaled validation features in [0, π] of shape (20000, 6).
        y_val: Binary validation labels (20000,).

    Returns:
        Dictionary containing diagnostics, alignments, condition numbers, and validation metrics.
    """
    n_features = X_train_scaled.shape[1]
    results = {}

    # -------------------------------------------------------------------------
    # Case A: 1-Repetition ZFeatureMap (Analytic Separable)
    # -------------------------------------------------------------------------
    logger.info("Computing Case A: 1-rep ZFeatureMap (Analytic Separable)...")
    K_train_a = compute_analytic_z_kernel_1rep(X_train_scaled, X_train_scaled)
    K_val_a = compute_analytic_z_kernel_1rep(X_val_scaled, X_train_scaled)

    diag_a = validate_kernel_matrix(K_train_a)
    kta_a = compute_kernel_target_alignment(K_train_a, y_train, centered=False)
    kta_a_c = compute_kernel_target_alignment(K_train_a, y_train, centered=True)

    clf_a = SVC(kernel="precomputed", class_weight="balanced", C=1.0, random_state=42)
    clf_a.fit(K_train_a, y_train)
    pred_a = clf_a.predict(K_val_a)
    score_a = clf_a.decision_function(K_val_a)
    metrics_a = compute_classification_metrics(y_val, pred_a, score_a)

    results["case_a_z_1rep_separable"] = {
        "description": "1-Repetition ZFeatureMap (Separable, analytic cosine product)",
        "circuit_type": "Separable Product State (0 two-qubit entangling gates)",
        "repetitions": 1,
        "diagnostics": diag_a,
        "kernel_target_alignment_uncentered": kta_a,
        "kernel_target_alignment_centered": kta_a_c,
        "validation_metrics": metrics_a,
    }

    # -------------------------------------------------------------------------
    # Case B: 2-Repetition ZFeatureMap (Separable Alternating Hadamard)
    # -------------------------------------------------------------------------
    logger.info("Computing Case B: 2-rep ZFeatureMap (Separable Statevector)...")
    fm_b = ZFeatureMap(feature_dimension=n_features, reps=2)
    K_train_b, sv_b, _ = compute_fidelity_kernel_matrix(X_train_scaled, feature_map=fm_b)
    K_val_b, _, _ = compute_fidelity_kernel_matrix(
        X_val_scaled, X_train_scaled, feature_map=fm_b, right_statevectors=sv_b
    )

    diag_b = validate_kernel_matrix(K_train_b)
    kta_b = compute_kernel_target_alignment(K_train_b, y_train, centered=False)
    kta_b_c = compute_kernel_target_alignment(K_train_b, y_train, centered=True)

    clf_b = SVC(kernel="precomputed", class_weight="balanced", C=1.0, random_state=42)
    clf_b.fit(K_train_b, y_train)
    pred_b = clf_b.predict(K_val_b)
    score_b = clf_b.decision_function(K_val_b)
    metrics_b = compute_classification_metrics(y_val, pred_b, score_b)

    results["case_b_z_2rep_separable"] = {
        "description": "2-Repetition ZFeatureMap (Separable, alternating Hadamard rotations)",
        "circuit_type": "Separable Product State (0 two-qubit entangling gates)",
        "repetitions": 2,
        "diagnostics": diag_b,
        "kernel_target_alignment_uncentered": kta_b,
        "kernel_target_alignment_centered": kta_b_c,
        "validation_metrics": metrics_b,
    }

    # -------------------------------------------------------------------------
    # Case C: 2-Repetition ZZFeatureMap (Canonical Entangled)
    # -------------------------------------------------------------------------
    logger.info("Computing Case C: 2-rep ZZFeatureMap (Canonical Entangled)...")
    fm_c = create_zz_feature_map(feature_dimension=n_features, reps=2, entanglement="linear")
    K_train_c, sv_c, _ = compute_fidelity_kernel_matrix(X_train_scaled, feature_map=fm_c)
    K_val_c, _, _ = compute_fidelity_kernel_matrix(
        X_val_scaled, X_train_scaled, feature_map=fm_c, right_statevectors=sv_c
    )

    diag_c = validate_kernel_matrix(K_train_c)
    kta_c = compute_kernel_target_alignment(K_train_c, y_train, centered=False)
    kta_c_c = compute_kernel_target_alignment(K_train_c, y_train, centered=True)

    clf_c = SVC(kernel="precomputed", class_weight="balanced", C=1.0, random_state=42)
    clf_c.fit(K_train_c, y_train)
    pred_c = clf_c.predict(K_val_c)
    score_c = clf_c.decision_function(K_val_c)
    metrics_c = compute_classification_metrics(y_val, pred_c, score_c)

    results["case_c_zz_2rep_entangled"] = {
        "description": "2-Repetition ZZFeatureMap (Canonical Entangled, linear CNOT ladders)",
        "circuit_type": "Entangled State (Two-qubit CNOT-P-CNOT phase interactions)",
        "repetitions": 2,
        "diagnostics": diag_c,
        "kernel_target_alignment_uncentered": kta_c,
        "kernel_target_alignment_centered": kta_c_c,
        "validation_metrics": metrics_c,
    }

    return results
