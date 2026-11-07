"""
Quantum Kernel and QSVM implementation module.

Implements ZZFeatureMap, FidelityQuantumKernel, fast statevector fidelity Gram matrix estimation,
numerical kernel matrix verification, pilot benchmarking, and precomputed QSVM classification.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
from qiskit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap
from qiskit.quantum_info import Statevector
from sklearn.svm import SVC

logger = logging.getLogger("quantum_fraud.quantum_kernel")


def create_zz_feature_map(
    feature_dimension: int = 6,
    reps: int = 2,
    entanglement: Literal["linear", "full", "circular"] = "linear",
) -> ZZFeatureMap:
    """Creates a canonical ZZFeatureMap for quantum kernel embedding.

    Args:
        feature_dimension: Number of qubits / features (default: 6).
        reps: Repetitions of the feature map circuit (default: 2).
        entanglement: Entanglement layout ('linear', 'full', 'circular').

    Returns:
        Instantiated ZZFeatureMap circuit.
    """
    return ZZFeatureMap(
        feature_dimension=feature_dimension,
        reps=reps,
        entanglement=entanglement,
    )


def compute_statevector_matrix(
    feature_map: ZZFeatureMap,
    X: np.ndarray,
) -> np.ndarray:
    """Precomputes exact statevectors for each sample in X.

    Computing statevectors individually scales linearly as O(N) rather than O(N^2),
    enabling subsequent pairwise inner products to execute via matrix multiplication in O(N^2 * d).

    Args:
        feature_map: Parametrized ZZFeatureMap.
        X: Scaled feature array of shape (N, feature_dimension).

    Returns:
        np.ndarray of complex statevectors with shape (N, 2^feature_dimension).
    """
    X_arr = np.asarray(X, dtype=np.float64)
    statevectors = []
    for x in X_arr:
        bound_qc = feature_map.assign_parameters(x)
        sv = Statevector.from_instruction(bound_qc)
        statevectors.append(sv.data)
    return np.array(statevectors, dtype=np.complex128)


def compute_fidelity_kernel_matrix(
    X_left: np.ndarray,
    X_right: Optional[np.ndarray] = None,
    feature_map: Optional[ZZFeatureMap] = None,
    left_statevectors: Optional[np.ndarray] = None,
    right_statevectors: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Computes exact fidelity quantum kernel matrix K(x, x') = |<phi(x')|phi(x)>|^2.

    Args:
        X_left: Feature matrix of shape (N_left, d).
        X_right: Feature matrix of shape (N_right, d), or None for symmetric train matrix.
        feature_map: ZZFeatureMap instance. Required if statevectors are not precomputed.
        left_statevectors: Optional precomputed statevectors for X_left.
        right_statevectors: Optional precomputed statevectors for X_right.

    Returns:
        Tuple of (kernel_matrix, left_statevectors, right_statevectors).
    """
    if feature_map is None and left_statevectors is None:
        feature_map = create_zz_feature_map(feature_dimension=X_left.shape[1], reps=2)

    if left_statevectors is None:
        left_statevectors = compute_statevector_matrix(feature_map, X_left)

    if X_right is None:
        # Symmetric Gram matrix: K_train = |Psi_train @ Psi_train^\dagger|^2
        inner_products = left_statevectors @ left_statevectors.conj().T
        K = np.abs(inner_products) ** 2
        # Ensure exact diagonal unity and symmetry despite floating-point rounding
        np.fill_diagonal(K, 1.0)
        K = 0.5 * (K + K.T)
        K = np.clip(K, 0.0, 1.0)
        return K, left_statevectors, None
    else:
        # Rectangular evaluation matrix: K_eval = |Psi_eval @ Psi_train^\dagger|^2
        if right_statevectors is None:
            right_statevectors = compute_statevector_matrix(feature_map, X_right)
        inner_products = left_statevectors @ right_statevectors.conj().T
        K = np.abs(inner_products) ** 2
        K = np.clip(K, 0.0, 1.0)
        return K, left_statevectors, right_statevectors


def validate_kernel_matrix(
    K: np.ndarray,
    is_symmetric: bool = True,
    atol: float = 1e-6,
) -> Dict[str, Any]:
    """Validates mathematical and numerical properties of a computed quantum kernel matrix.

    Checks:
    1. Finite values (no NaNs, no Infs).
    2. Boundedness within [0, 1].
    3. Diagonal unity (K_ii = 1.0 for symmetric matrices).
    4. Matrix symmetry (K = K^T).
    5. Positive semi-definiteness (minimum eigenvalue >= -atol).

    Args:
        K: Evaluated kernel matrix.
        is_symmetric: True if square training Gram matrix, False for rectangular test matrix.
        atol: Numerical tolerance for eigenvalues and symmetry.

    Returns:
        Dict[str, Any] with detailed verification diagnostics.
    """
    is_finite = bool(np.all(np.isfinite(K)))
    min_val = float(np.min(K))
    max_val = float(np.max(K))
    in_range = bool(min_val >= -atol and max_val <= 1.0 + atol)

    diagnostics: Dict[str, Any] = {
        "shape": list(K.shape),
        "is_finite": is_finite,
        "min_value": min_val,
        "max_value": max_val,
        "values_in_unit_interval": in_range,
    }

    if is_symmetric:
        symmetry_diff = float(np.max(np.abs(K - K.T)))
        is_symmetric_val = bool(symmetry_diff <= atol)
        diag = np.diag(K)
        diag_diff = float(np.max(np.abs(diag - 1.0)))
        diagonal_is_one = bool(diag_diff <= atol)

        # Eigenvalues check for positive semi-definiteness
        eigenvalues = np.linalg.eigvalsh(K)
        min_eigenval = float(np.min(eigenvalues))
        is_psd = bool(min_eigenval >= -atol)

        diagnostics.update({
            "is_symmetric": is_symmetric_val,
            "max_symmetry_discrepancy": symmetry_diff,
            "diagonal_is_one": diagonal_is_one,
            "max_diagonal_discrepancy": diag_diff,
            "min_eigenvalue": min_eigenval,
            "is_positive_semi_definite": is_psd,
        })
        is_valid = is_finite and in_range and is_symmetric_val and diagonal_is_one and is_psd
    else:
        is_valid = is_finite and in_range

    diagnostics["is_valid"] = is_valid
    return diagnostics


def run_tiny_pilot_benchmark(
    X_train_pilot: np.ndarray,
    full_n_train: int = 500,
    full_n_val: int = 1000,
    n_pilot: int = 10,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Runs a tiny pilot benchmark to measure execution speed and verify numerical kernel validity.

    Args:
        X_train_pilot: Scaled feature array of at least n_pilot samples.
        full_n_train: Target quantum training subset size (default: 500).
        full_n_val: Target quantum validation subset size (default: 1000).
        n_pilot: Sample size for pilot test (default: 10).
        random_state: Seed.

    Returns:
        Dict[str, Any] with benchmark timings and projected full experiment runtimes.
    """
    X_sub = X_train_pilot[:n_pilot]
    feature_map = create_zz_feature_map(feature_dimension=X_sub.shape[1], reps=2)

    logger.info("Executing tiny pilot benchmark on %d samples (dimension %d)...", n_pilot, X_sub.shape[1])
    t0 = time.perf_counter()
    K_pilot, pilot_sv, _ = compute_fidelity_kernel_matrix(X_sub, feature_map=feature_map)
    t1 = time.perf_counter()

    pilot_elapsed = t1 - t0
    num_pilot_pairs = (n_pilot * (n_pilot - 1)) // 2
    time_per_sample = pilot_elapsed / n_pilot

    # Validation diagnostics
    validation_diag = validate_kernel_matrix(K_pilot, is_symmetric=True)

    # Runtime projection for full N_train and N_val:
    # Statevector precomputation scales linearly O(N)
    est_train_sv_time = time_per_sample * full_n_train
    est_val_sv_time = time_per_sample * full_n_val
    # Inner products in BLAS: N_train x N_train takes negligible ms
    est_train_matmul_time = 0.05
    est_val_matmul_time = 0.05

    est_train_total = est_train_sv_time + est_train_matmul_time
    est_val_total = est_val_sv_time + est_val_matmul_time
    est_full_runtime = est_train_total + est_val_total

    benchmark_results: Dict[str, Any] = {
        "pilot_sample_size": n_pilot,
        "pilot_elapsed_seconds": round(pilot_elapsed, 4),
        "time_per_sample_seconds": round(time_per_sample, 6),
        "validation_diagnostics": validation_diag,
        "projected_full_experiment": {
            "full_n_train": full_n_train,
            "full_n_val": full_n_val,
            "estimated_train_kernel_seconds": round(est_train_total, 2),
            "estimated_val_kernel_seconds": round(est_val_total, 2),
            "estimated_total_runtime_seconds": round(est_full_runtime, 2),
            "estimated_total_runtime_minutes": round(est_full_runtime / 60.0, 2),
        },
    }

    logger.info(
        "Pilot completed in %.4fs. Matrix valid: %s. Estimated full runtime: %.2fs (~%.2f mins)",
        pilot_elapsed,
        validation_diag["is_valid"],
        est_full_runtime,
        est_full_runtime / 60.0,
    )
    return benchmark_results


class QuantumKernelClassifier:
    """Scikit-learn compatible Quantum Kernel Support Vector Classifier (QSVM).

    Evaluates the quantum feature map embedding into precomputed kernel matrices
    and trains a dual formulation SVM classifier.
    """

    def __init__(
        self,
        C: float = 1.0,
        class_weight: Optional[Literal["balanced"]] = "balanced",
        random_state: int = 42,
        reps: int = 2,
        entanglement: Literal["linear", "full", "circular"] = "linear",
    ) -> None:
        self.C = C
        self.class_weight = class_weight
        self.random_state = random_state
        self.reps = reps
        self.entanglement = entanglement

        self.svm = SVC(
            C=C,
            kernel="precomputed",
            class_weight=class_weight,
            random_state=random_state,
        )
        self.feature_map: Optional[ZZFeatureMap] = None
        self.train_statevectors: Optional[np.ndarray] = None
        self.train_X: Optional[np.ndarray] = None
        self.K_train: Optional[np.ndarray] = None

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> "QuantumKernelClassifier":
        """Fits the QSVM on training data X with labels y."""
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.int64)

        self.train_X = X_arr
        n_features = X_arr.shape[1]
        self.feature_map = create_zz_feature_map(
            feature_dimension=n_features,
            reps=self.reps,
            entanglement=self.entanglement,
        )

        K_train, train_sv, _ = compute_fidelity_kernel_matrix(
            X_left=X_arr,
            feature_map=self.feature_map,
        )
        self.train_statevectors = train_sv
        self.K_train = K_train

        self.svm.fit(K_train, y_arr)
        return self

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Predicts binary class labels for input samples X."""
        K_eval = self._compute_eval_kernel(X)
        return self.svm.predict(K_eval)

    def decision_function(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Computes continuous decision function scores for input samples X."""
        K_eval = self._compute_eval_kernel(X)
        return self.svm.decision_function(K_eval)

    def _compute_eval_kernel(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Computes the rectangular evaluation kernel matrix K(X, X_train)."""
        if self.train_statevectors is None or self.feature_map is None:
            raise ValueError("The QuantumKernelClassifier must be fitted before computing evaluation kernels.")

        X_arr = np.asarray(X, dtype=np.float64)
        K_eval, _, _ = compute_fidelity_kernel_matrix(
            X_left=X_arr,
            X_right=self.train_X,
            feature_map=self.feature_map,
            right_statevectors=self.train_statevectors,
        )
        return K_eval
