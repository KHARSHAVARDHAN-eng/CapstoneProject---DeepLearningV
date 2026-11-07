"""
Phase 11: Final Scientific Validation and Capstone Finalization Module.

Provides:
1. Leak-free stratified cross-validation with feature selection, resampling,
   and scaling strictly fitted inside each training fold.
2. Natural-prevalence validation fold evaluation resolving the CV vs Holdout PR-AUC gap.
3. Standardized fair evaluation across all 6 core models:
   - Classical Linear SVM
   - Classical RBF SVM
   - Canonical QSVM (ZZ-FeatureMap)
   - Optimized Separable QSVM (beta=0.50)
   - Anisotropic Separable QSVM (feature-weighted)
   - Projected Quantum Kernel (PQK)
4. Out-of-fold paired non-parametric bootstrap hypothesis testing (B=2,000).
5. Data-driven threshold calibration and financial cost optimization.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from qiskit.circuit.library import ZZFeatureMap
from qiskit.quantum_info import Statevector
from sklearn.feature_selection import f_classif
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC, SVC

logger = logging.getLogger(__name__)


# ==============================================================================
# 1. KERNEL COMPUTATION PRIMITIVES
# ==============================================================================

def compute_bandwidth_separable_kernel_matrix(
    X1: np.ndarray,
    X2: np.ndarray,
    beta: float = 0.50,
) -> np.ndarray:
    """Computes exact pure-state fidelity Gram matrix for bandwidth-scaled separable Z-kernel.

    Formula:
        k_beta(x, x') = Prod_{j=1}^d cos^2(beta * (x_j - x'_j) / 2)
    """
    diff = X1[:, np.newaxis, :] - X2[np.newaxis, :, :]
    cos_sq = np.cos(beta * diff / 2.0) ** 2
    K = np.prod(cos_sq, axis=2)
    return np.clip(K, 0.0, 1.0)


def compute_anisotropic_kernel_matrix(
    X1: np.ndarray,
    X2: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Computes feature-weighted anisotropic separable Z-kernel Gram matrix.

    Formula:
        k_w(x, x') = Prod_{j=1}^d cos^2(w_j * (x_j - x'_j) / 2)
    """
    w_arr = np.asarray(weights, dtype=float).reshape(1, 1, -1)
    diff = X1[:, np.newaxis, :] - X2[np.newaxis, :, :]
    cos_sq = np.cos(w_arr * diff / 2.0) ** 2
    K = np.prod(cos_sq, axis=2)
    return np.clip(K, 0.0, 1.0)


def compute_zz_statevectors(
    X: np.ndarray,
    reps: int = 1,
    entanglement: str = "linear",
) -> np.ndarray:
    """Computes statevectors for Canonical ZZFeatureMap in pure-state simulation.

    Output shape: (N, 2^d) complex128
    """
    n_qubits = X.shape[1]
    fmap = ZZFeatureMap(n_qubits, reps=reps, entanglement=entanglement)
    sv_list = [Statevector.from_instruction(fmap.assign_parameters(x)).data for x in X]
    return np.asarray(sv_list, dtype=np.complex128)


def compute_zz_fidelity_kernel_matrix(
    sv1: np.ndarray,
    sv2: np.ndarray,
) -> np.ndarray:
    """Computes pure-state fidelity matrix |<psi_1|psi_2>|^2 between two statevector arrays."""
    inner = np.dot(sv1, sv2.conj().T)
    K = np.abs(inner) ** 2
    return np.clip(K, 0.0, 1.0)


def compute_1qubit_bloch_vectors(
    sv_array: np.ndarray,
    n_qubits: int,
) -> np.ndarray:
    """Computes 1-qubit reduced Bloch vectors [Tr(rho_j sigma_x), Tr(rho_j sigma_y), Tr(rho_j sigma_z)]

    for all qubits j across statevector array. Output shape: (N, n_qubits * 3).
    """
    pauli_x = np.array([[0, 1], [1, 0]], dtype=complex)
    pauli_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    pauli_z = np.array([[1, 0], [0, -1]], dtype=complex)
    paulis = [pauli_x, pauli_y, pauli_z]

    N = len(sv_array)
    dim = 2**n_qubits
    bloch_feats = np.zeros((N, n_qubits * 3), dtype=float)

    for i in range(N):
        sv = sv_array[i]
        rho = np.outer(sv, sv.conj())
        feat_idx = 0
        for q in range(n_qubits):
            # Partial trace over other qubits
            other_qubits = [k for k in range(n_qubits) if k != q]
            dims = [2] * n_qubits
            rho_tensor = rho.reshape(dims + dims)

            # Contract axes
            axes_to_trace = sorted(other_qubits, reverse=True)
            rho_reduced = rho_tensor
            for ax in axes_to_trace:
                rho_reduced = np.trace(rho_reduced, axis1=ax, axis2=ax + len(rho_reduced.shape) // 2)

            # Expectation values of 3 Paulis
            for p_mat in paulis:
                val = np.real(np.trace(rho_reduced @ p_mat))
                bloch_feats[i, feat_idx] = val
                feat_idx += 1

    return bloch_feats


def compute_projected_quantum_kernel_matrix(
    bloch1: np.ndarray,
    bloch2: np.ndarray,
    gamma: float = 0.10,
) -> np.ndarray:
    """Computes Gaussian kernel between 1-qubit reduced Bloch representations (Huang et al., 2021)."""
    dists_sq = np.sum((bloch1[:, np.newaxis, :] - bloch2[np.newaxis, :, :]) ** 2, axis=2)
    K = np.exp(-gamma * dists_sq)
    return np.clip(K, 0.0, 1.0)


# ==============================================================================
# 2. LEAK-FREE CROSS-VALIDATION
# ==============================================================================

def run_leak_free_stratified_cv(
    X_train_full: pd.DataFrame,
    y_train_full: np.ndarray,
    n_splits: int = 5,
    n_quantum_train: int = 500,
    n_quantum_fraud: int = 100,
    random_state: int = 42,
    include_canonical: bool = True,
    include_projected: bool = True,
) -> Dict[str, Any]:
    """Executes mathematically leak-free 5-fold stratified cross-validation.

    Guarantees:
    1. Feature selection (ANOVA F-test) is fitted inside each fold on training split only.
    2. Resampling to 500 quantum training samples occurs strictly within each fold's training split.
    3. Scalers are fitted only on the fold's quantum training subset.
    4. Validation folds retain exact natural transaction prevalence (~0.166%).
    5. Out-of-fold decision function scores are accumulated for all training samples.
    """
    logger.info("Initializing leak-free Stratified %d-Fold Cross-Validation...", n_splits)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    n_total = len(X_train_full)

    # Containers for out-of-fold decision scores
    oof_scores = {
        "classical_linear_svm": np.zeros(n_total, dtype=float),
        "classical_rbf_svm": np.zeros(n_total, dtype=float),
        "opt_separable_qsvm": np.zeros(n_total, dtype=float),
        "anisotropic_separable_qsvm": np.zeros(n_total, dtype=float),
    }
    if include_canonical:
        oof_scores["canonical_qsvm"] = np.zeros(n_total, dtype=float)
    if include_projected:
        oof_scores["projected_quantum_kernel"] = np.zeros(n_total, dtype=float)

    fold_metrics: Dict[str, List[Dict[str, float]]] = {m: [] for m in oof_scores}
    fold_details: List[Dict[str, Any]] = []

    for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X_train_full, y_train_full)):
        t_fold_start = time.perf_counter()
        logger.info("--- Executing Fold %d/%d (Train=%d, Val=%d) ---",
                    fold_idx + 1, n_splits, len(tr_idx), len(val_idx))

        X_tr_raw = X_train_full.iloc[tr_idx]
        y_tr_raw = y_train_full[tr_idx]
        X_va_raw = X_train_full.iloc[val_idx]
        y_va_raw = y_train_full[val_idx]

        n_val_fraud = int(np.sum(y_va_raw == 1))
        n_val_legit = int(np.sum(y_va_raw == 0))
        val_prevalence = float(n_val_fraud / len(val_idx))

        # 1. Feature selection inside fold
        f_scores, _ = f_classif(X_tr_raw, y_tr_raw)
        top6_col_indices = np.argsort(f_scores)[::-1][:6]
        selected_cols = X_train_full.columns[top6_col_indices].tolist()

        # 2. Resampling inside fold for quantum training subset
        fraud_tr_idx = np.where(y_tr_raw == 1)[0]
        legit_tr_idx = np.where(y_tr_raw == 0)[0]

        rng_fold = np.random.RandomState(random_state + fold_idx * 100)
        chosen_fraud = rng_fold.choice(fraud_tr_idx, size=n_quantum_fraud, replace=False)
        chosen_legit = rng_fold.choice(legit_tr_idx, size=n_quantum_train - n_quantum_fraud, replace=False)
        q_tr_indices = np.concatenate([chosen_fraud, chosen_legit])

        X_tr_q = X_tr_raw.iloc[q_tr_indices][selected_cols].values
        y_tr_q = y_tr_raw[q_tr_indices]

        X_va = X_va_raw[selected_cols].values
        y_va = y_va_raw

        # 3. Scaling inside fold
        scaler = MinMaxScaler(feature_range=(0.0, np.pi))
        X_tr_s = scaler.fit_transform(X_tr_q)
        X_va_s = scaler.transform(X_va)

        # 4. Train Model 1: Classical Linear SVM
        clf_lin = LinearSVC(C=5.0, class_weight="balanced", random_state=random_state, max_iter=2000)
        clf_lin.fit(X_tr_s, y_tr_q)
        scores_lin = clf_lin.decision_function(X_va_s)
        oof_scores["classical_linear_svm"][val_idx] = scores_lin
        fold_metrics["classical_linear_svm"].append({
            "pr_auc": float(average_precision_score(y_va, scores_lin)),
            "roc_auc": float(roc_auc_score(y_va, scores_lin)),
        })

        # 5. Train Model 2: Classical RBF SVM
        clf_rbf = SVC(kernel="rbf", C=5.0, gamma=0.01, class_weight="balanced", random_state=random_state)
        clf_rbf.fit(X_tr_s, y_tr_q)
        scores_rbf = clf_rbf.decision_function(X_va_s)
        oof_scores["classical_rbf_svm"][val_idx] = scores_rbf
        fold_metrics["classical_rbf_svm"].append({
            "pr_auc": float(average_precision_score(y_va, scores_rbf)),
            "roc_auc": float(roc_auc_score(y_va, scores_rbf)),
        })

        # 6. Train Model 3: Optimized Separable QSVM (beta=0.50)
        K_tr_sep = compute_bandwidth_separable_kernel_matrix(X_tr_s, X_tr_s, beta=0.50)
        clf_sep = SVC(kernel="precomputed", C=0.5, class_weight="balanced", random_state=random_state)
        clf_sep.fit(K_tr_sep, y_tr_q)
        K_va_sep = compute_bandwidth_separable_kernel_matrix(X_va_s, X_tr_s, beta=0.50)
        scores_sep = clf_sep.decision_function(K_va_sep)
        oof_scores["opt_separable_qsvm"][val_idx] = scores_sep
        fold_metrics["opt_separable_qsvm"].append({
            "pr_auc": float(average_precision_score(y_va, scores_sep)),
            "roc_auc": float(roc_auc_score(y_va, scores_sep)),
        })

        # 7. Train Model 4: Anisotropic Separable QSVM
        f_sub, _ = f_classif(X_tr_s, y_tr_q)
        f_norm = f_sub / np.mean(f_sub) if np.mean(f_sub) > 0 else np.ones_like(f_sub)
        w_ani = np.clip(0.50 * f_norm, 0.1, 2.0)
        K_tr_ani = compute_anisotropic_kernel_matrix(X_tr_s, X_tr_s, weights=w_ani)
        clf_ani = SVC(kernel="precomputed", C=0.5, class_weight="balanced", random_state=random_state)
        clf_ani.fit(K_tr_ani, y_tr_q)
        K_va_ani = compute_anisotropic_kernel_matrix(X_va_s, X_tr_s, weights=w_ani)
        scores_ani = clf_ani.decision_function(K_va_ani)
        oof_scores["anisotropic_separable_qsvm"][val_idx] = scores_ani
        fold_metrics["anisotropic_separable_qsvm"].append({
            "pr_auc": float(average_precision_score(y_va, scores_ani)),
            "roc_auc": float(roc_auc_score(y_va, scores_ani)),
        })

        # 8. Train Model 5: Canonical QSVM (ZZ Feature Map, reps=1)
        if include_canonical:
            sv_tr = compute_zz_statevectors(X_tr_s, reps=1, entanglement="linear")
            sv_va = compute_zz_statevectors(X_va_s, reps=1, entanglement="linear")
            K_tr_can = compute_zz_fidelity_kernel_matrix(sv_tr, sv_tr)
            clf_can = SVC(kernel="precomputed", C=0.5, class_weight="balanced", random_state=random_state)
            clf_can.fit(K_tr_can, y_tr_q)
            K_va_can = compute_zz_fidelity_kernel_matrix(sv_va, sv_tr)
            scores_can = clf_can.decision_function(K_va_can)
            oof_scores["canonical_qsvm"][val_idx] = scores_can
            fold_metrics["canonical_qsvm"].append({
                "pr_auc": float(average_precision_score(y_va, scores_can)),
                "roc_auc": float(roc_auc_score(y_va, scores_can)),
            })

        # 9. Train Model 6: Projected Quantum Kernel
        if include_projected:
            if not include_canonical:
                sv_tr = compute_zz_statevectors(X_tr_s, reps=1, entanglement="linear")
                sv_va = compute_zz_statevectors(X_va_s, reps=1, entanglement="linear")
            bloch_tr = compute_1qubit_bloch_vectors(sv_tr, n_qubits=6)
            bloch_va = compute_1qubit_bloch_vectors(sv_va, n_qubits=6)
            K_tr_pqk = compute_projected_quantum_kernel_matrix(bloch_tr, bloch_tr, gamma=0.10)
            clf_pqk = SVC(kernel="precomputed", C=1.0, class_weight="balanced", random_state=random_state)
            clf_pqk.fit(K_tr_pqk, y_tr_q)
            K_va_pqk = compute_projected_quantum_kernel_matrix(bloch_va, bloch_tr, gamma=0.10)
            scores_pqk = clf_pqk.decision_function(K_va_pqk)
            oof_scores["projected_quantum_kernel"][val_idx] = scores_pqk
            fold_metrics["projected_quantum_kernel"].append({
                "pr_auc": float(average_precision_score(y_va, scores_pqk)),
                "roc_auc": float(roc_auc_score(y_va, scores_pqk)),
            })

        fold_elapsed = time.perf_counter() - t_fold_start
        fold_details.append({
            "fold_index": fold_idx,
            "train_samples": len(tr_idx),
            "val_samples": len(val_idx),
            "val_fraud_count": n_val_fraud,
            "val_legit_count": n_val_legit,
            "val_prevalence": round(val_prevalence, 6),
            "selected_features": selected_cols,
            "fold_runtime_seconds": round(fold_elapsed, 2),
        })
        logger.info("Fold %d completed in %.2fs (Val Fraud=%d, Legit=%d)",
                    fold_idx + 1, fold_elapsed, n_val_fraud, n_val_legit)

    return {
        "oof_scores": oof_scores,
        "fold_metrics": fold_metrics,
        "fold_details": fold_details,
        "y_train_full": y_train_full,
    }


# ==============================================================================
# 3. METRIC COMPUTATION & EVALUATION
# ==============================================================================

def compute_evaluation_summary(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float = 0.0,
) -> Dict[str, Any]:
    """Computes ranking and threshold-dependent metrics for a score array."""
    preds = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()

    pr_auc = float(average_precision_score(y_true, scores))
    roc_auc = float(roc_auc_score(y_true, scores))
    precision = float(precision_score(y_true, preds, zero_division=0))
    recall = float(recall_score(y_true, preds, zero_division=0))
    f1 = float(f1_score(y_true, preds, zero_division=0))

    return {
        "pr_auc": round(pr_auc, 4),
        "roc_auc": round(roc_auc, 4),
        "threshold": round(threshold, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "prevalence": round(float(np.mean(y_true)), 6),
    }


def calibrate_financial_cost_thresholds(
    y_true: np.ndarray,
    scores: np.ndarray,
    cost_ratios: List[float] = [5.0, 20.0, 50.0, 100.0],
    c_fp: float = 10.0,
    n_threshold_candidates: int = 200,
) -> Dict[str, Any]:
    """Derives cost-optimal decision cutoffs tau*(R) by minimizing C(tau) = c_fp * FP + c_fn * FN."""
    cutoffs = np.linspace(np.percentile(scores, 1.0), np.percentile(scores, 99.0), n_threshold_candidates)
    results = {}

    for R in cost_ratios:
        c_fn = c_fp * R
        best_cost = float("inf")
        best_tau = 0.0
        best_eval = {}

        for tau in cutoffs:
            preds = (scores >= tau).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
            cost = c_fp * fp + c_fn * fn

            if cost < best_cost:
                best_cost = cost
                best_tau = float(tau)
                best_eval = {
                    "cost_ratio_R": R,
                    "c_fp": c_fp,
                    "c_fn": c_fn,
                    "optimal_threshold": round(best_tau, 4),
                    "financial_cost": round(cost, 2),
                    "precision": round(float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0, 4),
                    "recall": round(float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0, 4),
                    "false_positives": int(fp),
                    "false_negatives": int(fn),
                    "true_positives": int(tp),
                    "true_negatives": int(tn),
                }

        results[str(int(R))] = best_eval

    return results


# ==============================================================================
# 4. PAIRED BOOTSTRAP HYPOTHESIS TESTING
# ==============================================================================

def compute_paired_bootstrap_comparison(
    y_true: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    model_a_name: str,
    model_b_name: str,
    n_bootstraps: int = 2000,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Computes paired non-parametric bootstrap test for Delta = Metric(A) - Metric(B)."""
    rng = np.random.RandomState(random_state)
    N = len(y_true)

    deltas_pr = np.zeros(n_bootstraps, dtype=float)
    deltas_roc = np.zeros(n_bootstraps, dtype=float)
    deltas_prec = np.zeros(n_bootstraps, dtype=float)
    deltas_rec = np.zeros(n_bootstraps, dtype=float)

    preds_a_def = (scores_a >= 0.0).astype(int)
    preds_b_def = (scores_b >= 0.0).astype(int)

    for b in range(n_bootstraps):
        b_idx = rng.randint(0, N, size=N)
        y_b = y_true[b_idx]

        # Stratified check for positive class
        if np.sum(y_b == 1) < 2:
            continue

        s_a_b = scores_a[b_idx]
        s_b_b = scores_b[b_idx]
        p_a_b = preds_a_def[b_idx]
        p_b_b = preds_b_def[b_idx]

        deltas_pr[b] = average_precision_score(y_b, s_a_b) - average_precision_score(y_b, s_b_b)
        deltas_roc[b] = roc_auc_score(y_b, s_a_b) - roc_auc_score(y_b, s_b_b)
        deltas_prec[b] = precision_score(y_b, p_a_b, zero_division=0) - precision_score(y_b, p_b_b, zero_division=0)
        deltas_rec[b] = recall_score(y_b, p_a_b, zero_division=0) - recall_score(y_b, p_b_b, zero_division=0)

    def _summarize(deltas: np.ndarray) -> Dict[str, Any]:
        mean_d = float(np.mean(deltas))
        std_d = float(np.std(deltas))
        ci_lower = float(np.percentile(deltas, 2.5))
        ci_upper = float(np.percentile(deltas, 97.5))
        is_sig = bool(ci_lower > 0.0 or ci_upper < 0.0)
        return {
            "mean": round(mean_d, 4),
            "std": round(std_d, 4),
            "ci_95": [round(ci_lower, 4), round(ci_upper, 4)],
            "statistically_significant": is_sig,
        }

    return {
        "comparison": f"{model_a_name} minus {model_b_name}",
        "subtraction_direction": f"Delta = Metric({model_a_name}) - Metric({model_b_name})",
        "delta_pr_auc": _summarize(deltas_pr),
        "delta_roc_auc": _summarize(deltas_roc),
        "delta_precision": _summarize(deltas_prec),
        "delta_recall": _summarize(deltas_rec),
    }
