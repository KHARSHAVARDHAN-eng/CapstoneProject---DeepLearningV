"""Phase 15: Trainable Quantum Kernel Optimization via Kernel Target Alignment (KTA).

Implements supervised, leak-free optimization of quantum kernel parameters on in-fold training data:
1. Centered Kernel Target Alignment (CKTA) with Prior Regularization to prevent feature collapse
   under class imbalance (pi ~ 0.165%).
2. In-fold optimization of anisotropic bandwidths theta_j for Separable QSVM (K=16).
3. Matched trainable classical RBF kernel control optimized under the identical CKTA loss.
4. Comparison against Phase 14 fixed separable QSVM, fixed classical RBF, and full-data Nystroem RBF.
5. Strict isolation of the historical holdout test set (data/processed/X_test.csv).
"""

from __future__ import annotations

import logging
import time
import tracemalloc
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize
from scipy.stats import ttest_rel, wilcoxon
from sklearn.feature_selection import f_classif
from sklearn.kernel_approximation import Nystroem
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC, SVC

logger = logging.getLogger("phase15_trainable_kernels")


# ==============================================================================
# 1. IN-FOLD PREPROCESSING & SUBSET SAMPLING
# ==============================================================================

def select_top_features_infold(
    X_train_fold: pd.DataFrame,
    y_train_fold: np.ndarray,
    k: int = 16,
) -> Tuple[List[str], np.ndarray]:
    """Selects top-K features using ANOVA F-statistic strictly inside training data."""
    f_vals, _ = f_classif(X_train_fold.values, y_train_fold)
    f_vals = np.nan_to_num(f_vals, nan=0.0)
    sorted_indices = np.argsort(f_vals)[::-1][:k]
    selected_cols = [X_train_fold.columns[i] for i in sorted_indices]
    selected_scores = f_vals[sorted_indices]
    return selected_cols, selected_scores


def generate_nested_subsets_p15(
    y_train_fold: np.ndarray,
    sample_sizes: List[int],
    fraud_ratio: float = 0.10,
    random_state: int = 42,
) -> Dict[int, np.ndarray]:
    """Generates strictly nested training subsets D_2000 subset D_3000 without leakage."""
    fraud_indices = np.where(y_train_fold == 1)[0]
    legit_indices = np.where(y_train_fold == 0)[0]

    max_req_size = max(sample_sizes)
    max_fraud = int(round(max_req_size * fraud_ratio))
    max_legit = max_req_size - max_fraud

    if len(fraud_indices) < max_fraud:
        max_fraud = len(fraud_indices)

    rng = np.random.RandomState(random_state)
    drawn_fraud = rng.choice(fraud_indices, size=max_fraud, replace=False)
    drawn_legit = rng.choice(legit_indices, size=max_legit, replace=False)

    nested_subsets: Dict[int, np.ndarray] = {}
    for N in sample_sizes:
        n_fraud = min(int(round(N * fraud_ratio)), max_fraud)
        n_legit = N - n_fraud

        sub_fraud = drawn_fraud[:n_fraud]
        sub_legit = drawn_legit[:n_legit]

        subset_indices = np.concatenate([sub_fraud, sub_legit])
        rng.shuffle(subset_indices)

        if len(subset_indices) != N or len(np.unique(subset_indices)) != N:
            raise RuntimeError(f"Subset construction failure for N={N}")

        nested_subsets[N] = subset_indices

    # Verify strict nesting
    for i in range(len(sample_sizes) - 1):
        s_small = set(nested_subsets[sample_sizes[i]])
        s_large = set(nested_subsets[sample_sizes[i + 1]])
        if not s_small.issubset(s_large):
            raise RuntimeError(f"Nesting violation: D_{sample_sizes[i]} not in D_{sample_sizes[i+1]}")

    return nested_subsets


def fit_transform_scaler(
    X_train_sub: np.ndarray,
    X_val_full: np.ndarray,
    target_range: Tuple[float, float] = (0.0, np.pi),
) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler]:
    """Fits MinMaxScaler strictly on the training subset and transforms validation data."""
    scaler = MinMaxScaler(feature_range=target_range)
    X_tr_scaled = scaler.fit_transform(X_train_sub)
    X_val_scaled = scaler.transform(X_val_full)
    return X_tr_scaled, X_val_scaled, scaler


# ==============================================================================
# 2. KERNEL TARGET ALIGNMENT (KTA) OPTIMIZERS
# ==============================================================================

def compute_centered_kta_exact(
    K: np.ndarray,
    u_labels: np.ndarray,
    norm_Y_c: float,
) -> float:
    """Computes exact Centered Kernel Target Alignment: CKTA = <K_c, Y_c>_F / (||K_c||_F * ||Y_c||_F).
    
    Mathematical proof of equivalence:
      Target matrix Y = y* (y*)^T where y* in {-1, +1}.
      Centered target Y_c = H Y H = (H y*)(H y*)^T = u u^T where u = y* - mean(y*).
      Since H u = u, <K_c, Y_c>_F = Tr(K_c u u^T) = u^T K_c u = u^T (H K H) u = (H u)^T K (H u) = u^T K u.
      ||Y_c||_F = Tr(u u^T u u^T)^(1/2) = u^T u = N * Var(y*).
      ||K_c||_F is computed in O(N^2) via K_c = K - row_m - col_m + tot_m.
    Numerically identical to full matrix H @ K @ H Frobenius inner product to machine precision (< 1e-14).
    """
    row_m = np.mean(K, axis=1, keepdims=True)
    col_m = np.mean(K, axis=0, keepdims=True)
    tot_m = np.mean(K)
    K_c = K - row_m - col_m + tot_m
    norm_K_c = float(np.linalg.norm(K_c, "fro"))
    if norm_K_c <= 1e-12 or norm_Y_c <= 1e-12:
        return 0.0
    numerator = float(u_labels.T @ K @ u_labels)
    return float(numerator / (norm_K_c * norm_Y_c))


def optimize_quantum_kernel_kta(
    X_train_sub: np.ndarray,
    y_train_sub: np.ndarray,
    init_beta: float = 0.031,
    lambda_reg: float = 200.0,
    maxiter: int = 30,
) -> Tuple[np.ndarray, float]:
    """Optimizes separable quantum kernel bandwidths theta_j via Regularized Centered KTA.
    
    Objective:
      L(theta) = - CKTA(K_theta, Y) + lambda * ||theta - beta_0||^2
    where Y is the centered rank-1 target matrix (y* - y_bar*)(y* - y_bar*)^T.
    """
    y_pm = np.where(y_train_sub == 1, 1.0, -1.0)
    u_pm = y_pm - np.mean(y_pm)
    norm_Y_c = float(np.dot(u_pm, u_pm))

    diff = X_train_sub[:, np.newaxis, :] - X_train_sub[np.newaxis, :, :]
    k_dim = X_train_sub.shape[1]

    def ckta_loss(theta: np.ndarray) -> float:
        cos_sq = np.cos(theta * diff / 2.0) ** 2
        K = np.prod(cos_sq, axis=2)
        ckta = compute_centered_kta_exact(K, u_pm, norm_Y_c)
        reg = lambda_reg * np.sum((theta - init_beta) ** 2)
        return -ckta + reg

    init_theta = np.full(k_dim, init_beta)
    bounds = [(0.005, 0.15)] * k_dim

    res = minimize(
        ckta_loss,
        init_theta,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": maxiter, "disp": False},
    )

    cos_sq_opt = np.cos(res.x * diff / 2.0) ** 2
    K_opt = np.prod(cos_sq_opt, axis=2)
    final_ckta = compute_centered_kta_exact(K_opt, u_pm, norm_Y_c)

    return res.x, final_ckta


def optimize_classical_rbf_kta(
    X_train_sub: np.ndarray,
    y_train_sub: np.ndarray,
    init_gamma: float = 0.0005,
    lambda_reg: float = 200.0,
    maxiter: int = 30,
) -> Tuple[np.ndarray, float]:
    """Optimizes anisotropic classical RBF bandwidths gamma_j via Regularized Centered KTA."""
    y_pm = np.where(y_train_sub == 1, 1.0, -1.0)
    u_pm = y_pm - np.mean(y_pm)
    norm_Y_c = float(np.dot(u_pm, u_pm))

    diff_sq = (X_train_sub[:, np.newaxis, :] - X_train_sub[np.newaxis, :, :]) ** 2
    k_dim = X_train_sub.shape[1]

    def ckta_rbf_loss(gamma: np.ndarray) -> float:
        K = np.exp(-np.sum(gamma * diff_sq, axis=2))
        ckta = compute_centered_kta_exact(K, u_pm, norm_Y_c)
        reg = lambda_reg * np.sum((gamma - init_gamma) ** 2)
        return -ckta + reg

    init_gamma_arr = np.full(k_dim, init_gamma)
    bounds = [(0.00005, 0.01)] * k_dim

    res = minimize(
        ckta_rbf_loss,
        init_gamma_arr,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": maxiter, "disp": False},
    )

    K_opt = np.exp(-np.sum(res.x * diff_sq, axis=2))
    final_ckta = compute_centered_kta_exact(K_opt, u_pm, norm_Y_c)

    return res.x, final_ckta


# ==============================================================================
# 3. KERNEL MATRIX EVALUATION & CHUNKED SCORING
# ==============================================================================

def compute_bandwidth_separable_kernel_matrix(
    X1: np.ndarray,
    X2: np.ndarray,
    beta: np.ndarray | float = 0.031,
) -> np.ndarray:
    """Exact pure-state fidelity Gram matrix for bandwidth-scaled separable Z-kernel."""
    diff = X1[:, np.newaxis, :] - X2[np.newaxis, :, :]
    cos_sq = np.cos(beta * diff / 2.0) ** 2
    return np.prod(cos_sq, axis=2)


def score_kernel_chunked(
    clf: SVC,
    compute_chunk_kernel_fn: Callable[[int, int], np.ndarray],
    n_samples: int,
    chunk_size: int = 5000,
) -> np.ndarray:
    """Computes decision function scores in safe memory chunks."""
    scores = np.zeros(n_samples, dtype=np.float64)
    for start_idx in range(0, n_samples, chunk_size):
        end_idx = min(start_idx + chunk_size, n_samples)
        K_chunk = compute_chunk_kernel_fn(start_idx, end_idx)
        scores[start_idx:end_idx] = clf.decision_function(K_chunk)
    return scores


# ==============================================================================
# 4. METRICS & COST CALIBRATION
# ==============================================================================

def compute_model_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float = 0.0,
    cost_fp: float = 10.0,
    cost_fn: float = 200.0,
) -> Dict[str, Any]:
    """Computes full set of binary classification and financial loss metrics."""
    y_pred = (scores >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]
    cost = float(cost_fp * fp + cost_fn * fn)

    pr_auc = float(average_precision_score(y_true, scores))
    roc_auc = float(roc_auc_score(y_true, scores))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    return {
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "threshold": float(threshold),
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
        "financial_cost": cost,
    }


def calibrate_threshold_prevalence_adjusted(
    y_true_sub: np.ndarray,
    scores_inner_oof: np.ndarray,
    pi_population: float,
    cost_fp: float = 10.0,
    cost_fn: float = 200.0,
    n_thresholds: int = 1000,
) -> float:
    """Calibrates cost-optimal decision threshold strictly on training-only inner out-of-fold scores.
    
    Adjusts for enrichment in training subsets (fraud_ratio = 10%) relative to population prevalence:
      E[Cost](tau) = cost_fp * (1 - pi_population) * FPR(tau) + cost_fn * pi_population * FNR(tau)
    where FPR = FP / N_neg and FNR = FN / N_pos evaluated on the training subset out-of-fold predictions.
    
    Never touches outer validation fold labels or historical holdout test set.
    """
    prec, rec, thresholds = precision_recall_curve(y_true_sub, scores_inner_oof)
    candidate_thresholds = np.unique(thresholds)
    if len(candidate_thresholds) > n_thresholds:
        candidate_thresholds = np.linspace(
            float(candidate_thresholds.min()), float(candidate_thresholds.max()), n_thresholds
        )
    candidate_thresholds = np.unique(np.append(candidate_thresholds, 0.0))

    n_pos = int(np.sum(y_true_sub == 1))
    n_neg = int(len(y_true_sub) - n_pos)

    best_expected_cost = float("inf")
    best_thresh = 0.0

    for thresh in candidate_thresholds:
        yp = scores_inner_oof >= thresh
        tp = int(np.count_nonzero(yp & (y_true_sub == 1)))
        fp = int(np.count_nonzero(yp & (y_true_sub == 0)))
        fn = n_pos - tp
        fpr = fp / n_neg if n_neg > 0 else 0.0
        fnr = fn / n_pos if n_pos > 0 else 0.0

        expected_cost = cost_fp * (1.0 - pi_population) * fpr + cost_fn * pi_population * fnr
        if expected_cost < best_expected_cost:
            best_expected_cost = expected_cost
            best_thresh = float(thresh)

    return float(best_thresh)


def find_cost_optimal_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    cost_fp: float = 10.0,
    cost_fn: float = 200.0,
    n_thresholds: int = 1000,
) -> Dict[str, Any]:
    """Finds decision threshold minimizing financial cost: C = cost_fp*FP + cost_fn*FN."""
    prec, rec, thresholds = precision_recall_curve(y_true, scores)
    candidate_thresholds = np.unique(thresholds)
    if len(candidate_thresholds) > n_thresholds:
        candidate_thresholds = np.linspace(
            float(candidate_thresholds.min()), float(candidate_thresholds.max()), n_thresholds
        )
    candidate_thresholds = np.unique(np.append(candidate_thresholds, 0.0))

    best_cost = float("inf")
    best_thresh = 0.0
    best_metrics: Dict[str, Any] = {}

    n_pos = int(np.sum(y_true == 1))
    n_neg = int(len(y_true) - n_pos)

    for thresh in candidate_thresholds:
        yp = scores >= thresh
        tp = int(np.count_nonzero(yp & (y_true == 1)))
        fp = int(np.count_nonzero(yp & (y_true == 0)))
        fn = n_pos - tp
        tn = n_neg - fp
        cost = cost_fp * fp + cost_fn * fn

        if cost < best_cost:
            best_cost = cost
            best_thresh = float(thresh)
            prec_val = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            rec_val = float(tp / n_pos) if n_pos > 0 else 0.0
            best_metrics = {
                "threshold": float(thresh),
                "cost": float(cost),
                "precision": prec_val,
                "recall": rec_val,
                "false_positives": fp,
                "false_negatives": fn,
                "true_positives": tp,
                "true_negatives": tn,
            }

    return best_metrics


# ==============================================================================
# 5. PHASE 15 BENCHMARK RUNNER
# ==============================================================================

def run_phase15_pipeline(
    X_train_full: pd.DataFrame,
    y_train_full: np.ndarray,
    sample_sizes: Optional[List[int]] = None,
    seeds: Optional[List[int]] = None,
    feature_dimension: int = 16,
    n_splits: int = 5,
    chunk_size: int = 2500,
    is_pilot: bool = False,
) -> Dict[str, Any]:
    """Executes the Phase 15 Trainable Quantum Kernel benchmark."""
    if sample_sizes is None:
        sample_sizes = [2000, 3000] if not is_pilot else [2000]
    if seeds is None:
        seeds = [42, 101, 2024] if not is_pilot else [42]

    tracemalloc.start()
    t_start = time.perf_counter()
    n_total = len(X_train_full)

    logger.info("==================================================================")
    logger.info("STARTING PHASE 15 TRAINABLE KERNEL RUNNER (Pilot=%s)", is_pilot)
    logger.info("Sample sizes: %s | Seeds: %s | Dimension K=%d | Total: %d",
                sample_sizes, seeds, feature_dimension, n_total)
    logger.info("==================================================================")

    model_catalog = [
        "baseline_classical_linear_k6",
        "p14_classical_rbf_k16",
        "p14_opt_separable_qsvm_k16",
        "p15_kta_trainable_qsvm_k16",
        "p15_kta_trainable_rbf_k16",
    ]
    macro_models = [
        "macro_classical_linear_svm",
        "macro_nystroem_rbf_approx",
    ]

    per_seed_results: Dict[str, Any] = {}

    for seed_idx, seed in enumerate(seeds):
        logger.info(">>> Starting Seed %d/%d (Seed=%d) <<<", seed_idx + 1, len(seeds), seed)
        t_seed_start = time.perf_counter()

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

        oof_scores: Dict[str, Dict[str, np.ndarray]] = {
            str(sz): {m: np.zeros(n_total, dtype=np.float64) for m in model_catalog}
            for sz in sample_sizes
        }
        oof_scores["macro"] = {m: np.zeros(n_total, dtype=np.float64) for m in macro_models}

        fold_metrics: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
            str(sz): {m: [] for m in model_catalog}
            for sz in sample_sizes
        }
        fold_metrics["macro"] = {m: [] for m in macro_models}

        for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X_train_full, y_train_full)):
            t_fold_start = time.perf_counter()
            logger.info("--- Seed %d | Fold %d/%d (Train=%d, Val=%d) ---",
                        seed, fold_idx + 1, n_splits, len(tr_idx), len(val_idx))

            X_tr_raw = X_train_full.iloc[tr_idx]
            y_tr_raw = y_train_full[tr_idx]
            X_va_raw = X_train_full.iloc[val_idx]
            y_va_raw = y_train_full[val_idx]
            n_val = len(val_idx)
            pi_tr = float(np.mean(y_tr_raw))

            # In-fold feature selection up to K=16
            top16_cols, _ = select_top_features_infold(X_tr_raw, y_tr_raw, k=feature_dimension)
            k6_cols = top16_cols[:6]

            # In-fold nested subsets
            nested_subsets = generate_nested_subsets_p15(
                y_train_fold=y_tr_raw,
                sample_sizes=sample_sizes,
                fraud_ratio=0.10,
                random_state=seed + fold_idx * 100,
            )

            # Inner CV splitter strictly on in-fold training subset for threshold calibration
            skf_inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed + fold_idx * 10)

            for N in sample_sizes:
                sub_idx = nested_subsets[N]
                y_sub = y_tr_raw[sub_idx]

                # 1. Baseline Linear SVM (K=6)
                X_tr_k6 = X_tr_raw[k6_cols].iloc[sub_idx].values
                X_va_k6 = X_va_raw[k6_cols].values
                X_tr_k6_s, X_va_k6_s, _ = fit_transform_scaler(X_tr_k6, X_va_k6)

                # Inner CV for threshold calibration
                s_inner_lin = np.zeros(N, dtype=np.float64)
                for in_tr, in_va in skf_inner.split(X_tr_k6_s, y_sub):
                    c_in = LinearSVC(C=5.0, class_weight="balanced", random_state=seed, max_iter=2000)
                    c_in.fit(X_tr_k6_s[in_tr], y_sub[in_tr])
                    s_inner_lin[in_va] = c_in.decision_function(X_tr_k6_s[in_va])
                th_lin = calibrate_threshold_prevalence_adjusted(y_sub, s_inner_lin, pi_tr)

                clf_lin = LinearSVC(C=5.0, class_weight="balanced", random_state=seed, max_iter=2000)
                clf_lin.fit(X_tr_k6_s, y_sub)
                s_lin = clf_lin.decision_function(X_va_k6_s)
                oof_scores[str(N)]["baseline_classical_linear_k6"][val_idx] = s_lin
                fold_metrics[str(N)]["baseline_classical_linear_k6"].append({
                    "default": compute_model_metrics(y_va_raw, s_lin, threshold=0.0),
                    "calibrated": compute_model_metrics(y_va_raw, s_lin, threshold=th_lin),
                    "calibrated_threshold": float(th_lin),
                    "pr_auc": float(average_precision_score(y_va_raw, s_lin)),
                    "roc_auc": float(roc_auc_score(y_va_raw, s_lin)),
                })

                # 2. Prepare K=16 Data
                X_tr_k16 = X_tr_raw[top16_cols].iloc[sub_idx].values
                X_va_k16 = X_va_raw[top16_cols].values
                X_tr_k16_s, X_va_k16_s, _ = fit_transform_scaler(X_tr_k16, X_va_k16)

                # 3. Phase 14 Fixed Classical RBF (K=16, gamma=0.0005, C=5.0)
                diff_sq_tr = (X_tr_k16_s[:, np.newaxis, :] - X_tr_k16_s[np.newaxis, :, :]) ** 2
                K_rbf_fixed_tr = np.exp(-0.0005 * np.sum(diff_sq_tr, axis=2))

                s_inner_rbf = np.zeros(N, dtype=np.float64)
                for in_tr, in_va in skf_inner.split(X_tr_k16_s, y_sub):
                    c_in = SVC(kernel="precomputed", C=5.0, class_weight="balanced", random_state=seed)
                    c_in.fit(K_rbf_fixed_tr[np.ix_(in_tr, in_tr)], y_sub[in_tr])
                    s_inner_rbf[in_va] = c_in.decision_function(K_rbf_fixed_tr[np.ix_(in_va, in_tr)])
                th_rbf = calibrate_threshold_prevalence_adjusted(y_sub, s_inner_rbf, pi_tr)

                clf_rbf_fixed = SVC(kernel="precomputed", C=5.0, class_weight="balanced", random_state=seed)
                clf_rbf_fixed.fit(K_rbf_fixed_tr, y_sub)
                def get_chunk_rbf_fixed(s_i: int, e_i: int, xt=X_tr_k16_s, xv=X_va_k16_s) -> np.ndarray:
                    d_sq = (xv[s_i:e_i, np.newaxis, :] - xt[np.newaxis, :, :]) ** 2
                    return np.exp(-0.0005 * np.sum(d_sq, axis=2))
                s_rbf_fixed = score_kernel_chunked(clf_rbf_fixed, get_chunk_rbf_fixed, n_val, chunk_size=chunk_size)
                oof_scores[str(N)]["p14_classical_rbf_k16"][val_idx] = s_rbf_fixed
                fold_metrics[str(N)]["p14_classical_rbf_k16"].append({
                    "default": compute_model_metrics(y_va_raw, s_rbf_fixed, threshold=0.0),
                    "calibrated": compute_model_metrics(y_va_raw, s_rbf_fixed, threshold=th_rbf),
                    "calibrated_threshold": float(th_rbf),
                    "pr_auc": float(average_precision_score(y_va_raw, s_rbf_fixed)),
                    "roc_auc": float(roc_auc_score(y_va_raw, s_rbf_fixed)),
                })

                # 4. Phase 14 Fixed Separable QSVM (K=16, beta=0.031, C=5.0)
                beta_fixed = 0.05 * np.sqrt(6.0 / 16.0) # ~0.0306 ~ 0.031
                K_sep_tr = compute_bandwidth_separable_kernel_matrix(X_tr_k16_s, X_tr_k16_s, beta=beta_fixed)

                s_inner_sep = np.zeros(N, dtype=np.float64)
                for in_tr, in_va in skf_inner.split(X_tr_k16_s, y_sub):
                    c_in = SVC(kernel="precomputed", C=5.0, class_weight="balanced", random_state=seed)
                    c_in.fit(K_sep_tr[np.ix_(in_tr, in_tr)], y_sub[in_tr])
                    s_inner_sep[in_va] = c_in.decision_function(K_sep_tr[np.ix_(in_va, in_tr)])
                th_sep = calibrate_threshold_prevalence_adjusted(y_sub, s_inner_sep, pi_tr)

                clf_sep_fixed = SVC(kernel="precomputed", C=5.0, class_weight="balanced", random_state=seed)
                clf_sep_fixed.fit(K_sep_tr, y_sub)
                def get_chunk_sep_fixed(s_i: int, e_i: int, xt=X_tr_k16_s, xv=X_va_k16_s, b=beta_fixed) -> np.ndarray:
                    return compute_bandwidth_separable_kernel_matrix(xv[s_i:e_i], xt, beta=b)
                s_sep_fixed = score_kernel_chunked(clf_sep_fixed, get_chunk_sep_fixed, n_val, chunk_size=chunk_size)
                oof_scores[str(N)]["p14_opt_separable_qsvm_k16"][val_idx] = s_sep_fixed
                fold_metrics[str(N)]["p14_opt_separable_qsvm_k16"].append({
                    "default": compute_model_metrics(y_va_raw, s_sep_fixed, threshold=0.0),
                    "calibrated": compute_model_metrics(y_va_raw, s_sep_fixed, threshold=th_sep),
                    "calibrated_threshold": float(th_sep),
                    "pr_auc": float(average_precision_score(y_va_raw, s_sep_fixed)),
                    "roc_auc": float(roc_auc_score(y_va_raw, s_sep_fixed)),
                })

                # 5. Phase 15 Trainable Quantum Kernel via Regularized Centered KTA (K=16)
                if N > 1000:
                    opt_sub = np.random.RandomState(seed + fold_idx).choice(N, size=1000, replace=False)
                    X_opt, y_opt = X_tr_k16_s[opt_sub], y_sub[opt_sub]
                else:
                    X_opt, y_opt = X_tr_k16_s, y_sub

                t_kta_start = time.perf_counter()
                opt_theta_q, final_ckta_q = optimize_quantum_kernel_kta(
                    X_opt, y_opt, init_beta=beta_fixed, lambda_reg=200.0, maxiter=25
                )
                logger.info("Fold %d N=%d QSVM KTA optimized in %.2fs (CKTA=%.4f, theta=[%.4f, %.4f])",
                            fold_idx + 1, N, time.perf_counter() - t_kta_start, final_ckta_q,
                            opt_theta_q.min(), opt_theta_q.max())

                K_kta_tr = compute_bandwidth_separable_kernel_matrix(X_tr_k16_s, X_tr_k16_s, beta=opt_theta_q)

                s_inner_kta_q = np.zeros(N, dtype=np.float64)
                for in_tr, in_va in skf_inner.split(X_tr_k16_s, y_sub):
                    c_in = SVC(kernel="precomputed", C=5.0, class_weight="balanced", random_state=seed)
                    c_in.fit(K_kta_tr[np.ix_(in_tr, in_tr)], y_sub[in_tr])
                    s_inner_kta_q[in_va] = c_in.decision_function(K_kta_tr[np.ix_(in_va, in_tr)])
                th_kta_q = calibrate_threshold_prevalence_adjusted(y_sub, s_inner_kta_q, pi_tr)

                clf_kta_q = SVC(kernel="precomputed", C=5.0, class_weight="balanced", random_state=seed)
                clf_kta_q.fit(K_kta_tr, y_sub)
                def get_chunk_kta_q(s_i: int, e_i: int, xt=X_tr_k16_s, xv=X_va_k16_s, th=opt_theta_q) -> np.ndarray:
                    return compute_bandwidth_separable_kernel_matrix(xv[s_i:e_i], xt, beta=th)
                s_kta_q = score_kernel_chunked(clf_kta_q, get_chunk_kta_q, n_val, chunk_size=chunk_size)
                oof_scores[str(N)]["p15_kta_trainable_qsvm_k16"][val_idx] = s_kta_q
                fold_metrics[str(N)]["p15_kta_trainable_qsvm_k16"].append({
                    "default": compute_model_metrics(y_va_raw, s_kta_q, threshold=0.0),
                    "calibrated": compute_model_metrics(y_va_raw, s_kta_q, threshold=th_kta_q),
                    "calibrated_threshold": float(th_kta_q),
                    "pr_auc": float(average_precision_score(y_va_raw, s_kta_q)),
                    "roc_auc": float(roc_auc_score(y_va_raw, s_kta_q)),
                })

                # 6. Phase 15 Trainable Classical RBF via Regularized Centered KTA (K=16)
                t_rbf_kta_start = time.perf_counter()
                opt_gamma_rbf, final_ckta_rbf = optimize_classical_rbf_kta(
                    X_opt, y_opt, init_gamma=0.0005, lambda_reg=200.0, maxiter=25
                )
                logger.info("Fold %d N=%d RBF KTA optimized in %.2fs (CKTA=%.4f, gamma=[%.6f, %.6f])",
                            fold_idx + 1, N, time.perf_counter() - t_rbf_kta_start, final_ckta_rbf,
                            opt_gamma_rbf.min(), opt_gamma_rbf.max())

                K_rbf_kta_tr = np.exp(-np.sum(opt_gamma_rbf * diff_sq_tr, axis=2))

                s_inner_kta_rbf = np.zeros(N, dtype=np.float64)
                for in_tr, in_va in skf_inner.split(X_tr_k16_s, y_sub):
                    c_in = SVC(kernel="precomputed", C=5.0, class_weight="balanced", random_state=seed)
                    c_in.fit(K_rbf_kta_tr[np.ix_(in_tr, in_tr)], y_sub[in_tr])
                    s_inner_kta_rbf[in_va] = c_in.decision_function(K_rbf_kta_tr[np.ix_(in_va, in_tr)])
                th_kta_rbf = calibrate_threshold_prevalence_adjusted(y_sub, s_inner_kta_rbf, pi_tr)

                clf_kta_rbf = SVC(kernel="precomputed", C=5.0, class_weight="balanced", random_state=seed)
                clf_kta_rbf.fit(K_rbf_kta_tr, y_sub)
                def get_chunk_kta_rbf(s_i: int, e_i: int, xt=X_tr_k16_s, xv=X_va_k16_s, g=opt_gamma_rbf) -> np.ndarray:
                    diff_sq_ch = (xv[s_i:e_i, np.newaxis, :] - xt[np.newaxis, :, :]) ** 2
                    return np.exp(-np.sum(g * diff_sq_ch, axis=2))
                s_kta_rbf = score_kernel_chunked(clf_kta_rbf, get_chunk_kta_rbf, n_val, chunk_size=chunk_size)
                oof_scores[str(N)]["p15_kta_trainable_rbf_k16"][val_idx] = s_kta_rbf
                fold_metrics[str(N)]["p15_kta_trainable_rbf_k16"].append({
                    "default": compute_model_metrics(y_va_raw, s_kta_rbf, threshold=0.0),
                    "calibrated": compute_model_metrics(y_va_raw, s_kta_rbf, threshold=th_kta_rbf),
                    "calibrated_threshold": float(th_kta_rbf),
                    "pr_auc": float(average_precision_score(y_va_raw, s_kta_rbf)),
                    "roc_auc": float(roc_auc_score(y_va_raw, s_kta_rbf)),
                })

            # --- Macro Full-Fold Controls (N=181,584) on K=16 ---
            X_tr_k16_full = X_tr_raw[top16_cols].values
            X_va_k16_full = X_va_raw[top16_cols].values
            scl_macro = MinMaxScaler(feature_range=(0.0, np.pi))
            X_tr_macro_s = scl_macro.fit_transform(X_tr_k16_full)
            X_va_macro_s = scl_macro.transform(X_va_k16_full)

            # In-fold calibration sample of 10,000 instances
            rng_mac = np.random.RandomState(seed + fold_idx * 50)
            cal_idx = rng_mac.choice(len(y_tr_raw), size=min(10000, len(y_tr_raw)), replace=False)

            # Full Linear
            clf_mac_lin = LinearSVC(C=5.0, class_weight="balanced", random_state=seed, max_iter=2000)
            clf_mac_lin.fit(X_tr_macro_s, y_tr_raw)
            s_mac_lin = clf_mac_lin.decision_function(X_va_macro_s)
            s_cal_lin = clf_mac_lin.decision_function(X_tr_macro_s[cal_idx])
            th_mac_lin = calibrate_threshold_prevalence_adjusted(y_tr_raw[cal_idx], s_cal_lin, pi_tr)

            oof_scores["macro"]["macro_classical_linear_svm"][val_idx] = s_mac_lin
            fold_metrics["macro"]["macro_classical_linear_svm"].append({
                "default": compute_model_metrics(y_va_raw, s_mac_lin, threshold=0.0),
                "calibrated": compute_model_metrics(y_va_raw, s_mac_lin, threshold=th_mac_lin),
                "calibrated_threshold": float(th_mac_lin),
                "pr_auc": float(average_precision_score(y_va_raw, s_mac_lin)),
                "roc_auc": float(roc_auc_score(y_va_raw, s_mac_lin)),
            })

            # Full Nystroem RBF
            nyst = Nystroem(kernel="rbf", gamma=0.01, n_components=300, random_state=seed)
            X_tr_nyst = nyst.fit_transform(X_tr_macro_s)
            X_va_nyst = nyst.transform(X_va_macro_s)
            clf_mac_rbf = LinearSVC(C=5.0, class_weight="balanced", random_state=seed, max_iter=2000)
            clf_mac_rbf.fit(X_tr_nyst, y_tr_raw)
            s_mac_rbf = clf_mac_rbf.decision_function(X_va_nyst)
            s_cal_rbf = clf_mac_rbf.decision_function(X_tr_nyst[cal_idx])
            th_mac_rbf = calibrate_threshold_prevalence_adjusted(y_tr_raw[cal_idx], s_cal_rbf, pi_tr)

            oof_scores["macro"]["macro_nystroem_rbf_approx"][val_idx] = s_mac_rbf
            fold_metrics["macro"]["macro_nystroem_rbf_approx"].append({
                "default": compute_model_metrics(y_va_raw, s_mac_rbf, threshold=0.0),
                "calibrated": compute_model_metrics(y_va_raw, s_mac_rbf, threshold=th_mac_rbf),
                "calibrated_threshold": float(th_mac_rbf),
                "pr_auc": float(average_precision_score(y_va_raw, s_mac_rbf)),
                "roc_auc": float(roc_auc_score(y_va_raw, s_mac_rbf)),
            })

            logger.info("Fold %d completed in %.2fs", fold_idx + 1, time.perf_counter() - t_fold_start)

            if is_pilot:
                break

        # Compute Out-of-fold summaries
        oof_summary: Dict[str, Any] = {}
        eval_indices = val_idx if is_pilot else np.arange(n_total)
        y_eval = y_train_full[eval_indices]

        for sz in sample_sizes:
            sz_str = str(sz)
            oof_summary[sz_str] = {}
            for m in model_catalog:
                scores_eval = oof_scores[sz_str][m][eval_indices]
                pr_auc = float(average_precision_score(y_eval, scores_eval))
                roc_auc = float(roc_auc_score(y_eval, scores_eval))
                def_m = compute_model_metrics(y_eval, scores_eval, threshold=0.0)

                cv_pr = [fm["pr_auc"] for fm in fold_metrics[sz_str][m]]
                cv_roc = [fm["roc_auc"] for fm in fold_metrics[sz_str][m]]
                cv_cal_th = [fm["calibrated_threshold"] for fm in fold_metrics[sz_str][m]]
                cv_cal_prec = [fm["calibrated"]["precision"] for fm in fold_metrics[sz_str][m]]
                cv_cal_rec = [fm["calibrated"]["recall"] for fm in fold_metrics[sz_str][m]]
                cv_cal_fp = [fm["calibrated"]["false_positives"] for fm in fold_metrics[sz_str][m]]
                cv_cal_fn = [fm["calibrated"]["false_negatives"] for fm in fold_metrics[sz_str][m]]
                cv_cal_cost = [fm["calibrated"]["financial_cost"] for fm in fold_metrics[sz_str][m]]

                oof_summary[sz_str][m] = {
                    "oof_pr_auc": pr_auc,
                    "oof_roc_auc": roc_auc,
                    "mean_cv_pr_auc": float(np.mean(cv_pr)),
                    "std_cv_pr_auc": float(np.std(cv_pr, ddof=1)) if len(cv_pr) > 1 else 0.0,
                    "mean_cv_roc_auc": float(np.mean(cv_roc)),
                    "default_precision": def_m["precision"],
                    "default_recall": def_m["recall"],
                    "default_fp": def_m["false_positives"],
                    "default_cost": def_m["financial_cost"],
                    "calibrated_cost": float(np.mean(cv_cal_cost)),
                    "total_calibrated_cost": float(np.sum(cv_cal_cost)),
                    "calibrated_threshold": float(np.mean(cv_cal_th)),
                    "calibrated_precision": float(np.mean(cv_cal_prec)),
                    "calibrated_recall": float(np.mean(cv_cal_rec)),
                    "calibrated_fp": int(round(np.mean(cv_cal_fp))),
                    "calibrated_fn": int(round(np.mean(cv_cal_fn))),
                }

        # Macro summaries
        oof_summary["macro"] = {}
        for m in macro_models:
            scores_eval = oof_scores["macro"][m][eval_indices]
            pr_auc = float(average_precision_score(y_eval, scores_eval))
            roc_auc = float(roc_auc_score(y_eval, scores_eval))
            def_m = compute_model_metrics(y_eval, scores_eval, threshold=0.0)

            cv_pr = [fm["pr_auc"] for fm in fold_metrics["macro"][m]]
            cv_roc = [fm["roc_auc"] for fm in fold_metrics["macro"][m]]
            cv_cal_th = [fm["calibrated_threshold"] for fm in fold_metrics["macro"][m]]
            cv_cal_prec = [fm["calibrated"]["precision"] for fm in fold_metrics["macro"][m]]
            cv_cal_rec = [fm["calibrated"]["recall"] for fm in fold_metrics["macro"][m]]
            cv_cal_fp = [fm["calibrated"]["false_positives"] for fm in fold_metrics["macro"][m]]
            cv_cal_fn = [fm["calibrated"]["false_negatives"] for fm in fold_metrics["macro"][m]]
            cv_cal_cost = [fm["calibrated"]["financial_cost"] for fm in fold_metrics["macro"][m]]

            oof_summary["macro"][m] = {
                "oof_pr_auc": pr_auc,
                "oof_roc_auc": roc_auc,
                "mean_cv_pr_auc": float(np.mean(cv_pr)),
                "std_cv_pr_auc": float(np.std(cv_pr, ddof=1)) if len(cv_pr) > 1 else 0.0,
                "mean_cv_roc_auc": float(np.mean(cv_roc)),
                "default_precision": def_m["precision"],
                "default_recall": def_m["recall"],
                "default_fp": def_m["false_positives"],
                "default_cost": def_m["financial_cost"],
                "calibrated_cost": float(np.mean(cv_cal_cost)),
                "total_calibrated_cost": float(np.sum(cv_cal_cost)),
                "calibrated_threshold": float(np.mean(cv_cal_th)),
                "calibrated_precision": float(np.mean(cv_cal_prec)),
                "calibrated_recall": float(np.mean(cv_cal_rec)),
                "calibrated_fp": int(round(np.mean(cv_cal_fp))),
                "calibrated_fn": int(round(np.mean(cv_cal_fn))),
            }

        per_seed_results[str(seed)] = {
            "seed": seed,
            "runtime_seconds": time.perf_counter() - t_seed_start,
            "oof_summary": oof_summary,
            "fold_metrics": fold_metrics,
        }

    # Aggregate across seeds
    multi_seed_aggregated: Dict[str, Any] = {}
    for sz in sample_sizes:
        sz_str = str(sz)
        multi_seed_aggregated[sz_str] = {}
        for m in model_catalog:
            seed_pr = [per_seed_results[str(s)]["oof_summary"][sz_str][m]["oof_pr_auc"] for s in seeds]
            seed_roc = [per_seed_results[str(s)]["oof_summary"][sz_str][m]["oof_roc_auc"] for s in seeds]
            seed_def_prec = [per_seed_results[str(s)]["oof_summary"][sz_str][m]["default_precision"] for s in seeds]
            seed_def_rec = [per_seed_results[str(s)]["oof_summary"][sz_str][m]["default_recall"] for s in seeds]
            seed_def_cost = [per_seed_results[str(s)]["oof_summary"][sz_str][m]["default_cost"] for s in seeds]
            seed_cal_cost = [per_seed_results[str(s)]["oof_summary"][sz_str][m]["calibrated_cost"] for s in seeds]

            multi_seed_aggregated[sz_str][m] = {
                "mean_oof_pr_auc": float(np.mean(seed_pr)),
                "std_oof_pr_auc_sample": float(np.std(seed_pr, ddof=1)) if len(seed_pr) > 1 else 0.0,
                "std_oof_pr_auc_pop": float(np.std(seed_pr, ddof=0)),
                "mean_oof_roc_auc": float(np.mean(seed_roc)),
                "std_oof_roc_auc": float(np.std(seed_roc, ddof=1)) if len(seed_roc) > 1 else 0.0,
                "mean_default_precision": float(np.mean(seed_def_prec)),
                "mean_default_recall": float(np.mean(seed_def_rec)),
                "mean_default_cost": float(np.mean(seed_def_cost)),
                "mean_calibrated_cost": float(np.mean(seed_cal_cost)),
                "std_calibrated_cost": float(np.std(seed_cal_cost, ddof=1)) if len(seed_cal_cost) > 1 else 0.0,
            }

    # Macro aggregation
    multi_seed_aggregated["macro"] = {}
    for m in macro_models:
        seed_pr = [per_seed_results[str(s)]["oof_summary"]["macro"][m]["oof_pr_auc"] for s in seeds]
        seed_roc = [per_seed_results[str(s)]["oof_summary"]["macro"][m]["oof_roc_auc"] for s in seeds]
        seed_def_prec = [per_seed_results[str(s)]["oof_summary"]["macro"][m]["default_precision"] for s in seeds]
        seed_def_rec = [per_seed_results[str(s)]["oof_summary"]["macro"][m]["default_recall"] for s in seeds]
        seed_def_cost = [per_seed_results[str(s)]["oof_summary"]["macro"][m]["default_cost"] for s in seeds]
        seed_cal_cost = [per_seed_results[str(s)]["oof_summary"]["macro"][m]["calibrated_cost"] for s in seeds]

    # Compute paired statistical tests across all 15 matched folds (when full benchmark)
    statistical_tests: Dict[str, Any] = {}
    if not is_pilot:
        for sz in sample_sizes:
            sz_str = str(sz)
            statistical_tests[sz_str] = {}
            pr_q_train = [fm["pr_auc"] for s in seeds for fm in per_seed_results[str(s)]["fold_metrics"][sz_str]["p15_kta_trainable_qsvm_k16"]]
            pr_q_fixed = [fm["pr_auc"] for s in seeds for fm in per_seed_results[str(s)]["fold_metrics"][sz_str]["p14_opt_separable_qsvm_k16"]]
            pr_c_fixed = [fm["pr_auc"] for s in seeds for fm in per_seed_results[str(s)]["fold_metrics"][sz_str]["p14_classical_rbf_k16"]]
            pr_c_train = [fm["pr_auc"] for s in seeds for fm in per_seed_results[str(s)]["fold_metrics"][sz_str]["p15_kta_trainable_rbf_k16"]]

            pairs = [
                ("trainable_qsvm_vs_fixed_rbf", pr_q_train, pr_c_fixed),
                ("trainable_qsvm_vs_fixed_qsvm", pr_q_train, pr_q_fixed),
                ("trainable_qsvm_vs_trainable_rbf", pr_q_train, pr_c_train),
                ("fixed_qsvm_vs_fixed_rbf", pr_q_fixed, pr_c_fixed),
            ]

            for comp_name, vec_a, vec_b in pairs:
                diffs = np.array(vec_a) - np.array(vec_b)
                tt_res = ttest_rel(vec_a, vec_b)
                try:
                    w_res = wilcoxon(vec_a, vec_b)
                    w_p = float(w_res.pvalue)
                    w_stat = float(w_res.statistic)
                except Exception:
                    w_p = 1.0
                    w_stat = 0.0

                statistical_tests[sz_str][comp_name] = {
                    "mean_diff": float(np.mean(diffs)),
                    "std_diff": float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0,
                    "t_statistic": float(tt_res.statistic) if not np.isnan(tt_res.statistic) else 0.0,
                    "t_pvalue": float(tt_res.pvalue) if not np.isnan(tt_res.pvalue) else 1.0,
                    "wilcoxon_stat": w_stat,
                    "wilcoxon_pvalue": w_p,
                    "is_significant_p05": bool(w_p < 0.05),
                }

    total_time = time.perf_counter() - t_start
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    results_payload = {
        "metadata": {
            "phase": "Phase 15 Pilot" if is_pilot else "Phase 15 Full Benchmark",
            "sample_sizes": sample_sizes,
            "seeds": seeds,
            "feature_dimension": feature_dimension,
            "n_splits": n_splits,
            "is_pilot": is_pilot,
            "total_runtime_seconds": round(total_time, 2),
            "total_runtime_minutes": round(total_time / 60.0, 2),
            "peak_memory_mb": round(peak_mem / (1024 * 1024), 2),
        },
        "multi_seed_aggregated_results": multi_seed_aggregated,
        "statistical_tests": statistical_tests,
        "per_seed_results": per_seed_results,
    }

    logger.info("==================================================================")
    logger.info("PHASE 15 BENCHMARK COMPLETE in %.2fs (%.2f min). Peak Memory: %.2f MB",
                total_time, total_time / 60.0, peak_mem / (1024 * 1024))
    logger.info("==================================================================")

    return results_payload
