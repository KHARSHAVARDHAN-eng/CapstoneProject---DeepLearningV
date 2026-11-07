"""Mentor Demonstration Script: Quantum Machine Learning for Financial Fraud Detection.

This script executes a reproducible, leak-free training and evaluation experiment
comparing Quantum Kernel Methods (Fixed Separable QSVM, Trainable KTA QSVM)
against Classical Baselines (Fixed Classical RBF, Trainable Classical RBF, Linear SVM, Macro Nystroem).

Protocol:
- Training Fold: 181,584 transactions, balanced subsample N=2,000 (10% fraud ratio, 200 frauds, 1800 legit).
- Validation Fold: 45,396 transactions, natural fraud prevalence (pi = 0.1652%, 75 frauds, 45,321 legit).
- Strict in-fold feature selection (ANOVA F-statistic, K=16).
- Strict in-fold angle scaling (MinMaxScaler [0, pi]).
- Strict in-training threshold calibration with analytical population prevalence adjustment.
- Zero test set leakage (data/processed/X_test.csv is completely untouched).
- Evaluates both default threshold (tau = 0.0) and prevalence-adjusted calibrated threshold (tau*).
- Outputs metrics CSV, comparison figures, and comprehensive logs.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
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

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.experiments.phase15_trainable_kernels import (
    calibrate_threshold_prevalence_adjusted,
    compute_bandwidth_separable_kernel_matrix,
    compute_model_metrics,
    generate_nested_subsets_p15,
    optimize_classical_rbf_kta,
    optimize_quantum_kernel_kta,
    score_kernel_chunked,
    select_top_features_infold,
)

OUTPUT_DIR = REPO_ROOT / "reports" / "mentor_demo"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = OUTPUT_DIR / "mentor_demo.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(LOG_FILE), mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger("mentor_demo")


def run_demonstration(seed: int = 42, chunk_size: int = 2500) -> Dict[str, Any]:
    """Runs the fresh matched-data training and evaluation benchmark."""
    logger.info("=" * 75)
    logger.info("QUANTUM FRAUD DETECTION: CAPSTONE MENTOR DEMONSTRATION BENCHMARK")
    logger.info("=" * 75)
    logger.info("Random Seed: %d | Chunk Size: %d", seed, chunk_size)
    logger.info("Output Directory: %s", OUTPUT_DIR)

    tracemalloc.start()
    t_start = time.perf_counter()

    # 1. Load data
    train_x_path = REPO_ROOT / "data" / "processed" / "X_train.csv"
    train_y_path = REPO_ROOT / "data" / "processed" / "y_train.csv"
    test_x_path = REPO_ROOT / "data" / "processed" / "X_test.csv"

    if not train_x_path.exists() or not train_y_path.exists():
        raise FileNotFoundError(f"Training data missing: {train_x_path}")

    logger.info("Security Verification: Test dataset %s is strictly isolated (UNTOUCHED).", test_x_path)
    logger.info("Loading training database from %s...", train_x_path)
    X_train_full = pd.read_csv(train_x_path)
    y_train_full = pd.read_csv(train_y_path).squeeze().to_numpy()
    n_train_total = len(y_train_full)
    n_frauds_total = int(np.sum(y_train_full == 1))
    logger.info("Loaded Training DB: %d transactions (%d frauds, prevalence = %.4f%%)",
                n_train_total, n_frauds_total, 100.0 * n_frauds_total / n_train_total)

    # 2. Setup Stratified 5-Fold Split and extract Fold 0
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    train_idx, val_idx = next(skf.split(X_train_full, y_train_full))

    X_tr_raw = X_train_full.iloc[train_idx]
    y_tr_raw = y_train_full[train_idx]
    X_va_raw = X_train_full.iloc[val_idx]
    y_va_raw = y_train_full[val_idx]

    n_val = len(y_va_raw)
    n_val_frauds = int(np.sum(y_va_raw == 1))
    pi_val = n_val_frauds / n_val
    logger.info("Fold 0 Partition: Train=%d (Frauds=%d) | Val=%d (Frauds=%d, prevalence=%.4f%%)",
                len(y_tr_raw), int(np.sum(y_tr_raw == 1)), n_val, n_val_frauds, 100.0 * pi_val)

    # 3. In-fold Feature Selection via ANOVA F-statistic
    t_feat = time.perf_counter()
    top16_cols, top16_scores = select_top_features_infold(X_tr_raw, y_tr_raw, k=16)
    top6_cols = top16_cols[:6]
    logger.info("In-Fold ANOVA Feature Selection completed in %.2fs:", time.perf_counter() - t_feat)
    logger.info("  Top 6 Features: %s", top6_cols)
    logger.info("  Top 16 Features: %s", top16_cols)

    # 4. In-fold Nested Subsampling (N=2000, 10% fraud ratio)
    target_sample_size = 2000
    subsets = generate_nested_subsets_p15(y_tr_raw, sample_sizes=[target_sample_size], fraud_ratio=0.10, random_state=seed)
    sub_idx = subsets[target_sample_size]
    X_tr_sub_raw = X_tr_raw.iloc[sub_idx]
    y_sub = y_tr_raw[sub_idx]
    n_sub_frauds = int(np.sum(y_sub == 1))
    pi_tr = n_sub_frauds / len(y_sub)
    logger.info("Training Subsample: N=%d (Frauds=%d, enriched prevalence=%.2f%%)",
                len(y_sub), n_sub_frauds, 100.0 * pi_tr)

    # 5. In-fold Feature Transformations
    # K=6 scaling
    scaler_k6 = MinMaxScaler(feature_range=(0.0, np.pi))
    X_tr_k6_s = scaler_k6.fit_transform(X_tr_sub_raw[top6_cols].values)
    X_va_k6_s = scaler_k6.transform(X_va_raw[top6_cols].values)

    # K=16 scaling
    scaler_k16 = MinMaxScaler(feature_range=(0.0, np.pi))
    X_tr_k16_s = scaler_k16.fit_transform(X_tr_sub_raw[top16_cols].values)
    X_va_k16_s = scaler_k16.transform(X_va_raw[top16_cols].values)

    # Inner CV for threshold calibration
    skf_inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)

    # Container for model evaluation records
    models_data: Dict[str, Dict[str, Any]] = {}

    # -------------------------------------------------------------------------
    # MODEL 1: Classical Linear SVM Baseline (K=6)
    # -------------------------------------------------------------------------
    logger.info("--- [1/6] Training Classical Linear SVM Baseline (K=6) ---")
    t0 = time.perf_counter()
    s_inner_lin = np.zeros(target_sample_size, dtype=np.float64)
    for in_tr, in_va in skf_inner.split(X_tr_k6_s, y_sub):
        c_in = LinearSVC(C=5.0, class_weight="balanced", random_state=seed, max_iter=5000)
        c_in.fit(X_tr_k6_s[in_tr], y_sub[in_tr])
        s_inner_lin[in_va] = c_in.decision_function(X_tr_k6_s[in_va])
    th_lin = calibrate_threshold_prevalence_adjusted(y_sub, s_inner_lin, pi_tr)

    clf_lin = LinearSVC(C=5.0, class_weight="balanced", random_state=seed, max_iter=5000)
    clf_lin.fit(X_tr_k6_s, y_sub)
    fit_time_lin = time.perf_counter() - t0

    t_eval = time.perf_counter()
    s_lin = clf_lin.decision_function(X_va_k6_s)
    eval_time_lin = time.perf_counter() - t_eval
    per_sample_lin_ms = (eval_time_lin / n_val) * 1000.0

    models_data["baseline_classical_linear_k6"] = {
        "name": "Classical Linear SVM (K=6)",
        "family": "Classical Linear",
        "features": 6,
        "fit_time_s": fit_time_lin,
        "eval_time_s": eval_time_lin,
        "latency_per_sample_ms": per_sample_lin_ms,
        "calibrated_threshold": float(th_lin),
        "scores": s_lin,
        "y_true": y_va_raw,
    }
    logger.info("  Linear SVM fit in %.2fs, scored in %.3fs (%.5f ms/sample)", fit_time_lin, eval_time_lin, per_sample_lin_ms)

    # -------------------------------------------------------------------------
    # MODEL 2: Fixed Classical RBF SVM (K=16, gamma=0.0005)
    # -------------------------------------------------------------------------
    logger.info("--- [2/6] Training Fixed Classical RBF SVM (K=16, gamma=0.0005) ---")
    t0 = time.perf_counter()
    gamma_fixed = 0.0005
    diff_sq_tr = (X_tr_k16_s[:, np.newaxis, :] - X_tr_k16_s[np.newaxis, :, :]) ** 2
    K_rbf_tr = np.exp(-gamma_fixed * np.sum(diff_sq_tr, axis=2))

    s_inner_rbf = np.zeros(target_sample_size, dtype=np.float64)
    for in_tr, in_va in skf_inner.split(X_tr_k16_s, y_sub):
        c_in = SVC(kernel="precomputed", C=5.0, class_weight="balanced", random_state=seed)
        c_in.fit(K_rbf_tr[np.ix_(in_tr, in_tr)], y_sub[in_tr])
        s_inner_rbf[in_va] = c_in.decision_function(K_rbf_tr[np.ix_(in_va, in_tr)])
    th_rbf = calibrate_threshold_prevalence_adjusted(y_sub, s_inner_rbf, pi_tr)

    clf_rbf = SVC(kernel="precomputed", C=5.0, class_weight="balanced", random_state=seed)
    clf_rbf.fit(K_rbf_tr, y_sub)
    fit_time_rbf = time.perf_counter() - t0

    t_eval = time.perf_counter()
    def get_chunk_rbf(s_i: int, e_i: int, xt=X_tr_k16_s, xv=X_va_k16_s, g=gamma_fixed) -> np.ndarray:
        diff_sq_ch = (xv[s_i:e_i, np.newaxis, :] - xt[np.newaxis, :, :]) ** 2
        return np.exp(-g * np.sum(diff_sq_ch, axis=2))
    s_rbf = score_kernel_chunked(clf_rbf, get_chunk_rbf, n_val, chunk_size=chunk_size)
    eval_time_rbf = time.perf_counter() - t_eval
    per_sample_rbf_ms = (eval_time_rbf / n_val) * 1000.0

    models_data["p14_classical_rbf_k16"] = {
        "name": "Fixed Classical RBF SVM (K=16)",
        "family": "Classical RBF",
        "features": 16,
        "fit_time_s": fit_time_rbf,
        "eval_time_s": eval_time_rbf,
        "latency_per_sample_ms": per_sample_rbf_ms,
        "calibrated_threshold": float(th_rbf),
        "scores": s_rbf,
        "y_true": y_va_raw,
    }
    logger.info("  Fixed RBF fit in %.2fs, scored in %.3fs (%.5f ms/sample)", fit_time_rbf, eval_time_rbf, per_sample_rbf_ms)

    # -------------------------------------------------------------------------
    # MODEL 3: Phase 14 Optimized Separable QSVM (K=16, beta=0.0306)
    # -------------------------------------------------------------------------
    logger.info("--- [3/6] Training Fixed Separable QSVM (K=16, beta=0.0306) ---")
    t0 = time.perf_counter()
    beta_fixed = 0.05 * np.sqrt(6.0 / 16.0)  # ~0.0306
    K_sep_tr = compute_bandwidth_separable_kernel_matrix(X_tr_k16_s, X_tr_k16_s, beta=beta_fixed)

    s_inner_sep = np.zeros(target_sample_size, dtype=np.float64)
    for in_tr, in_va in skf_inner.split(X_tr_k16_s, y_sub):
        c_in = SVC(kernel="precomputed", C=5.0, class_weight="balanced", random_state=seed)
        c_in.fit(K_sep_tr[np.ix_(in_tr, in_tr)], y_sub[in_tr])
        s_inner_sep[in_va] = c_in.decision_function(K_sep_tr[np.ix_(in_va, in_tr)])
    th_sep = calibrate_threshold_prevalence_adjusted(y_sub, s_inner_sep, pi_tr)

    clf_sep = SVC(kernel="precomputed", C=5.0, class_weight="balanced", random_state=seed)
    clf_sep.fit(K_sep_tr, y_sub)
    fit_time_sep = time.perf_counter() - t0

    t_eval = time.perf_counter()
    def get_chunk_sep(s_i: int, e_i: int, xt=X_tr_k16_s, xv=X_va_k16_s, b=beta_fixed) -> np.ndarray:
        return compute_bandwidth_separable_kernel_matrix(xv[s_i:e_i], xt, beta=b)
    s_sep = score_kernel_chunked(clf_sep, get_chunk_sep, n_val, chunk_size=chunk_size)
    eval_time_sep = time.perf_counter() - t_eval
    per_sample_sep_ms = (eval_time_sep / n_val) * 1000.0

    models_data["p14_opt_separable_qsvm_k16"] = {
        "name": "Fixed Separable QSVM (K=16)",
        "family": "Quantum Separable",
        "features": 16,
        "fit_time_s": fit_time_sep,
        "eval_time_s": eval_time_sep,
        "latency_per_sample_ms": per_sample_sep_ms,
        "calibrated_threshold": float(th_sep),
        "scores": s_sep,
        "y_true": y_va_raw,
    }
    logger.info("  Fixed QSVM fit in %.2fs, scored in %.3fs (%.5f ms/sample)", fit_time_sep, eval_time_sep, per_sample_sep_ms)

    # -------------------------------------------------------------------------
    # MODEL 4: Phase 15 Trainable Quantum Kernel via Centered Regularized KTA
    # -------------------------------------------------------------------------
    logger.info("--- [4/6] Training Trainable QSVM via Centered Regularized KTA (K=16, lambda=200.0) ---")
    t0 = time.perf_counter()
    # KTA optimization on 1000-sample pilot subset for efficiency
    opt_sub = np.random.RandomState(seed).choice(target_sample_size, size=1000, replace=False)
    X_opt, y_opt = X_tr_k16_s[opt_sub], y_sub[opt_sub]

    t_opt_q = time.perf_counter()
    opt_theta_q, final_ckta_q = optimize_quantum_kernel_kta(
        X_opt, y_opt, init_beta=beta_fixed, lambda_reg=200.0, maxiter=25
    )
    opt_time_q = time.perf_counter() - t_opt_q
    logger.info("  KTA Optimization in %.2fs: CKTA=%.4f, theta range=[%.4f, %.4f]",
                opt_time_q, final_ckta_q, opt_theta_q.min(), opt_theta_q.max())

    K_kta_q_tr = compute_bandwidth_separable_kernel_matrix(X_tr_k16_s, X_tr_k16_s, beta=opt_theta_q)

    s_inner_kta_q = np.zeros(target_sample_size, dtype=np.float64)
    for in_tr, in_va in skf_inner.split(X_tr_k16_s, y_sub):
        c_in = SVC(kernel="precomputed", C=5.0, class_weight="balanced", random_state=seed)
        c_in.fit(K_kta_q_tr[np.ix_(in_tr, in_tr)], y_sub[in_tr])
        s_inner_kta_q[in_va] = c_in.decision_function(K_kta_q_tr[np.ix_(in_va, in_tr)])
    th_kta_q = calibrate_threshold_prevalence_adjusted(y_sub, s_inner_kta_q, pi_tr)

    clf_kta_q = SVC(kernel="precomputed", C=5.0, class_weight="balanced", random_state=seed)
    clf_kta_q.fit(K_kta_q_tr, y_sub)
    fit_time_kta_q = time.perf_counter() - t0

    t_eval = time.perf_counter()
    def get_chunk_kta_q(s_i: int, e_i: int, xt=X_tr_k16_s, xv=X_va_k16_s, th=opt_theta_q) -> np.ndarray:
        return compute_bandwidth_separable_kernel_matrix(xv[s_i:e_i], xt, beta=th)
    s_kta_q = score_kernel_chunked(clf_kta_q, get_chunk_kta_q, n_val, chunk_size=chunk_size)
    eval_time_kta_q = time.perf_counter() - t_eval
    per_sample_kta_q_ms = (eval_time_kta_q / n_val) * 1000.0

    models_data["p15_kta_trainable_qsvm_k16"] = {
        "name": "Trainable KTA QSVM (K=16)",
        "family": "Quantum Trainable",
        "features": 16,
        "fit_time_s": fit_time_kta_q,
        "eval_time_s": eval_time_kta_q,
        "latency_per_sample_ms": per_sample_kta_q_ms,
        "calibrated_threshold": float(th_kta_q),
        "scores": s_kta_q,
        "y_true": y_va_raw,
    }
    logger.info("  Trainable QSVM fit in %.2fs (opt: %.2fs), scored in %.3fs (%.5f ms/sample)",
                fit_time_kta_q, opt_time_q, eval_time_kta_q, per_sample_kta_q_ms)

    # -------------------------------------------------------------------------
    # MODEL 5: Phase 15 Trainable Classical RBF via Centered Regularized KTA
    # -------------------------------------------------------------------------
    logger.info("--- [5/6] Training Trainable Classical RBF via Centered Regularized KTA (K=16, lambda=200.0) ---")
    t0 = time.perf_counter()
    t_opt_rbf = time.perf_counter()
    opt_gamma_rbf, final_ckta_rbf = optimize_classical_rbf_kta(
        X_opt, y_opt, init_gamma=0.0005, lambda_reg=200.0, maxiter=25
    )
    opt_time_rbf = time.perf_counter() - t_opt_rbf
    logger.info("  KTA Optimization in %.2fs: CKTA=%.4f, gamma range=[%.6f, %.6f]",
                opt_time_rbf, final_ckta_rbf, opt_gamma_rbf.min(), opt_gamma_rbf.max())

    K_kta_rbf_tr = np.exp(-np.sum(opt_gamma_rbf * diff_sq_tr, axis=2))

    s_inner_kta_rbf = np.zeros(target_sample_size, dtype=np.float64)
    for in_tr, in_va in skf_inner.split(X_tr_k16_s, y_sub):
        c_in = SVC(kernel="precomputed", C=5.0, class_weight="balanced", random_state=seed)
        c_in.fit(K_kta_rbf_tr[np.ix_(in_tr, in_tr)], y_sub[in_tr])
        s_inner_kta_rbf[in_va] = c_in.decision_function(K_kta_rbf_tr[np.ix_(in_va, in_tr)])
    th_kta_rbf = calibrate_threshold_prevalence_adjusted(y_sub, s_inner_kta_rbf, pi_tr)

    clf_kta_rbf = SVC(kernel="precomputed", C=5.0, class_weight="balanced", random_state=seed)
    clf_kta_rbf.fit(K_kta_rbf_tr, y_sub)
    fit_time_kta_rbf = time.perf_counter() - t0

    t_eval = time.perf_counter()
    def get_chunk_kta_rbf(s_i: int, e_i: int, xt=X_tr_k16_s, xv=X_va_k16_s, g=opt_gamma_rbf) -> np.ndarray:
        diff_sq_ch = (xv[s_i:e_i, np.newaxis, :] - xt[np.newaxis, :, :]) ** 2
        return np.exp(-np.sum(g * diff_sq_ch, axis=2))
    s_kta_rbf = score_kernel_chunked(clf_kta_rbf, get_chunk_kta_rbf, n_val, chunk_size=chunk_size)
    eval_time_kta_rbf = time.perf_counter() - t_eval
    per_sample_kta_rbf_ms = (eval_time_kta_rbf / n_val) * 1000.0

    models_data["p15_kta_trainable_rbf_k16"] = {
        "name": "Trainable Classical RBF (K=16)",
        "family": "Classical Trainable",
        "features": 16,
        "fit_time_s": fit_time_kta_rbf,
        "eval_time_s": eval_time_kta_rbf,
        "latency_per_sample_ms": per_sample_kta_rbf_ms,
        "calibrated_threshold": float(th_kta_rbf),
        "scores": s_kta_rbf,
        "y_true": y_va_raw,
    }
    logger.info("  Trainable RBF fit in %.2fs (opt: %.2fs), scored in %.3fs (%.5f ms/sample)",
                fit_time_kta_rbf, opt_time_rbf, eval_time_kta_rbf, per_sample_kta_rbf_ms)

    # -------------------------------------------------------------------------
    # MODEL 6: Macro Nystroem RBF Approximation (Full Data Reference, N=181,584)
    # -------------------------------------------------------------------------
    logger.info("--- [6/6] Training Macro Nystroem RBF Reference (Full Fold Train N=181,584, K=16) ---")
    t0 = time.perf_counter()
    X_tr_k16_full = X_tr_raw[top16_cols].values
    X_va_k16_full = X_va_raw[top16_cols].values
    scl_macro = MinMaxScaler(feature_range=(0.0, np.pi))
    X_tr_macro_s = scl_macro.fit_transform(X_tr_k16_full)
    X_va_macro_s = scl_macro.transform(X_va_k16_full)

    nys = Nystroem(kernel="rbf", gamma=gamma_fixed, n_components=500, random_state=seed)
    X_tr_nys = nys.fit_transform(X_tr_macro_s)
    X_va_nys = nys.transform(X_va_macro_s)

    # In-fold calibration on 10k random samples
    rng_mac = np.random.RandomState(seed)
    cal_idx = rng_mac.choice(len(y_tr_raw), size=min(10000, len(y_tr_raw)), replace=False)
    clf_nys = LinearSVC(C=5.0, class_weight="balanced", random_state=seed, max_iter=5000)
    clf_nys.fit(X_tr_nys, y_tr_raw)
    fit_time_nys = time.perf_counter() - t0

    s_cal_nys = clf_nys.decision_function(X_tr_nys[cal_idx])
    th_nys = calibrate_threshold_prevalence_adjusted(y_tr_raw[cal_idx], s_cal_nys, np.mean(y_tr_raw[cal_idx]))

    t_eval = time.perf_counter()
    s_nys = clf_nys.decision_function(X_va_nys)
    eval_time_nys = time.perf_counter() - t_eval
    per_sample_nys_ms = (eval_time_nys / n_val) * 1000.0

    models_data["macro_nystroem_rbf_approx"] = {
        "name": "Macro Nystroem RBF (N=181k Ref)",
        "family": "Macro Classical Control",
        "features": 16,
        "fit_time_s": fit_time_nys,
        "eval_time_s": eval_time_nys,
        "latency_per_sample_ms": per_sample_nys_ms,
        "calibrated_threshold": float(th_nys),
        "scores": s_nys,
        "y_true": y_va_raw,
    }
    logger.info("  Macro Nystroem fit in %.2fs, scored in %.3fs (%.5f ms/sample)",
                fit_time_nys, eval_time_nys, per_sample_nys_ms)

    total_runtime = time.perf_counter() - t_start
    _, peak_mem_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mem_mb = peak_mem_bytes / (1024 * 1024)
    logger.info("=" * 75)
    logger.info("Benchmark Execution Complete in %.2fs (~%.2f min). Peak Memory: %.2f MB",
                total_runtime, total_runtime / 60.0, peak_mem_mb)
    logger.info("=" * 75)

    # 6. Compute Comprehensive Model Metrics Table
    results_rows = []
    for model_id, mdata in models_data.items():
        y_true = mdata["y_true"]
        scores = mdata["scores"]
        th_cal = mdata["calibrated_threshold"]

        # Default threshold (tau = 0.0)
        m_def = compute_model_metrics(y_true, scores, threshold=0.0)
        # Calibrated threshold (tau = th_cal)
        m_cal = compute_model_metrics(y_true, scores, threshold=th_cal)

        pr_auc = float(average_precision_score(y_true, scores))
        roc_auc = float(roc_auc_score(y_true, scores))

        # Additional fraud diagnostics
        tn_def, fp_def, fn_def, tp_def = confusion_matrix(y_true, (scores >= 0.0).astype(int)).ravel()
        tn_cal, fp_cal, fn_cal, tp_cal = confusion_matrix(y_true, (scores >= th_cal).astype(int)).ravel()

        fpr_def = fp_def / (fp_def + tn_def)
        fnr_def = fn_def / (fn_def + tp_def)
        fpr_cal = fp_cal / (fp_cal + tn_cal)
        fnr_cal = fn_cal / (fn_cal + tp_cal)

        results_rows.append({
            "model_id": model_id,
            "model_name": mdata["name"],
            "model_family": mdata["family"],
            "features_k": mdata["features"],
            "train_sample_size": 181584 if "macro" in model_id else target_sample_size,
            "val_sample_size": n_val,
            "val_frauds": n_val_frauds,
            "val_prevalence_pct": round(pi_val * 100.0, 4),
            "pr_auc": round(pr_auc, 4),
            "roc_auc": round(roc_auc, 4),
            "fit_time_seconds": round(mdata["fit_time_s"], 2),
            "eval_time_seconds": round(mdata["eval_time_s"], 3),
            "latency_per_sample_ms": round(mdata["latency_per_sample_ms"], 5),
            "default_threshold": 0.0,
            "default_precision_pct": round(m_def["precision"] * 100.0, 2),
            "default_recall_pct": round(m_def["recall"] * 100.0, 2),
            "default_f1": round(m_def["f1"], 4),
            "default_tp": int(tp_def),
            "default_fp": int(fp_def),
            "default_tn": int(tn_def),
            "default_fn": int(fn_def),
            "default_fpr_pct": round(fpr_def * 100.0, 4),
            "default_fnr_pct": round(fnr_def * 100.0, 2),
            "default_cost_dollars": float(m_def["financial_cost"]),
            "calibrated_threshold": round(th_cal, 4),
            "calibrated_precision_pct": round(m_cal["precision"] * 100.0, 2),
            "calibrated_recall_pct": round(m_cal["recall"] * 100.0, 2),
            "calibrated_f1": round(m_cal["f1"], 4),
            "calibrated_tp": int(tp_cal),
            "calibrated_fp": int(fp_cal),
            "calibrated_tn": int(tn_cal),
            "calibrated_fn": int(fn_cal),
            "calibrated_fpr_pct": round(fpr_cal * 100.0, 4),
            "calibrated_fnr_pct": round(fnr_cal * 100.0, 2),
            "calibrated_cost_dollars": float(m_cal["financial_cost"]),
        })

    df_results = pd.DataFrame(results_rows)
    csv_path = OUTPUT_DIR / "mentor_demo_results.csv"
    df_results.to_csv(csv_path, index=False)
    logger.info("Saved fresh benchmark results to %s", csv_path)

    # 7. Generate Mentor Demonstration Figures
    generate_figures(models_data, df_results, pi_val)

    return {
        "results_df": df_results,
        "models_data": models_data,
        "total_runtime_s": total_runtime,
        "peak_mem_mb": peak_mem_mb,
        "n_val": n_val,
        "n_val_frauds": n_val_frauds,
    }


def generate_figures(models_data: Dict[str, Dict[str, Any]], df_results: pd.DataFrame, pi_val: float) -> None:
    """Generates four publication-quality diagnostic plots for mentor presentation."""
    sns.set_theme(style="whitegrid", font_scale=1.1)
    palette = {
        "baseline_classical_linear_k6": "#7f7f7f",      # Grey
        "p14_classical_rbf_k16": "#1f77b4",             # Blue
        "p14_opt_separable_qsvm_k16": "#2ca02c",        # Green
        "p15_kta_trainable_qsvm_k16": "#d62728",        # Red / Crimson
        "p15_kta_trainable_rbf_k16": "#ff7f0e",         # Orange
        "macro_nystroem_rbf_approx": "#9467bd",         # Purple
    }

    # -------------------------------------------------------------------------
    # FIGURE 1: Precision-Recall Curves
    # -------------------------------------------------------------------------
    plt.figure(figsize=(9, 6.5), dpi=300)
    for model_id, mdata in models_data.items():
        y_true = mdata["y_true"]
        scores = mdata["scores"]
        prec, rec, _ = precision_recall_curve(y_true, scores)
        pr_auc = average_precision_score(y_true, scores)
        label = f"{mdata['name']} (PR-AUC = {pr_auc:.4f})"
        color = palette.get(model_id, "#333333")
        lw = 2.5 if "qsvm" in model_id or "rbf" in model_id else 1.8
        ls = "-" if "macro" not in model_id else "--"
        plt.plot(rec, prec, label=label, color=color, linewidth=lw, linestyle=ls)

    plt.axhline(pi_val, color="black", linestyle=":", linewidth=1.5,
                label=f"Natural Fraud Prevalence ({pi_val*100.0:.3f}%)")
    plt.title("Precision-Recall Curves: Quantum vs. Classical SVMs\n(Evaluated on N=45,396 Transactions, 75 Frauds, Natural Prevalence)",
              fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Recall (Fraction of Detected Frauds)", fontsize=11, fontweight="bold")
    plt.ylabel("Precision (Fraction of Legitimate Alerts)", fontsize=11, fontweight="bold")
    plt.xlim([0.0, 1.02])
    plt.ylim([0.0, 1.02])
    plt.legend(loc="upper right", frameon=True, framealpha=0.92, fontsize=9.5)
    plt.tight_layout()
    fig1_path = OUTPUT_DIR / "pr_curves.png"
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    logger.info("Saved Figure 1: %s", fig1_path)

    # -------------------------------------------------------------------------
    # FIGURE 2: Confusion Matrices (Default vs Calibrated Threshold)
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5), dpi=300)
    models_to_plot = [
        "baseline_classical_linear_k6",
        "p14_classical_rbf_k16",
        "p14_opt_separable_qsvm_k16",
        "p15_kta_trainable_qsvm_k16",
        "p15_kta_trainable_rbf_k16",
        "macro_nystroem_rbf_approx",
    ]

    for idx, model_id in enumerate(models_to_plot):
        ax = axes[idx // 3, idx % 3]
        mdata = models_data[model_id]
        scores = mdata["scores"]
        y_true = mdata["y_true"]
        th_cal = mdata["calibrated_threshold"]
        preds = (scores >= th_cal).astype(int)
        cm = confusion_matrix(y_true, preds)

        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    annot_kws={"size": 11, "weight": "bold"})
        ax.set_title(f"{mdata['name']}\n(Calibrated Threshold $\\tau^*={th_cal:.2f}$)",
                     fontsize=10.5, fontweight="bold")
        ax.set_xlabel("Predicted Label (0: Legit, 1: Fraud)", fontsize=9.5)
        ax.set_ylabel("True Label (0: Legit, 1: Fraud)", fontsize=9.5)
        ax.set_xticklabels(["Legit", "Fraud"])
        ax.set_yticklabels(["Legit", "Fraud"])

    fig.suptitle("Confusion Matrices under Prevalence-Adjusted Calibrated Operating Thresholds\n($N_{\\text{val}}=45,396$, 75 Frauds, 45,321 Legitimate)",
                 fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig2_path = OUTPUT_DIR / "confusion_matrices.png"
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    logger.info("Saved Figure 2: %s", fig2_path)

    # -------------------------------------------------------------------------
    # FIGURE 3: Model Metrics Comparison Bar Chart
    # -------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), dpi=300)
    short_names = [
        "Linear K6",
        "Fixed RBF K16",
        "Fixed QSVM K16",
        "Trainable QSVM",
        "Trainable RBF",
        "Macro Nystroem",
    ]
    x_pos = np.arange(len(short_names))
    width = 0.35

    prauc_vals = df_results["pr_auc"].values
    rocauc_vals = df_results["roc_auc"].values

    ax1.bar(x_pos - width/2, prauc_vals, width, label="PR-AUC (Primary)", color="#1f77b4")
    ax1.bar(x_pos + width/2, rocauc_vals, width, label="ROC-AUC", color="#aec7e8")
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(short_names, rotation=25, ha="right", fontsize=9.5)
    ax1.set_ylim([0.0, 1.05])
    ax1.set_ylabel("Metric Value", fontsize=11, fontweight="bold")
    ax1.set_title("Discriminative Capacity: PR-AUC vs. ROC-AUC", fontsize=12, fontweight="bold")
    ax1.legend(loc="lower right", frameon=True)
    for i, v in enumerate(prauc_vals):
        ax1.text(i - width/2, v + 0.02, f"{v:.3f}", ha="center", fontsize=8.5, fontweight="bold")

    cost_cal = df_results["calibrated_cost_dollars"].values
    cost_def = df_results["default_cost_dollars"].values
    ax2.bar(x_pos - width/2, cost_cal, width, label="Calibrated Cost ($)", color="#2ca02c")
    ax2.bar(x_pos + width/2, cost_def, width, label="Default Cost ($)", color="#ff7f0e")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(short_names, rotation=25, ha="right", fontsize=9.5)
    ax2.set_ylabel("Financial Loss ($: 10FP + 200FN)", fontsize=11, fontweight="bold")
    ax2.set_title("Operational Financial Cost ($R=20$)", fontsize=12, fontweight="bold")
    ax2.legend(loc="upper right", frameon=True)
    for i, v in enumerate(cost_cal):
        ax2.text(i - width/2, v + 200, f"${int(v)}", ha="center", fontsize=8, fontweight="bold")

    fig.suptitle("Model Evaluation Summary: Detection Ranking & Financial Cost",
                 fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig3_path = OUTPUT_DIR / "model_metric_comparison.png"
    plt.savefig(fig3_path, dpi=300)
    plt.close()
    logger.info("Saved Figure 3: %s", fig3_path)

    # -------------------------------------------------------------------------
    # FIGURE 4: Runtime and Inference Latency Comparison
    # -------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2), dpi=300)
    fit_times = df_results["fit_time_seconds"].values
    latencies = df_results["latency_per_sample_ms"].values

    colors = [palette.get(m, "#333333") for m in df_results["model_id"]]
    ax1.bar(x_pos, fit_times, color=colors, edgecolor="black", linewidth=0.8)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(short_names, rotation=25, ha="right", fontsize=9.5)
    ax1.set_ylabel("Training Time (seconds)", fontsize=11, fontweight="bold")
    ax1.set_title("Training Execution Time", fontsize=12, fontweight="bold")
    for i, v in enumerate(fit_times):
        ax1.text(i, v + max(fit_times)*0.02, f"{v:.1f}s", ha="center", fontsize=8.5, fontweight="bold")

    ax2.bar(x_pos, latencies, color=colors, edgecolor="black", linewidth=0.8)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(short_names, rotation=25, ha="right", fontsize=9.5)
    ax2.set_yscale("log")
    ax2.set_ylabel("Inference Latency (ms/sample, Log Scale)", fontsize=11, fontweight="bold")
    ax2.set_title("Per-Sample Scoring Latency (CPU Simulation)", fontsize=12, fontweight="bold")
    for i, v in enumerate(latencies):
        ax2.text(i, v * 1.25, f"{v:.4f}ms", ha="center", fontsize=8, fontweight="bold")

    fig.suptitle("Computational Efficiency: Training Runtime & Inference Latency",
                 fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig4_path = OUTPUT_DIR / "runtime_comparison.png"
    plt.savefig(fig4_path, dpi=300)
    plt.close()
    logger.info("Saved Figure 4: %s", fig4_path)


def print_summary_table(results_df: pd.DataFrame) -> None:
    """Prints a clean ASCII summary table to stdout."""
    print("\n" + "=" * 90)
    print("MENTOR DEMO RESULTS: FRESH RUN (FOLD 0, N_VAL=45,396, FRAUDS=75, PREVALENCE=0.165%)")
    print("=" * 90)
    cols = ["model_name", "pr_auc", "roc_auc", "default_precision_pct", "default_recall_pct",
            "calibrated_cost_dollars", "fit_time_seconds", "latency_per_sample_ms"]
    display_df = results_df[cols].copy()
    display_df.columns = ["Model Architecture", "PR-AUC", "ROC-AUC", "Prec(%)", "Recall(%)",
                          "Cost($)", "Fit(s)", "Lat(ms)"]
    print(display_df.to_string(index=False))
    print("=" * 90 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Capstone Mentor Demonstration Benchmark")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for demonstration split")
    parser.add_argument("--chunk-size", type=int, default=2500, help="Scoring chunk size")
    args = parser.parse_args()

    demo_output = run_demonstration(seed=args.seed, chunk_size=args.chunk_size)
    print_summary_table(demo_output["results_df"])


if __name__ == "__main__":
    main()
