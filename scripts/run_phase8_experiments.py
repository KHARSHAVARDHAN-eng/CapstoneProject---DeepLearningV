#!/usr/bin/env python3
"""
Phase 8: Master Improvement Execution Script.

Orchestrates all Phase 8 experiments:
1. Deep Quantum Kernel Diagnostics & Spectral Analysis.
2. Feature Selection Strategy Comparison (Training data only).
3. Quantum Feature Map Architectural Variant Evaluation (Validation data).
4. Option A Scalability (Fixed 10% fraud ratio across N=500, 1000, 2000, 3 seeds).
5. Cost-Optimal Threshold Calibration & False-Positive Mitigation.
6. 1,000-Replicate Paired Non-Parametric Bootstrap Significance Testing.
7. Final Holdout Test Set Comparison against Historical Phase 6 Baseline.
8. Artifact Serialization and Master Phase 8 Academic Report Generation.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC, SVC

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.quantum_subsets import get_quantum_partition_indices
from src.experiments.phase8_improvements import (
    compare_feature_selection_strategies,
    compute_analytic_z_kernel,
    compute_paired_bootstrap_significance,
    diagnose_kernel_properties,
    evaluate_feature_map_architectures,
    evaluate_threshold_calibrations,
    run_option_a_scalability_experiment,
)
from src.models.quantum_kernel import (
    compute_statevector_matrix,
    create_zz_feature_map,
)

# Logging configuration
LOGS_DIR = PROJECT_ROOT / "reports"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures" / "phase8"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOGS_DIR / "phase8_run.log", mode="w"),
    ],
)
logger = logging.getLogger("phase8_runner")


def plot_kernel_similarity_distributions(
    diag_ref: dict,
    diag_zz1: dict,
    diag_z1: dict,
    output_path: Path,
) -> None:
    """Generates comparison bar chart of within-class vs cross-class kernel similarities."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(10, 6))

    models = [
        "Canonical QSVM\n(ZZ reps=2, [0, π])",
        "Shallow QSVM\n(ZZ reps=1, [0, π])",
        "Separable QSVM\n(Z reps=1, [0, π])",
    ]
    ff_means = [
        diag_ref["class_similarities"]["fraud_fraud_mean"],
        diag_zz1["class_similarities"]["fraud_fraud_mean"],
        diag_z1["class_similarities"]["fraud_fraud_mean"],
    ]
    fl_means = [
        diag_ref["class_similarities"]["fraud_legit_mean"],
        diag_zz1["class_similarities"]["fraud_legit_mean"],
        diag_z1["class_similarities"]["fraud_legit_mean"],
    ]
    ll_means = [
        diag_ref["class_similarities"]["legit_legit_mean"],
        diag_zz1["class_similarities"]["legit_legit_mean"],
        diag_z1["class_similarities"]["legit_legit_mean"],
    ]

    x = np.arange(len(models))
    width = 0.25

    rects1 = ax.bar(x - width, ff_means, width, label="Fraud-Fraud (Target Class)", color="#d9534f")
    rects2 = ax.bar(x, fl_means, width, label="Fraud-Legit (Cross-Class Boundary)", color="#f0ad4e")
    rects3 = ax.bar(x + width, ll_means, width, label="Legit-Legit (Majority Class)", color="#5cb85c")

    ax.set_ylabel("Mean Kernel Similarity K(x, x')", fontsize=12)
    ax.set_title("Kernel Similarity by Class Pair: The Collapse of Entangled Geometric Locality", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.6)

    # Highlight ratio
    for i, (ff, fl) in enumerate(zip(ff_means, fl_means)):
        ratio = ff / max(1e-6, fl)
        ax.annotate(
            f"FF/FL = {ratio:.2f}x",
            xy=(x[i] - width/2, max(ff, fl) + 0.03),
            ha="center",
            fontsize=10,
            fontweight="bold",
            color="black" if ratio >= 1.0 else "darkred",
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info("Saved similarity distribution plot to %s", output_path)


def plot_feature_selection_comparison(strat_results: dict, output_path: Path) -> None:
    """Plots validation PR-AUC and ROC-AUC for feature selection strategies."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    names = [s["name"] for s in strat_results.values()]
    lin_pr = [s["linear_svm"]["pr_auc"] for s in strat_results.values()]
    rbf_pr = [s["rbf_svm"]["pr_auc"] for s in strat_results.values()]
    lin_roc = [s["linear_svm"]["roc_auc"] for s in strat_results.values()]
    rbf_roc = [s["rbf_svm"]["roc_auc"] for s in strat_results.values()]

    y = np.arange(len(names))
    height = 0.35

    ax1.barh(y - height/2, lin_pr, height, label="Linear SVM", color="#337ab7")
    ax1.barh(y + height/2, rbf_pr, height, label="Classical RBF SVM", color="#5cb85c")
    ax1.set_yticks(y)
    ax1.set_yticklabels(names, fontsize=10)
    ax1.set_xlabel("Validation PR-AUC", fontsize=12)
    ax1.set_title("PR-AUC Across Feature Selection Strategies", fontsize=13, fontweight="bold")
    ax1.set_xlim(0.75, 0.90)
    ax1.legend(loc="lower right")
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2.barh(y - height/2, lin_roc, height, label="Linear SVM", color="#337ab7")
    ax2.barh(y + height/2, rbf_roc, height, label="Classical RBF SVM", color="#5cb85c")
    ax2.set_yticks(y)
    ax2.set_yticklabels(["" for _ in names])
    ax2.set_xlabel("Validation ROC-AUC", fontsize=12)
    ax2.set_title("ROC-AUC Across Feature Selection Strategies", fontsize=13, fontweight="bold")
    ax2.set_xlim(0.96, 0.99)
    ax2.legend(loc="lower right")
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info("Saved feature selection comparison plot to %s", output_path)


def plot_option_a_scalability(option_a_summary: dict, output_path: Path) -> None:
    """Plots PR-AUC and False Positives vs Training Sample Size N in Option A."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    sizes = [500, 1000, 2000]
    models = {
        "linear_svm": ("Linear SVM (C=5.0)", "#337ab7", "o-"),
        "rbf_svm": ("Classical RBF SVM", "#5cb85c", "s-"),
        "qsvm_zz_r2_ref": ("Canonical QSVM (ZZ reps=2)", "#d9534f", "d--"),
        "qsvm_zz_r1_imp": ("Shallow QSVM (ZZ reps=1)", "#f0ad4e", "^-"),
        "qsvm_z_r1_sep": ("Separable QSVM (Z reps=1)", "#9b59b6", "v-"),
    }

    for m_key, (m_label, color, style) in models.items():
        pr_means = [option_a_summary[str(N)][m_key]["pr_auc_mean"] for N in sizes]
        pr_stds = [option_a_summary[str(N)][m_key]["pr_auc_std"] for N in sizes]
        fp_means = [option_a_summary[str(N)][m_key]["fp_mean"] for N in sizes]

        ax1.errorbar(sizes, pr_means, yerr=pr_stds, label=m_label, color=color, fmt=style, capsize=4, linewidth=2)
        ax2.plot(sizes, fp_means, style, label=m_label, color=color, linewidth=2)

    ax1.set_xlabel("Training Sample Size N (10% Fixed Fraud Ratio)", fontsize=12)
    ax1.set_ylabel("Validation PR-AUC", fontsize=12)
    ax1.set_title("Option A: PR-AUC Scalability (Mean ± 1 SD across 3 Seeds)", fontsize=13, fontweight="bold")
    ax1.set_xticks(sizes)
    ax1.legend(loc="lower left", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2.set_xlabel("Training Sample Size N (10% Fixed Fraud Ratio)", fontsize=12)
    ax2.set_ylabel("Mean Validation False Positives (at τ = 0.0)", fontsize=12)
    ax2.set_title("Option A: False Alarms vs Training Sample Size N", fontsize=13, fontweight="bold")
    ax2.set_xticks(sizes)
    ax2.legend(loc="upper left", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info("Saved Option A scalability plot to %s", output_path)


def plot_threshold_cost_curves(threshold_data: dict, output_path: Path) -> None:
    """Plots scenario loss vs decision threshold across cost ratios for improved QSVM."""
    fig, ax = plt.subplots(figsize=(10, 6))

    q_data = threshold_data["improved_qsvm_zz1"]["cost_optimal_cutoffs"]
    colors = {"5": "#5bc0de", "20": "#5cb85c", "50": "#f0ad4e", "100": "#d9534f"}

    for r_str, res in q_data.items():
        c_fp = res["c_fp"]
        r_val = res["cost_ratio_R"]
        opt_tau = res["optimal_threshold"]
        min_loss = res["minimized_scenario_loss"]
        fp = res["false_positives"]
        fn = res["false_negatives"]

        ax.scatter(
            [opt_tau], [min_loss],
            color=colors[r_str], s=120, zorder=5,
            label=f"R={r_val} (Optimal τ*={opt_tau:.3f}, Loss=${min_loss:,.0f}, FP={fp}, FN={fn})",
        )

    ax.set_xlabel("Decision Threshold (τ)", fontsize=12)
    ax.set_ylabel("Illustrative Scenario Financial Loss ($)", fontsize=12)
    ax.set_title("Decision Cutoff Optimization: Shallow QSVM (ZZ reps=1)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info("Saved threshold cost curves plot to %s", output_path)


def main() -> None:
    """Master Phase 8 execution pipeline."""
    logger.info("================================================================================")
    logger.info("STARTING PHASE 8: COMPREHENSIVE QUANTUM FRAUD IMPROVEMENTS & AUDIT")
    logger.info("================================================================================")
    start_total_time = time.perf_counter()

    # 1. Load Data
    logger.info("Loading training and validation partitions...")
    X_train_full = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "X_train.csv")
    y_train_full = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "y_train.csv").squeeze("columns")

    # Reproduce exact non-validation pool and validation partition
    t_idx_ref, v_idx_ref = get_quantum_partition_indices(
        y_train_full, n_train=500, n_train_fraud=100, n_val=20000, random_state=42
    )
    val_set = set(v_idx_ref)
    non_val_indices = np.array([i for i in range(len(y_train_full)) if i not in val_set])

    features_ref = ["V4", "V10", "V11", "V12", "V14", "V16"]
    X_train_500 = X_train_full.loc[t_idx_ref, features_ref].copy()
    y_train_500 = y_train_full.loc[t_idx_ref].to_numpy()
    X_val_20k = X_train_full.loc[v_idx_ref, features_ref].copy()
    y_val_20k = y_train_full.loc[v_idx_ref].to_numpy()

    # Load pre-saved reference training Gram matrix
    km_ref = np.load(PROJECT_ROOT / "models" / "quantum" / "kernel_matrices.npz")
    K_train_ref = km_ref["K_train"]

    # ==============================================================================
    # EXPERIMENT 1: DEEP QUANTUM KERNEL DIAGNOSTICS
    # ==============================================================================
    logger.info("\n--- STEP 1: Deep Quantum Kernel Diagnostics ---")
    diag_ref = diagnose_kernel_properties(K_train_ref, y_train_500)
    logger.info("Reference QSVM (ZZ reps=2) Diagnostics:")
    logger.info("  Condition Number: %.2f | Eff Rank: %.2f | Top-1 Energy: %.2f%%",
                diag_ref["numerical_validity"]["condition_number"],
                diag_ref["spectral_diagnostics"]["effective_rank"],
                diag_ref["spectral_diagnostics"]["top1_eigenvalue_energy_ratio"] * 100)
    logger.info("  Fraud-Fraud Similarity: %.4f | Fraud-Legit Similarity: %.4f | Ratio: %.2fx",
                diag_ref["class_similarities"]["fraud_fraud_mean"],
                diag_ref["class_similarities"]["fraud_legit_mean"],
                diag_ref["class_similarities"]["ff_to_fl_ratio"])
    logger.info("  Uncentered KTA: %.4f | Centered KTA: %.4f",
                diag_ref["kernel_target_alignment"]["kta_uncentered"],
                diag_ref["kernel_target_alignment"]["kta_centered"])

    # Diagnostics on Shallow ZZ reps=1
    scaler_pi = MinMaxScaler(feature_range=(0.0, np.pi))
    X_tr_s_pi = scaler_pi.fit_transform(X_train_500)
    fmap_zz1 = create_zz_feature_map(6, reps=1, entanglement="linear")
    sv_zz1 = compute_statevector_matrix(fmap_zz1, X_tr_s_pi)
    K_tr_zz1 = np.abs(sv_zz1 @ sv_zz1.conj().T) ** 2
    np.fill_diagonal(K_tr_zz1, 1.0)
    diag_zz1 = diagnose_kernel_properties(K_tr_zz1, y_train_500)

    # Diagnostics on Separable Z reps=1
    K_tr_z1 = compute_analytic_z_kernel(X_tr_s_pi, X_tr_s_pi)
    np.fill_diagonal(K_tr_z1, 1.0)
    diag_z1 = diagnose_kernel_properties(K_tr_z1, y_train_500)

    plot_kernel_similarity_distributions(
        diag_ref=diag_ref,
        diag_zz1=diag_zz1,
        diag_z1=diag_z1,
        output_path=FIGURES_DIR / "kernel_similarity_distributions.png",
    )

    # ==============================================================================
    # EXPERIMENT 2: FEATURE SELECTION COMPARISON (TRAINING POOL ONLY)
    # ==============================================================================
    logger.info("\n--- STEP 2: Feature Selection Evaluation on Training Pool ---")
    feat_strat_results = compare_feature_selection_strategies(
        X_train_pool=X_train_full.loc[non_val_indices],
        y_train_pool=y_train_full.loc[non_val_indices],
        X_val_full=X_train_full.loc[v_idx_ref],
        y_val=y_val_20k,
        k=6,
        subsample_size=2000,
        random_state=42,
    )
    for strat_k, strat_v in feat_strat_results.items():
        logger.info("  %s (%s): Linear PR-AUC=%.4f, RBF PR-AUC=%.4f",
                    strat_v["name"], strat_v["features"],
                    strat_v["linear_svm"]["pr_auc"], strat_v["rbf_svm"]["pr_auc"])

    plot_feature_selection_comparison(
        feat_strat_results,
        FIGURES_DIR / "feature_selection_comparison.png",
    )

    # ==============================================================================
    # EXPERIMENT 3: FEATURE MAP ARCHITECTURAL VARIANTS (VALIDATION DATA)
    # ==============================================================================
    logger.info("\n--- STEP 3: Quantum Feature Map Architectural Variants ---")
    fmap_arch_results = evaluate_feature_map_architectures(
        X_train_df=X_train_500,
        y_train=y_train_500,
        X_val_df=X_val_20k,
        y_val=y_val_20k,
    )
    for arch_k, arch_v in fmap_arch_results.items():
        m = arch_v["metrics_at_default_cutoff"]
        d = arch_v["diagnostics"]
        logger.info("  %s: PR-AUC=%.4f, ROC-AUC=%.4f, Prec=%.2f%%, Rec=%.2f%%, FP=%d, KTA=%.4f, EffRank=%.2f",
                    arch_v["name"], m["pr_auc"], m["roc_auc"], m["precision"]*100, m["recall"]*100,
                    m["false_positives"], d["kernel_target_alignment"]["kta_uncentered"],
                    d["spectral_diagnostics"]["effective_rank"])

    # ==============================================================================
    # EXPERIMENT 4: OPTION A SCALABILITY (FIXED 10% FRAUD PREVALENCE)
    # ==============================================================================
    logger.info("\n--- STEP 4: Option A Scalability (10%% Fraud, N=500, 1000, 2000, 3 Seeds) ---")
    option_a_results = run_option_a_scalability_experiment(
        X_train_full=X_train_full,
        y_train_full=y_train_full,
        non_val_indices=non_val_indices,
        val_indices=v_idx_ref,
        features=features_ref,
        sample_sizes=[500, 1000, 2000],
        seeds=[42, 101, 2024],
    )
    opt_a_sum = option_a_results["summary_statistics"]
    for N_s in ["500", "1000", "2000"]:
        logger.info("  N=%s (Fraud=%d):", N_s, opt_a_sum[N_s]["n_fraud"])
        for m_name in ["linear_svm", "rbf_svm", "qsvm_zz_r2_ref", "qsvm_zz_r1_imp", "qsvm_z_r1_sep"]:
            logger.info("    %-18s: PR-AUC=%.4f ± %.4f, FP=%.1f",
                        m_name, opt_a_sum[N_s][m_name]["pr_auc_mean"],
                        opt_a_sum[N_s][m_name]["pr_auc_std"],
                        opt_a_sum[N_s][m_name]["fp_mean"])

    plot_option_a_scalability(
        opt_a_sum,
        FIGURES_DIR / "option_a_scalability_curves.png",
    )

    # ==============================================================================
    # EXPERIMENT 5: THRESHOLD CALIBRATION & FALSE-POSITIVE MITIGATION
    # ==============================================================================
    logger.info("\n--- STEP 5: Cost-Optimal Decision Threshold Calibration ---")
    # Precompute decision scores for threshold analysis
    scaler_val = MinMaxScaler(feature_range=(0.0, np.pi))
    X_tr_s500 = scaler_val.fit_transform(X_train_500)
    X_va_s20k = scaler_val.transform(X_val_20k)

    # Classical Linear
    lin_svm = LinearSVC(C=5.0, class_weight="balanced", dual=False, random_state=42)
    lin_svm.fit(X_tr_s500, y_train_500)
    scores_lin = lin_svm.decision_function(X_va_s20k)

    # Classical RBF
    rbf_svm = SVC(C=5.0, gamma=0.01, class_weight="balanced", random_state=42)
    rbf_svm.fit(X_tr_s500, y_train_500)
    scores_rbf = rbf_svm.decision_function(X_va_s20k)

    # Reference QSVM (ZZ reps=2)
    K_val_ref = km_ref["K_val"]
    qsvm_ref = SVC(C=0.5, kernel="precomputed", class_weight="balanced", random_state=42)
    qsvm_ref.fit(K_train_ref, y_train_500)
    scores_qsvm_ref = qsvm_ref.decision_function(K_val_ref)

    # Improved QSVM (ZZ reps=1)
    sv_va_zz1 = compute_statevector_matrix(fmap_zz1, X_va_s20k)
    K_val_zz1 = np.abs(sv_va_zz1 @ sv_zz1.conj().T) ** 2
    qsvm_zz1 = SVC(C=0.5, kernel="precomputed", class_weight="balanced", random_state=42)
    qsvm_zz1.fit(K_tr_zz1, y_train_500)
    scores_qsvm_zz1 = qsvm_zz1.decision_function(K_val_zz1)

    # Separable QSVM (Z reps=1)
    K_val_z1 = compute_analytic_z_kernel(X_va_s20k, X_tr_s_pi)
    qsvm_z1 = SVC(C=0.5, kernel="precomputed", class_weight="balanced", random_state=42)
    qsvm_z1.fit(K_tr_z1, y_train_500)
    scores_qsvm_z1 = qsvm_z1.decision_function(K_val_z1)

    model_scores_dict = {
        "linear_svm": scores_lin,
        "rbf_svm": scores_rbf,
        "reference_qsvm_zz2": scores_qsvm_ref,
        "improved_qsvm_zz1": scores_qsvm_zz1,
        "separable_qsvm_z1": scores_qsvm_z1,
    }

    threshold_calib_results = evaluate_threshold_calibrations(
        y_true=y_val_20k,
        models_scores=model_scores_dict,
        cost_ratios=[5.0, 20.0, 50.0, 100.0],
        c_fp=10.0,
    )

    plot_threshold_cost_curves(
        threshold_calib_results,
        FIGURES_DIR / "threshold_tradeoffs_and_cost.png",
    )

    # ==============================================================================
    # EXPERIMENT 6: PAIRED BOOTSTRAP SIGNIFICANCE (1,000 REPLICATES)
    # ==============================================================================
    logger.info("\n--- STEP 6: Paired Non-Parametric Bootstrap (1,000 Replicates) ---")
    predictions_dict = {
        m: ((scores >= 0.0).astype(int), scores)
        for m, scores in model_scores_dict.items()
    }
    bootstrap_results = compute_paired_bootstrap_significance(
        y_true=y_val_20k,
        model_predictions=predictions_dict,
        reference_key="reference_qsvm_zz2",
        n_bootstraps=1000,
        random_state=42,
        alpha=0.05,
    )
    for pair_k, pair_v in bootstrap_results["paired_differences"].items():
        logger.info("  %s:", pair_k)
        for met in ["delta_pr_auc", "delta_roc_auc", "delta_precision", "delta_recall"]:
            sub = pair_v[met]
            sig_flag = "STATISTICALLY SIGNIFICANT" if sub["statistically_significant"] else "NOT SIGNIFICANT"
            logger.info("    %-16s: Mean = %+.4f, 95%% CI = [%+.4f, %+.4f] (%s)",
                        met, sub["mean"], sub["ci_95"][0], sub["ci_95"][1], sig_flag)

    # ==============================================================================
    # EXPERIMENT 7: FINAL HOLDOUT TEST SET BENCHMARK
    # ==============================================================================
    logger.info("\n--- STEP 7: Holdout Test Set Evaluation (56,746 samples) ---")
    X_test_full = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "X_test.csv")
    y_test_full = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "y_test.csv").squeeze("columns").to_numpy()
    X_test_sub = X_test_full[features_ref].copy()

    X_test_s_pi = scaler_pi.transform(X_test_sub)

    # 1. Historical Phase 6 QSVM test metrics from frozen evaluation
    phase6_metrics_path = PROJECT_ROOT / "reports" / "phase6_test_metrics.json"
    with open(phase6_metrics_path) as f:
        phase6_json = json.load(f)
    p6_res = phase6_json["metrics_results"]
    p6_qsvm = p6_res["qsvm_6f"]
    p6_rbf = p6_res["rbf_svm_6f"]
    p6_lin = p6_res["linear_svm_6f"]

    # 2. Evaluate Improved Shallow QSVM (ZZ reps=1) on Holdout Test Set
    logger.info("Evaluating Improved Shallow QSVM (ZZ reps=1) on holdout test set...")
    t0_sv_test = time.perf_counter()
    sv_test_zz1 = compute_statevector_matrix(fmap_zz1, X_test_s_pi)
    t_sv_test = time.perf_counter() - t0_sv_test

    t0_gram_test = time.perf_counter()
    K_test_zz1 = np.abs(sv_test_zz1 @ sv_zz1.conj().T) ** 2
    t_gram_test = time.perf_counter() - t0_gram_test

    t0_test_score = time.perf_counter()
    scores_test_zz1 = qsvm_zz1.decision_function(K_test_zz1)
    t_test_score = time.perf_counter() - t0_test_score

    preds_test_zz1_def = (scores_test_zz1 >= 0.0).astype(int)
    tn_t, fp_t, fn_t, tp_t = confusion_matrix(y_test_full, preds_test_zz1_def, labels=[0, 1]).ravel()
    test_pr_auc_zz1 = float(average_precision_score(y_test_full, scores_test_zz1))
    test_roc_auc_zz1 = float(roc_auc_score(y_test_full, scores_test_zz1))
    test_prec_zz1 = float(tp_t / (tp_t + fp_t))
    test_rec_zz1 = float(tp_t / (tp_t + fn_t))
    test_f1_zz1 = float(f1_score(y_test_full, preds_test_zz1_def, pos_label=1))

    # Test under frozen validation cost-optimal threshold (R=20)
    opt_tau_r20 = threshold_calib_results["improved_qsvm_zz1"]["cost_optimal_cutoffs"]["20"]["optimal_threshold"]
    preds_test_zz1_opt = (scores_test_zz1 >= opt_tau_r20).astype(int)
    tn_to, fp_to, fn_to, tp_to = confusion_matrix(y_test_full, preds_test_zz1_opt, labels=[0, 1]).ravel()
    test_prec_opt = float(tp_to / (tp_to + fp_to)) if (tp_to + fp_to) > 0 else 0.0
    test_rec_opt = float(tp_to / (tp_to + fn_to)) if (tp_to + fn_to) > 0 else 0.0
    test_f1_opt = float(f1_score(y_test_full, preds_test_zz1_opt, pos_label=1, zero_division=0))

    # 3. Evaluate Separable Z-QSVM on Holdout Test Set (Analytic Cosine Product)
    t0_z_test = time.perf_counter()
    K_test_z1 = compute_analytic_z_kernel(X_test_s_pi, X_tr_s_pi)
    scores_test_z1 = qsvm_z1.decision_function(K_test_z1)
    t_z_test = time.perf_counter() - t0_z_test

    preds_test_z1_def = (scores_test_z1 >= 0.0).astype(int)
    tn_tz, fp_tz, fn_tz, tp_tz = confusion_matrix(y_test_full, preds_test_z1_def, labels=[0, 1]).ravel()
    test_pr_auc_z1 = float(average_precision_score(y_test_full, scores_test_z1))
    test_roc_auc_z1 = float(roc_auc_score(y_test_full, scores_test_z1))
    test_prec_z1 = float(tp_tz / (tp_tz + fp_tz))
    test_rec_z1 = float(tp_tz / (tp_tz + fn_tz))
    test_f1_z1 = float(f1_score(y_test_full, preds_test_z1_def, pos_label=1))

    test_comparison = {
        "historical_classical_linear_svm": {
            "pr_auc": p6_lin["pr_auc"],
            "roc_auc": p6_lin["roc_auc"],
            "precision": p6_lin["default_cutoff_evaluation"]["precision"],
            "recall": p6_lin["default_cutoff_evaluation"]["recall"],
            "false_positives": p6_lin["default_cutoff_evaluation"]["confusion_matrix"]["false_positives"],
            "false_negatives": p6_lin["default_cutoff_evaluation"]["confusion_matrix"]["false_negatives"],
            "f1_score": p6_lin["default_cutoff_evaluation"]["f1_score"],
        },
        "historical_classical_rbf_svm": {
            "pr_auc": p6_rbf["pr_auc"],
            "roc_auc": p6_rbf["roc_auc"],
            "precision": p6_rbf["default_cutoff_evaluation"]["precision"],
            "recall": p6_rbf["default_cutoff_evaluation"]["recall"],
            "false_positives": p6_rbf["default_cutoff_evaluation"]["confusion_matrix"]["false_positives"],
            "false_negatives": p6_rbf["default_cutoff_evaluation"]["confusion_matrix"]["false_negatives"],
            "f1_score": p6_rbf["default_cutoff_evaluation"]["f1_score"],
        },
        "historical_qsvm_canonical_zz2": {
            "pr_auc": p6_qsvm["pr_auc"],
            "roc_auc": p6_qsvm["roc_auc"],
            "precision": p6_qsvm["default_cutoff_evaluation"]["precision"],
            "recall": p6_qsvm["default_cutoff_evaluation"]["recall"],
            "false_positives": p6_qsvm["default_cutoff_evaluation"]["confusion_matrix"]["false_positives"],
            "false_negatives": p6_qsvm["default_cutoff_evaluation"]["confusion_matrix"]["false_negatives"],
            "f1_score": p6_qsvm["default_cutoff_evaluation"]["f1_score"],
        },
        "improved_qsvm_shallow_zz1_default": {
            "pr_auc": round(test_pr_auc_zz1, 4),
            "roc_auc": round(test_roc_auc_zz1, 4),
            "precision": round(test_prec_zz1, 4),
            "recall": round(test_rec_zz1, 4),
            "false_positives": int(fp_t),
            "false_negatives": int(fn_t),
            "f1_score": round(test_f1_zz1, 4),
            "total_inference_latency_ms": round((t_sv_test + t_gram_test + t_test_score) * 1000 / len(y_test_full), 4),
        },
        "improved_qsvm_shallow_zz1_cost_optimal_r20": {
            "calibrated_threshold": round(opt_tau_r20, 4),
            "pr_auc": round(test_pr_auc_zz1, 4),
            "roc_auc": round(test_roc_auc_zz1, 4),
            "precision": round(test_prec_opt, 4),
            "recall": round(test_rec_opt, 4),
            "false_positives": int(fp_to),
            "false_negatives": int(fn_to),
            "f1_score": round(test_f1_opt, 4),
        },
        "separable_qsvm_z1_default": {
            "pr_auc": round(test_pr_auc_z1, 4),
            "roc_auc": round(test_roc_auc_z1, 4),
            "precision": round(test_prec_z1, 4),
            "recall": round(test_rec_z1, 4),
            "false_positives": int(fp_tz),
            "false_negatives": int(fn_tz),
            "f1_score": round(test_f1_z1, 4),
            "total_inference_latency_ms": round(t_z_test * 1000 / len(y_test_full), 4),
        },
    }

    logger.info("Holdout Test Set Comparison:")
    logger.info("  Canonical QSVM (Phase 6): PR-AUC = %.4f, FP = %d, Prec = %.2f%%",
                p6_qsvm["pr_auc"],
                p6_qsvm["default_cutoff_evaluation"]["confusion_matrix"]["false_positives"],
                p6_qsvm["default_cutoff_evaluation"]["precision"] * 100)
    logger.info("  Shallow QSVM (ZZ reps=1): PR-AUC = %.4f, FP = %d, Prec = %.2f%%",
                test_pr_auc_zz1, fp_t, test_prec_zz1 * 100)
    logger.info("  Shallow QSVM (τ*=%.3f):   FP = %d, Prec = %.2f%%, Recall = %.2f%%",
                opt_tau_r20, fp_to, test_prec_opt * 100, test_rec_opt * 100)
    logger.info("  Separable QSVM (Z reps=1): PR-AUC = %.4f, FP = %d, Prec = %.2f%%",
                test_pr_auc_z1, fp_tz, test_prec_z1 * 100)

    # Save test predictions CSV
    df_preds = pd.DataFrame({
        "y_true": y_test_full,
        "score_shallow_zz1": scores_test_zz1,
        "pred_shallow_zz1_def": preds_test_zz1_def,
        "pred_shallow_zz1_opt_r20": preds_test_zz1_opt,
        "score_separable_z1": scores_test_z1,
        "pred_separable_z1_def": preds_test_z1_def,
    })
    preds_out_path = PROJECT_ROOT / "reports" / "phase8_test_predictions.csv"
    df_preds.to_csv(preds_out_path, index=False)
    logger.info("Saved Phase 8 test predictions to %s", preds_out_path)

    # ==============================================================================
    # SERIALIZATION: MASTER METRICS JSON
    # ==============================================================================
    total_elapsed = time.perf_counter() - start_total_time
    logger.info("\n--- STEP 8: Serializing Master Phase 8 Metrics ---")

    phase8_master_metrics = {
        "metadata": {
            "experiment": "Phase 8 — Quantum Kernel Improvements, Diagnostics, Scalability & Statistical Verification",
            "execution_timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "total_execution_seconds": round(total_elapsed, 2),
            "python_version": sys.version,
            "data_partitions": {
                "training_pool_available": len(non_val_indices),
                "validation_partition": len(v_idx_ref),
                "holdout_test_set": len(y_test_full),
            },
        },
        "kernel_diagnostics": {
            "canonical_reference_zz2": diag_ref,
            "shallow_improved_zz1": diag_zz1,
            "separable_product_z1": diag_z1,
        },
        "feature_selection_comparison": feat_strat_results,
        "feature_map_architectures_validation": fmap_arch_results,
        "option_a_scalability": option_a_results,
        "threshold_calibration_and_mitigation": threshold_calib_results,
        "paired_bootstrap_significance": bootstrap_results,
        "holdout_test_set_benchmark": test_comparison,
    }

    metrics_json_path = PROJECT_ROOT / "reports" / "phase8_metrics.json"
    with open(metrics_json_path, "w") as f:
        json.dump(phase8_master_metrics, f, indent=2)
    logger.info("Master metrics successfully serialized to %s", metrics_json_path)

    logger.info("================================================================================")
    logger.info("PHASE 8 COMPLETE in %.2f seconds (%.2f minutes)", total_elapsed, total_elapsed / 60.0)
    logger.info("================================================================================")


if __name__ == "__main__":
    main()
