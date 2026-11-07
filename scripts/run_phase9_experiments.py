#!/usr/bin/env python3
"""
Phase 9: Comprehensive Quantum Kernel Optimization Pipeline Script.

Executes:
1. Mathematical analysis and geometric diagnosis of Separable vs Canonical QSVM.
2. Bandwidth optimization grid search via Stratified 5-Fold Cross-Validation on training data.
3. Anisotropic (feature-weighted) separable quantum kernel evaluation.
4. Continuous entanglement strength sweep (alpha in [0.0, 1.0]).
5. Projected Quantum Kernel (PQK) tuning and evaluation (Huang et al., 2021).
6. Validation-set comparison and cost-optimal threshold calibration (R in {5, 20, 50, 100}).
7. Final holdout test set evaluation (56,746 samples, 95 frauds) and 1,000-replicate paired bootstrap.
8. Latency profiling and master JSON serialization.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC, SVC

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.quantum_subsets import get_quantum_partition_indices
from src.experiments.phase8_improvements import (
    compute_paired_bootstrap_significance,
    diagnose_kernel_properties,
    evaluate_threshold_calibrations,
)
from src.experiments.phase9_optimization import (
    analyze_separable_vs_canonical_geometry,
    compute_anisotropic_separable_kernel,
    compute_bandwidth_separable_kernel,
    compute_controlled_entanglement_kernel,
    compute_controlled_entanglement_statevectors,
    compute_projected_quantum_kernel,
    compute_1qubit_bloch_representations,
    cross_validate_projected_quantum_kernel,
    cross_validate_separable_kernel_bandwidth,
    sweep_entanglement_coupling,
)

# Logging configuration
LOG_FILE = PROJECT_ROOT / "reports" / "phase9_run.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] phase9_runner: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, mode="w"),
    ],
)
logger = logging.getLogger("phase9_runner")

FIGURES_DIR = PROJECT_ROOT / "reports" / "figures" / "phase9"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# PLOTTING FUNCTIONS
# ==============================================================================

def plot_separable_vs_canonical_geometry(analysis: dict, output_path: Path) -> None:
    """Plots spectral and class similarity contrasts between Separable and Canonical kernels."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    geom = analysis["geometric_metrics"]
    models = ["Separable Z-Kernel\n(reps=1, product)", "Canonical ZZ-Kernel\n(reps=2, entangled)"]
    ff_sims = [geom["separable_z"]["fraud_fraud_similarity"], geom["canonical_zz"]["fraud_fraud_similarity"]]
    fl_sims = [geom["separable_z"]["fraud_legit_similarity"], geom["canonical_zz"]["fraud_legit_similarity"]]

    x = np.arange(len(models))
    width = 0.35

    ax1.bar(x - width/2, ff_sims, width, label="Fraud–Fraud (Within-Class)", color="#5cb85c")
    ax1.bar(x + width/2, fl_sims, width, label="Fraud–Legit (Cross-Class)", color="#d9534f")
    ax1.set_ylabel("Mean Pairwise Kernel Similarity", fontsize=12)
    ax1.set_title("Geometric Separation & Class Overlap", fontsize=13, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=11)
    ax1.legend(fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Highlight FF/FL ratio
    for i, (ff, fl) in enumerate(zip(ff_sims, fl_sims)):
        ratio = ff / max(1e-6, fl)
        ax1.annotate(
            f"Ratio FF/FL = {ratio:.2f}x\n({ 'Cluster Preserved' if ratio >= 0.95 else 'Inverted Overlap' })",
            xy=(x[i], max(ff, fl) + 0.02),
            ha="center",
            fontsize=10,
            fontweight="bold",
            color="darkgreen" if ratio >= 0.95 else "darkred",
        )

    # Effective rank and SV saturation
    eff_ranks = [geom["separable_z"]["effective_rank"], geom["canonical_zz"]["effective_rank"]]
    sv_fracs = [geom["separable_z"]["support_vector_fraction"] * 100, geom["canonical_zz"]["support_vector_fraction"] * 100]

    ax2_twin = ax2.twinx()
    ax2.bar(x - width/2, eff_ranks, width, label="Effective Rank (PR)", color="#337ab7")
    ax2_twin.bar(x + width/2, sv_fracs, width, label="Support Vector Fraction (%)", color="#f0ad4e")

    ax2.set_ylabel("Spectral Effective Rank (out of 500)", fontsize=12, color="#337ab7")
    ax2_twin.set_ylabel("Dual SV Saturation (%)", fontsize=12, color="#f0ad4e")
    ax2.set_title("Spectral Dispersion & Dual SV Saturation", fontsize=13, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info("Saved geometry comparison plot to %s", output_path)


def plot_bandwidth_optimization_heatmap(cv_results: dict, output_path: Path) -> None:
    """Plots CV PR-AUC heatmap across bandwidth beta and regularization C."""
    all_configs = cv_results["all_configurations"]
    betas = sorted(list(set(c["beta"] for c in all_configs.values())))
    c_vals = sorted(list(set(c["C"] for c in all_configs.values())))

    matrix = np.zeros((len(betas), len(c_vals)))
    for i, b in enumerate(betas):
        for j, c in enumerate(c_vals):
            key = f"beta_{b:.2f}_C_{c:.1f}"
            matrix[i, j] = all_configs[key]["cv_pr_auc_mean"]

    fig, ax = plt.subplots(figsize=(9, 7))
    cax = ax.imshow(matrix, cmap="viridis", aspect="auto", origin="lower")
    fig.colorbar(cax, label="Stratified 5-Fold Mean PR-AUC")

    ax.set_xticks(range(len(c_vals)))
    ax.set_xticklabels([f"C={c}" for c in c_vals], fontsize=11)
    ax.set_yticks(range(len(betas)))
    ax.set_yticklabels([f"β={b:.2f}" for b in betas], fontsize=11)
    ax.set_xlabel("SVM Regularization Parameter (C)", fontsize=12)
    ax.set_ylabel("Quantum Bandwidth Scale (β)", fontsize=12)
    ax.set_title("5-Fold Cross-Validation PR-AUC Grid Search\nSeparable Quantum Kernel", fontsize=13, fontweight="bold")

    # Annotate values
    for i in range(len(betas)):
        for j in range(len(c_vals)):
            val = matrix[i, j]
            ax.text(j, i, f"{val:.4f}", ha="center", va="center",
                    color="white" if val < np.mean(matrix) else "black", fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info("Saved bandwidth optimization heatmap to %s", output_path)


def plot_entanglement_coupling_sweep(sweep_results: dict, output_path: Path) -> None:
    """Plots validation PR-AUC, false alarms, and condition number vs coupling alpha."""
    alphas = [res["alpha"] for res in sweep_results.values()]
    pr_aucs = [res["pr_auc"] for res in sweep_results.values()]
    fps = [res["false_positives"] for res in sweep_results.values()]
    ktas = [res["kta"] for res in sweep_results.values()]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: PR-AUC and KTA vs Alpha
    ax1.plot(alphas, pr_aucs, "o-", color="#337ab7", linewidth=2.5, label="Validation PR-AUC")
    ax1_twin = ax1.twinx()
    ax1_twin.plot(alphas, ktas, "s--", color="#5cb85c", linewidth=2.5, label="Kernel-Target Alignment (KTA)")

    ax1.set_xlabel("Entanglement Coupling Strength (α)", fontsize=12)
    ax1.set_ylabel("Validation PR-AUC", fontsize=12, color="#337ab7")
    ax1_twin.set_ylabel("Kernel-Target Alignment", fontsize=12, color="#5cb85c")
    ax1.set_title("Impact of Quantum Entanglement on Ranking Performance", fontsize=13, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Right: False Positives vs Alpha
    ax2.plot(alphas, fps, "^-", color="#d9534f", linewidth=2.5, label="Validation False Positives (τ=0)")
    ax2.set_xlabel("Entanglement Coupling Strength (α)", fontsize=12)
    ax2.set_ylabel("False Positives on Validation Set", fontsize=12, color="#d9534f")
    ax2.set_title("False Alarm Proliferation vs Entanglement Strength", fontsize=13, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info("Saved entanglement sweep plot to %s", output_path)


# ==============================================================================
# MAIN PIPELINE
# ==============================================================================

def main() -> None:
    """Master Phase 9 execution pipeline."""
    logger.info("================================================================================")
    logger.info("STARTING PHASE 9: QUANTUM KERNEL OPTIMIZATION & RIGOROUS MODEL SELECTION")
    logger.info("================================================================================")
    t_start = time.perf_counter()

    # 1. Load Data
    logger.info("Loading training and validation partitions...")
    X_train_full = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "X_train.csv")
    y_train_full = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "y_train.csv").squeeze("columns")

    t_idx_ref, v_idx_ref = get_quantum_partition_indices(
        y_train_full, n_train=500, n_train_fraud=100, n_val=20000, random_state=42
    )
    features_6f = ["V4", "V10", "V11", "V12", "V14", "V16"]

    X_train_500 = X_train_full.loc[t_idx_ref, features_6f].copy()
    y_train_500 = y_train_full.loc[t_idx_ref].to_numpy()
    X_val_20k = X_train_full.loc[v_idx_ref, features_6f].copy()
    y_val_20k = y_train_full.loc[v_idx_ref].to_numpy()

    # Standard [0, pi] angle scaling
    scaler_pi = MinMaxScaler(feature_range=(0.0, np.pi))
    X_tr_s = scaler_pi.fit_transform(X_train_500)
    X_va_s = scaler_pi.transform(X_val_20k)

    # ==============================================================================
    # STEP 1: ANALYTICAL & GEOMETRIC DIAGNOSIS: SEPARABLE VS CANONICAL
    # ==============================================================================
    logger.info("\n--- STEP 1: Analytical & Geometric Contrast (Separable vs Canonical) ---")
    geom_analysis = analyze_separable_vs_canonical_geometry(X_tr_s, y_train_500)
    logger.info("Separable Z-kernel condition number: %.2f | Eff Rank: %.2f | FF/FL: %.2fx | KTA: %.4f | SVs: %d",
                geom_analysis["geometric_metrics"]["separable_z"]["condition_number"],
                geom_analysis["geometric_metrics"]["separable_z"]["effective_rank"],
                geom_analysis["geometric_metrics"]["separable_z"]["ff_to_fl_ratio"],
                geom_analysis["geometric_metrics"]["separable_z"]["kta_uncentered"],
                geom_analysis["geometric_metrics"]["separable_z"]["support_vector_count"])
    logger.info("Canonical ZZ-kernel condition number: %.2f | Eff Rank: %.2f | FF/FL: %.2fx | KTA: %.4f | SVs: %d",
                geom_analysis["geometric_metrics"]["canonical_zz"]["condition_number"],
                geom_analysis["geometric_metrics"]["canonical_zz"]["effective_rank"],
                geom_analysis["geometric_metrics"]["canonical_zz"]["ff_to_fl_ratio"],
                geom_analysis["geometric_metrics"]["canonical_zz"]["kta_uncentered"],
                geom_analysis["geometric_metrics"]["canonical_zz"]["support_vector_count"])

    plot_separable_vs_canonical_geometry(geom_analysis, FIGURES_DIR / "separable_vs_canonical_geometry.png")

    # ==============================================================================
    # STEP 2: BANDWIDTH OPTIMIZATION VIA STRATIFIED 5-FOLD CROSS-VALIDATION
    # ==============================================================================
    logger.info("\n--- STEP 2: Bandwidth Tuning Grid Search (5-Fold Stratified CV on Training) ---")
    betas_to_test = [0.10, 0.20, 0.35, 0.50, 0.75, 1.00, 1.50]
    c_to_test = [0.1, 0.5, 1.0, 5.0, 10.0]

    cv_separable = cross_validate_separable_kernel_bandwidth(
        X_train=X_tr_s,
        y_train=y_train_500,
        betas=betas_to_test,
        C_values=c_to_test,
        n_splits=5,
        random_state=42,
    )
    best_beta = cv_separable["best_config"]["beta"]
    best_C = cv_separable["best_config"]["C"]
    best_cv_pr = cv_separable["best_config"]["best_cv_pr_auc"]
    logger.info("Optimal Separable Quantum Kernel Configuration: beta=%.2f, C=%.1f (5-Fold Mean PR-AUC = %.4f)",
                best_beta, best_C, best_cv_pr)

    plot_bandwidth_optimization_heatmap(cv_separable, FIGURES_DIR / "bandwidth_optimization_heatmap.png")

    # ==============================================================================
    # STEP 3: ANISOTROPIC (FEATURE-WEIGHTED) QUANTUM KERNEL
    # ==============================================================================
    logger.info("\n--- STEP 3: Anisotropic (Feature-Weighted) Separable Quantum Kernel ---")
    f_scores, _ = f_classif(X_tr_s, y_train_500)
    norm_f_weights = best_beta * (f_scores / np.mean(f_scores))
    logger.info("Derived feature weights w_j: %s", {col: round(w, 4) for col, w in zip(features_6f, norm_f_weights)})

    K_tr_aniso = compute_anisotropic_separable_kernel(X_tr_s, X_tr_s, weights=norm_f_weights)
    np.fill_diagonal(K_tr_aniso, 1.0)
    K_va_aniso = compute_anisotropic_separable_kernel(X_va_s, X_tr_s, weights=norm_f_weights)

    clf_aniso = SVC(kernel="precomputed", C=best_C, class_weight="balanced", random_state=42)
    clf_aniso.fit(K_tr_aniso, y_train_500)
    s_aniso = clf_aniso.decision_function(K_va_aniso)
    p_aniso = (s_aniso >= 0.0).astype(int)
    pr_aniso = float(average_precision_score(y_val_20k, s_aniso))
    roc_aniso = float(roc_auc_score(y_val_20k, s_aniso))
    prec_aniso = float(precision_score(y_val_20k, p_aniso, zero_division=0))
    rec_aniso = float(recall_score(y_val_20k, p_aniso, zero_division=0))
    _, fp_aniso, fn_aniso, tp_aniso = confusion_matrix(y_val_20k, p_aniso, labels=[0, 1]).ravel()
    logger.info("Anisotropic QSVM Validation: PR-AUC=%.4f, ROC-AUC=%.4f, Prec=%.2f%%, Rec=%.2f%%, FP=%d",
                pr_aniso, roc_aniso, prec_aniso * 100, rec_aniso * 100, fp_aniso)

    # ==============================================================================
    # STEP 4: CONTINUOUS ENTANGLEMENT COUPLING SWEEP (alpha in [0, 1])
    # ==============================================================================
    logger.info("\n--- STEP 4: Continuous Entanglement Sweep (alpha in [0.0, 1.0]) ---")
    alphas_to_sweep = [0.0, 0.05, 0.10, 0.25, 0.50, 0.75, 1.00]
    entanglement_sweep = sweep_entanglement_coupling(
        X_train=X_tr_s,
        y_train=y_train_500,
        X_val=X_va_s,
        y_val=y_val_20k,
        alphas=alphas_to_sweep,
        C=0.5,
    )
    for a_key, a_res in entanglement_sweep.items():
        logger.info("  alpha=%.2f: PR-AUC=%.4f, ROC-AUC=%.4f, Prec=%.2f%%, FP=%d, KTA=%.4f, Cond=%.1f",
                    a_res["alpha"], a_res["pr_auc"], a_res["roc_auc"], a_res["precision"]*100,
                    a_res["false_positives"], a_res["kta"], a_res["condition_number"])

    plot_entanglement_coupling_sweep(entanglement_sweep, FIGURES_DIR / "entanglement_coupling_phase_transition.png")

    # ==============================================================================
    # STEP 5: PROJECTED QUANTUM KERNELS (PQK)
    # ==============================================================================
    logger.info("\n--- STEP 5: Projected Quantum Kernel (PQK) Optimization ---")
    gammas_to_test = [0.05, 0.10, 0.25, 0.50, 1.00, 2.00]
    c_pqk_test = [0.5, 1.0, 5.0]

    cv_pqk = cross_validate_projected_quantum_kernel(
        X_train=X_tr_s,
        y_train=y_train_500,
        gammas=gammas_to_test,
        C_values=c_pqk_test,
        alpha=0.0,  # Unentangled Bloch projections
        n_splits=5,
        random_state=42,
    )
    best_gamma_pqk = cv_pqk["best_config"]["gamma"]
    best_C_pqk = cv_pqk["best_config"]["C"]
    best_cv_pr_pqk = cv_pqk["best_config"]["best_cv_pr_auc"]
    logger.info("Optimal PQK Configuration: gamma=%.2f, C=%.1f (5-Fold Mean PR-AUC = %.4f)",
                best_gamma_pqk, best_C_pqk, best_cv_pr_pqk)

    # Evaluate optimal PQK on validation set
    sv_tr_unent = compute_controlled_entanglement_statevectors(X_tr_s, alpha=0.0, reps=1)
    sv_va_unent = compute_controlled_entanglement_statevectors(X_va_s, alpha=0.0, reps=1)
    bloch_tr = compute_1qubit_bloch_representations(sv_tr_unent, n_qubits=6)
    bloch_va = compute_1qubit_bloch_representations(sv_va_unent, n_qubits=6)

    K_tr_pqk = compute_projected_quantum_kernel(bloch_tr, bloch_tr, gamma=best_gamma_pqk)
    np.fill_diagonal(K_tr_pqk, 1.0)
    K_va_pqk = compute_projected_quantum_kernel(bloch_va, bloch_tr, gamma=best_gamma_pqk)

    clf_pqk = SVC(kernel="precomputed", C=best_C_pqk, class_weight="balanced", random_state=42)
    clf_pqk.fit(K_tr_pqk, y_train_500)
    s_pqk = clf_pqk.decision_function(K_va_pqk)
    p_pqk = (s_pqk >= 0.0).astype(int)
    pr_pqk = float(average_precision_score(y_val_20k, s_pqk))
    roc_pqk = float(roc_auc_score(y_val_20k, s_pqk))
    prec_pqk = float(precision_score(y_val_20k, p_pqk, zero_division=0))
    rec_pqk = float(recall_score(y_val_20k, p_pqk, zero_division=0))
    _, fp_pqk, fn_pqk, tp_pqk = confusion_matrix(y_val_20k, p_pqk, labels=[0, 1]).ravel()
    logger.info("Projected Quantum Kernel Validation: PR-AUC=%.4f, ROC-AUC=%.4f, Prec=%.2f%%, Rec=%.2f%%, FP=%d",
                pr_pqk, roc_pqk, prec_pqk * 100, rec_pqk * 100, fp_pqk)

    # ==============================================================================
    # STEP 6: VALIDATION SET COMPREHENSIVE BENCHMARK & THRESHOLD CALIBRATION
    # ==============================================================================
    logger.info("\n--- STEP 6: Validation Set Benchmark & Threshold Calibration ---")
    # Evaluate Optimal Bandwidth Separable QSVM on validation set
    K_tr_opt_sep = compute_bandwidth_separable_kernel(X_tr_s, X_tr_s, beta=best_beta)
    np.fill_diagonal(K_tr_opt_sep, 1.0)
    K_va_opt_sep = compute_bandwidth_separable_kernel(X_va_s, X_tr_s, beta=best_beta)

    clf_opt_sep = SVC(kernel="precomputed", C=best_C, class_weight="balanced", random_state=42)
    clf_opt_sep.fit(K_tr_opt_sep, y_train_500)
    s_opt_sep = clf_opt_sep.decision_function(K_va_opt_sep)
    p_opt_sep = (s_opt_sep >= 0.0).astype(int)
    pr_opt_sep = float(average_precision_score(y_val_20k, s_opt_sep))
    roc_opt_sep = float(roc_auc_score(y_val_20k, s_opt_sep))
    prec_opt_sep = float(precision_score(y_val_20k, p_opt_sep, zero_division=0))
    rec_opt_sep = float(recall_score(y_val_20k, p_opt_sep, zero_division=0))
    _, fp_opt_sep, fn_opt_sep, tp_opt_sep = confusion_matrix(y_val_20k, p_opt_sep, labels=[0, 1]).ravel()

    # Classical Controls for direct comparison
    lin_svm = LinearSVC(C=5.0, class_weight="balanced", dual=False, random_state=42)
    lin_svm.fit(X_tr_s, y_train_500)
    s_lin = lin_svm.decision_function(X_va_s)

    rbf_svm = SVC(C=5.0, gamma=0.01, class_weight="balanced", random_state=42)
    rbf_svm.fit(X_tr_s, y_train_500)
    s_rbf = rbf_svm.decision_function(X_va_s)

    # Reference QSVM from Phase 8
    km_ref = np.load(PROJECT_ROOT / "models" / "quantum" / "kernel_matrices.npz")
    clf_ref = SVC(C=0.5, kernel="precomputed", class_weight="balanced", random_state=42)
    clf_ref.fit(km_ref["K_train"], y_train_500)
    s_ref = clf_ref.decision_function(km_ref["K_val"])

    validation_scores_dict = {
        "classical_rbf_svm": s_rbf,
        "classical_linear_svm": s_lin,
        "reference_qsvm_canonical": s_ref,
        "opt_bandwidth_separable_qsvm": s_opt_sep,
        "anisotropic_separable_qsvm": s_aniso,
        "projected_quantum_kernel": s_pqk,
    }

    # Calibrate thresholds across R in {5, 20, 50, 100}
    threshold_calib_p9 = evaluate_threshold_calibrations(
        y_true=y_val_20k,
        models_scores=validation_scores_dict,
        cost_ratios=[5.0, 20.0, 50.0, 100.0],
        c_fp=10.0,
    )

    logger.info("Validation Comparison (at default cutoff tau=0):")
    logger.info("  Classical RBF SVM:         PR-AUC=%.4f, Prec=%.2f%%, FP=%d",
                average_precision_score(y_val_20k, s_rbf),
                precision_score(y_val_20k, (s_rbf>=0).astype(int), zero_division=0)*100,
                confusion_matrix(y_val_20k, (s_rbf>=0).astype(int))[0, 1])
    logger.info("  Canonical QSVM (Phase 6):  PR-AUC=%.4f, Prec=%.2f%%, FP=%d",
                average_precision_score(y_val_20k, s_ref),
                precision_score(y_val_20k, (s_ref>=0).astype(int), zero_division=0)*100,
                confusion_matrix(y_val_20k, (s_ref>=0).astype(int))[0, 1])
    logger.info("  Opt Bandwidth QSVM (β=%.2f): PR-AUC=%.4f, Prec=%.2f%%, FP=%d",
                best_beta, pr_opt_sep, prec_opt_sep*100, fp_opt_sep)
    logger.info("  Anisotropic QSVM:          PR-AUC=%.4f, Prec=%.2f%%, FP=%d",
                pr_aniso, prec_aniso*100, fp_aniso)
    logger.info("  Projected Quantum Kernel:  PR-AUC=%.4f, Prec=%.2f%%, FP=%d",
                pr_pqk, prec_pqk*100, fp_pqk)

    # ==============================================================================
    # STEP 7: HOLDOUT TEST SET BENCHMARK & 1,000 BOOTSTRAP SIGNIFICANCE
    # ==============================================================================
    logger.info("\n--- STEP 7: Final Holdout Test Set Evaluation (56,746 samples) ---")
    X_test_full = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "X_test.csv")
    y_test_full = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "y_test.csv").squeeze("columns").to_numpy()
    X_test_6f = X_test_full[features_6f].copy()
    X_te_s = scaler_pi.transform(X_test_6f)

    # 1. Classical RBF SVM Holdout Evaluation
    t0_rbf_inf = time.perf_counter()
    s_rbf_test = rbf_svm.decision_function(X_te_s)
    t_rbf_inf_total = time.perf_counter() - t0_rbf_inf
    lat_rbf_per_sample_ms = (t_rbf_inf_total / len(y_test_full)) * 1000.0

    # 2. Optimal Bandwidth Separable QSVM Holdout Evaluation
    t0_opt_inf = time.perf_counter()
    K_test_opt_sep = compute_bandwidth_separable_kernel(X_te_s, X_tr_s, beta=best_beta)
    s_opt_sep_test = clf_opt_sep.decision_function(K_test_opt_sep)
    t_opt_inf_total = time.perf_counter() - t0_opt_inf
    lat_opt_per_sample_ms = (t_opt_inf_total / len(y_test_full)) * 1000.0

    # 3. Anisotropic Separable QSVM Holdout Evaluation
    t0_ani_inf = time.perf_counter()
    K_test_ani = compute_anisotropic_separable_kernel(X_te_s, X_tr_s, weights=norm_f_weights)
    s_ani_test = clf_aniso.decision_function(K_test_ani)
    t_ani_inf_total = time.perf_counter() - t0_ani_inf
    lat_ani_per_sample_ms = (t_ani_inf_total / len(y_test_full)) * 1000.0

    # 4. Projected Quantum Kernel (PQK) Holdout Evaluation
    t0_pqk_inf = time.perf_counter()
    sv_te_unent = compute_controlled_entanglement_statevectors(X_te_s, alpha=0.0, reps=1)
    bloch_te = compute_1qubit_bloch_representations(sv_te_unent, n_qubits=6)
    K_test_pqk = compute_projected_quantum_kernel(bloch_te, bloch_tr, gamma=best_gamma_pqk)
    s_pqk_test = clf_pqk.decision_function(K_test_pqk)
    t_pqk_inf_total = time.perf_counter() - t0_pqk_inf
    lat_pqk_per_sample_ms = (t_pqk_inf_total / len(y_test_full)) * 1000.0

    # Test Metrics Helper
    def calc_test_metrics(scores: np.ndarray, threshold: float = 0.0) -> dict:
        preds = (scores >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test_full, preds, labels=[0, 1]).ravel()
        return {
            "pr_auc": round(float(average_precision_score(y_test_full, scores)), 4),
            "roc_auc": round(float(roc_auc_score(y_test_full, scores)), 4),
            "precision": round(float(precision_score(y_test_full, preds, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test_full, preds, zero_division=0)), 4),
            "f1_score": round(float(f1_score(y_test_full, preds, zero_division=0)), 4),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
            "true_negatives": int(tn),
        }

    # Frozen threshold from validation R=20
    tau_opt_sep_r20 = threshold_calib_p9["opt_bandwidth_separable_qsvm"]["cost_optimal_cutoffs"]["20"]["optimal_threshold"]
    tau_ani_r20 = threshold_calib_p9["anisotropic_separable_qsvm"]["cost_optimal_cutoffs"]["20"]["optimal_threshold"]
    tau_pqk_r20 = threshold_calib_p9["projected_quantum_kernel"]["cost_optimal_cutoffs"]["20"]["optimal_threshold"]

    holdout_results = {
        "classical_rbf_svm": {
            **calc_test_metrics(s_rbf_test, threshold=0.0),
            "latency_ms_per_sample": round(lat_rbf_per_sample_ms, 5),
        },
        "phase6_canonical_qsvm_zz2": {
            "pr_auc": 0.4966,
            "roc_auc": 0.9600,
            "precision": 0.0469,
            "recall": 0.8421,
            "f1_score": 0.0888,
            "false_positives": 1627,
            "false_negatives": 15,
            "true_positives": 80,
            "true_negatives": 55024,
            "latency_ms_per_sample": 1.6420,
        },
        "phase8_shallow_qsvm_zz1": {
            "pr_auc": 0.5652,
            "roc_auc": 0.9624,
            "precision": 0.0792,
            "recall": 0.8421,
            "f1_score": 0.1448,
            "false_positives": 930,
            "false_negatives": 15,
            "true_positives": 80,
            "true_negatives": 55721,
            "latency_ms_per_sample": 0.8249,
        },
        "phase9_opt_bandwidth_separable_default": {
            **calc_test_metrics(s_opt_sep_test, threshold=0.0),
            "beta": best_beta,
            "C": best_C,
            "latency_ms_per_sample": round(lat_opt_per_sample_ms, 5),
        },
        "phase9_opt_bandwidth_separable_calibrated_r20": {
            **calc_test_metrics(s_opt_sep_test, threshold=tau_opt_sep_r20),
            "calibrated_threshold": round(tau_opt_sep_r20, 4),
            "beta": best_beta,
            "C": best_C,
            "latency_ms_per_sample": round(lat_opt_per_sample_ms, 5),
        },
        "phase9_anisotropic_separable_default": {
            **calc_test_metrics(s_ani_test, threshold=0.0),
            "latency_ms_per_sample": round(lat_ani_per_sample_ms, 5),
        },
        "phase9_anisotropic_separable_calibrated_r20": {
            **calc_test_metrics(s_ani_test, threshold=tau_ani_r20),
            "calibrated_threshold": round(tau_ani_r20, 4),
            "latency_ms_per_sample": round(lat_ani_per_sample_ms, 5),
        },
        "phase9_projected_quantum_kernel_default": {
            **calc_test_metrics(s_pqk_test, threshold=0.0),
            "gamma": best_gamma_pqk,
            "C": best_C_pqk,
            "latency_ms_per_sample": round(lat_pqk_per_sample_ms, 5),
        },
        "phase9_projected_quantum_kernel_calibrated_r20": {
            **calc_test_metrics(s_pqk_test, threshold=tau_pqk_r20),
            "calibrated_threshold": round(tau_pqk_r20, 4),
            "gamma": best_gamma_pqk,
            "C": best_C_pqk,
            "latency_ms_per_sample": round(lat_pqk_per_sample_ms, 5),
        },
    }

    logger.info("Holdout Test Results Summary:")
    for m_name, m_res in holdout_results.items():
        logger.info("  %-45s: PR-AUC=%.4f, Prec=%.2f%%, Rec=%.2f%%, FP=%d, F1=%.4f, Latency=%.4f ms",
                    m_name, m_res["pr_auc"], m_res["precision"]*100, m_res["recall"]*100,
                    m_res["false_positives"], m_res["f1_score"], m_res["latency_ms_per_sample"])

    # 1,000 Paired Bootstrap Significance on Holdout Test Predictions
    logger.info("\nComputing 1,000 paired bootstrap iterations on holdout test set...")
    test_preds_dict = {
        "classical_rbf_svm": ((s_rbf_test >= 0.0).astype(int), s_rbf_test),
        "phase9_opt_bandwidth_separable": ((s_opt_sep_test >= 0.0).astype(int), s_opt_sep_test),
        "phase9_anisotropic_separable": ((s_ani_test >= 0.0).astype(int), s_ani_test),
        "phase9_projected_quantum_kernel": ((s_pqk_test >= 0.0).astype(int), s_pqk_test),
    }

    bootstrap_significance = compute_paired_bootstrap_significance(
        y_true=y_test_full,
        model_predictions=test_preds_dict,
        reference_key="phase9_opt_bandwidth_separable",
        n_bootstraps=1000,
        random_state=42,
    )

    for pair_name, diffs in bootstrap_significance["paired_differences"].items():
        logger.info("Bootstrap Paired Difference: %s", pair_name)
        for met in ["delta_pr_auc", "delta_roc_auc", "delta_precision", "delta_recall"]:
            sub = diffs[met]
            sig_flag = "STATISTICALLY SIGNIFICANT" if sub["statistically_significant"] else "NOT SIGNIFICANT"
            logger.info("  %-16s: Mean = %+.4f, 95%% CI = [%+.4f, %+.4f] (%s)",
                        met, sub["mean"], sub["ci_95"][0], sub["ci_95"][1], sig_flag)

    # Save test predictions CSV
    df_preds_test = pd.DataFrame({
        "y_true": y_test_full,
        "score_rbf": s_rbf_test,
        "score_opt_separable": s_opt_sep_test,
        "score_anisotropic_separable": s_ani_test,
        "score_projected_quantum": s_pqk_test,
    })
    preds_csv_path = PROJECT_ROOT / "reports" / "phase9_test_predictions.csv"
    df_preds_test.to_csv(preds_csv_path, index=False)
    logger.info("Saved Phase 9 test predictions to %s", preds_csv_path)

    # ==============================================================================
    # STEP 8: MASTER SERIALIZATION
    # ==============================================================================
    master_phase9_metrics = {
        "metadata": {
            "phase": "Phase 9 — Quantum Kernel Optimization",
            "date": "October 2026",
            "runtime_seconds": round(time.perf_counter() - t_start, 2),
            "features_selected": features_6f,
            "n_train_quantum": len(y_train_500),
            "n_val_quantum": len(y_val_20k),
            "n_test_holdout": len(y_test_full),
        },
        "separable_vs_canonical_geometry_analysis": geom_analysis,
        "bandwidth_cv_grid_search": cv_separable,
        "anisotropic_separable_results": {
            "weights": {col: round(w, 4) for col, w in zip(features_6f, norm_f_weights)},
            "validation_pr_auc": round(pr_aniso, 4),
            "validation_precision": round(prec_aniso, 4),
            "validation_false_positives": int(fp_aniso),
        },
        "entanglement_coupling_sweep": entanglement_sweep,
        "projected_quantum_kernel_cv": cv_pqk,
        "threshold_calibration_validation": threshold_calib_p9,
        "holdout_test_set_benchmark": holdout_results,
        "bootstrap_hypothesis_testing": bootstrap_significance,
    }

    metrics_json_path = PROJECT_ROOT / "reports" / "phase9_metrics.json"
    with open(metrics_json_path, "w") as f:
        json.dump(master_phase9_metrics, f, indent=2)
    logger.info("Successfully serialized master Phase 9 metrics to %s", metrics_json_path)

    logger.info("================================================================================")
    logger.info("PHASE 9 COMPLETE in %.2f seconds (%.2f minutes)",
                time.perf_counter() - t_start, (time.perf_counter() - t_start) / 60.0)
    logger.info("================================================================================")


if __name__ == "__main__":
    main()
