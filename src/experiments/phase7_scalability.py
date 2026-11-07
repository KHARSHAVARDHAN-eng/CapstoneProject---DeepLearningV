"""
Phase 7: Training-Size Scalability Experiment module.

Implements:
1. Verification of data availability and nested subset partitioning (500, 1000, 2000).
2. Fair model comparison across training sample sizes for Linear SVM, RBF SVM, and QSVM.
3. Preprocessing isolation (scaler fitted on training subset only, transform on validation only).
4. Exact statevector fidelity quantum kernel matrix computation with memory tracking.
5. Multi-metric evaluation on the 20,000-sample validation set (33 fraud cases).
6. Theoretical and computational scaling projections for larger sample sizes.
7. Publication-quality figure generation.
"""

from __future__ import annotations

import json
import logging
import time
import tracemalloc
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from qiskit.circuit.library import ZZFeatureMap
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC, SVC

from src.data.quantum_subsets import get_quantum_partition_indices
from src.models.quantum_kernel import (
    compute_statevector_matrix,
    create_zz_feature_map,
    validate_kernel_matrix,
)

logger = logging.getLogger("quantum_fraud.phase7")

SELECTED_FEATURES = ["V4", "V10", "V11", "V12", "V14", "V16"]


def verify_sample_availability(
    y_train: pd.Series,
    val_indices: np.ndarray,
    requested_sizes: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Verifies whether enough fraud and legitimate instances exist to sample without replacement.

    Checks:
    - Total fraud cases in dataset.
    - Available fraud cases outside the held-out validation set.
    - Feasibility of each requested training size assuming an 80:20 legitimate:fraud ratio.
    """
    if requested_sizes is None:
        requested_sizes = [500, 1000, 2000]

    y_arr = y_train.to_numpy()
    n_total = len(y_arr)
    total_fraud = int(np.sum(y_arr == 1))
    total_legit = int(np.sum(y_arr == 0))

    val_set = set(val_indices)
    non_val_indices = [i for i in range(n_total) if i not in val_set]
    y_non_val = y_arr[non_val_indices]
    avail_fraud = int(np.sum(y_non_val == 1))
    avail_legit = int(np.sum(y_non_val == 0))

    availability_report: Dict[str, Any] = {
        "dataset_total": {
            "total_samples": n_total,
            "total_fraud": total_fraud,
            "total_legit": total_legit,
        },
        "validation_held_out": {
            "total_samples": len(val_indices),
            "fraud_count": int(np.sum(y_arr[val_indices] == 1)),
            "legit_count": int(np.sum(y_arr[val_indices] == 0)),
        },
        "non_validation_training_pool": {
            "total_samples": len(non_val_indices),
            "available_fraud": avail_fraud,
            "available_legit": avail_legit,
        },
        "requested_size_evaluations": {},
    }

    for size in requested_sizes:
        req_fraud = int(round(size * 0.20))
        req_legit = size - req_fraud
        is_feasible_non_val = (avail_fraud >= req_fraud) and (avail_legit >= req_legit)
        is_feasible_global = (total_fraud >= req_fraud) and (total_legit >= req_legit)

        availability_report["requested_size_evaluations"][str(size)] = {
            "total_samples": size,
            "required_fraud": req_fraud,
            "required_legit": req_legit,
            "available_fraud_pool": avail_fraud,
            "available_legit_pool": avail_legit,
            "feasible_without_validation_leakage": is_feasible_non_val,
            "feasible_globally_in_ytrain": is_feasible_global,
            "shortfall_fraud": max(0, req_fraud - avail_fraud),
            "status": "FEASIBLE" if is_feasible_non_val else "INSUFFICIENT_FRAUD_EXAMPLES",
        }

    return availability_report


def generate_nested_training_subsets(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    val_indices: np.ndarray,
    train_500_indices: np.ndarray,
    random_state: int = 42,
) -> Dict[int, Dict[str, Any]]:
    """Generates nested, reproducible subsets for N=500 and N=1,000 preserving an 80:20 ratio.

    Ensures D_500 is strictly contained within D_1000 (D_500 ⊂ D_1000).
    """
    val_set = set(val_indices)
    t500_set = set(train_500_indices)

    # 1. Base N=500 subset
    X_500 = X_train.loc[train_500_indices, SELECTED_FEATURES].copy().reset_index(drop=True)
    y_500 = y_train.loc[train_500_indices].copy().reset_index(drop=True)

    # 2. Additional candidate samples from non-validation, non-t500 pool
    fraud_indices = y_train[y_train == 1].index.to_numpy()
    legit_indices = y_train[y_train == 0].index.to_numpy()

    avail_fraud = [i for i in fraud_indices if i not in val_set and i not in t500_set]
    avail_legit = [i for i in legit_indices if i not in val_set and i not in t500_set]

    # Sample 100 additional fraud and 400 additional legit to expand N=500 -> N=1000
    rng = np.random.RandomState(random_state)
    add_fraud_100 = rng.choice(avail_fraud, size=100, replace=False)
    add_legit_400 = rng.choice(avail_legit, size=400, replace=False)

    train_1000_indices = np.concatenate([train_500_indices, add_fraud_100, add_legit_400])
    rng.shuffle(train_1000_indices)

    X_1000 = X_train.loc[train_1000_indices, SELECTED_FEATURES].copy().reset_index(drop=True)
    y_1000 = y_train.loc[train_1000_indices].copy().reset_index(drop=True)

    # Verify nested property
    assert set(train_500_indices).issubset(set(train_1000_indices)), "Nested property violated!"
    assert len(set(train_1000_indices).intersection(val_set)) == 0, "Validation leakage detected!"

    return {
        500: {
            "indices": train_500_indices.tolist(),
            "X_unscaled": X_500,
            "y": y_500,
            "n_total": 500,
            "n_legit": 400,
            "n_fraud": 100,
        },
        1000: {
            "indices": train_1000_indices.tolist(),
            "X_unscaled": X_1000,
            "y": y_1000,
            "n_total": 1000,
            "n_legit": 800,
            "n_fraud": 200,
        },
    }


def compute_quantum_kernels(
    fmap: ZZFeatureMap,
    X_tr_scaled: np.ndarray,
    X_val_scaled: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Computes exact statevector fidelity kernel Gram matrices K_train and K_val."""
    n_train = len(X_tr_scaled)
    n_val = len(X_val_scaled)

    # 1. Training Gram Matrix
    t0_tr = time.perf_counter()
    sv_train = compute_statevector_matrix(fmap, X_tr_scaled)
    K_train = np.abs(sv_train @ sv_train.conj().T) ** 2
    np.fill_diagonal(K_train, 1.0)
    K_train = 0.5 * (K_train + K_train.T)
    t_train_kernel = time.perf_counter() - t0_tr

    diag_tr = validate_kernel_matrix(K_train, is_symmetric=True)
    if not diag_tr["is_valid"]:
        raise ValueError(f"Training kernel failed validity check for N={n_train}: {diag_tr}")

    # 2. Validation Cross-Kernel
    t0_val = time.perf_counter()
    sv_val = compute_statevector_matrix(fmap, X_val_scaled)
    K_val = np.abs(sv_val @ sv_train.conj().T) ** 2
    t_val_kernel = time.perf_counter() - t0_val

    diag_val = validate_kernel_matrix(K_val, is_symmetric=False)
    if not diag_val["is_valid"]:
        raise ValueError(f"Validation kernel failed validity check for N={n_train}: {diag_val}")

    timings = {
        "train_kernel_seconds": round(t_train_kernel, 4),
        "val_kernel_seconds": round(t_val_kernel, 4),
        "train_throughput_samples_per_sec": round(n_train / max(1e-6, t_train_kernel), 2),
        "val_throughput_samples_per_sec": round(n_val / max(1e-6, t_val_kernel), 2),
    }

    return K_train, K_val, timings


def train_and_evaluate_models_for_size(
    X_tr_unscaled: pd.DataFrame,
    y_tr: pd.Series,
    X_val_unscaled: pd.DataFrame,
    y_val: pd.Series,
    fmap: ZZFeatureMap,
) -> Dict[str, Any]:
    """Trains and evaluates Linear SVM, RBF SVM, and QSVM for a specific training size."""
    tracemalloc.start()
    n_train = len(X_tr_unscaled)
    n_val = len(X_val_unscaled)
    y_val_arr = y_val.to_numpy()
    y_tr_arr = y_tr.to_numpy()

    # Preprocessing: Fit scaler strictly on training subset
    t0_prep = time.perf_counter()
    scaler = MinMaxScaler(feature_range=(0.0, np.pi))
    X_tr_scaled = scaler.fit_transform(X_tr_unscaled)
    X_val_scaled = scaler.transform(X_val_unscaled)
    t_prep = time.perf_counter() - t0_prep

    # Quantum Kernels
    K_train, K_val, kernel_timings = compute_quantum_kernels(fmap, X_tr_scaled, X_val_scaled)

    # 1. Model: Tuned Linear SVM (C=5.0)
    t0_lin_fit = time.perf_counter()
    linear_svm = LinearSVC(C=5.0, class_weight="balanced", dual=False, random_state=42)
    linear_svm.fit(X_tr_scaled, y_tr_arr)
    t_lin_fit = time.perf_counter() - t0_lin_fit

    t0_lin_infer = time.perf_counter()
    scores_lin = linear_svm.decision_function(X_val_scaled)
    t_lin_infer = time.perf_counter() - t0_lin_infer

    # 2. Model: Tuned RBF SVM (C=5.0, gamma=0.01)
    t0_rbf_fit = time.perf_counter()
    rbf_svm = SVC(C=5.0, gamma=0.01, class_weight="balanced", random_state=42)
    rbf_svm.fit(X_tr_scaled, y_tr_arr)
    t_rbf_fit = time.perf_counter() - t0_rbf_fit

    t0_rbf_infer = time.perf_counter()
    scores_rbf = rbf_svm.decision_function(X_val_scaled)
    t_rbf_infer = time.perf_counter() - t0_rbf_infer

    # 3. Model: Tuned QSVM (C=0.5, precomputed kernel)
    t0_qsvm_fit = time.perf_counter()
    qsvm = SVC(C=0.5, kernel="precomputed", class_weight="balanced", random_state=42)
    qsvm.fit(K_train, y_tr_arr)
    t_qsvm_fit = time.perf_counter() - t0_qsvm_fit

    t0_qsvm_infer = time.perf_counter()
    scores_qsvm = qsvm.decision_function(K_val)
    t_qsvm_infer = time.perf_counter() - t0_qsvm_infer

    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    def evaluate_scores(scores: np.ndarray) -> Dict[str, Any]:
        pr_auc = float(average_precision_score(y_val_arr, scores))
        roc_auc = float(roc_auc_score(y_val_arr, scores))
        preds_0 = (scores >= 0.0).astype(int)
        cm = confusion_matrix(y_val_arr, preds_0, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        prec = float(precision_score(y_val_arr, preds_0, pos_label=1, zero_division=0))
        rec = float(recall_score(y_val_arr, preds_0, pos_label=1, zero_division=0))
        f1 = float(f1_score(y_val_arr, preds_0, pos_label=1, zero_division=0))
        return {
            "pr_auc": round(pr_auc, 4),
            "roc_auc": round(roc_auc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "confusion_matrix": {
                "true_negatives": int(tn),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_positives": int(tp),
            },
        }

    results: Dict[str, Any] = {
        "sample_size": n_train,
        "preprocessing_time_seconds": round(t_prep, 4),
        "peak_memory_mb": round(peak_mem / (1024 * 1024), 2),
        "quantum_kernel_computation": kernel_timings,
        "models": {
            "linear_svm": {
                "name": "Linear SVM (C=5.0)",
                "fit_time_seconds": round(t_lin_fit, 4),
                "decision_function_time_seconds": round(t_lin_infer, 6),
                "latency_ms_per_sample": round((t_lin_infer / n_val) * 1000.0, 6),
                "total_inference_pipeline_seconds": round(t_lin_infer, 6),
                "metrics": evaluate_scores(scores_lin),
            },
            "rbf_svm": {
                "name": "RBF SVM (C=5.0, gamma=0.01)",
                "fit_time_seconds": round(t_rbf_fit, 4),
                "decision_function_time_seconds": round(t_rbf_infer, 6),
                "latency_ms_per_sample": round((t_rbf_infer / n_val) * 1000.0, 4),
                "total_inference_pipeline_seconds": round(t_rbf_infer, 6),
                "support_vectors_count": len(rbf_svm.support_),
                "metrics": evaluate_scores(scores_rbf),
            },
            "qsvm": {
                "name": "Quantum SVM (C=0.5, ZZ-Kernel)",
                "fit_time_seconds": round(t_qsvm_fit, 4),
                "decision_function_time_seconds": round(t_qsvm_infer, 6),
                "decision_latency_ms_per_sample": round((t_qsvm_infer / n_val) * 1000.0, 4),
                "kernel_generation_time_seconds": kernel_timings["val_kernel_seconds"],
                "total_inference_pipeline_seconds": round(kernel_timings["val_kernel_seconds"] + t_qsvm_infer, 4),
                "total_inference_latency_ms_per_sample": round(
                    ((kernel_timings["val_kernel_seconds"] + t_qsvm_infer) / n_val) * 1000.0, 4
                ),
                "support_vectors_count": len(qsvm.support_),
                "metrics": evaluate_scores(scores_qsvm),
            },
        },
    }

    return results


def compute_theoretical_scaling_projections(
    empirical_results: Dict[int, Any],
) -> Dict[str, Any]:
    """Computes theoretical quadratic scaling projections for training sample sizes up to N=2,000."""
    t_ktr_500 = empirical_results[500]["quantum_kernel_computation"]["train_kernel_seconds"]
    t_ktr_1000 = empirical_results[1000]["quantum_kernel_computation"]["train_kernel_seconds"]

    # Quadratic growth factor: entries = N^2
    entries_500 = 500 * 500
    entries_1000 = 1000 * 1000
    entries_2000 = 2000 * 2000

    mem_mb_500 = (entries_500 * 8) / (1024 * 1024)
    mem_mb_1000 = (entries_1000 * 8) / (1024 * 1024)
    mem_mb_2000 = (entries_2000 * 8) / (1024 * 1024)

    # Project N=2000 time based on measured statevector generation + BLAS scaling
    est_time_2000 = t_ktr_1000 * (entries_2000 / entries_1000)

    return {
        "scaling_law": "O(N^2 * 2^n)",
        "empirical_measurements": {
            "500": {
                "training_kernel_entries": entries_500,
                "kernel_memory_mb": round(mem_mb_500, 3),
                "measured_train_kernel_seconds": t_ktr_500,
            },
            "1000": {
                "training_kernel_entries": entries_1000,
                "kernel_memory_mb": round(mem_mb_1000, 3),
                "measured_train_kernel_seconds": t_ktr_1000,
            },
        },
        "projected_scaling_n2000": {
            "training_kernel_entries": entries_2000,
            "kernel_memory_mb": round(mem_mb_2000, 3),
            "estimated_train_kernel_seconds": round(est_time_2000, 2),
            "entry_increase_factor_vs_500": round(entries_2000 / entries_500, 1),
            "entry_increase_factor_vs_1000": round(entries_2000 / entries_1000, 1),
        },
    }


def generate_phase7_figures(
    metrics_data: Dict[str, Any],
    figures_dir: Path,
) -> List[Path]:
    """Generates the 5 required Phase 7 publication figures."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    generated: List[Path] = []

    sizes = [500, 1000]
    res_500 = metrics_data["empirical_results"]["500"]
    res_1000 = metrics_data["empirical_results"]["1000"]

    # Palette
    c_lin = "#2980b9"
    c_rbf = "#27ae60"
    c_qsvm = "#8e44ad"

    # 1. Figure 1: PR-AUC vs Training Set Size
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    pr_lin = [res_500["models"]["linear_svm"]["metrics"]["pr_auc"], res_1000["models"]["linear_svm"]["metrics"]["pr_auc"]]
    pr_rbf = [res_500["models"]["rbf_svm"]["metrics"]["pr_auc"], res_1000["models"]["rbf_svm"]["metrics"]["pr_auc"]]
    pr_qsvm = [res_500["models"]["qsvm"]["metrics"]["pr_auc"], res_1000["models"]["qsvm"]["metrics"]["pr_auc"]]

    ax.plot(sizes, pr_lin, marker="o", color=c_lin, linewidth=2.5, label="Linear SVM (C=5.0)")
    ax.plot(sizes, pr_rbf, marker="s", color=c_rbf, linewidth=2.5, label="RBF SVM (C=5.0, γ=0.01)")
    ax.plot(sizes, pr_qsvm, marker="^", color=c_qsvm, linewidth=2.5, linestyle="--", label="QSVM (C=0.5, ZZ-Kernel)")

    ax.set_title("PR-AUC vs. Training-Set Size (Validation Set, N=20,000)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Training Set Size (N)", fontsize=11)
    ax.set_ylabel("Precision-Recall AUC (Average Precision)", fontsize=11)
    ax.set_xticks(sizes)
    ax.set_ylim(0.2, 0.95)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, facecolor="white", edgecolor="none")
    p1 = figures_dir / "scalability_pr_auc.png"
    plt.tight_layout()
    plt.savefig(p1)
    plt.close()
    generated.append(p1)

    # 2. Figure 2: Recall and Precision vs Training Set Size
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
    rec_lin = [res_500["models"]["linear_svm"]["metrics"]["recall"], res_1000["models"]["linear_svm"]["metrics"]["recall"]]
    rec_rbf = [res_500["models"]["rbf_svm"]["metrics"]["recall"], res_1000["models"]["rbf_svm"]["metrics"]["recall"]]
    rec_qsvm = [res_500["models"]["qsvm"]["metrics"]["recall"], res_1000["models"]["qsvm"]["metrics"]["recall"]]

    prec_lin = [res_500["models"]["linear_svm"]["metrics"]["precision"], res_1000["models"]["linear_svm"]["metrics"]["precision"]]
    prec_rbf = [res_500["models"]["rbf_svm"]["metrics"]["precision"], res_1000["models"]["rbf_svm"]["metrics"]["precision"]]
    prec_qsvm = [res_500["models"]["qsvm"]["metrics"]["precision"], res_1000["models"]["qsvm"]["metrics"]["precision"]]

    ax1.plot(sizes, [r * 100 for r in rec_lin], marker="o", color=c_lin, linewidth=2.2, label="Linear SVM")
    ax1.plot(sizes, [r * 100 for r in rec_rbf], marker="s", color=c_rbf, linewidth=2.2, label="RBF SVM")
    ax1.plot(sizes, [r * 100 for r in rec_qsvm], marker="^", color=c_qsvm, linewidth=2.2, linestyle="--", label="QSVM")
    ax1.set_title("Fraud Recall vs. Training Size (τ = 0.0)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Training Set Size (N)", fontsize=10)
    ax1.set_ylabel("Recall (%)", fontsize=10)
    ax1.set_xticks(sizes)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()

    ax2.plot(sizes, [p * 100 for p in prec_lin], marker="o", color=c_lin, linewidth=2.2, label="Linear SVM")
    ax2.plot(sizes, [p * 100 for p in prec_rbf], marker="s", color=c_rbf, linewidth=2.2, label="RBF SVM")
    ax2.plot(sizes, [p * 100 for p in prec_qsvm], marker="^", color=c_qsvm, linewidth=2.2, linestyle="--", label="QSVM")
    ax2.set_title("Fraud Precision vs. Training Size (τ = 0.0)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Training Set Size (N)", fontsize=10)
    ax2.set_ylabel("Precision (%)", fontsize=10)
    ax2.set_xticks(sizes)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()

    p2 = figures_dir / "scalability_recall_precision.png"
    plt.tight_layout()
    plt.savefig(p2)
    plt.close()
    generated.append(p2)

    # 3. Figure 3: Quantum Kernel Computation Time vs Training Size
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    t_train_kernel = [
        res_500["quantum_kernel_computation"]["train_kernel_seconds"],
        res_1000["quantum_kernel_computation"]["train_kernel_seconds"],
    ]
    t_val_kernel = [
        res_500["quantum_kernel_computation"]["val_kernel_seconds"],
        res_1000["quantum_kernel_computation"]["val_kernel_seconds"],
    ]

    width = 80
    ax.bar([s - width / 2 for s in sizes], t_train_kernel, width=width, label="Training Gram Matrix (N × N)", color="#3498db")
    ax.bar([s + width / 2 for s in sizes], t_val_kernel, width=width, label="Validation Cross-Kernel (20,000 × N)", color="#9b59b6")
    ax.set_title("Quantum Kernel Computation Time vs. Training Size", fontsize=13, fontweight="bold")
    ax.set_xlabel("Training Set Size (N)", fontsize=11)
    ax.set_ylabel("Wall-Clock Time (seconds)", fontsize=11)
    ax.set_xticks(sizes)
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")
    ax.legend(frameon=True, facecolor="white")
    p3 = figures_dir / "scalability_quantum_kernel_time.png"
    plt.tight_layout()
    plt.savefig(p3)
    plt.close()
    generated.append(p3)

    # 4. Figure 4: Total Inference Pipeline Time vs Training Size
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    inf_lin = [res_500["models"]["linear_svm"]["total_inference_pipeline_seconds"], res_1000["models"]["linear_svm"]["total_inference_pipeline_seconds"]]
    inf_rbf = [res_500["models"]["rbf_svm"]["total_inference_pipeline_seconds"], res_1000["models"]["rbf_svm"]["total_inference_pipeline_seconds"]]
    inf_qsvm = [res_500["models"]["qsvm"]["total_inference_pipeline_seconds"], res_1000["models"]["qsvm"]["total_inference_pipeline_seconds"]]

    ax.plot(sizes, inf_lin, marker="o", color=c_lin, linewidth=2.5, label="Linear SVM Inference")
    ax.plot(sizes, inf_rbf, marker="s", color=c_rbf, linewidth=2.5, label="RBF SVM Inference")
    ax.plot(sizes, inf_qsvm, marker="^", color=c_qsvm, linewidth=2.5, linestyle="--", label="QSVM Full Pipeline (Kernel + Solver)")

    ax.set_yscale("log")
    ax.set_title("Total Inference Time (20,000 Validation Samples, Log Scale)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Training Set Size (N)", fontsize=11)
    ax.set_ylabel("Inference Pipeline Time (seconds, log scale)", fontsize=11)
    ax.set_xticks(sizes)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, facecolor="white")
    p4 = figures_dir / "scalability_total_inference_time.png"
    plt.tight_layout()
    plt.savefig(p4)
    plt.close()
    generated.append(p4)

    # 5. Figure 5: Performance vs Computational Cost Trade-off
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    for model_name, col, pr_vals, times in [
        ("Linear SVM", c_lin, pr_lin, inf_lin),
        ("RBF SVM", c_rbf, pr_rbf, inf_rbf),
        ("QSVM", c_qsvm, pr_qsvm, inf_qsvm),
    ]:
        ax.scatter(times, pr_vals, color=col, s=120, zorder=5, label=model_name)
        for s_idx, sz in enumerate(sizes):
            ax.annotate(
                f"{model_name} (N={sz})",
                (times[s_idx], pr_vals[s_idx]),
                textcoords="offset points",
                xytext=(8, 5),
                fontsize=9,
            )

    ax.set_xscale("log")
    ax.set_title("PR-AUC vs. Total Inference Time Trade-Off", fontsize=13, fontweight="bold")
    ax.set_xlabel("Total Inference Time (seconds, log scale)", fontsize=11)
    ax.set_ylabel("Validation PR-AUC", fontsize=11)
    ax.set_ylim(0.2, 0.95)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, facecolor="white", loc="lower left")
    p5 = figures_dir / "scalability_performance_vs_cost.png"
    plt.tight_layout()
    plt.savefig(p5)
    plt.close()
    generated.append(p5)

    return generated
