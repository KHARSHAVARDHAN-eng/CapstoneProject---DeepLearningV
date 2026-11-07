"""
Phase 5: Multi-Seed Resampling Stability Experiment Module.

Executes:
- 5 independent stratified 500-sample training draws from the 206,980-row training pool.
- Strict exclusion of the fixed 20,000 validation row indices from every draw.
- Independent fold-local feature selection (k=6) and angle scaling ([0, π]) per seed.
- Evaluation on the fixed 20,000-sample validation partition.
- Mean and standard deviation tracking across seeds for all three model families.
"""

import gc
import logging
import time
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC, SVC

from src.data.quantum_subsets import fit_feature_selector_and_scaler, transform_features
from src.models.quantum_kernel import (
    create_zz_feature_map,
    compute_fidelity_kernel_matrix,
)
from src.models.evaluate import compute_classification_metrics

logger = logging.getLogger("quantum_fraud.resampling")


def extract_resampled_training_subset(
    X_pool: pd.DataFrame,
    y_pool: pd.Series,
    n_train: int = 500,
    n_train_fraud: int = 100,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Draws a reproducible stratified training subset from the available training pool.

    Args:
        X_pool: Available training pool features (disjoint from validation).
        y_pool: Available training pool labels.
        n_train: Total training samples (default: 500).
        n_train_fraud: Positive fraud count (default: 100).
        random_state: Random seed for stratified sampling.

    Returns:
        (X_train_sub, y_train_sub) tuple.
    """
    fraud_indices = y_pool[y_pool == 1].index
    legit_indices = y_pool[y_pool == 0].index

    rng = np.random.RandomState(random_state)
    sampled_fraud = rng.choice(fraud_indices, size=n_train_fraud, replace=False)
    n_legit = n_train - n_train_fraud
    sampled_legit = rng.choice(legit_indices, size=n_legit, replace=False)

    all_indices = np.concatenate([sampled_fraud, sampled_legit])
    rng.shuffle(all_indices)

    X_train_sub = X_pool.loc[all_indices].copy()
    y_train_sub = y_pool.loc[all_indices].copy()
    return X_train_sub, y_train_sub


def run_multi_seed_resampling_experiment(
    X_train_full: pd.DataFrame,
    y_train_full: pd.Series,
    val_indices: np.ndarray,
    seeds: List[int] = [42, 101, 2024, 7, 99],
    n_train: int = 500,
    n_train_fraud: int = 100,
) -> Dict[str, Any]:
    """Runs the 5-seed resampling stability evaluation.

    Discloses:
    - 20,000 validation indices are strictly excluded from all draws.
    - Training subsets are drawn from the remaining 206,980 rows and may overlap across seeds.
    - Feature selection and scaling are fit independently on each seed's 500 training samples.

    Args:
        X_train_full: Full 226,980-row training feature matrix.
        y_train_full: Full 226,980-row training labels.
        val_indices: Indices of the locked 20,000-row validation partition.
        seeds: List of random seeds (default: [42, 101, 2024, 7, 99]).
        n_train: Training samples per seed (default: 500).
        n_train_fraud: Fraud samples per seed (default: 100).

    Returns:
        Dictionary containing individual seed runs, timing breakdowns, and aggregated summary stats.
    """
    # Create the disjoint available training pool (206,980 rows)
    val_idx_set = set(val_indices)
    pool_indices = [idx for idx in X_train_full.index if idx not in val_idx_set]
    X_pool = X_train_full.loc[pool_indices]
    y_pool = y_train_full.loc[pool_indices]

    assert len(X_pool) == len(X_train_full) - len(val_indices), "Pool size mismatch!"
    logger.info("Resampling pool created: %d rows (disjoint from %d validation rows)", len(X_pool), len(val_indices))

    # Validation raw features and labels
    X_val_raw = X_train_full.loc[val_indices]
    y_val_raw = y_train_full.loc[val_indices]
    y_val_arr = y_val_raw.to_numpy()

    fm = create_zz_feature_map(feature_dimension=6, reps=2, entanglement="linear")

    seed_runs = {}
    aggregated_metrics = {
        m: {metric: [] for metric in ["recall", "precision", "f1_score", "pr_auc", "roc_auc", "false_positives"]}
        for m in ["linear_svm", "rbf_svm", "qsvm"]
    }

    for seed in seeds:
        t0 = time.perf_counter()
        logger.info("--- Executing Resampling Run for Seed %d ---", seed)

        # 1. Draw stratified training subset
        X_tr_seed, y_tr_seed = extract_resampled_training_subset(
            X_pool, y_pool, n_train=n_train, n_train_fraud=n_train_fraud, random_state=seed
        )
        # Verify strict disjointness from validation
        overlap = set(X_tr_seed.index).intersection(val_idx_set)
        assert len(overlap) == 0, f"Critical leakage: Seed {seed} overlaps validation indices!"

        y_tr_arr = y_tr_seed.to_numpy()

        # 2. Fit feature selector and scaler strictly on this seed's 500 rows
        selector, scaler, selected_cols = fit_feature_selector_and_scaler(X_tr_seed, y_tr_arr, k=6)
        X_tr_scaled = transform_features(X_tr_seed, scaler, selected_cols).to_numpy()
        X_val_scaled = transform_features(X_val_raw, scaler, selected_cols).to_numpy()

        # 3. Model Training and Prediction
        # Linear SVM
        lin = LinearSVC(C=1.0, class_weight="balanced", dual=False, random_state=42)
        lin.fit(X_tr_scaled, y_tr_arr)
        pred_lin = lin.predict(X_val_scaled)
        score_lin = lin.decision_function(X_val_scaled)
        metrics_lin = compute_classification_metrics(y_val_arr, pred_lin, score_lin)

        # RBF SVM
        rbf = SVC(C=1.0, kernel="rbf", class_weight="balanced", random_state=42)
        rbf.fit(X_tr_scaled, y_tr_arr)
        pred_rbf = rbf.predict(X_val_scaled)
        score_rbf = rbf.decision_function(X_val_scaled)
        metrics_rbf = compute_classification_metrics(y_val_arr, pred_rbf, score_rbf)

        # QSVM
        t_q_start = time.perf_counter()
        K_tr, sv_tr, _ = compute_fidelity_kernel_matrix(X_tr_scaled, feature_map=fm)
        K_val, _, _ = compute_fidelity_kernel_matrix(
            X_val_scaled, X_tr_scaled, feature_map=fm, right_statevectors=sv_tr
        )
        t_kernel = time.perf_counter() - t_q_start

        qsvm = SVC(C=1.0, kernel="precomputed", class_weight="balanced", random_state=42)
        qsvm.fit(K_tr, y_tr_arr)
        pred_q = qsvm.predict(K_val)
        score_q = qsvm.decision_function(K_val)
        metrics_q = compute_classification_metrics(y_val_arr, pred_q, score_q)

        elapsed_seed = time.perf_counter() - t0

        seed_label = f"resampling_seed_{seed}"
        seed_runs[seed_label] = {
            "seed": seed,
            "selected_features": selected_cols,
            "elapsed_seconds": elapsed_seed,
            "kernel_construction_seconds": t_kernel,
            "linear_svm": metrics_lin,
            "rbf_svm": metrics_rbf,
            "qsvm": metrics_q,
        }

        # Accumulate metrics
        for m_key, m_dict in [("linear_svm", metrics_lin), ("rbf_svm", metrics_rbf), ("qsvm", metrics_q)]:
            for k in ["recall", "precision", "f1_score", "pr_auc", "roc_auc"]:
                aggregated_metrics[m_key][k].append(m_dict[k])
            aggregated_metrics[m_key]["false_positives"].append(m_dict["confusion_matrix"]["false_positives"])

        # Garbage collect intermediate kernel matrices to conserve RAM
        del K_tr, K_val
        gc.collect()

    # Calculate summary statistics across seeds
    summary_stats = {}
    for m_key in ["linear_svm", "rbf_svm", "qsvm"]:
        summary_stats[m_key] = {}
        for metric, values in aggregated_metrics[m_key].items():
            arr = np.array(values)
            summary_stats[m_key][metric] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "values": [float(v) for v in arr],
            }

    return {
        "n_seeds": len(seeds),
        "seeds_evaluated": seeds,
        "seed_runs": seed_runs,
        "aggregated_statistics": summary_stats,
    }
