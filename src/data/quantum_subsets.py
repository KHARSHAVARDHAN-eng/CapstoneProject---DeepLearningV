"""
Quantum subset extraction, feature selection, and angle scaling module.

Extracts reproducible quantum training and validation subsets from the existing training partition,
fits ANOVA F-statistic feature selection (SelectKBest, k=6) strictly on the quantum training subset,
and scales selected features to [0, π] for angle encoding in quantum feature maps.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger("quantum_fraud.quantum_subsets")


def get_quantum_partition_indices(
    y_train: pd.Series,
    n_train: int = 500,
    n_train_fraud: int = 100,
    n_val: int = 20000,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Computes reproducible original row indices for quantum training and validation subsets.

    Args:
        y_train: Training labels Series with original dataset index.
        n_train: Total quantum training sample size (default: 500).
        n_train_fraud: Number of fraud samples in training subset (default: 100).
        n_val: Total quantum validation sample size (default: 20000).
        random_state: Seed for reproducibility.

    Returns:
        Tuple of (train_q_idx, val_q_idx) containing original row indices.
    """
    n_train_legit = n_train - n_train_fraud
    if n_train_fraud <= 0 or n_train_legit <= 0:
        raise ValueError("Both fraud and legit samples must be positive in quantum training subset.")

    rng = np.random.RandomState(random_state)
    fraud_indices = y_train[y_train == 1].index.to_numpy()
    legit_indices = y_train[y_train == 0].index.to_numpy()

    if len(fraud_indices) < n_train_fraud:
        raise ValueError(
            f"Requested {n_train_fraud} fraud samples, but only {len(fraud_indices)} available in training set."
        )

    # 1. Quantum training subset indices
    shuffled_fraud = rng.permutation(fraud_indices)
    shuffled_legit = rng.permutation(legit_indices)

    train_fraud_idx = shuffled_fraud[:n_train_fraud]
    train_legit_idx = shuffled_legit[:n_train_legit]
    train_q_idx = np.concatenate([train_fraud_idx, train_legit_idx])
    rng.shuffle(train_q_idx)

    # 2. Remaining training pool for validation indices
    rem_fraud_idx = shuffled_fraud[n_train_fraud:]
    rem_legit_idx = shuffled_legit[n_train_legit:]

    natural_prevalence = len(fraud_indices) / len(y_train)
    n_val_fraud = max(1, int(round(n_val * natural_prevalence)))
    n_val_legit = n_val - n_val_fraud

    val_fraud_idx = rem_fraud_idx[:n_val_fraud]
    val_legit_idx = rem_legit_idx[:n_val_legit]
    val_q_idx = np.concatenate([val_fraud_idx, val_legit_idx])
    rng.shuffle(val_q_idx)

    return train_q_idx, val_q_idx


def extract_quantum_subsets(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_train: int = 500,
    n_train_fraud: int = 100,
    n_val: int = 1000,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Extracts reproducible quantum training and validation subsets strictly from the training partition.

    Args:
        X_train: Preprocessed training feature matrix.
        y_train: Training labels (0 = Legitimate, 1 = Fraud).
        n_train: Total quantum training sample size (default: 500).
        n_train_fraud: Number of fraud samples in quantum training subset (default: 100).
        n_val: Total quantum validation sample size (default: 1000).
        random_state: Seed for reproducibility.

    Returns:
        Tuple of (X_train_q, y_train_q, X_val_q, y_val_q).
    """
    if len(X_train) != len(y_train):
        raise ValueError("X_train and y_train must have the same length.")

    train_q_idx, val_q_idx = get_quantum_partition_indices(
        y_train, n_train=n_train, n_train_fraud=n_train_fraud, n_val=n_val, random_state=random_state
    )

    X_train_q = X_train.loc[train_q_idx].copy().reset_index(drop=True)
    y_train_q = y_train.loc[train_q_idx].copy().reset_index(drop=True)
    X_val_q = X_train.loc[val_q_idx].copy().reset_index(drop=True)
    y_val_q = y_train.loc[val_q_idx].copy().reset_index(drop=True)

    n_train_fraud_actual = int(np.sum(y_train_q == 1))
    n_train_legit_actual = int(np.sum(y_train_q == 0))
    n_val_fraud_actual = int(np.sum(y_val_q == 1))
    n_val_legit_actual = int(np.sum(y_val_q == 0))

    logger.info(
        "Quantum subsets created: Train N=%d (Legit=%d, Fraud=%d, %.2f%% fraud), "
        "Val N=%d (Legit=%d, Fraud=%d, %.2f%% fraud)",
        len(X_train_q),
        n_train_legit_actual,
        n_train_fraud_actual,
        (n_train_fraud_actual / len(X_train_q)) * 100,
        len(X_val_q),
        n_val_legit_actual,
        n_val_fraud_actual,
        (n_val_fraud_actual / len(X_val_q)) * 100,
    )

    return X_train_q, y_train_q, X_val_q, y_val_q


def fit_feature_selector_and_scaler(
    X_train_q: pd.DataFrame,
    y_train_q: pd.Series,
    k: int = 6,
) -> Tuple[SelectKBest, MinMaxScaler, List[str]]:
    """Fits ANOVA F-test feature selection and [0, π] angle scaler strictly on the quantum training subset.

    Args:
        X_train_q: Quantum training features.
        y_train_q: Quantum training labels.
        k: Number of features to select (qubit count).

    Returns:
        Tuple of (fitted SelectKBest, fitted MinMaxScaler, list of selected feature names).
    """
    selector = SelectKBest(score_func=f_classif, k=k)
    selector.fit(X_train_q, y_train_q)

    support_mask = selector.get_support()
    selected_features = list(X_train_q.columns[support_mask])

    X_train_selected = X_train_q[selected_features]

    scaler = MinMaxScaler(feature_range=(0.0, np.pi))
    scaler.fit(X_train_selected)

    logger.info("Selected %d features: %s", k, selected_features)
    for feat, score in zip(selected_features, selector.scores_[support_mask]):
        logger.info("  Feature %s: ANOVA F-statistic = %.2f", feat, score)

    return selector, scaler, selected_features


def transform_features(
    X: pd.DataFrame,
    scaler: MinMaxScaler,
    selected_features: List[str],
) -> pd.DataFrame:
    """Transforms feature matrix using pre-selected features and pre-fitted angle scaler.

    Args:
        X: Feature matrix containing the selected feature columns.
        scaler: Fitted MinMaxScaler with range [0, π].
        selected_features: List of feature names selected by SelectKBest.

    Returns:
        Transformed pd.DataFrame with scaled angles in [0, π].
    """
    X_sub = X[selected_features]
    X_scaled = scaler.transform(X_sub)
    return pd.DataFrame(X_scaled, columns=selected_features)


def prepare_and_save_quantum_data(
    train_features_path: Path,
    train_labels_path: Path,
    output_dir: Path,
    n_train: int = 500,
    n_train_fraud: int = 100,
    n_val: int = 1000,
    k_features: int = 6,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Full workflow to extract, transform, and persist quantum subsets and metadata."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading training partitions from %s and %s", train_features_path, train_labels_path)
    X_train = pd.read_csv(train_features_path)
    y_train = pd.read_csv(train_labels_path).squeeze("columns")

    # Extract subsets
    X_train_q, y_train_q, X_val_q, y_val_q = extract_quantum_subsets(
        X_train=X_train,
        y_train=y_train,
        n_train=n_train,
        n_train_fraud=n_train_fraud,
        n_val=n_val,
        random_state=random_state,
    )

    # Fit feature selection and angle scaling strictly on training subset
    selector, scaler, selected_features = fit_feature_selector_and_scaler(
        X_train_q=X_train_q,
        y_train_q=y_train_q,
        k=k_features,
    )

    # Transform both splits
    X_train_q_scaled = transform_features(X_train_q, scaler, selected_features)
    X_val_q_scaled = transform_features(X_val_q, scaler, selected_features)

    # Save artifacts
    X_train_q_scaled.to_csv(output_dir / "X_train_q_scaled.csv", index=False)
    y_train_q.to_frame(name="Class").to_csv(output_dir / "y_train_q.csv", index=False)
    X_val_q_scaled.to_csv(output_dir / "X_val_q_scaled.csv", index=False)
    y_val_q.to_frame(name="Class").to_csv(output_dir / "y_val_q.csv", index=False)

    # Save raw unscaled selected subsets for reference
    X_train_q[selected_features].to_csv(output_dir / "X_train_q_selected_unscaled.csv", index=False)
    X_val_q[selected_features].to_csv(output_dir / "X_val_q_selected_unscaled.csv", index=False)

    joblib.dump(selector, output_dir / "feature_selector.joblib")
    joblib.dump(scaler, output_dir / "angle_scaler.joblib")

    scores_dict = {
        feat: float(score)
        for feat, score in zip(
            selected_features,
            selector.scores_[selector.get_support()],
        )
    }

    metadata = {
        "pipeline_phase": "Phase 4B - Quantum Subset Preparation",
        "random_state": random_state,
        "k_features": k_features,
        "selected_features": selected_features,
        "anova_f_scores": scores_dict,
        "angle_scaling_range": [0.0, float(np.pi)],
        "train_subset": {
            "total_samples": int(len(X_train_q)),
            "legitimate_count": int((y_train_q == 0).sum()),
            "fraud_count": int((y_train_q == 1).sum()),
            "fraud_percentage": float((y_train_q == 1).mean() * 100),
        },
        "val_subset": {
            "total_samples": int(len(X_val_q)),
            "legitimate_count": int((y_val_q == 0).sum()),
            "fraud_count": int((y_val_q == 1).sum()),
            "fraud_percentage": float((y_val_q == 1).mean() * 100),
        },
    }

    with open(output_dir / "quantum_subset_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Quantum subsets and metadata successfully saved to %s", output_dir)
    return metadata
