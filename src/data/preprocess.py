"""
Data preprocessing and splitting pipeline for financial fraud detection.

Handles duplicate investigation and removal, feature-target separation,
reproducible stratified train/test partitioning, and leak-free feature scaling.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Literal, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

logger = logging.getLogger("quantum_fraud.preprocess")


def load_raw_data(data_path: Path) -> pd.DataFrame:
    """Loads the raw credit card fraud dataset in read-only fashion.

    Args:
        data_path: Path to the raw creditcard.csv file.

    Returns:
        pd.DataFrame: Loaded dataset.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not data_path.is_file():
        raise FileNotFoundError(f"Raw dataset not found at: {data_path.resolve()}")

    logger.info("Loading raw dataset from %s (read-only)...", data_path)
    df = pd.read_csv(data_path)
    logger.info("Raw dataset loaded: %d rows, %d columns.", len(df), len(df.columns))
    return df


def handle_duplicates(
    df: pd.DataFrame, drop: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Analyzes and conditionally handles duplicate rows.

    Args:
        df: Input DataFrame.
        drop: Whether to drop duplicate rows (keeping first occurrence).

    Returns:
        Tuple[pd.DataFrame, Dict[str, Any]]: Cleaned DataFrame and duplicate statistics.
    """
    dup_mask = df.duplicated(keep="first")
    n_duplicates = int(dup_mask.sum())
    dup_by_class = df.loc[dup_mask, "Class"].value_counts().to_dict() if "Class" in df.columns else {}

    dup_stats = {
        "total_duplicates": n_duplicates,
        "duplicate_percentage": round((n_duplicates / len(df)) * 100, 4),
        "duplicates_by_class": {
            "legitimate (0)": int(dup_by_class.get(0, 0)),
            "fraudulent (1)": int(dup_by_class.get(1, 0)),
        },
        "strategy": "drop_first" if drop else "retained",
    }

    if drop and n_duplicates > 0:
        logger.info(
            "Dropping %d duplicate rows (%d legitimate, %d fraudulent) to prevent train-test data leakage.",
            n_duplicates,
            dup_stats["duplicates_by_class"]["legitimate (0)"],
            dup_stats["duplicates_by_class"]["fraudulent (1)"],
        )
        df_cleaned = df.drop_duplicates(keep="first").reset_index(drop=True)
    else:
        df_cleaned = df.copy()

    logger.info("Dataset record count after duplicate handling: %d rows.", len(df_cleaned))
    return df_cleaned, dup_stats


def separate_features_and_target(
    df: pd.DataFrame, target_col: str = "Class"
) -> Tuple[pd.DataFrame, pd.Series]:
    """Separates feature matrix X from target vector y.

    Args:
        df: Input DataFrame.
        target_col: Name of the target column.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: Feature matrix X and target Series y.

    Raises:
        KeyError: If target_col is not found in df.
    """
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in dataset columns.")

    y = df[target_col].copy()
    X = df.drop(columns=[target_col]).copy()

    logger.info(
        "Features and target separated: X shape = %s (features: %d), y shape = %s.",
        X.shape,
        X.shape[1],
        y.shape,
    )
    return X, y


def create_stratified_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.20,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Creates a reproducible, stratified train/test split.

    Args:
        X: Feature matrix.
        y: Target series.
        test_size: Proportion of the dataset to allocate to the test split (default 0.20).
        random_state: Random seed for exact reproducibility (default 42).

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
            X_train, X_test, y_train, y_test.
    """
    logger.info(
        "Splitting data with test_size=%.2f, random_state=%d, stratified by target...",
        test_size,
        random_state,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    logger.info("Train split: %d samples (%d features)", len(X_train), X_train.shape[1])
    logger.info("Test split:  %d samples (%d features)", len(X_test), X_test.shape[1])

    return X_train, X_test, y_train, y_test


def get_scaler(scaler_type: Literal["robust", "standard", "minmax"]) -> Any:
    """Factory function for feature scalers.

    Args:
        scaler_type: Scaler identifier ('robust', 'standard', 'minmax').

    Returns:
        Fitted or unfitted scaler object.
    """
    scaler_type = scaler_type.lower()
    if scaler_type == "robust":
        return RobustScaler()
    elif scaler_type == "standard":
        return StandardScaler()
    elif scaler_type == "minmax":
        return MinMaxScaler()
    else:
        raise ValueError(
            f"Unsupported scaler_type '{scaler_type}'. Choose from 'robust', 'standard', 'minmax'."
        )


def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    scaler_type: Literal["robust", "standard", "minmax"] = "robust",
) -> Tuple[pd.DataFrame, pd.DataFrame, Any]:
    """Scales features by fitting on X_train only to prevent data leakage.

    Args:
        X_train: Training features.
        X_test: Testing features.
        scaler_type: Scaler to use (default 'robust' for outlier robustness).

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, Any]:
            Scaled X_train, scaled X_test, and the fitted scaler object.
    """
    scaler = get_scaler(scaler_type)
    logger.info("Fitting %s strictly on training features (leak-free)...", type(scaler).__name__)

    # Fit on training data ONLY
    X_train_scaled_arr = scaler.fit_transform(X_train)
    # Transform test data using the fitted training statistics
    X_test_scaled_arr = scaler.transform(X_test)

    # Reconstruct DataFrames to preserve column names and index
    X_train_scaled = pd.DataFrame(
        X_train_scaled_arr,
        columns=X_train.columns,
        index=X_train.index,
    )
    X_test_scaled = pd.DataFrame(
        X_test_scaled_arr,
        columns=X_test.columns,
        index=X_test.index,
    )

    logger.info("Feature scaling completed successfully.")
    return X_train_scaled, X_test_scaled, scaler


def save_processed_data(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    scaler: Any,
    output_dir: Path,
    metadata: Dict[str, Any],
) -> Dict[str, Path]:
    """Saves processed data splits, scaler artifact, and metadata.

    Args:
        X_train: Scaled training features.
        X_test: Scaled testing features.
        y_train: Training target labels.
        y_test: Testing target labels.
        scaler: Fitted scaler object.
        output_dir: Output directory (data/processed).
        metadata: Preprocessing configuration and metrics.

    Returns:
        Dict[str, Path]: Dictionary mapping artifact names to their file paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_files: Dict[str, Path] = {}

    # 1. Feature matrices
    x_train_path = output_dir / "X_train.csv"
    x_test_path = output_dir / "X_test.csv"
    X_train.to_csv(x_train_path, index=False)
    X_test.to_csv(x_test_path, index=False)
    saved_files["X_train"] = x_train_path
    saved_files["X_test"] = x_test_path

    # 2. Target labels
    y_train_path = output_dir / "y_train.csv"
    y_test_path = output_dir / "y_test.csv"
    y_train.to_csv(y_train_path, index=False, header=True)
    y_test.to_csv(y_test_path, index=False, header=True)
    saved_files["y_train"] = y_train_path
    saved_files["y_test"] = y_test_path

    # 3. Combined splits (convenience for classical baselines)
    train_combined = X_train.copy()
    train_combined["Class"] = y_train.values
    train_path = output_dir / "train.csv"
    train_combined.to_csv(train_path, index=False)
    saved_files["train_combined"] = train_path

    test_combined = X_test.copy()
    test_combined["Class"] = y_test.values
    test_path = output_dir / "test.csv"
    test_combined.to_csv(test_path, index=False)
    saved_files["test_combined"] = test_path

    # 4. Fitted Scaler object
    scaler_path = output_dir / "scaler.joblib"
    joblib.dump(scaler, scaler_path)
    saved_files["scaler"] = scaler_path

    # 5. Preprocessing metadata JSON
    metadata_path = output_dir / "preprocessing_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    saved_files["metadata"] = metadata_path

    logger.info("All processed artifacts successfully saved to %s", output_dir)
    return saved_files


def run_preprocessing_pipeline(
    data_path: Path,
    output_dir: Path,
    drop_duplicates: bool = True,
    test_size: float = 0.20,
    random_state: int = 42,
    scaler_type: Literal["robust", "standard", "minmax"] = "robust",
) -> Dict[str, Any]:
    """Orchestrates the complete Phase 2 preprocessing and preparation pipeline.

    Args:
        data_path: Path to raw creditcard.csv.
        output_dir: Target directory for processed artifacts (data/processed).
        drop_duplicates: Whether to drop duplicate rows to prevent train-test leakage.
        test_size: Test split ratio (default 0.20 for 80/20 split).
        random_state: Random seed for reproducibility (default 42).
        scaler_type: Feature scaling method ('robust', 'standard', 'minmax').

    Returns:
        Dict[str, Any]: Summary metrics, distribution counts, and saved artifact paths.
    """
    logger.info("Starting Phase 2 preprocessing pipeline...")

    # Step 1: Load raw data
    raw_df = load_raw_data(data_path)
    initial_rows = len(raw_df)

    # Step 2: Handle duplicate rows
    cleaned_df, dup_stats = handle_duplicates(raw_df, drop=drop_duplicates)

    # Step 3: Separate features from target
    X, y = separate_features_and_target(cleaned_df, target_col="Class")

    # Step 4: Stratified train/test split
    X_train, X_test, y_train, y_test = create_stratified_split(
        X=X,
        y=y,
        test_size=test_size,
        random_state=random_state,
    )

    # Step 5: Feature scaling (fit on train only, transform test)
    X_train_scaled, X_test_scaled, scaler = scale_features(
        X_train=X_train,
        X_test=X_test,
        scaler_type=scaler_type,
    )

    # Calculate distributions
    train_counts = y_train.value_counts().to_dict()
    test_counts = y_test.value_counts().to_dict()

    train_legit = int(train_counts.get(0, 0))
    train_fraud = int(train_counts.get(1, 0))
    test_legit = int(test_counts.get(0, 0))
    test_fraud = int(test_counts.get(1, 0))

    train_fraud_pct = (train_fraud / len(y_train)) * 100 if len(y_train) > 0 else 0.0
    test_fraud_pct = (test_fraud / len(y_test)) * 100 if len(y_test) > 0 else 0.0

    metadata = {
        "pipeline_version": "phase-2-preprocessing-v1",
        "random_state": random_state,
        "test_size": test_size,
        "scaler_type": scaler_type,
        "initial_rows": initial_rows,
        "final_rows": len(cleaned_df),
        "duplicate_handling": dup_stats,
        "feature_count": X.shape[1],
        "feature_names": list(X.columns),
        "target_col": "Class",
        "splits": {
            "train": {
                "total_samples": len(y_train),
                "legitimate_count": train_legit,
                "fraud_count": train_fraud,
                "fraud_percentage": round(train_fraud_pct, 4),
            },
            "test": {
                "total_samples": len(y_test),
                "legitimate_count": test_legit,
                "fraud_count": test_fraud,
                "fraud_percentage": round(test_fraud_pct, 4),
            },
        },
    }

    # Step 6: Save processed artifacts
    saved_files = save_processed_data(
        X_train=X_train_scaled,
        X_test=X_test_scaled,
        y_train=y_train,
        y_test=y_test,
        scaler=scaler,
        output_dir=output_dir,
        metadata=metadata,
    )

    logger.info("Phase 2 preprocessing pipeline finished successfully.")
    return {
        "status": "SUCCESS",
        "metadata": metadata,
        "saved_files": {k: str(v) for k, v in saved_files.items()},
    }
