"""
Automated test suite for Phase 2: Data Preprocessing and Classical Baseline Preparation.

Verifies:
1. Label separation (target column excluded from features)
2. Exact reproducibility of stratified partitioning given fixed random seed
3. Data leakage prevention (disjoint index sets, no train-test cross-contamination, scaler fit on train only)
4. Stratified class distribution preservation across train and test splits
5. Preprocessing artifact persistence and integrity
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import RobustScaler

from src.data.preprocess import (
    create_stratified_split,
    handle_duplicates,
    load_raw_data,
    run_preprocessing_pipeline,
    save_processed_data,
    scale_features,
    separate_features_and_target,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "creditcard.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


@pytest.fixture
def sample_imbalanced_data() -> pd.DataFrame:
    """Creates a controlled synthetic DataFrame with duplicate rows and extreme imbalance."""
    np.random.seed(42)
    n_samples = 1000
    n_features = 30

    feature_cols = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]
    data = np.random.randn(n_samples, n_features)

    df = pd.DataFrame(data, columns=feature_cols)
    # Binary class with ~1% fraud
    y = np.zeros(n_samples, dtype=int)
    fraud_indices = np.random.choice(n_samples, size=10, replace=False)
    y[fraud_indices] = 1
    df["Class"] = y

    # Inject 5 duplicate rows (4 legitimate, 1 fraud)
    dup_rows = pd.concat([df.iloc[[0, 1, 2, 3]], df.iloc[[fraud_indices[0]]]], ignore_index=True)
    df_with_dups = pd.concat([df, dup_rows], ignore_index=True)
    return df_with_dups


# ============================================================================
# 1. Label Separation Tests
# ============================================================================

def test_label_separation(sample_imbalanced_data: pd.DataFrame):
    """Verifies features X and target y are strictly separated."""
    X, y = separate_features_and_target(sample_imbalanced_data, target_col="Class")

    # Target column must not exist in feature matrix
    assert "Class" not in X.columns, "Target column 'Class' was leaked into feature matrix X!"
    assert X.shape[1] == 30, f"Expected 30 features, but found {X.shape[1]}"
    assert len(X) == len(y), "Row counts for X and y must match"
    assert isinstance(y, pd.Series), "Target y must be a pandas Series"
    assert set(y.unique()).issubset({0, 1}), "Target values must be binary (0 or 1)"


def test_label_separation_missing_column():
    """Verifies KeyError is raised if target column is absent."""
    df = pd.DataFrame({"feat1": [1, 2], "feat2": [3, 4]})
    with pytest.raises(KeyError):
        separate_features_and_target(df, target_col="Class")


# ============================================================================
# 2. Reproducibility Tests
# ============================================================================

def test_split_reproducibility(sample_imbalanced_data: pd.DataFrame):
    """Verifies running create_stratified_split with the same seed yields identical splits."""
    df_cleaned, _ = handle_duplicates(sample_imbalanced_data, drop=True)
    X, y = separate_features_and_target(df_cleaned, target_col="Class")

    X_train_1, X_test_1, y_train_1, y_test_1 = create_stratified_split(
        X, y, test_size=0.20, random_state=42
    )
    X_train_2, X_test_2, y_train_2, y_test_2 = create_stratified_split(
        X, y, test_size=0.20, random_state=42
    )

    pd.testing.assert_frame_equal(X_train_1, X_train_2)
    pd.testing.assert_frame_equal(X_test_1, X_test_2)
    pd.testing.assert_series_equal(y_train_1, y_train_2)
    pd.testing.assert_series_equal(y_test_1, y_test_2)


def test_different_seeds_yield_different_splits(sample_imbalanced_data: pd.DataFrame):
    """Verifies different seeds produce distinct sample assignments."""
    df_cleaned, _ = handle_duplicates(sample_imbalanced_data, drop=True)
    X, y = separate_features_and_target(df_cleaned, target_col="Class")

    X_train_seed42, _, _, _ = create_stratified_split(X, y, test_size=0.20, random_state=42)
    X_train_seed99, _, _, _ = create_stratified_split(X, y, test_size=0.20, random_state=99)

    assert not X_train_seed42.index.equals(X_train_seed99.index)


# ============================================================================
# 3. Data Leakage Prevention Tests
# ============================================================================

def test_disjoint_train_test_indices(sample_imbalanced_data: pd.DataFrame):
    """Verifies training and test sets have zero index overlap."""
    df_cleaned, _ = handle_duplicates(sample_imbalanced_data, drop=True)
    X, y = separate_features_and_target(df_cleaned, target_col="Class")

    X_train, X_test, _, _ = create_stratified_split(X, y, test_size=0.20, random_state=42)

    overlap = set(X_train.index).intersection(set(X_test.index))
    assert len(overlap) == 0, f"Data leakage detected! Overlapping indices: {overlap}"
    assert len(X_train) + len(X_test) == len(X)


def test_scaler_fitted_only_on_training_data():
    """Verifies the scaler statistics match training set ONLY and are not influenced by test set."""
    np.random.seed(123)
    # Training set centered around 10.0, test set centered around 100.0
    X_train = pd.DataFrame({
        "feat1": np.random.normal(loc=10.0, scale=2.0, size=500),
        "feat2": np.random.normal(loc=10.0, scale=2.0, size=500),
    })
    X_test = pd.DataFrame({
        "feat1": np.random.normal(loc=100.0, scale=2.0, size=100),
        "feat2": np.random.normal(loc=100.0, scale=2.0, size=100),
    })

    X_train_scaled, X_test_scaled, scaler = scale_features(
        X_train, X_test, scaler_type="robust"
    )

    # Scaler center_ must equal median of X_train, NOT combined
    expected_train_median = X_train.median().values
    np.testing.assert_allclose(scaler.center_, expected_train_median, rtol=1e-5)

    # Combined median would be significantly higher
    combined_median = pd.concat([X_train, X_test]).median().values
    assert not np.allclose(scaler.center_, combined_median, rtol=1e-2), (
        "Data leakage! Scaler parameters match combined data rather than train-only data."
    )


def test_no_identical_duplicate_leakage(sample_imbalanced_data: pd.DataFrame):
    """Verifies that dropping duplicates prevents identical records from spanning across train and test."""
    df_cleaned, dup_stats = handle_duplicates(sample_imbalanced_data, drop=True)
    assert dup_stats["total_duplicates"] == 5

    X, y = separate_features_and_target(df_cleaned, target_col="Class")
    X_train, X_test, _, _ = create_stratified_split(X, y, test_size=0.20, random_state=42)

    # Merge on all features to check for cross-split duplicates
    cross_dups = pd.merge(X_train, X_test, on=list(X.columns), how="inner")
    assert len(cross_dups) == 0, (
        f"Data leakage detected! {len(cross_dups)} identical records appear in both train and test."
    )


# ============================================================================
# 4. Class Distribution & Stratification Tests
# ============================================================================

def test_stratification_preserves_class_ratio(sample_imbalanced_data: pd.DataFrame):
    """Verifies class proportion is preserved across train and test splits."""
    df_cleaned, _ = handle_duplicates(sample_imbalanced_data, drop=True)
    X, y = separate_features_and_target(df_cleaned, target_col="Class")

    X_train, X_test, y_train, y_test = create_stratified_split(
        X, y, test_size=0.20, random_state=42
    )

    overall_fraud_pct = (y.sum() / len(y)) * 100
    train_fraud_pct = (y_train.sum() / len(y_train)) * 100
    test_fraud_pct = (y_test.sum() / len(y_test)) * 100

    # Proportions should match within 0.1% tolerance
    assert abs(train_fraud_pct - overall_fraud_pct) < 0.15
    assert abs(test_fraud_pct - overall_fraud_pct) < 0.15
    assert y_train.sum() > 0, "Training set contains zero fraud instances!"
    assert y_test.sum() > 0, "Test set contains zero fraud instances!"


# ============================================================================
# 5. Pipeline Integration Tests on Real Dataset (if present)
# ============================================================================

@pytest.mark.skipif(not RAW_DATA_PATH.exists(), reason="Raw dataset not found at data/raw/creditcard.csv")
def test_full_pipeline_on_real_dataset():
    """Integration test validating output files and metadata created by the full pipeline."""
    # Ensure processed files exist
    expected_files = [
        PROCESSED_DIR / "X_train.csv",
        PROCESSED_DIR / "X_test.csv",
        PROCESSED_DIR / "y_train.csv",
        PROCESSED_DIR / "y_test.csv",
        PROCESSED_DIR / "train.csv",
        PROCESSED_DIR / "test.csv",
        PROCESSED_DIR / "scaler.joblib",
        PROCESSED_DIR / "preprocessing_metadata.json",
    ]

    for f in expected_files:
        assert f.exists(), f"Expected artifact {f.name} does not exist!"
        assert f.stat().st_size > 0, f"Expected artifact {f.name} is empty!"

    # Verify metadata contents
    meta_path = PROCESSED_DIR / "preprocessing_metadata.json"
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))

    assert metadata["initial_rows"] == 284807
    assert metadata["final_rows"] == 283726
    assert metadata["duplicate_handling"]["total_duplicates"] == 1081
    assert metadata["duplicate_handling"]["duplicates_by_class"]["fraudulent (1)"] == 19
    assert metadata["duplicate_handling"]["duplicates_by_class"]["legitimate (0)"] == 1062

    # Verify train and test dimensions
    train_samples = metadata["splits"]["train"]["total_samples"]
    test_samples = metadata["splits"]["test"]["total_samples"]
    assert train_samples == 226980
    assert test_samples == 56746

    # Verify fraud counts in splits
    assert metadata["splits"]["train"]["fraud_count"] == 378
    assert metadata["splits"]["test"]["fraud_count"] == 95
    assert metadata["splits"]["train"]["fraud_count"] + metadata["splits"]["test"]["fraud_count"] == 473

    # Verify fitted scaler can be loaded and used
    scaler = joblib.load(PROCESSED_DIR / "scaler.joblib")
    assert isinstance(scaler, RobustScaler)
    assert len(scaler.center_) == 30
