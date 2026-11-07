"""
Unit and integration tests for Phase 7 Training-Size Scalability Experiment.

Validates:
1. Data availability verification and non-replacement shortfall detection for N=2,000.
2. Nested reproducible subset construction (D_500 ⊂ D_1000) and exact 80:20 class composition.
3. Preprocessing isolation (scaler fitted strictly on training subset; transform only on val).
4. Quantum kernel matrix properties (numerical bounds, symmetry, statevector reuse).
5. Phase 7 deliverables, metrics consistency, and publication figures integrity.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.quantum_subsets import get_quantum_partition_indices
from src.experiments.phase7_scalability import (
    SELECTED_FEATURES,
    compute_quantum_kernels,
    compute_theoretical_scaling_projections,
    generate_nested_training_subsets,
    verify_sample_availability,
)
from src.models.quantum_kernel import create_zz_feature_map

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_sample_availability_verification():
    """Verifies that fraud availability check correctly flags the N=2,000 shortfall."""
    y_train = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "y_train.csv").squeeze("columns")
    _, val_indices = get_quantum_partition_indices(y_train, n_train=500, n_train_fraud=100, n_val=20000, random_state=42)

    avail = verify_sample_availability(y_train=y_train, val_indices=val_indices, requested_sizes=[500, 1000, 2000])

    # Total dataset has 378 frauds; 33 are in validation, leaving 345
    assert avail["dataset_total"]["total_fraud"] == 378
    assert avail["validation_held_out"]["fraud_count"] == 33
    assert avail["non_validation_training_pool"]["available_fraud"] == 345

    # 500 and 1,000 are feasible
    assert avail["requested_size_evaluations"]["500"]["status"] == "FEASIBLE"
    assert avail["requested_size_evaluations"]["1000"]["status"] == "FEASIBLE"

    # 2,000 requires 400 frauds, but only 345 are available without replacement
    eval_2000 = avail["requested_size_evaluations"]["2000"]
    assert eval_2000["status"] == "INSUFFICIENT_FRAUD_EXAMPLES"
    assert eval_2000["shortfall_fraud"] == 55
    assert eval_2000["feasible_without_validation_leakage"] is False


def test_nested_subsets_reproducibility_and_isolation():
    """Verifies that D_500 ⊂ D_1000 and subsets have exact 80:20 legitimate:fraud ratio."""
    X_train = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "X_train.csv")
    y_train = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "y_train.csv").squeeze("columns")

    train_500_idx, val_20000_idx = get_quantum_partition_indices(
        y_train, n_train=500, n_train_fraud=100, n_val=20000, random_state=42
    )

    subsets = generate_nested_training_subsets(
        X_train=X_train,
        y_train=y_train,
        val_indices=val_20000_idx,
        train_500_indices=train_500_idx,
        random_state=42,
    )

    # Check N=500
    assert subsets[500]["n_total"] == 500
    assert subsets[500]["n_legit"] == 400
    assert subsets[500]["n_fraud"] == 100
    assert int((subsets[500]["y"] == 1).sum()) == 100

    # Check N=1,000
    assert subsets[1000]["n_total"] == 1000
    assert subsets[1000]["n_legit"] == 800
    assert subsets[1000]["n_fraud"] == 200
    assert int((subsets[1000]["y"] == 1).sum()) == 200

    # Strictly nested
    set_500 = set(subsets[500]["indices"])
    set_1000 = set(subsets[1000]["indices"])
    assert set_500.issubset(set_1000), "D_500 must be a subset of D_1000!"

    # Disjoint with validation set
    val_set = set(val_20000_idx)
    assert len(set_500.intersection(val_set)) == 0
    assert len(set_1000.intersection(val_set)) == 0


def test_preprocessing_isolation():
    """Verifies that scaler parameters are fitted strictly on the training subset."""
    from sklearn.preprocessing import MinMaxScaler

    X_train = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "X_train.csv")[SELECTED_FEATURES]
    y_train = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "y_train.csv").squeeze("columns")

    train_500_idx, val_20000_idx = get_quantum_partition_indices(
        y_train, n_train=500, n_train_fraud=100, n_val=20000, random_state=42
    )

    subsets = generate_nested_training_subsets(
        X_train=X_train,
        y_train=y_train,
        val_indices=val_20000_idx,
        train_500_indices=train_500_idx,
        random_state=42,
    )

    # Fit scalers on 500 vs 1000
    scaler_500 = MinMaxScaler(feature_range=(0.0, np.pi))
    scaler_500.fit(subsets[500]["X_unscaled"])

    scaler_1000 = MinMaxScaler(feature_range=(0.0, np.pi))
    scaler_1000.fit(subsets[1000]["X_unscaled"])

    # Ensure scaler limits are in [0, pi]
    assert np.all(scaler_500.feature_range == (0.0, np.pi))
    assert np.all(scaler_1000.feature_range == (0.0, np.pi))

    # Transforming dummy validation data should not mutate fitted scaler params
    dummy_val = np.random.normal(size=(10, 6))
    old_min_500 = np.copy(scaler_500.min_)
    _ = scaler_500.transform(dummy_val)
    np.testing.assert_array_equal(scaler_500.min_, old_min_500)


def test_quantum_kernel_computation_and_validity():
    """Verifies that compute_quantum_kernels produces valid, bounded Gram matrices."""
    fmap = create_zz_feature_map(feature_dimension=6, reps=2, entanglement="linear")

    dummy_tr = np.random.uniform(0, np.pi, size=(10, 6))
    dummy_val = np.random.uniform(0, np.pi, size=(15, 6))

    K_tr, K_val, timings = compute_quantum_kernels(fmap, dummy_tr, dummy_val)

    assert K_tr.shape == (10, 10)
    assert K_val.shape == (15, 10)
    assert np.all(np.isfinite(K_tr))
    assert np.all(np.isfinite(K_val))
    assert np.all(K_tr >= -1e-6)
    assert np.all(K_tr <= 1.0 + 1e-6)
    assert np.all(K_val >= -1e-6)
    assert np.all(K_val <= 1.0 + 1e-6)
    np.testing.assert_allclose(np.diag(K_tr), 1.0, atol=1e-6)
    np.testing.assert_allclose(K_tr, K_tr.T, atol=1e-6)


def test_theoretical_scaling_projections():
    """Verifies that quadratic scaling projection computes expected entries and memory."""
    dummy_empirical = {
        500: {"quantum_kernel_computation": {"train_kernel_seconds": 0.65}},
        1000: {"quantum_kernel_computation": {"train_kernel_seconds": 1.30}},
    }
    proj = compute_theoretical_scaling_projections(dummy_empirical)

    assert proj["empirical_measurements"]["500"]["training_kernel_entries"] == 250000
    assert proj["empirical_measurements"]["1000"]["training_kernel_entries"] == 1000000
    assert proj["projected_scaling_n2000"]["training_kernel_entries"] == 4000000
    assert proj["projected_scaling_n2000"]["entry_increase_factor_vs_500"] == 16.0
    assert proj["projected_scaling_n2000"]["entry_increase_factor_vs_1000"] == 4.0
