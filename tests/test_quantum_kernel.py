"""
Automated tests for Phase 4B: Quantum Kernel, Subsets, Scaler, and QSVM.

Tests cover:
1. Kernel validity (symmetry, PSD, diagonal unity, bounded in [0, 1]).
2. Reproducibility of subsets and transformations.
3. Feature selection and [0, π] angle transformations without validation leakage.
4. Pilot benchmark execution and diagnostics.
5. QSVM classifier prediction output and decision scores.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC, SVC

from src.data.quantum_subsets import (
    extract_quantum_subsets,
    fit_feature_selector_and_scaler,
    transform_features,
)
from src.models.quantum_kernel import (
    QuantumKernelClassifier,
    compute_fidelity_kernel_matrix,
    create_zz_feature_map,
    run_tiny_pilot_benchmark,
    validate_kernel_matrix,
)


@pytest.fixture
def synthetic_data():
    """Generates synthetic dataset mimicking preprocessed training split."""
    rng = np.random.RandomState(42)
    n_samples = 1200
    n_features = 10
    X = pd.DataFrame(
        rng.normal(0, 1, size=(n_samples, n_features)),
        columns=[f"V{i}" for i in range(n_features)],
    )
    # 20 fraud cases (severe imbalance)
    y = pd.Series(np.zeros(n_samples, dtype=int), name="Class")
    fraud_indices = rng.choice(n_samples, size=20, replace=False)
    y.iloc[fraud_indices] = 1
    # Introduce discriminatory signal into features V0, V1 for fraud
    X.iloc[fraud_indices, 0] += 3.0
    X.iloc[fraud_indices, 1] -= 3.0
    return X, y


# =============================================================================
# 1. Quantum Subsets and Transformations Tests
# =============================================================================

def test_quantum_subsets_extraction_and_disjoint(synthetic_data):
    X, y = synthetic_data
    n_train = 50
    n_train_fraud = 10
    n_val = 100

    X_train_q, y_train_q, X_val_q, y_val_q = extract_quantum_subsets(
        X_train=X,
        y_train=y,
        n_train=n_train,
        n_train_fraud=n_train_fraud,
        n_val=n_val,
        random_state=42,
    )

    assert len(X_train_q) == n_train
    assert len(y_train_q) == n_train
    assert (y_train_q == 1).sum() == n_train_fraud
    assert (y_train_q == 0).sum() == n_train - n_train_fraud

    assert len(X_val_q) == n_val
    assert len(y_val_q) == n_val

    # Verify no sample index collision between train and val
    train_hashes = set(pd.util.hash_pandas_object(X_train_q).values)
    val_hashes = set(pd.util.hash_pandas_object(X_val_q).values)
    assert len(train_hashes.intersection(val_hashes)) == 0


def test_quantum_subsets_reproducibility(synthetic_data):
    X, y = synthetic_data
    res1 = extract_quantum_subsets(X, y, n_train=40, n_train_fraud=8, n_val=60, random_state=42)
    res2 = extract_quantum_subsets(X, y, n_train=40, n_train_fraud=8, n_val=60, random_state=42)

    pd.testing.assert_frame_equal(res1[0], res2[0])
    pd.testing.assert_series_equal(res1[1], res2[1])
    pd.testing.assert_frame_equal(res1[2], res2[2])
    pd.testing.assert_series_equal(res1[3], res2[3])


def test_feature_selection_and_angle_scaling(synthetic_data):
    X, y = synthetic_data
    X_train_q, y_train_q, X_val_q, _ = extract_quantum_subsets(
        X, y, n_train=60, n_train_fraud=10, n_val=80, random_state=42
    )

    k = 4
    selector, scaler, selected_features = fit_feature_selector_and_scaler(
        X_train_q, y_train_q, k=k
    )

    assert len(selected_features) == k
    assert isinstance(selector, SelectKBest)
    assert isinstance(scaler, MinMaxScaler)
    # The top discriminating features V0 and V1 must be selected
    assert "V0" in selected_features
    assert "V1" in selected_features

    # Test transformation
    X_train_scaled = transform_features(X_train_q, scaler, selected_features)
    X_val_scaled = transform_features(X_val_q, scaler, selected_features)

    assert X_train_scaled.shape == (60, k)
    assert X_val_scaled.shape == (80, k)

    # Values in training set must be strictly bounded in [0, π]
    assert np.all(X_train_scaled.values >= 0.0 - 1e-9)
    assert np.all(X_train_scaled.values <= np.pi + 1e-9)
    # Features must reach 0 and π on train
    assert np.allclose(X_train_scaled.min().values, 0.0, atol=1e-7)
    assert np.allclose(X_train_scaled.max().values, np.pi, atol=1e-7)


def test_no_validation_data_leakage_in_scaler(synthetic_data):
    """Verifies that scaler parameters depend exclusively on training subset data."""
    X, y = synthetic_data
    X_train_q, y_train_q, _, _ = extract_quantum_subsets(
        X, y, n_train=50, n_train_fraud=10, n_val=50, random_state=42
    )

    _, scaler, selected_features = fit_feature_selector_and_scaler(
        X_train_q, y_train_q, k=4
    )

    # Scaler min and max should match X_train_q[selected_features]
    expected_min = X_train_q[selected_features].min().values
    expected_max = X_train_q[selected_features].max().values

    np.testing.assert_allclose(scaler.data_min_, expected_min, atol=1e-7)
    np.testing.assert_allclose(scaler.data_max_, expected_max, atol=1e-7)


# =============================================================================
# 2. Quantum Kernel Matrix & Mathematics Tests
# =============================================================================

def test_zz_feature_map_structure():
    fmap = create_zz_feature_map(feature_dimension=6, reps=2, entanglement="linear")
    assert fmap.num_qubits == 6
    assert fmap.num_parameters == 6


def test_kernel_matrix_mathematical_validity():
    rng = np.random.RandomState(42)
    # 8 samples in [0, π] with 6 features
    X_scaled = rng.uniform(0, np.pi, size=(8, 6))
    fmap = create_zz_feature_map(feature_dimension=6, reps=2, entanglement="linear")

    K_train, train_sv, _ = compute_fidelity_kernel_matrix(X_scaled, feature_map=fmap)

    # 1. Dimension
    assert K_train.shape == (8, 8)
    # 2. Finite
    assert np.all(np.isfinite(K_train))
    # 3. Symmetry
    assert np.allclose(K_train, K_train.T, atol=1e-7)
    # 4. Diagonal unity
    assert np.allclose(np.diag(K_train), 1.0, atol=1e-7)
    # 5. Unit interval bounds
    assert np.all(K_train >= 0.0 - 1e-7)
    assert np.all(K_train <= 1.0 + 1e-7)
    # 6. Positive semi-definite (all eigenvalues >= -1e-7)
    eigenvalues = np.linalg.eigvalsh(K_train)
    assert np.all(eigenvalues >= -1e-7)


def test_validate_kernel_matrix_diagnostics():
    # Valid matrix
    K_valid = np.eye(4)
    diag_valid = validate_kernel_matrix(K_valid, is_symmetric=True)
    assert diag_valid["is_valid"] is True
    assert diag_valid["is_symmetric"] is True
    assert diag_valid["diagonal_is_one"] is True
    assert diag_valid["is_positive_semi_definite"] is True

    # Invalid: non-symmetric
    K_asym = np.array([[1.0, 0.5], [0.1, 1.0]])
    diag_asym = validate_kernel_matrix(K_asym, is_symmetric=True)
    assert diag_asym["is_symmetric"] is False
    assert diag_asym["is_valid"] is False

    # Invalid: non-unit diagonal
    K_bad_diag = np.array([[0.5, 0.2], [0.2, 0.8]])
    diag_bad = validate_kernel_matrix(K_bad_diag, is_symmetric=True)
    assert diag_bad["diagonal_is_one"] is False
    assert diag_bad["is_valid"] is False


def test_tiny_pilot_benchmark():
    rng = np.random.RandomState(42)
    X_pilot = rng.uniform(0, np.pi, size=(10, 6))

    benchmark = run_tiny_pilot_benchmark(
        X_train_pilot=X_pilot,
        full_n_train=500,
        full_n_val=1000,
        n_pilot=10,
        random_state=42,
    )

    assert benchmark["pilot_sample_size"] == 10
    assert benchmark["pilot_elapsed_seconds"] > 0
    assert benchmark["validation_diagnostics"]["is_valid"] is True
    assert "projected_full_experiment" in benchmark
    assert benchmark["projected_full_experiment"]["estimated_total_runtime_seconds"] > 0


# =============================================================================
# 3. QSVM Classifier and Controls Prediction Tests
# =============================================================================

def test_qsvm_classifier_fit_predict():
    rng = np.random.RandomState(42)
    X_tr = rng.uniform(0, np.pi, size=(16, 6))
    y_tr = np.array([0] * 12 + [1] * 4)
    X_test = rng.uniform(0, np.pi, size=(8, 6))

    qsvm = QuantumKernelClassifier(C=1.0, class_weight="balanced", random_state=42)
    qsvm.fit(X_tr, y_tr)

    preds = qsvm.predict(X_test)
    scores = qsvm.decision_function(X_test)

    assert len(preds) == 8
    assert len(scores) == 8
    assert set(np.unique(preds)).issubset({0, 1})
    assert np.all(np.isfinite(scores))


def test_matched_classical_controls_interface():
    rng = np.random.RandomState(42)
    X_tr = rng.uniform(0, np.pi, size=(16, 6))
    y_tr = np.array([0] * 12 + [1] * 4)
    X_test = rng.uniform(0, np.pi, size=(8, 6))

    lin = LinearSVC(C=1.0, class_weight="balanced", dual=False, random_state=42)
    lin.fit(X_tr, y_tr)
    lin_pred = lin.predict(X_test)
    lin_score = lin.decision_function(X_test)

    rbf = SVC(C=1.0, kernel="rbf", class_weight="balanced", random_state=42)
    rbf.fit(X_tr, y_tr)
    rbf_pred = rbf.predict(X_test)
    rbf_score = rbf.decision_function(X_test)

    assert len(lin_pred) == 8
    assert len(rbf_pred) == 8
    assert len(lin_score) == 8
    assert len(rbf_score) == 8


# =============================================================================
# 4. Audit & Verification Tests (Phase 4B Audit Requirements)
# =============================================================================

def test_kernel_implementation_consistency_pure_fidelity():
    """Audits kernel implementation consistency: verifies exact pure-state fidelity overlap."""
    rng = np.random.RandomState(42)
    X = rng.uniform(0, np.pi, size=(5, 6))
    fmap = create_zz_feature_map(feature_dimension=6, reps=2, entanglement="linear")

    K_matrix, statevectors, _ = compute_fidelity_kernel_matrix(X, feature_map=fmap)

    # Recompute manual pure-state fidelities: |<psi_i | psi_j>|^2
    from qiskit.quantum_info import Statevector
    manual_svs = [Statevector.from_instruction(fmap.assign_parameters(x)).data for x in X]

    for i in range(len(X)):
        for j in range(len(X)):
            expected_fidelity = float(np.abs(np.vdot(manual_svs[j], manual_svs[i])) ** 2)
            assert np.isclose(K_matrix[i, j], expected_fidelity, atol=1e-7)


def test_cross_kernel_matrix_shapes_and_boundedness():
    """Audits cross-kernel matrix rectangular shape, finite check, and range [0, 1]."""
    rng = np.random.RandomState(42)
    n_left = 35
    n_right = 15
    d = 6
    X_left = rng.uniform(0, np.pi, size=(n_left, d))
    X_right = rng.uniform(0, np.pi, size=(n_right, d))
    fmap = create_zz_feature_map(feature_dimension=d, reps=2, entanglement="linear")

    K_cross, _, _ = compute_fidelity_kernel_matrix(X_left=X_left, X_right=X_right, feature_map=fmap)

    assert K_cross.shape == (n_left, n_right)
    assert np.all(np.isfinite(K_cross))
    assert np.all(K_cross >= 0.0 - 1e-7)
    assert np.all(K_cross <= 1.0 + 1e-7)

    diag = validate_kernel_matrix(K_cross, is_symmetric=False)
    assert diag["is_valid"] is True
    assert diag["is_finite"] is True


def test_preprocessing_leakage_audit_no_val_target(synthetic_data):
    """Audits data leakage: verifies that validation labels and data have zero effect on training selector/scaler."""
    X, y = synthetic_data

    # Draw two different validation sets with different targets from remaining pool
    X_tr, y_tr, X_val_1, y_val_1 = extract_quantum_subsets(
        X, y, n_train=40, n_train_fraud=8, n_val=50, random_state=42
    )
    # Fit selector and scaler strictly on training set
    sel_1, sc_1, feat_1 = fit_feature_selector_and_scaler(X_tr, y_tr, k=4)

    # Perturb validation set drastically (mocking an alternate validation sample)
    X_val_perturbed = X_val_1 * 100.0
    y_val_perturbed = 1 - y_val_1

    # Verify that transforming the perturbed validation data does not mutate selector or scaler params
    _ = transform_features(X_val_perturbed, sc_1, feat_1)

    np.testing.assert_allclose(sc_1.data_min_, X_tr[feat_1].min().values, atol=1e-7)
    np.testing.assert_allclose(sc_1.data_max_, X_tr[feat_1].max().values, atol=1e-7)


def test_matched_evaluation_samples_consistency(synthetic_data):
    """Audits that QSVM, Linear SVM, and RBF SVM are evaluated on identical matched samples."""
    X, y = synthetic_data
    X_tr_q, y_tr_q, X_val_q, y_val_q = extract_quantum_subsets(
        X, y, n_train=40, n_train_fraud=8, n_val=60, random_state=42
    )

    selector, scaler, selected_features = fit_feature_selector_and_scaler(X_tr_q, y_tr_q, k=4)
    X_tr_scaled = transform_features(X_tr_q, scaler, selected_features)
    X_val_scaled = transform_features(X_val_q, scaler, selected_features)

    # Train QSVM
    qsvm = QuantumKernelClassifier(C=1.0, class_weight="balanced", random_state=42)
    qsvm.fit(X_tr_scaled, y_tr_q)
    q_pred = qsvm.predict(X_val_scaled)
    q_score = qsvm.decision_function(X_val_scaled)

    # Train Linear SVM
    lin = LinearSVC(C=1.0, class_weight="balanced", dual=False, random_state=42)
    lin.fit(X_tr_scaled, y_tr_q)
    lin_pred = lin.predict(X_val_scaled)
    lin_score = lin.decision_function(X_val_scaled)

    # Train RBF SVM
    rbf = SVC(C=1.0, kernel="rbf", class_weight="balanced", random_state=42)
    rbf.fit(X_tr_scaled, y_tr_q)
    rbf_pred = rbf.predict(X_val_scaled)
    rbf_score = rbf.decision_function(X_val_scaled)

    # All predictions and scores must have the exact same shape matching len(X_val_q)
    assert len(q_pred) == len(lin_pred) == len(rbf_pred) == len(y_val_q) == 60
    assert len(q_score) == len(lin_score) == len(rbf_score) == 60
    assert set(np.unique(q_pred)).issubset({0, 1})
    assert set(np.unique(lin_pred)).issubset({0, 1})
    assert set(np.unique(rbf_pred)).issubset({0, 1})


def test_saved_artifacts_and_predictions_consistency():
    """Audits persisted Phase 4B artifacts: predictions CSV, metrics JSON, kernel matrices, and models."""
    import joblib
    import json
    from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

    preds_path = Path("data/processed/quantum/predictions_validation.csv")
    metrics_path = Path("reports/quantum/experiment_metrics.json")
    kernels_path = Path("models/quantum/kernel_matrices.npz")
    qsvm_path = Path("models/quantum/qsvm_model.joblib")
    linear_path = Path("models/quantum/matched_linear_svm.joblib")
    rbf_path = Path("models/quantum/matched_rbf_svm.joblib")
    x_val_path = Path("data/processed/quantum/X_val_q_scaled.csv")

    if not (
        preds_path.exists()
        and metrics_path.exists()
        and kernels_path.exists()
        and qsvm_path.exists()
        and linear_path.exists()
        and rbf_path.exists()
        and x_val_path.exists()
    ):
        pytest.skip("Persisted Phase 4B artifacts not found; skipping artifact audit test.")

    df = pd.read_csv(preds_path)
    assert len(df) == 20000
    assert (df["True_Class"] == 0).sum() == 19967
    assert (df["True_Class"] == 1).sum() == 33
    assert set(df.columns) == {
        "True_Class",
        "QSVM_Pred",
        "QSVM_Score",
        "Linear_SVM_Pred",
        "Linear_SVM_Score",
        "RBF_SVM_Pred",
        "RBF_SVM_Score",
    }

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    y_true = df["True_Class"].to_numpy()

    # 1. Verify confusion matrices and metrics for all 3 models against saved JSON
    for model_key, pred_col in [
        ("qsvm", "QSVM_Pred"),
        ("linear_svm", "Linear_SVM_Pred"),
        ("rbf_svm", "RBF_SVM_Pred"),
    ]:
        cm = confusion_matrix(y_true, df[pred_col].to_numpy(), labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        expected_cm = metrics["results"][model_key]["confusion_matrix"]
        assert tn == expected_cm["true_negatives"]
        assert fp == expected_cm["false_positives"]
        assert fn == expected_cm["false_negatives"]
        assert tp == expected_cm["true_positives"]

        calc_rec = recall_score(y_true, df[pred_col].to_numpy(), pos_label=1)
        calc_prec = precision_score(y_true, df[pred_col].to_numpy(), pos_label=1)
        calc_f1 = f1_score(y_true, df[pred_col].to_numpy(), pos_label=1)
        assert np.isclose(calc_rec, metrics["results"][model_key]["recall"], atol=1e-4)
        assert np.isclose(calc_prec, metrics["results"][model_key]["precision"], atol=1e-4)
        assert np.isclose(calc_f1, metrics["results"][model_key]["f1_score"], atol=1e-4)

    # 2. Verify kernel matrices shapes
    npz = np.load(kernels_path)
    assert npz["K_train"].shape == (500, 500)
    assert npz["K_val"].shape == (20000, 500)

    # 3. Verify model re-prediction equivalence
    qsvm = joblib.load(qsvm_path)
    np.testing.assert_array_equal(qsvm.predict(npz["K_val"]), df["QSVM_Pred"].to_numpy())

    X_val_scaled = pd.read_csv(x_val_path)
    lin_model = joblib.load(linear_path)
    np.testing.assert_array_equal(lin_model.predict(X_val_scaled.values), df["Linear_SVM_Pred"].to_numpy())

    rbf_model = joblib.load(rbf_path)
    np.testing.assert_array_equal(rbf_model.predict(X_val_scaled.values), df["RBF_SVM_Pred"].to_numpy())
