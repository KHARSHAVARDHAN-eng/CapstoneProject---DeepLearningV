"""
Unit and integration tests for Phase 6 Final Holdout Evaluation and Capstone Synthesis.

Validates:
1. Frozen artifact restoration and configuration manifest verification against Phase 5.
2. Six-feature ordering and leak-free preprocessing (transform() only, zero test-set fitting).
3. Training statevector fidelity kernel matching saved K_train.
4. Batch boundary partitioning, rectangular kernel dimensions, and [0, 1] numerical boundedness.
5. Small pilot benchmark diagnostics and throughput calculation.
6. Multi-model holdout evaluation, cost-scenario loss calculation, and zero test-tuning compliance.
7. Checkpoint persistence and resumption behavior.
"""

import json
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from src.experiments.phase6_evaluation import (
    benchmark_small_test_batch,
    compute_batched_test_kernel,
    compute_file_sha256,
    compute_illustrative_scenario_loss,
    recalibrate_validation_cost_optimal_cutoffs,
    verify_and_freeze_configuration_manifest,
)
from src.models.quantum_kernel import (
    compute_statevector_matrix,
    create_zz_feature_map,
    validate_kernel_matrix,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_frozen_artifact_restoration_and_manifest_verification():
    """Verifies that Phase 5 tuned models, scalers, selectors, and manifests match exactly."""
    models_dir = PROJECT_ROOT / "models" / "quantum"
    data_dir = PROJECT_ROOT / "data" / "processed" / "quantum"
    reports_dir = PROJECT_ROOT / "reports"

    manifest = verify_and_freeze_configuration_manifest(
        models_dir=models_dir,
        data_dir=data_dir,
        reports_dir=reports_dir,
    )

    # 1. Verify manifest structure
    assert manifest["status"] == "FROZEN_VALIDATION_CALIBRATED"
    assert manifest["manifest_version"] == "2.0 (Threshold Corrected)"
    assert "models" in manifest
    assert "preprocessing" in manifest
    assert "frozen_decision_cutoffs" in manifest
    assert "audit_provenance_trail" in manifest

    # 2. Check tuned hyperparameters
    assert manifest["models"]["tuned_linear_svm"]["hyperparameters"]["C"] == 5.0
    assert manifest["models"]["tuned_rbf_svm"]["hyperparameters"]["C"] == 5.0
    assert manifest["models"]["tuned_rbf_svm"]["hyperparameters"]["gamma"] == 0.01
    assert manifest["models"]["tuned_qsvm"]["hyperparameters"]["C"] == 0.5
    assert manifest["models"]["tuned_qsvm"]["hyperparameters"]["kernel"] == "precomputed"

    # 3. Verify validation-calibrated cutoffs in manifest
    val_cutoffs = manifest["frozen_decision_cutoffs"]["cost_optimal_by_scenario_ratio"]
    assert np.isclose(val_cutoffs["linear_svm"][5], 1.0246, atol=1e-3)
    assert np.isclose(val_cutoffs["linear_svm"][20], 1.0246, atol=1e-3)
    assert np.isclose(val_cutoffs["rbf_svm"][5], 0.4656, atol=1e-3)
    assert np.isclose(val_cutoffs["rbf_svm"][20], 0.4656, atol=1e-3)
    assert np.isclose(val_cutoffs["qsvm"][5], 0.8651, atol=1e-3)
    assert np.isclose(val_cutoffs["qsvm"][20], 0.7180, atol=1e-3)
    assert np.isclose(val_cutoffs["qsvm"][50], 0.7180, atol=1e-3)
    assert np.isclose(val_cutoffs["qsvm"][100], 0.7180, atol=1e-3)

    # 4. Verify historical baseline disclosure
    hist_model = manifest["models"]["historical_classical_svm_30f"]
    assert hist_model["n_features"] == 30
    assert hist_model["holdout_evaluation_status"] == "HISTORICAL_BASELINE_REUSE_DISCLOSED"
    assert "Phase 3" in hist_model["disclosure"]


def test_recalibrate_validation_cost_optimal_cutoffs_execution():
    """Verifies that validation-set cutoff recalibration executes without holdout data access."""
    models_dir = PROJECT_ROOT / "models" / "quantum"
    data_dir = PROJECT_ROOT / "data" / "processed" / "quantum"

    cutoffs, diags = recalibrate_validation_cost_optimal_cutoffs(
        data_dir=data_dir,
        models_dir=models_dir,
        cost_ratios=[5.0, 20.0, 50.0, 100.0],
    )

    assert "linear_svm" in cutoffs
    assert "rbf_svm" in cutoffs
    assert "qsvm" in cutoffs

    for mod in ["linear_svm", "rbf_svm", "qsvm"]:
        for r in [5, 20, 50, 100]:
            assert r in cutoffs[mod]
            assert isinstance(cutoffs[mod][r], float)
            assert np.isfinite(cutoffs[mod][r])


def test_preprocessing_isolation_and_feature_ordering():
    """Verifies 6-feature ordering, target scaling range, and zero-fit compliance."""
    selector_path = PROJECT_ROOT / "data" / "processed" / "quantum" / "feature_selector.joblib"
    scaler_path = PROJECT_ROOT / "data" / "processed" / "quantum" / "angle_scaler.joblib"

    selector = joblib.load(selector_path)
    scaler = joblib.load(scaler_path)

    # Verify 6-feature names and ordering
    selected_features = selector.get_feature_names_out().tolist()
    expected_features = ["V4", "V10", "V11", "V12", "V14", "V16"]
    assert selected_features == expected_features, f"Feature ordering mismatch: {selected_features}"

    # Verify scaling bounds
    assert float(scaler.feature_range[0]) == 0.0
    assert np.isclose(scaler.feature_range[1], np.pi)

    # Ensure transforming dummy test data does NOT modify fitted scaler parameters
    old_min = np.copy(scaler.min_)
    old_scale = np.copy(scaler.scale_)

    dummy_test = pd.DataFrame(
        np.random.normal(loc=10.0, scale=5.0, size=(20, 30)),
        columns=[f"V{i}" if i != 0 and i != 29 else ("Time" if i == 0 else "Amount") for i in range(30)],
    )
    dummy_sel = pd.DataFrame(selector.transform(dummy_test), columns=selected_features)
    dummy_scaled = scaler.transform(dummy_sel)

    assert dummy_scaled.shape == (20, 6)
    np.testing.assert_array_equal(scaler.min_, old_min)
    np.testing.assert_array_equal(scaler.scale_, old_scale)


def test_training_statevector_kernel_fidelity_consistency():
    """Verifies that precomputing training statevectors exactly reproduces saved K_train."""
    fmap = create_zz_feature_map(feature_dimension=6, reps=2, entanglement="linear")
    X_train_q_scaled = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "quantum" / "X_train_q_scaled.csv").to_numpy()

    # Precompute statevectors
    sv_train = compute_statevector_matrix(fmap, X_train_q_scaled)
    assert sv_train.shape == (500, 64)

    # Pairwise inner product Gram matrix
    K_train_calc = np.abs(sv_train @ sv_train.conj().T) ** 2
    np.fill_diagonal(K_train_calc, 1.0)
    K_train_calc = 0.5 * (K_train_calc + K_train_calc.T)

    km = np.load(PROJECT_ROOT / "models" / "quantum" / "kernel_matrices.npz")
    K_train_saved = km["K_train"]

    max_diff = float(np.max(np.abs(K_train_calc - K_train_saved)))
    assert max_diff < 1e-10, f"Statevectors do not reproduce saved K_train: discrepancy={max_diff}"


def test_batch_boundaries_and_rectangular_kernel_dimensions():
    """Verifies batched quantum kernel computation handles arbitrary batch boundaries and valid shapes."""
    fmap = create_zz_feature_map(feature_dimension=6, reps=2, entanglement="linear")
    dummy_train = np.random.uniform(0, np.pi, size=(10, 6))
    dummy_test = np.random.uniform(0, np.pi, size=(25, 6))

    sv_train = compute_statevector_matrix(fmap, dummy_train)

    # Test batching with batch_size=10 (chunks: 10, 10, 5)
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "test_kernel_chunk.npy"
        K_batched, perf_meta = compute_batched_test_kernel(
            X_test_scaled=dummy_test,
            sv_train=sv_train,
            fmap=fmap,
            batch_size=10,
            checkpoint_path=ckpt_path,
            force_recompute=True,
        )

        assert K_batched.shape == (25, 10)
        assert np.all(np.isfinite(K_batched))
        assert np.all(K_batched >= -1e-6)
        assert np.all(K_batched <= 1.0 + 1e-6)
        assert ckpt_path.exists()

        # Test resumption from checkpoint
        K_resumed, perf_resumed = compute_batched_test_kernel(
            X_test_scaled=dummy_test,
            sv_train=sv_train,
            fmap=fmap,
            batch_size=10,
            checkpoint_path=ckpt_path,
            force_recompute=False,
        )
        assert perf_resumed["source"] == "restored_from_checkpoint"
        np.testing.assert_array_equal(K_batched, K_resumed)


def test_pilot_benchmark_numerical_validity():
    """Verifies that pilot benchmarking computes finite, [0, 1] bounded metrics and non-zero throughput."""
    fmap = create_zz_feature_map(feature_dimension=6, reps=2, entanglement="linear")
    dummy_train = np.random.uniform(0, np.pi, size=(20, 6))
    dummy_test = np.random.uniform(0, np.pi, size=(15, 6))

    sv_train = compute_statevector_matrix(fmap, dummy_train)
    bench_results = benchmark_small_test_batch(
        X_test_scaled=dummy_test,
        sv_train=sv_train,
        fmap=fmap,
        n_bench=5,
    )

    assert bench_results["n_samples"] == 5
    assert bench_results["measured_throughput_samples_per_sec"] > 0
    assert bench_results["total_elapsed_seconds"] > 0
    assert bench_results["validation_diagnostics"]["is_valid"] is True
    assert bench_results["validation_diagnostics"]["min_value"] >= -1e-6
    assert bench_results["validation_diagnostics"]["max_value"] <= 1.0 + 1e-6


def test_illustrative_scenario_loss_calculation():
    """Verifies the standard capstone scenario loss formula L = c_fp * FP + (R * c_fp) * FN."""
    # Example: FP = 10, FN = 5, R = 20, c_fp = 10.0
    # c_fn = 20 * 10 = 200
    # Loss = 10 * 10 + 200 * 5 = 100 + 1000 = 1100
    loss_r20 = compute_illustrative_scenario_loss(fp=10, fn=5, cost_ratio_R=20.0, c_fp=10.0)
    assert loss_r20 == 1100.0

    # Example: FP = 0, FN = 2, R = 50, c_fp = 10.0 -> Loss = 500 * 2 = 1000
    loss_r50 = compute_illustrative_scenario_loss(fp=0, fn=2, cost_ratio_R=50.0, c_fp=10.0)
    assert loss_r50 == 1000.0


def test_sha256_provenance_calculation():
    """Verifies SHA-256 calculation for file integrity checking."""
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(b"quantum_fraud_detection_test_content")
        f_path = Path(f.name)

    try:
        h1 = compute_file_sha256(f_path)
        assert len(h1) == 64
        assert isinstance(h1, str)
    finally:
        f_path.unlink()


def test_phase6_deliverables_and_metrics_integrity():
    """Verifies all Phase 6 deliverables, predictions file structure, and metrics consistency."""
    from sklearn.metrics import confusion_matrix, precision_score, recall_score
    # 1. Manifest
    manifest_path = PROJECT_ROOT / "reports" / "phase6_configuration_manifest.json"
    assert manifest_path.exists(), "Manifest missing"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    assert manifest["status"] == "FROZEN_VALIDATION_CALIBRATED"
    assert manifest["manifest_version"] == "2.0 (Threshold Corrected)"

    # 2. Kernel Matrix Checkpoint
    k_test_path = PROJECT_ROOT / "data" / "processed" / "quantum" / "K_test.npy"
    assert k_test_path.exists(), "K_test.npy missing"
    k_test = np.load(k_test_path)
    assert k_test.shape == (56746, 500)
    assert np.all(np.isfinite(k_test))
    assert np.all(k_test >= -1e-6)
    assert np.all(k_test <= 1.0 + 1e-6)

    # 3. Predictions File
    preds_path = PROJECT_ROOT / "data" / "processed" / "quantum" / "predictions_test.csv"
    assert preds_path.exists(), "predictions_test.csv missing"
    preds_df = pd.read_csv(preds_path)
    assert len(preds_df) == 56746
    assert "y_true" in preds_df.columns
    assert "score_classical_svm_30f" in preds_df.columns
    assert "score_linear_svm_6f" in preds_df.columns
    assert "score_rbf_svm_6f" in preds_df.columns
    assert "score_qsvm_6f" in preds_df.columns
    assert int(preds_df["y_true"].sum()) == 95

    # Check presence of corrected and initial audit prediction columns
    for mod_key in ["linear_svm_6f", "rbf_svm_6f", "qsvm_6f"]:
        for r in [5, 20, 50, 100]:
            assert f"pred_{mod_key}_cost_opt_R{r}" in preds_df.columns
            assert f"pred_{mod_key}_initial_audit_cost_opt_R{r}" in preds_df.columns

    # 4. Metrics JSON
    metrics_path = PROJECT_ROOT / "reports" / "phase6_test_metrics.json"
    assert metrics_path.exists(), "phase6_test_metrics.json missing"
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics_data = json.load(f)
    assert "metrics_results" in metrics_data
    assert "qsvm_6f" in metrics_data["metrics_results"]
    assert "rbf_svm_6f" in metrics_data["metrics_results"]
    assert "linear_svm_6f" in metrics_data["metrics_results"]
    assert "classical_svm_30f" in metrics_data["metrics_results"]

    # 5. Bit-for-bit consistency between predictions_test.csv and phase6_test_metrics.json
    y_true_test = preds_df["y_true"].to_numpy()
    for mod_key in ["linear_svm_6f", "rbf_svm_6f", "qsvm_6f"]:
        for r in [5, 20, 50, 100]:
            col_pred = preds_df[f"pred_{mod_key}_cost_opt_R{r}"].to_numpy()
            ev = metrics_data["metrics_results"][mod_key]["frozen_cost_optimal_evaluation"][str(r)]

            tn, fp, fn, tp = confusion_matrix(y_true_test, col_pred, labels=[0, 1]).ravel()
            prec = float(precision_score(y_true_test, col_pred, pos_label=1, zero_division=0))
            rec = float(recall_score(y_true_test, col_pred, pos_label=1, zero_division=0))
            loss = compute_illustrative_scenario_loss(fp, fn, float(r))

            assert int(tn) == ev["confusion_matrix"]["true_negatives"]
            assert int(fp) == ev["confusion_matrix"]["false_positives"]
            assert int(fn) == ev["confusion_matrix"]["false_negatives"]
            assert int(tp) == ev["confusion_matrix"]["true_positives"]
            assert np.isclose(prec, ev["precision"], atol=1e-4)
            assert np.isclose(rec, ev["recall"], atol=1e-4)
            assert np.isclose(loss, ev["illustrative_scenario_loss"], atol=1e-4)

    # 6. Figures
    fig_dir = PROJECT_ROOT / "reports" / "figures" / "phase6"
    expected_figures = [
        "holdout_precision_recall_curves.png",
        "holdout_roc_curves.png",
        "holdout_confusion_matrices.png",
        "holdout_cost_comparison.png",
        "inference_latency_comparison.png",
    ]
    for fig_name in expected_figures:
        assert (fig_dir / fig_name).exists(), f"Missing figure: {fig_name}"

    # 7. Synthesis Report
    report_path = PROJECT_ROOT / "reports" / "phase6_final_evaluation_and_synthesis_report.md"
    assert report_path.exists(), "Synthesis report missing"
    report_text = report_path.read_text(encoding="utf-8")
    assert "Prior Test Set Evaluation Disclosure (No Clean Untouched Holdout Claim)" in report_text
    assert "Validation Subset Reuse & Optimism Warning" in report_text
    assert "Threshold Calibration Provenance & Versioned Audit Trail" in report_text
    assert "Small Fraud-Case Count in Holdout (95 Fraud Cases)" in report_text
    assert "Absence of Physical Quantum Hardware Noise Evaluation" in report_text

