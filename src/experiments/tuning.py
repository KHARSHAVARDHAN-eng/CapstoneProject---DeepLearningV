"""
Phase 5: Genuinely Leak-Free Training Cross-Validation for Regularization Tuning.

Executes:
- Stratified 5-Fold Cross-Validation strictly inside the 500-sample training draw.
- Fold-local feature selection (SelectKBest, k=6) and angle scaling (MinMaxScaler, [0, π]).
- Fold-local quantum Gram matrix (400 x 400) and cross-kernel (100 x 400) generation.
- Exact 180 dual solver fits:
  * Linear SVM: 6 C values * 5 folds = 30 fits
  * QSVM: 6 C values * 5 folds = 30 fits
  * RBF SVM: 6 C values * 4 gamma values = 24 configs * 5 folds = 120 fits
- Zero access to the 20,000-sample validation partition during parameter selection.
"""

import logging
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import average_precision_score, f1_score

from src.data.quantum_subsets import fit_feature_selector_and_scaler, transform_features
from src.models.quantum_kernel import (
    create_zz_feature_map,
    compute_fidelity_kernel_matrix,
)
from src.models.evaluate import compute_classification_metrics

logger = logging.getLogger("quantum_fraud.tuning")


def run_leak_free_training_cross_validation(
    X_train_unselected: pd.DataFrame,
    y_train: pd.Series,
    X_val_unselected: pd.DataFrame,
    y_val: pd.Series,
    c_grid: List[float] = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0],
    gamma_grid: List[Any] = ["scale", 0.01, 0.1, 1.0],
    n_splits: int = 5,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Runs genuinely leak-free 5-fold cross-validation inside the training subset.

    Ensures that feature selection, scaling, and quantum Gram matrices are strictly
    computed per-fold using only that fold's training indices.

    Args:
        X_train_unselected: Unselected features of training subset (500 samples, 30 features).
        y_train: Binary labels of training subset (500 samples).
        X_val_unselected: Unselected features of validation partition (20,000 samples, 30 features).
        y_val: Binary labels of validation partition (20,000 samples).
        c_grid: List of candidate C values (6 candidates).
        gamma_grid: List of candidate gamma values for RBF SVM (4 candidates).
        n_splits: Number of stratified folds (default: 5).
        random_state: Random state for fold splitting.

    Returns:
        Dictionary containing fold scores, selected parameters, solver fit counts,
        and validation performance for selected models.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    y_train_arr = y_train.to_numpy()

    # Track scores per configuration
    # Linear SVM: {C: [fold1_pr_auc, ...]}
    linear_cv_scores = {c: [] for c in c_grid}
    # QSVM: {C: [fold1_pr_auc, ...]}
    qsvm_cv_scores = {c: [] for c in c_grid}
    # RBF SVM: {(C, gamma): [fold1_pr_auc, ...]}
    rbf_cv_scores = {(c, g): [] for c in c_grid for g in gamma_grid}

    total_solver_fits = 0
    fold_idx = 0

    fm = create_zz_feature_map(feature_dimension=6, reps=2, entanglement="linear")

    logger.info("Beginning leak-free %d-fold cross-validation on 500 training samples...", n_splits)

    for train_idx, val_idx in skf.split(X_train_unselected, y_train_arr):
        fold_idx += 1
        logger.info("Executing Fold %d/%d (Train: %d, Val: %d)...", fold_idx, n_splits, len(train_idx), len(val_idx))

        X_fold_tr_raw = X_train_unselected.iloc[train_idx]
        y_fold_tr = y_train_arr[train_idx]
        X_fold_val_raw = X_train_unselected.iloc[val_idx]
        y_fold_val = y_train_arr[val_idx]

        # 1. Fold-local feature selection and angle scaling
        selector_k, scaler_k, selected_cols_k = fit_feature_selector_and_scaler(
            X_fold_tr_raw, y_fold_tr, k=6
        )
        X_fold_tr_scaled = transform_features(X_fold_tr_raw, scaler_k, selected_cols_k).to_numpy()
        X_fold_val_scaled = transform_features(X_fold_val_raw, scaler_k, selected_cols_k).to_numpy()

        # 2. Fold-local Linear SVM sweep (6 fits)
        for c_val in c_grid:
            clf_lin = LinearSVC(C=c_val, class_weight="balanced", dual=False, random_state=random_state)
            clf_lin.fit(X_fold_tr_scaled, y_fold_tr)
            total_solver_fits += 1
            scores_lin = clf_lin.decision_function(X_fold_val_scaled)
            pr_auc_lin = float(average_precision_score(y_fold_val, scores_lin))
            linear_cv_scores[c_val].append(pr_auc_lin)

        # 3. Fold-local RBF SVM sweep (24 fits)
        for c_val in c_grid:
            for g_val in gamma_grid:
                clf_rbf = SVC(C=c_val, kernel="rbf", gamma=g_val, class_weight="balanced", random_state=random_state)
                clf_rbf.fit(X_fold_tr_scaled, y_fold_tr)
                total_solver_fits += 1
                scores_rbf = clf_rbf.decision_function(X_fold_val_scaled)
                pr_auc_rbf = float(average_precision_score(y_fold_val, scores_rbf))
                rbf_cv_scores[(c_val, g_val)].append(pr_auc_rbf)

        # 4. Fold-local QSVM kernel construction and sweep (6 fits)
        K_fold_tr, sv_fold_tr, _ = compute_fidelity_kernel_matrix(X_fold_tr_scaled, feature_map=fm)
        K_fold_val, _, _ = compute_fidelity_kernel_matrix(
            X_fold_val_scaled, X_fold_tr_scaled, feature_map=fm, right_statevectors=sv_fold_tr
        )

        for c_val in c_grid:
            clf_q = SVC(C=c_val, kernel="precomputed", class_weight="balanced", random_state=random_state)
            clf_q.fit(K_fold_tr, y_fold_tr)
            total_solver_fits += 1
            scores_q = clf_q.decision_function(K_fold_val)
            pr_auc_q = float(average_precision_score(y_fold_val, scores_q))
            qsvm_cv_scores[c_val].append(pr_auc_q)

    logger.info("Completed CV. Total solver fits executed: %d (expected 180)", total_solver_fits)

    # 5. Determine optimal hyperparameters based on mean internal validation PR-AUC
    best_c_linear = max(linear_cv_scores.keys(), key=lambda c: np.mean(linear_cv_scores[c]))
    best_config_rbf = max(rbf_cv_scores.keys(), key=lambda cg: np.mean(rbf_cv_scores[cg]))
    best_c_rbf, best_gamma_rbf = best_config_rbf
    best_c_qsvm = max(qsvm_cv_scores.keys(), key=lambda c: np.mean(qsvm_cv_scores[c]))

    cv_summary = {
        "linear_svm": {
            str(c): {"mean_pr_auc": float(np.mean(scores)), "std_pr_auc": float(np.std(scores))}
            for c, scores in linear_cv_scores.items()
        },
        "rbf_svm": {
            f"C={c}_gamma={g}": {"mean_pr_auc": float(np.mean(scores)), "std_pr_auc": float(np.std(scores))}
            for (c, g), scores in rbf_cv_scores.items()
        },
        "qsvm": {
            str(c): {"mean_pr_auc": float(np.mean(scores)), "std_pr_auc": float(np.std(scores))}
            for c, scores in qsvm_cv_scores.items()
        },
    }

    selected_hyperparameters = {
        "linear_svm": {"C": best_c_linear, "mean_cv_pr_auc": float(np.mean(linear_cv_scores[best_c_linear]))},
        "rbf_svm": {
            "C": best_c_rbf,
            "gamma": best_gamma_rbf,
            "mean_cv_pr_auc": float(np.mean(rbf_cv_scores[best_config_rbf])),
        },
        "qsvm": {"C": best_c_qsvm, "mean_cv_pr_auc": float(np.mean(qsvm_cv_scores[best_c_qsvm]))},
    }

    # 6. Refit selected models on the full 500-sample training set
    logger.info("Refitting selected models on full 500-sample training set...")
    sel_full, sc_full, cols_full = fit_feature_selector_and_scaler(X_train_unselected, y_train_arr, k=6)
    X_tr_full_scaled = transform_features(X_train_unselected, sc_full, cols_full).to_numpy()
    X_val_full_scaled = transform_features(X_val_unselected, sc_full, cols_full).to_numpy()
    y_val_arr = y_val.to_numpy()

    # Linear SVM refit
    best_lin_model = LinearSVC(C=best_c_linear, class_weight="balanced", dual=False, random_state=random_state)
    best_lin_model.fit(X_tr_full_scaled, y_train_arr)
    lin_val_pred = best_lin_model.predict(X_val_full_scaled)
    lin_val_score = best_lin_model.decision_function(X_val_full_scaled)
    lin_val_metrics = compute_classification_metrics(y_val_arr, lin_val_pred, lin_val_score)

    # RBF SVM refit
    best_rbf_model = SVC(
        C=best_c_rbf, kernel="rbf", gamma=best_gamma_rbf, class_weight="balanced", random_state=random_state
    )
    best_rbf_model.fit(X_tr_full_scaled, y_train_arr)
    rbf_val_pred = best_rbf_model.predict(X_val_full_scaled)
    rbf_val_score = best_rbf_model.decision_function(X_val_full_scaled)
    rbf_val_metrics = compute_classification_metrics(y_val_arr, rbf_val_pred, rbf_val_score)

    # QSVM refit
    K_tr_full, sv_full, _ = compute_fidelity_kernel_matrix(X_tr_full_scaled, feature_map=fm)
    K_val_full, _, _ = compute_fidelity_kernel_matrix(
        X_val_full_scaled, X_tr_full_scaled, feature_map=fm, right_statevectors=sv_full
    )
    best_q_model = SVC(C=best_c_qsvm, kernel="precomputed", class_weight="balanced", random_state=random_state)
    best_q_model.fit(K_tr_full, y_train_arr)
    q_val_pred = best_q_model.predict(K_val_full)
    q_val_score = best_q_model.decision_function(K_val_full)
    q_val_metrics = compute_classification_metrics(y_val_arr, q_val_pred, q_val_score)

    return {
        "n_folds": n_splits,
        "total_solver_fits": total_solver_fits,
        "cv_grid_results": cv_summary,
        "selected_hyperparameters": selected_hyperparameters,
        "validation_evaluation_with_tuned_hyperparameters": {
            "linear_svm": lin_val_metrics,
            "rbf_svm": rbf_val_metrics,
            "qsvm": q_val_metrics,
        },
    }
