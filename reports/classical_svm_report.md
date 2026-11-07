# Phase 3: Classical SVM Baseline Evaluation Report

**Project Title**: Quantum Machine Learning for Financial Fraud Detection using Quantum Kernel Methods  
**Phase**: Phase 3 — Classical SVM Baseline  
**Model Type**: Support Vector Machine (`kernel='rbf', C=1.0`)  
**Class Weight Configuration**: `balanced`  
**Random Seed**: `42`  

---

## 1. Executive Summary & Runtime Profile

- **Training Set Size**: 226,980 samples (378 Fraud, 0.1665%)
- **Test Set Size**: 56,746 samples (95 Fraud, 0.1674%)
- **Training Time**: **59.83 seconds**
- **Prediction / Inference Time**: **37.6865 seconds** (1,506 transactions/sec)

## 2. Test Set Evaluation Metrics (Positive Class = Fraud)

| Evaluation Metric | Score | Interpretation |
| :--- | :---: | :--- |
| **Precision (PPV)** | **0.1970** | Percentage of predicted frauds that are actual frauds |
| **Recall (Sensitivity)** | **0.6947** | Percentage of actual frauds successfully caught |
| **F1-Score** | **0.3070** | Harmonic mean of precision and recall |
| **PR-AUC (Average Precision)** | **0.4138** | Area under Precision-Recall Curve (primary imbalanced metric) |
| **ROC-AUC** | **0.9651** | Area under Receiver Operating Characteristic curve |
| **Specificity (TNR)** | **0.9953** | Percentage of legitimate transactions correctly classified |
| **Balanced Accuracy** | **0.8450** | Unweighted average of recall and specificity |

## 3. Confusion Matrix Breakdown

| | Predicted Legitimate (0) | Predicted Fraudulent (1) | Total |
| :--- | :---: | :---: | :---: |
| **Actual Legitimate (0)** | **56,382** (TN) | **269** (FP) | 56,651 |
| **Actual Fraudulent (1)** | **29** (FN) | **66** (TP) | 95 |
| **Total** | 56,411 | 335 | 56,746 |

```text
precision    recall  f1-score   support

Legitimate (0)     0.9995    0.9953    0.9974     56651
     Fraud (1)     0.1970    0.6947    0.3070        95

      accuracy                         0.9947     56746
     macro avg     0.5983    0.8450    0.6522     56746
  weighted avg     0.9981    0.9947    0.9962     56746
```

## 4. Class Imbalance Strategy Evaluation (Internal Training CV)

To prevent data leakage, `class_weight='balanced'` was evaluated against the unweighted configuration strictly on the training partition:

| Strategy | Mean Recall | Mean Precision | Mean F1 | Mean PR-AUC | Mean Fit Time |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `unweighted` | 0.5846 | 0.8741 | 0.6990 | 0.7622 | 0.41s |
| `balanced` | 0.9021 | 0.0702 | 0.1303 | 0.7350 | 0.68s |

> **Key Trade-off**: `class_weight='balanced'` applies an inverse frequency penalty (~578x for fraud errors), boosting recall at the cost of precision. Unweighted SVM maximizes precision and F1 by penalizing false alarms.

## 5. Artifacts and Visualizations

- **Trained Model**: `models/classical_svm.joblib`
- **Model Config**: `models/classical_svm_config.json`
- **Evaluation Metrics**: `reports/classical_svm_metrics.json`
- **Confusion Matrix Plot**: `reports/figures/classical_svm_confusion_matrix.png`
- **Precision-Recall Curve**: `reports/figures/classical_svm_pr_curve.png`
- **ROC Curve**: `reports/figures/classical_svm_roc_curve.png`

## 6. Significance as Baseline for Quantum Kernel Methods

1. **Mathematical Parallelism with Quantum Kernel SVM (QSVM)**:
   - The classical RBF SVM optimizes a dual quadratic program over the classical Gram matrix $K_{ij} = \exp(-\gamma ||x_i - x_j||^2)$.
   - The Quantum Kernel SVM replaces this classical kernel with a quantum state fidelity kernel $K_{ij} = |\langle \phi(x_i) | \phi(x_j) \rangle|^2$ computed via parameterized quantum circuits.
   - By establishing this classical RBF SVM baseline on the exact same dataset splits, we establish a rigorous classical benchmark for later comparison with quantum kernels.

2. **Computational Complexity Benchmark**:
   - Classical RBF SVM scales as $O(N^2)$ to $O(N^3)$ in sample size ($N = 226,980$).
   - Quantum kernel matrix calculation on current NISQ simulators also scales as $O(N^2)$ quantum circuit executions, highlighting the necessity of dimensionality reduction and sub-sampling for quantum algorithms.
