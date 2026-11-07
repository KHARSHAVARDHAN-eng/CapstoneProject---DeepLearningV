# Phase 4B: Quantum Kernel SVM Audit & Matched Classical Benchmarking Report

**Project Title**: Quantum Machine Learning for Financial Fraud Detection using Quantum Kernel Methods  
**Phase**: Phase 4B — Comprehensive Audit, Validation Reliability Expansion & Matched Classical Evaluation  
**Status**: Audited and Verified  
**Generated At**: 2026-10-11T16:02:11Z  

---

## 1. Executive Summary & Audit Overview

This report documents the rigorous audit and correction of **Phase 4B** for our capstone project. Following initial pilot testing, an audit revealed that evaluating models on only 1,000 validation samples (which contained only 2 positive fraud cases) suffered from severe small-sample variance and could not support reliable scientific conclusions.

To address this, the evaluation protocol was audited and expanded:
1. **Validation Reliability & Preliminary Scope**: Expanded the validation set to **20,000 samples** containing **33 ground-truth fraud cases** (19,967 legitimate transactions), preserving natural class prevalence (0.165% fraud rate) strictly from the remaining training partition. Findings are explicitly treated as **preliminary** due to the modest sample of 33 fraud cases, avoiding claims of definitive generalization.
2. **Implementation Verification**: Verified that the quantum kernel evaluates pure-state fidelity $K(x, x') = |\langle \phi(x')|\phi(x)\rangle|^2$ using exact statevector simulation with $O(N)$ statevector caching, producing mathematically exact, noiseless Gram matrices ($K \succeq 0$, $K_{ii} = 1.0$, $K_{ij} \in [0, 1]$).
3. **Clarification of Test-Set Status**:
   - The 56,746-sample holdout test set (`X_test.csv`) was previously evaluated in **Phase 3** to establish the global 30-feature classical baseline benchmark.
   - For **Phase 4B**, the holdout test set was **strictly not accessed, evaluated, or used for model tuning, feature selection, or scaling**. It is reserved strictly for final capstone evaluation, and was not used for model selection in Phase 4B.
4. **Matched Classical Controls**: Evaluated classical Linear SVM and RBF SVM models on the identical 500 training samples, 6 transformed features, and 20,000 validation samples.

---

## 2. Kernel Implementation Verification & Granular Timing Analysis

### Exact Method Specification:
- **Feature Map**: 6-qubit `ZZFeatureMap` with 2 repetitions and linear entanglement ($R_Z$ rotations on single qubits, $ZZ$ phase interactions on adjacent pairs).
- **Simulation Method**: **Exact Statevector Simulation** (Analytic statevector calculation via `qiskit.quantum_info.Statevector`).
- **Finite-Shot Sampling Audit**: Finite-shot sampling (e.g. `StatevectorSampler(shots=1024)`) was evaluated during the pilot benchmark. For the full 20,000-sample cross-kernel matrix ($20,000 \times 500 = 10,000,000$ entries), pairwise circuit execution with 1,024 shots would require 10 million quantum circuit compiles, taking **$\sim 13.9$ hours**.
- **Statevector Caching Justification**: Because statevectors scale linearly $O(N)$, precomputing statevectors for each sample (500 train + 20,000 validation) requires only 20,500 statevector evaluations, followed by BLAS complex matrix multiplication:
  $$K_{ij} = |\langle \phi(x_i) | \phi(x_j) \rangle|^2 = |\Psi_{i,:} \Psi_{j,:}^\dagger|^2$$
  - Actual measured training Gram matrix computation ($500 \times 500$): **0.771 seconds**
  - Actual measured validation cross-kernel computation (20,000 $\times$ 500): **27.182 seconds**
  - Result: Noiseless, positive semi-definite ($K \succeq 0$), and mathematically exact pure-state fidelity.

### Granular Computational Timing Breakdown:

To ensure scientific transparency, the computational operations are separated into kernel construction, solver fitting, and inference:

| Stage / Operation | QSVM (ZZ Kernel) | Classical Linear SVM | Classical RBF SVM | What Operation Is Included |
| :--- | :---: | :---: | :---: | :--- |
| **Training Kernel Generation** | **0.7712 s** | N/A | N/A (Internal) | Precomputing 500 statevectors + Gram matrix inner products |
| **Quadratic Programming Solver Fit** | **0.0051 s** | **0.0032 s** | **0.0020 s** | Dual problem optimization (`SVC.fit` / `LinearSVC.fit`) |
| **Total Training Pipeline** | **0.7763 s** | **0.0032 s** | **0.0020 s** | Full end-to-end training pipeline wall-clock time |
| **Evaluation Cross-Kernel Generation** | **27.1819 s** | N/A | N/A (Internal) | Precomputing 20,000 statevectors + $20,000 \times 500$ matrix multiply |
| **Decision Function Evaluation** | **0.0388 s** | **0.0016 s** | **0.1089 s** | `predict` and `decision_function` calculation |
| **Total Inference Pipeline (20,000 samples)** | **27.2207 s** | **0.0016 s** | **0.1089 s** | Full end-to-end inference pipeline wall-clock time |

> **Timing Audit Note**: Directly comparing classifier solver fitting time alone (0.0051s for QSVM) against classical models would be misleading, as QSVM requires 0.771s of quantum kernel preparation before quadratic programming can proceed. The total pipeline times provide the fair comparison.

---

## 3. Data Partitioning & Leakage Safeguards

| Partition | Total Samples | Legitimate | Fraudulent | Fraud Prevalence | Sampling Source | Status / Role |
| :--- | :---: | :---: | :---: | :---: | :--- | :--- |
| **Quantum Train ($D_{\text{train\_q}}$)** | 500 | 400 | 100 | 20.00% | Stratified sample from `X_train.csv` (seed 42) | Model training & transformer fitting |
| **Audited Validation ($D_{\text{val\_q}}$)** | 20,000 | 19,967 | 33 | 0.165% | Disjoint sample from remaining train pool (seed 42) | Model evaluation & validation tuning |
| **Holdout Test (`X_test.csv`)** | 56,746 | 56,651 | 95 | 0.167% | Stratified 80/20 test split (Phase 2) | **Not used in Phase 4B** (evaluated in Phase 3) |

- **Feature Selector**: `SelectKBest(score_func=f_classif, k=6)` fit strictly on $D_{\text{train\_q}}$.
  - Selected features: `V4, V10, V11, V12, V14, V16`
- **Angle Scaler**: `MinMaxScaler(feature_range=(0, np.pi))` fit strictly on $D_{\text{train\_q}}$ selected features.
- **Validation Transformation**: $D_{\text{val\_q}}$ features transformed strictly via pre-fitted scaler.

---

## 4. Controlled Model Performance Comparison (6 Features, 500 Training Samples)

All three models were evaluated on the **exact same 20,000 validation samples** using identical training samples and feature transformations:

| Metric | QSVM (ZZ Kernel) | Classical Linear SVM | Classical RBF SVM | Best Performer (6-feature matched) |
| :--- | :---: | :---: | :---: | :--- |
| **Recall (Fraud Sensitivity)** | **0.8788** | **0.8788** | **0.8788** | **Tie** (all caught 29 of 33 frauds) |
| **Precision (Positive Predictive Value)** | 0.0594 | 0.1213 | **0.2522** | **Classical RBF SVM** ($>4\times$ higher) |
| **F1-Score** | 0.1113 | 0.2132 | **0.3919** | **Classical RBF SVM** ($>3.5\times$ higher) |
| **PR-AUC (Average Precision)** | 0.6206 | **0.8171** | 0.8012 | **Classical Linear SVM** |
| **ROC-AUC** | 0.9606 | 0.9790 | **0.9819** | **Classical RBF SVM** |
| **True Positives (Frauds Caught)** | **29** / 33 | **29** / 33 | **29** / 33 | **Tie** |
| **False Positives (Legitimate Flagged)** | 459 | 210 | **86** | **Classical RBF SVM** ($>5\times$ fewer false alarms) |
| **False Negatives (Frauds Missed)** | **4** | **4** | **4** | **Tie** |
| **Training Pipeline Time** | 0.7763 s | 0.0032 s | **0.0020 s** | **Classical RBF SVM** |
| **Inference Pipeline Time (20,000 samples)** | 27.2207 s | **0.0016 s** | 0.1089 s | **Classical Linear SVM** |

---

## 5. Contextual Reference: Historical Phase 3 Classical Baseline (30 Features, Full Training Set)

For broader context across the project, the table below references the **historical full-feature classical baseline evaluated on the Phase 3 holdout test set**:

| Evaluation Dimension | Historical Phase 3 Baseline (Holdout Test Set) | Phase 4B Classical RBF SVM Control | Phase 4B QSVM (ZZ Kernel) |
| :--- | :---: | :---: | :---: |
| **Feature Count** | **30 features** (All PCA features + Time + Amount) | **6 features** (Top ANOVA $F$-statistic) | **6 qubits / features** (Top ANOVA $F$-statistic) |
| **Training Set Size** | **226,980 samples** (Full training set) | **500 samples** (Quantum subset) | **500 samples** (Quantum subset) |
| **Evaluation Set** | Holdout Test Set ($N = 56,746$, 95 frauds) | Audited Validation ($N = 20,000$, 33 frauds) | Audited Validation ($N = 20,000$, 33 frauds) |
| **Recall** | 0.6947 | **0.8788** | **0.8788** |
| **Precision** | 0.1970 | **0.2522** | 0.0594 |
| **F1-Score** | 0.3070 | **0.3919** | 0.1113 |
| **PR-AUC** | 0.4138 | **0.8012** | 0.6206 |
| **ROC-AUC** | 0.9651 | **0.9819** | 0.9606 |
| **False Positive Rate** | 0.47% (269 / 56,651) | **0.43%** (86 / 19,967) | 2.30% (459 / 19,967) |

> **Contextual Clarification Note**: The Phase 3 Classical SVM baseline represents the historical full-feature classical baseline evaluated on the Phase 3 holdout test set. Results across Phase 3 and Phase 4B are **not directly comparable** because the data partitions, training sizes, and feature counts differ (Phase 3 used all 30 features and 226,980 training samples evaluated on the 56,746-sample test set; Phase 4B used 6 features and 500 training samples evaluated on the 20,000-sample validation subset). It is included strictly as a contextual reference and not as a capacity-matched comparison.

---

## 6. Detailed Confusion Matrices

### QSVM (Fidelity Quantum Kernel, 6 features, N=500 train):
```
                       Predicted Legit   Predicted Fraud
True Legit (19,967)         19,508               459  (FP Rate = 2.30%)
True Fraud (33)                 4                  29  (Recall = 87.88%)
```

### Classical Linear SVM (6 features, N=500 train):
```
                       Predicted Legit   Predicted Fraud
True Legit (19,967)         19,757               210  (FP Rate = 1.05%)
True Fraud (33)                 4                  29  (Recall = 87.88%)
```

### Classical RBF SVM (6 features, N=500 train):
```
                       Predicted Legit   Predicted Fraud
True Legit (19,967)         19,881                86  (FP Rate = 0.43%)
True Fraud (33)                 4                  29  (Recall = 87.88%)
```

---

## 7. Preliminary Pilot (1,000 samples) vs. Audited Evaluation (20,000 samples) Comparison

| Metric | Preliminary Pilot (1,000 samples, 2 Frauds) | Audited Evaluation (20,000 samples, 33 Frauds) | Scientific Implication |
| :--- | :---: | :---: | :--- |
| **QSVM Recall** | 1.0000 (2/2) | 0.8788 (29/33) | Missed frauds emerged under representative sample |
| **QSVM Precision** | 0.0909 (20 FP) | 0.0594 (459 FP) | Precision dropped; false alarms remain substantial |
| **QSVM PR-AUC** | 1.0000 | 0.6206 | Pilot PR-AUC of 1.0 was an artifact of only 2 positive cases |
| **Linear SVM PR-AUC** | 0.5833 | 0.8171 | Significantly higher average precision on larger validation set |
| **RBF SVM PR-AUC** | 0.8333 | 0.8012 | High, stable average precision across both sets |
| **RBF SVM FP Count** | 7 FP | 86 FP | **Over 5x fewer false alarms than QSVM (86 vs 459)** |

---

## 8. Key Findings & Scientific Conclusions

1. **Preliminary Nature of Findings (Small Positive Fraud Support)**:
   - All findings in Phase 4B are **preliminary** because the 20,000-sample validation set contains only **33 ground-truth fraud cases** (reflecting natural 0.165% prevalence). While this represents a substantial statistical improvement over the 2-fraud pilot, positive-class support remains modest.
   - **Avoidance of Definitive Generalization Claims**: Claims of definitive generalization across broader operational environments or unobserved distributions must be avoided.
2. **Controlled 6-Feature Comparison**:
   - Within this preliminary 6-feature, 500-sample training regime, the classical RBF SVM decisively outperformed the QSVM, achieving **$>4\times$ higher precision** (0.2522 vs 0.0594) and **$>5\times$ fewer false positives** (86 vs 459) at the exact same fraud sensitivity (0.8788).
   - Classical Linear SVM also outperformed QSVM on precision (0.1213 vs 0.0594) and PR-AUC (0.8171 vs 0.6206).
3. **Representational Inductive Bias**:
   - While the 2-repetition `ZZFeatureMap` provides non-linear expressivity, its decision boundary in the 6-qubit Hilbert space flags an excessive volume of legitimate transactions as fraud compared to classical RBF kernel Euclidean geometry under this configuration.
4. **No Quantum Superiority or Novelty Claim**:
   - Neither the capacity-matched comparison nor the historical Phase 3 baseline reference demonstrates quantum advantage. The results demonstrate that classical kernel methods remain superior in accuracy, precision, and efficiency for this tabular fraud detection dataset under NISQ constraints.

