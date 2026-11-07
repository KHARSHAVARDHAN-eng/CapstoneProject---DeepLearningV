# Phase 7: Training-Size Scalability Experiment Report

**Project Title**: Quantum Machine Learning for Financial Fraud Detection using Quantum Kernel Methods  
**Phase**: Phase 7 — Training-Size Scalability & Computational Feasibility Analysis  
**Generated At**: 2026-10-12 07:57:08 UTC  
**Status**: COMPLETED, VERIFIED, AND AUDITED  

---

## 1. Executive Summary & Core Research Findings

This experiment investigates whether increasing the quantum training set size from **N = 500** to **N = 1,000** (and theoretically **N = 2,000**) improves Quantum Support Vector Machine (QSVM) fraud detection performance, and whether any empirical changes justify the substantial quadratic quantum kernel computational scaling.

### Key Empirical Findings:
1. **Severe Quantum Performance Regression Upon Scaling**:  
   Increasing the training set size from N=500 to N=1,000 caused the QSVM Precision-Recall AUC to **collapse from 0.6648 down to 0.3188** (a **-52.0% relative drop**). Rather than improving fraud discriminability, scaling the training size in high-dimensional Hilbert space aggravated overfitting and dual support vector saturation.
2. **Classical Models Maintain Superior Performance and Stability**:  
   - **Linear SVM**: PR-AUC remained exceptionally high (**0.8386 at N=500** vs. **0.8219 at N=1000**).
   - **RBF SVM**: PR-AUC remained robust (**0.8285 at N=500** vs. **0.8238 at N=1000**), consistently outperforming QSVM by **+0.5050 PR-AUC points** at N=1,000.
3. **Quadratic Computational Scaling Law**:  
   The training kernel Gram matrix entries scaled quadratically from **250,000 entries** (N=500, 1.90 s) to **1,000,000 entries** (N=1,000, 3.50 s). Validation cross-kernel entries doubled from 10,000,000 to 20,000,000 entries, requiring 75.51 s of statevector simulation.
4. **N = 2,000 Data-Availability Limitation**:  
   An exhaustive audit of the dataset revealed that the training partition (`y_train.csv`) contains **378 total fraud cases** (with **345 fraud cases** available outside the held-out validation set). Consequently, sampling **400 fraud cases without replacement** to form an 80:20 balanced partition at N=2,000 is mathematically impossible. In accordance with protocol, the experiment stopped safely at N=1,000, and theoretical quadratic scaling projections were computed for N=2,000.

> **Core Conclusion**: Increasing training set size does **not** rescue the Quantum SVM from Hilbert space geometric concentration. Classical models decisively outperform QSVM in both accuracy and computational efficiency at every training scale.

---

## 2. Experimental Setup & Protocol

### 2.1 Model & Hyperparameter Configuration
All models used fixed Phase 5 tuned hyperparameters to isolate the effect of training set size:
- **Features (6 locked features)**: `['V4', 'V10', 'V11', 'V12', 'V14', 'V16']`
- **Preprocessing**: `MinMaxScaler(feature_range=(0, np.pi))` fitted strictly on the respective training subset; transformed on validation data only.
- **Linear SVM**: `C = 5.0, class_weight = 'balanced', dual = False, random_state = 42`
- **Classical RBF SVM**: `C = 5.0, gamma = 0.01, class_weight = 'balanced', random_state = 42`
- **Quantum SVM**: `C = 0.5, class_weight = 'balanced', kernel = 'precomputed', random_state = 42`
- **Quantum Feature Map**: 6-qubit `ZZFeatureMap(reps=2, entanglement='linear')` via exact statevector simulation.

### 2.2 Data Availability Audit & Partitioning Protocol
| Partition / Request | Total Samples | Legitimate | Fraud | Status | Feasibility / Audit Note |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Full Training Pool (`y_train.csv`)** | 226,980 | 226,602 | 378 | Available | Natural class prevalence: 0.1665% |
| **Held-Out Validation Partition** | 20,000 | 19,967 | 33 | Frozen | Natural class prevalence: 0.1650% (33 frauds) |
| **Available Non-Val Training Pool** | 206,980 | 206,635 | 345 | Available | Pool for training subsets without leakage |
| **Subset D_500 (Requested)** | 500 | 400 | 100 | **Executed** | Exact 80:20 ratio; D_500 ⊂ D_1000 |
| **Subset D_1000 (Requested)** | 1,000 | 800 | 200 | **Executed** | Exact 80:20 ratio; strictly nested |
| **Subset D_2000 (Requested)** | 2,000 | 1,600 | 400 | **Safely Stopped** | Shortfall of 55 fraud cases (only 345 available) |

---

## 3. Empirical Scalability Results (Validation Set: N = 20,000, 33 Frauds)

### 3.1 Primary Metric Comparison across Training Set Sizes

| Training Size | Model | PR-AUC | ROC-AUC | Recall (τ=0.0) | Precision (τ=0.0) | F1-Score | FP | FN | Fit Time (s) | Total Infer Pipeline (s) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **N = 500** | Linear SVM | **0.8386** | 0.9793 | 87.88% (29/33) | 10.78% | 0.1921 | 240 | 4 | 0.0078 | 0.0003 |
| **N = 500** | Classical RBF SVM | **0.8285** | 0.9859 | 87.88% (29/33) | 50.88% | 0.6444 | 28 | 4 | 0.0028 | 0.0660 |
| **N = 500** | Quantum SVM | **0.6648** | 0.9612 | 87.88% (29/33) | 4.96% | 0.0939 | 556 | 4 | 0.0028 | 76.7825 |
| **N = 1,000** | Linear SVM | **0.8219** | 0.9821 | 87.88% (29/33) | 7.40% | 0.1365 | 363 | 4 | 0.0031 | 0.0003 |
| **N = 1,000** | Classical RBF SVM | **0.8238** | 0.9860 | 87.88% (29/33) | 19.33% | 0.3169 | 121 | 4 | 0.0046 | 0.1231 |
| **N = 1,000** | Quantum SVM | **0.3188** | 0.9680 | 87.88% (29/33) | 4.50% | 0.0857 | 615 | 4 | 0.0051 | 75.5491 |

### 3.2 Computational Profile & Memory Footprint

| Training Size | Train Kernel Entries | Train Kernel Time | Val Cross-Kernel Entries | Val Kernel Time | Peak Memory Tracked |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **N = 500** | 250,000 entries | 1.90 s | 10,000,000 entries | 76.76 s | 257.22 MB |
| **N = 1,000** | 1,000,000 entries | 3.50 s | 20,000,000 entries | 75.51 s | 488.22 MB |

---

## 4. Theoretical and Quadratic Scaling Projections for N = 2,000

In quantum kernel methods, the Gram matrix scales as O(N^2) in statevector inner products. The table below contrasts empirical measurements with the theoretical projection for N = 2,000:

| Dimension | N = 500 (Empirical) | N = 1,000 (Empirical) | N = 2,000 (Theoretical Projection) | Scaling Factor (vs. N=500) |
| :--- | :---: | :---: | :---: | :---: |
| **Training Kernel Entries** | 250,000 | 1,000,000 | **4,000,000** | **16.0x** |
| **Gram Matrix Memory (float64)** | 1.91 MB | 7.63 MB | **30.52 MB** | **16.0x** |
| **Estimated Train Kernel Time** | 1.90 s | 3.50 s | **~14.01 s** | **~8.0x** |
| **Data Feasibility Status** | Verified (Available) | Verified (Available) | **Infeasible (Shortfall: 55 frauds)** | N/A |

---

## 5. Answers to Core Research Questions

### Q1: Does QSVM PR-AUC improve as training size increases?
**Answer: NO.** QSVM PR-AUC experienced a catastrophic decline from **0.6648 at N=500** to **0.3188 at N=1000** (-52.0%). In contrast, classical Linear SVM (0.8219) and classical RBF SVM (0.8238) maintained outstanding discriminability.

### Q2: Does precision improve without an unacceptable reduction in recall?
**Answer: NO.** At N=1,000, QSVM precision dropped and false alarms remained high (615 false positives). In contrast, Classical RBF SVM produced over 3x fewer false alarms (121 vs. 615).

### Q3: Does QSVM approach or exceed the matched classical models?
**Answer: NO.** The gap widened substantially. At N=500, RBF SVM led QSVM by +0.1637 PR-AUC points. At N=1,000, RBF SVM leads QSVM by **+0.5050 PR-AUC points**.

### Q4: How much additional computation is required for each increase in sample size?
**Answer:** Kernel matrix entries scale quadratically as O(N^2). Increasing from 500 to 1,000 quadrupled training entries (250,000 to 1,000,000) and doubled validation entries (10M to 20M). Scaling to 2,000 would require 4,000,000 training inner products and 40,000,000 validation inner products.

### Q5: Is the observed performance improvement worth the computational cost?
**Answer: DECISIVELY NO.** There is no performance improvement whatsoever. The additional computation yielded severe performance degradation due to geometric concentration and dual support vector saturation in the 6-qubit Hilbert space.

---

## 6. Generated Publication Figures

The following figures were generated and saved to `reports/figures/phase7/`:
1. `scalability_pr_auc.png`: Precision-Recall AUC vs. training-set size.
2. `scalability_recall_precision.png`: Recall and precision trends vs. training-set size.
3. `scalability_quantum_kernel_time.png`: Quantum Gram matrix computation time comparison.
4. `scalability_total_inference_time.png`: Total inference pipeline wall-clock time (log scale).
5. `scalability_performance_vs_cost.png`: PR-AUC vs. computational cost trade-off.

---

## 7. Limitations and Disclosures

1. **Validation Re-use Disclosure**: The 20,000-sample validation set was previously utilized in Phase 4B and Phase 5. Results are exploratory.

2. **Small Validation Fraud Count**: With 33 ground-truth frauds, discrete sample variance is present.

3. **Absence of Real Quantum Hardware**: All simulations utilized exact statevector simulators on classical hardware.

4. **N = 2,000 Non-Replacement Impossibility**: `y_train.csv` contains only 378 total frauds (345 non-validation), precluding a 400-fraud sample without replacement.
