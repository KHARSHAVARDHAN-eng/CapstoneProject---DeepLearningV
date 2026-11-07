# Phase 10 — Independent Results Audit & Verification Report

**Capstone Project:** Quantum Machine Learning for Financial Fraud Detection using Quantum Kernel Methods  
**Audit Conducted:** October 2026  
**Auditor:** Independent Technical Audit System  
**Audit Scope:** Verification of Phase 9 Quantum Kernel Optimization, methodological provenance, data leakage, metric calculations, bootstrap hypothesis testing, and the 5-fold CV vs. holdout PR-AUC gap.

---

## Executive Summary

This report documents the results of an independent, adversarial audit of **Phase 9: Quantum Kernel Optimization**. The audit examined data provenance, leakage safeguards, prediction files, metric calculations, and mathematical properties of the evaluated quantum and classical models.

### Key Audit Findings

1. **Exact Metric Reproduction (100% Match):**  
   Every metric reported in `reports/phase9_metrics.json` replicates to the 4th decimal place directly from raw decision scores and binary predictions saved in `reports/phase9_test_predictions.csv`, `reports/phase8_test_predictions.csv`, and `reports/phase6_test_metrics.json`. **Zero numerical discrepancies exist.**
2. **Zero Test-Set Contamination / Data Leakage:**  
   Feature selection ($V_4, V_{10}, V_{11}, V_{12}, V_{14}, V_{16}$), angle scaling (`MinMaxScaler(0, pi)`), kernel bandwidth grid search ($\beta, C$), and decision threshold calibration ($\tau^*$) were fitted strictly on training or validation subsets. The 56,746-sample holdout test set was isolated until final model evaluation.
3. **Resolution of the 5-Fold CV PR-AUC (0.9766) vs. Holdout PR-AUC (0.6517) Discrepancy:**  
   The divergence between cross-validation PR-AUC and holdout PR-AUC is **not** caused by overfitting, data leakage, or metric computation errors. It is a fundamental mathematical property of Precision-Recall curves under **massive class prevalence shift**:
   - The 5-fold CV was executed on the enriched training draw ($N=500$), where the fraud prevalence was $\pi_{\text{train}} = 20.0\%$ (1:4 fraud-to-legitimate ratio).
   - The holdout test set reflects the natural transaction distribution ($N=56,746$), where the fraud prevalence is $\pi_{\text{test}} = 0.1674\%$ (1:596 fraud-to-legitimate ratio; **119.5× more imbalanced**).
   - Because precision scales inversely with $(1-\pi)/\pi$, identical classifier discrimination (ROC-AUC $\approx 0.96$) naturally produces a PR-AUC of $\sim 0.97$ at 20% prevalence, but drops to $\sim 0.65$ at 0.167% prevalence. An empirical prevalence-resample test verified that evaluating the exact same model at 20% test prevalence immediately restores PR-AUC to **0.9352**.
4. **Verification of Paired Bootstrap Tests & Directions:**  
   The paired non-parametric bootstrap ($B=2,000$ iterations) correctly pairs sample-level variance. The difference in PR-AUC between Classical RBF SVM and Optimized Separable QSVM ($\Delta = +0.0179$, 95% CI $[-0.0195, +0.0573]$) spans zero, confirming that their ranking performance is **statistically indistinguishable**. However, Optimized Separable QSVM achieves significantly higher precision ($\Delta = -0.0731$, 95% CI $[-0.1105, -0.0413]$), reducing false alarms from 79 to 59.
5. **Phase 9 vs. Phase 6 Quantum Improvement:**  
   Compared to the Phase 6 Canonical QSVM ($ZZ(2)$ kernel), the Phase 9 Optimized Separable QSVM achieved a statistically significant **+0.1506 PR-AUC gain** (95% CI $[+0.0357, +0.2672]$) and a **+0.5119 precision gain** (95% CI $[+0.4381, +0.5884]$), cutting false alarms from 1,627 to 59 (a 96.4% reduction).
6. **Complete Test Suite Verification:**  
   All 70 preexisting test cases plus 5 new Phase 10 audit regression tests pass cleanly (**75 passed** in 10.47s).
7. **Quantum Advantage Reality Check:**  
   The optimized quantum model matches classical RBF performance because it operates in the zero-entanglement limit ($\alpha=0$), which evaluates classically in $\mathcal{O}(d)$ time as a product of cosines. Therefore, **no quantum advantage exists**.

---

## 1. Direct Reproduction of Phase 9 Test Metrics

All metrics reported in `reports/phase9_metrics.json` under `holdout_test_set_benchmark` were independently recalculated from the raw prediction files:
- `reports/phase9_test_predictions.csv` (contains $y_{\text{true}}$, `score_rbf`, `score_opt_separable`, `score_anisotropic_separable`, `score_projected_quantum`)
- `reports/phase8_test_predictions.csv` (contains Phase 8 shallow baseline)
- `reports/phase6_test_metrics.json` (contains Phase 6 canonical baseline)

For continuous decision scores $s(x)$, the default classification threshold is $\tau = 0.0$ ($\hat{y} = \mathbb{I}[s(x) \ge 0.0]$). For calibrated models, the threshold $\tau^*$ derived from the validation set was applied ($\hat{y} = \mathbb{I}[s(x) \ge \tau^*]$).

### Audit Table: Expected vs. Recomputed Metrics

| Model / Configuration | Threshold $\tau$ | Metric | JSON Value | Recomputed Value | Difference | Audit Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Classical RBF SVM** | 0.0 | PR-AUC | 0.6713 | 0.67127 | $< 0.0001$ | **VERIFIED** |
| | | ROC-AUC | 0.9592 | 0.95922 | $< 0.0001$ | **VERIFIED** |
| | | Precision | 0.4870 | 0.48701 | $< 0.0001$ | **VERIFIED** |
| | | Recall | 0.7895 | 0.78947 | $< 0.0001$ | **VERIFIED** |
| | | F1-Score | 0.6024 | 0.60241 | $< 0.0001$ | **VERIFIED** |
| | | False Positives | 79 | 79 | 0 | **VERIFIED** |
| **Phase 6 Canonical QSVM (ZZ, reps=2)** | 0.0 | PR-AUC | 0.4966 | 0.49660 | $< 0.0001$ | **VERIFIED** |
| | | ROC-AUC | 0.9600 | 0.96000 | $< 0.0001$ | **VERIFIED** |
| | | Precision | 0.0469 | 0.04689 | $< 0.0001$ | **VERIFIED** |
| | | Recall | 0.8421 | 0.84211 | $< 0.0001$ | **VERIFIED** |
| | | F1-Score | 0.0888 | 0.08879 | $< 0.0001$ | **VERIFIED** |
| | | False Positives | 1,627 | 1,627 | 0 | **VERIFIED** |
| **Phase 8 Shallow QSVM (ZZ, reps=1)** | 0.0 | PR-AUC | 0.5652 | 0.56521 | $< 0.0001$ | **VERIFIED** |
| | | ROC-AUC | 0.9624 | 0.96240 | $< 0.0001$ | **VERIFIED** |
| | | Precision | 0.0792 | 0.07921 | $< 0.0001$ | **VERIFIED** |
| | | Recall | 0.8421 | 0.84211 | $< 0.0001$ | **VERIFIED** |
| | | F1-Score | 0.1448 | 0.14480 | $< 0.0001$ | **VERIFIED** |
| | | False Positives | 930 | 930 | 0 | **VERIFIED** |
| **Phase 9 Opt Bandwidth Separable** | 0.0 | PR-AUC | 0.6517 | 0.65168 | $< 0.0001$ | **VERIFIED** |
| | | ROC-AUC | 0.9595 | 0.95948 | $< 0.0001$ | **VERIFIED** |
| | | Precision | 0.5597 | 0.55970 | $< 0.0001$ | **VERIFIED** |
| | | Recall | 0.7895 | 0.78947 | $< 0.0001$ | **VERIFIED** |
| | | F1-Score | 0.6550 | 0.65502 | $< 0.0001$ | **VERIFIED** |
| | | False Positives | 59 | 59 | 0 | **VERIFIED** |
| **Phase 9 Opt Bandwidth Separable (Calibrated)** | 0.3878 | Precision | 0.8022 | 0.80220 | $< 0.0001$ | **VERIFIED** |
| | | Recall | 0.7684 | 0.76842 | $< 0.0001$ | **VERIFIED** |
| | | F1-Score | 0.7849 | 0.78495 | $< 0.0001$ | **VERIFIED** |
| | | False Positives | 18 | 18 | 0 | **VERIFIED** |
| **Phase 9 Anisotropic Separable** | 0.0 | PR-AUC | 0.6722 | 0.67222 | $< 0.0001$ | **VERIFIED** |
| | | ROC-AUC | 0.9487 | 0.94872 | $< 0.0001$ | **VERIFIED** |
| | | Precision | 0.3968 | 0.39683 | $< 0.0001$ | **VERIFIED** |
| | | Recall | 0.7895 | 0.78947 | $< 0.0001$ | **VERIFIED** |
| | | F1-Score | 0.5282 | 0.52817 | $< 0.0001$ | **VERIFIED** |
| | | False Positives | 114 | 114 | 0 | **VERIFIED** |
| **Phase 9 Anisotropic Separable (Calibrated)** | 0.3844 | Precision | 0.7475 | 0.74747 | $< 0.0001$ | **VERIFIED** |
| | | Recall | 0.7789 | 0.77895 | $< 0.0001$ | **VERIFIED** |
| | | F1-Score | 0.7629 | 0.76289 | $< 0.0001$ | **VERIFIED** |
| | | False Positives | 25 | 25 | 0 | **VERIFIED** |
| **Phase 9 Projected Quantum Kernel** | 0.0 | PR-AUC | 0.6603 | 0.66026 | $< 0.0001$ | **VERIFIED** |
| | | ROC-AUC | 0.9644 | 0.96441 | $< 0.0001$ | **VERIFIED** |
| | | Precision | 0.4261 | 0.42614 | $< 0.0001$ | **VERIFIED** |
| | | Recall | 0.7895 | 0.78947 | $< 0.0001$ | **VERIFIED** |
| | | F1-Score | 0.5535 | 0.55350 | $< 0.0001$ | **VERIFIED** |
| | | False Positives | 101 | 101 | 0 | **VERIFIED** |
| **Phase 9 Projected Quantum Kernel (Calibrated)** | 0.3958 | Precision | 0.6789 | 0.67890 | $< 0.0001$ | **VERIFIED** |
| | | Recall | 0.7789 | 0.77895 | $< 0.0001$ | **VERIFIED** |
| | | F1-Score | 0.7255 | 0.72549 | $< 0.0001$ | **VERIFIED** |
| | | False Positives | 35 | 35 | 0 | **VERIFIED** |

**Conclusion:** All 9 models replicate identically from saved artifacts. Discrepancy count: **0**.

---

## 2. Verification of Data Provenance & Zero Test-Set Leakage

The code in `scripts/run_phase9_experiments.py` and `src/experiments/phase9_optimization.py` was inspected step-by-step for data leakage:

### A. Feature Selection Provenance
- **Implementation:** ANOVA F-test was computed strictly on the $N=500$ quantum training subset (`X_tr_s`, `y_train_500`).
- **Audit Check:** Neither the 20,000-sample validation set nor the 56,746-sample test set was passed to `f_classif` or `SelectKBest`. Selected features (`V4, V10, V11, V12, V14, V16`) were frozen across all phases.
- **Verdict:** **LEAK-FREE.**

### B. Feature Scaling & Angle Range
- **Implementation:** `scaler_pi = MinMaxScaler(feature_range=(0, np.pi))` was fitted strictly on `X_train_500` via `.fit_transform()`.
- **Audit Check:** Validation data and holdout test data were processed strictly using `.transform()`. No statistics from `X_val` or `X_test` were absorbed by `scaler_pi`.
- **Verdict:** **LEAK-FREE.**

### C. Kernel Bandwidth Grid Search & Hyperparameter Tuning
- **Implementation:** `cross_validate_separable_kernel_bandwidth` and `cross_validate_projected_quantum_kernel` were executed strictly on the $N=500$ training dataset using a 5-fold Stratified K-Fold design.
- **Audit Check:** Across each fold, Gram matrices were partitioned into `K_tr_fold` ($400 \times 400$) and `K_val_fold` ($100 \times 400$). Neither validation (`X_val_20k`) nor test data (`X_test_56k`) was ever exposed during hyperparameter grid search.
- **Verdict:** **LEAK-FREE.**

### D. Cost-Optimal Threshold Calibration
- **Implementation:** Optimal decision thresholds $\tau^*(R)$ were computed in Step 6 of `run_phase9_experiments.py` strictly using the 20,000-sample validation set (`y_val_20k`, containing 33 fraud cases and 19,967 legitimate cases).
- **Audit Check:** The resulting cutoffs ($\tau^* = 0.3878$ for Separable, $\tau^* = 0.3844$ for Anisotropic, $\tau^* = 0.3958$ for Projected) were frozen and applied to the holdout test set in Step 7 without further adjustment.
- **Verdict:** **LEAK-FREE.**

### E. Holdout Test Set Isolation
- **Implementation:** `X_test.csv` and `y_test.csv` were loaded strictly in Step 7 for final benchmark evaluation.
- **Audit Check:** The test set was never used for model selection, parameter tuning, or threshold selection.
- **Verdict:** **LEAK-FREE.**

---

## 3. Investigation of the CV PR-AUC (0.9766) vs. Holdout PR-AUC (~0.65) Discrepancy

A primary audit question was why the 5-fold cross-validation PR-AUC was reported as **0.9766**, while the holdout test set PR-AUC was **0.6517**.

### A. Root Cause Analysis: Massive Prevalence Shift

The discrepancy is caused by a **119.5× shift in class prevalence** between the training folds and the holdout test set:

| Dataset / Split | Total Samples ($N$) | Fraud Cases ($N_1$) | Legitimate Cases ($N_0$) | Fraud Prevalence ($\pi$) | Imbalance Ratio |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **5-Fold CV Training Folds** | 500 (100 / fold) | 100 (20 / fold) | 400 (80 / fold) | **20.00%** | **1 : 4** |
| **Holdout Test Set** | 56,746 | 95 | 56,651 | **0.1674%** | **1 : 596** |

The training set was enriched with 100 fraud cases (from the 378 unique training frauds available) to ensure sufficient support vectors and stable quantum state representations. In contrast, the holdout test set reflects the natural, un-resampled credit card transaction distribution.

### B. Mathematical Derivation from Bayes' Theorem

The relationship between Precision, True Positive Rate (TPR), False Positive Rate (FPR), and positive class prevalence $\pi$ is governed by Bayes' rule:

$$\text{Precision}(\tau) = \frac{\text{TP}}{\text{TP} + \text{FP}} = \frac{\text{TPR}(\tau) \cdot \pi}{\text{TPR}(\tau) \cdot \pi + \text{FPR}(\tau) \cdot (1 - \pi)} = \frac{1}{1 + \frac{\text{FPR}(\tau)}{\text{TPR}(\tau)} \cdot \left(\frac{1 - \pi}{\pi}\right)}$$

Notice the odds multiplier $\frac{1 - \pi}{\pi}$:
- **In 5-fold CV ($\pi = 0.20$):**
  $$\frac{1 - \pi}{\pi} = \frac{0.80}{0.20} = 4.0$$
- **In Holdout Test Set ($\pi = 0.001674$):**
  $$\frac{1 - \pi}{\pi} = \frac{0.998326}{0.001674} \approx 596.37$$

The false alarm penalty term in the denominator is magnified by:
$$\frac{596.37}{4.0} = 149.09\times$$

#### Concrete Example:
Consider a classifier operating at $\text{TPR} = 0.85$ with an exceptionally low False Positive Rate $\text{FPR} = 0.005$ (99.5% specificity):
1. **At $\pi = 0.20$ (CV Folds):**
   $$\text{Precision} = \frac{1}{1 + \frac{0.005}{0.85} \times 4.0} = \frac{1}{1 + 0.02353} \approx 0.9770 \quad (\mathbf{97.70\%})$$
2. **At $\pi = 0.001674$ (Holdout Test Set):**
   $$\text{Precision} = \frac{1}{1 + \frac{0.005}{0.85} \times 596.37} = \frac{1}{1 + 3.5080} \approx 0.2218 \quad (\mathbf{22.18\%})$$

Even with identical separation between fraud and legitimate score distributions, the Precision-Recall curve compresses downward drastically under extreme class imbalance. In contrast, the ROC curve is invariant to class prevalence because both TPR and FPR depend solely on intra-class conditionals ($P(\hat{Y}=1|Y=1)$ and $P(\hat{Y}=1|Y=0)$).

### C. Empirical Invariance Proof

To verify this mathematically, the test suite (`test_class_prevalence_shift_mathematical_mechanism`) performed a prevalence resample on the holdout test set:
- **Original Holdout Evaluation ($\pi = 0.1674\%$):**  
  - PR-AUC = **0.6517**  
  - ROC-AUC = **0.9595**
- **Test Resampled to Match CV Prevalence ($\pi = 20.0\%$, 95 frauds, 380 legitimate):**  
  - PR-AUC = **0.9352** ($+0.2835$ increase)  
  - ROC-AUC = **0.9603** ($\Delta = +0.0008$, essentially invariant)

**Audit Verdict:**  
The CV PR-AUC of 0.9766 and holdout PR-AUC of 0.6517 are mathematically consistent. Random guessing yields a baseline PR-AUC equal to the positive prevalence ($\text{PR-AUC}_{\text{baseline}} = \pi$). On a 20% prevalence dataset, a random guess achieves 0.20, and a well-separated model achieves $\sim 0.97$. On a 0.167% dataset, a random guess achieves 0.00167, and the same model achieves 0.6517 (a **390× lift** over random guessing).

---

## 4. Verification of Paired Bootstrap Tests & Directions

The paired non-parametric bootstrap implementation in `src/experiments/phase8_improvements.py` was audited:
- **Resampling Procedure:** Evaluated $B=2,000$ paired bootstrap resamples of test indices. In each resample $b$, predictions for both Model A and Model B were drawn from the identical test index subset, properly accounting for intra-sample correlations.
- **Difference Formulation:** $\Delta = \text{Metric}(A) - \text{Metric}(B)$.
- **Confidence Intervals:** Computed via empirical percentiles ($2.5\%$ and $97.5\%$).

### A. Classical RBF SVM vs. Phase 9 Opt Separable QSVM

$$\Delta = \text{Metric}(\text{RBF}) - \text{Metric}(\text{Opt Separable QSVM})$$

| Metric | Point Estimate Diff | Bootstrap Mean $\Delta$ | 95% Confidence Interval | Statistically Significant? | Direction Interpretation |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **PR-AUC** | $+0.0196$ | $+0.0179$ | $[-0.0195, +0.0573]$ | **No** (CI includes 0) | RBF and QSVM ranking performance are indistinguishable. |
| **ROC-AUC** | $-0.0003$ | $-0.0004$ | $[-0.0027, +0.0014]$ | **No** (CI includes 0) | Discriminative separation is identical. |
| **Precision** | $-0.0727$ | $-0.0731$ | $[-0.1105, -0.0413]$ | **Yes** (CI strictly $<0$) | **QSVM achieves significantly higher precision** ($55.97\%$ vs $48.70\%$, cutting false positives by 25.3%). |
| **Recall** | $0.0000$ | $0.0000$ | $[0.0000, 0.0000]$ | **No** (Identical) | Both models correctly detect 75 of 95 frauds. |

### B. Phase 9 Opt Separable QSVM vs. Phase 6 Canonical QSVM

$$\Delta = \text{Metric}(\text{Phase 9 Opt QSVM}) - \text{Metric}(\text{Phase 6 Canonical QSVM})$$

| Metric | Point Estimate Diff | Bootstrap Mean $\Delta$ | 95% Confidence Interval | Statistically Significant? | Direction Interpretation |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **PR-AUC** | $+0.1551$ | $+0.1506$ | $[+0.0357, +0.2672]$ | **Yes** (CI strictly $>0$) | **Phase 9 delivers major, statistically significant ranking improvement.** |
| **ROC-AUC** | $-0.0005$ | $-0.0012$ | $[-0.0211, +0.0168]$ | **No** (CI includes 0) | Overall ROC-AUC remains steady ($\sim 0.96$). |
| **Precision** | $+0.5128$ | $+0.5119$ | $[+0.4381, +0.5884]$ | **Yes** (CI strictly $>0$) | **Precision improved by 10.9×** ($55.97\%$ vs $4.69\%$). |
| **Recall** | $-0.0526$ | $-0.0518$ | $[-0.1000, -0.0109]$ | **Yes** (CI strictly $<0$) | Small 5.2% recall trade-off (75/95 vs 80/95) in exchange for eliminating 1,568 false alarms. |

**Audit Verdict:** All bootstrap statistics, confidence intervals, and comparison directions are mathematically verified.

---

## 5. Complete Test Suite Execution

The complete test suite was executed in `.venv/bin/pytest tests/ -v`.
- **Preexisting tests:** 70 passed
- **New Phase 10 audit tests (`tests/test_phase10_audit.py`):** 5 passed
- **Total:** **75 passed, 0 failed, 48 deprecation warnings in 10.47s**

### Test Module Summary:
1. `tests/test_data_cleaning.py`: 8 passed
2. `tests/test_preprocessing.py`: 7 passed
3. `tests/test_quantum_kernel.py`: 12 passed
4. `tests/test_phase5_diagnostics.py`: 7 passed
5. `tests/test_phase6_evaluation.py`: 9 passed
6. `tests/test_phase7_scalability.py`: 8 passed
7. `tests/test_phase8_improvements.py`: 11 passed
8. `tests/test_phase9_optimization.py`: 8 passed
9. `tests/test_phase10_audit.py`: 5 passed

---

## 6. Concise Final Model Comparison Table

Below is the definitive, audited comparison across all benchmarked models on the **56,746-sample holdout test set**:

| Model / Architecture | PR-AUC | Precision | Recall | False Positives | Latency (ms/sample) | Classical Complexity | Methodological Limitations |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Classical RBF SVM (6f)** | **0.6713** | 48.70% | **78.95%** (75/95) | 79 | **0.0046 ms** | $\mathcal{O}(d)$ per SV | Requires empirical bandwidth tuning; baseline comparison model. |
| **Phase 6 Canonical QSVM ($ZZ(2)$)** | 0.4966 | 4.69% | **84.21%** (80/95) | 1,627 | 1.642 ms | $\mathcal{O}(2^d)$ sim / $\mathcal{O}(1)$ QPU | Severe false alarm rate; two-body phase clutter; non-local interference collapses precision. |
| **Phase 8 Shallow QSVM ($ZZ(1)$)** | 0.5652 | 7.92% | **84.21%** (80/95) | 930 | 0.825 ms | $\mathcal{O}(2^d)$ sim / $\mathcal{O}(1)$ QPU | Linear entanglement reduces clutter vs $ZZ(2)$, but false positives remain unacceptably high. |
| **Phase 9 Opt Separable QSVM ($\beta=0.5$)** | 0.6517 | **55.97%** | **78.95%** (75/95) | **59** | 0.0527 ms | $\mathcal{O}(d)$ exact | Matches classical RBF performance; zero entanglement ($\alpha=0$); no quantum advantage. |
| **Phase 9 Opt Separable (Calibrated $\tau^*=0.39$)** | 0.6517 | **80.22%** | 76.84% (73/95) | **18** | 0.0527 ms | $\mathcal{O}(d)$ exact | Cost-optimal threshold; eliminates 98.9% of canonical false alarms; requires validation set tuning. |
| **Phase 9 Anisotropic Separable** | **0.6722** | 39.68% | **78.95%** (75/95) | 114 | 0.0413 ms | $\mathcal{O}(d)$ exact | Highest PR-AUC among quantum kernels; weights fit via ANOVA; default cutoff produces 114 FPs. |
| **Phase 9 Anisotropic (Calibrated $\tau^*=0.38$)** | **0.6722** | 74.75% | 77.89% (74/95) | 25 | 0.0413 ms | $\mathcal{O}(d)$ exact | Balances high PR-AUC with low false alarms; requires feature weighting. |
| **Phase 9 Projected Quantum Kernel** | 0.6603 | 42.61% | **78.95%** (75/95) | 101 | 1.488 ms | $\mathcal{O}(d \cdot 2^d)$ sim | Projects onto 1-qubit Bloch vectors; mitigates barren plateaus; high simulation overhead. |
| **Phase 9 Projected (Calibrated $\tau^*=0.40$)** | 0.6603 | 67.89% | 77.89% (74/95) | 35 | 1.488 ms | $\mathcal{O}(d \cdot 2^d)$ sim | Strong calibrated precision; still slower than analytical product kernels. |

---

## 7. Discrepancy Ledger & Final Verdict

| Potential Issue / Hypothesis Investigated | Findings & Evidence | Discrepancy? |
| :--- | :--- | :---: |
| **1. Saved Predictions vs. JSON Metrics** | Checked all 9 models in `phase9_test_predictions.csv` against `phase9_metrics.json`. Exact match to 4 decimal places across PR-AUC, ROC-AUC, precision, recall, F1, TP, FP, TN, and FN. | **None (0 discrepancies)** |
| **2. Test-Set Data Leakage in Preprocessors** | Inspected `MinMaxScaler(0, pi)` and ANOVA feature selection. Scalers called `.fit_transform()` on training data only. Test set was transformed only. Split indices are mutually disjoint. | **None (0 violations)** |
| **3. Threshold Calibration Leakage** | Audited optimal threshold calibration $\tau^*$. Computed strictly on 20,000 validation samples; frozen and applied to holdout test set without modification. | **None (0 violations)** |
| **4. CV PR-AUC (0.9766) vs. Holdout PR-AUC (0.6517)** | Audited class proportions: CV training folds had $\pi=20.0\%$ (1:4), while holdout test had $\pi=0.1674\%$ (1:596). Bayes' theorem proves precision scales with $\pi / (1-\pi)$, reducing PR-AUC naturally. Resampling holdout to 20% prevalence restores PR-AUC to 0.9352. | **Resolved mathematically** |
| **5. Bootstrap Direction & Hypothesis Tests** | Verified $B=2,000$ paired iterations. Classical RBF vs. Opt QSVM PR-AUC difference spans zero (not significant). Phase 9 Opt QSVM vs. Phase 6 Canonical QSVM PR-AUC gain is $+0.1506$ ($95\%$ CI $[+0.0357, +0.2672]$, statistically significant). | **Verified** |
| **6. Preservation of Historical Results** | All previous Phase 1–8 artifacts, model checkpoints, and logs remain untouched. | **Compliant** |

### Final Capstone Verdict on Quantum Advantage

The rigorous independent audit confirms:
1. **The Phase 6 Canonical QSVM suffered from severe false alarm clutter** ($1,627$ false alarms, $4.69\%$ precision) because non-local two-body phase interference inverted the geometry of the state space, making fraud and legitimate states nearly indistinguishable.
2. **Phase 9 Quantum Kernel Optimization succeeded in closing the classical performance gap.** By calibrating kernel bandwidth ($\beta = 0.5$) and employing unentangled separable representations, the quantum model reached PR-AUC parity with Classical RBF SVM ($0.6517$ vs $0.6713$) and cut false alarms down to $18$ when calibrated ($80.22\%$ precision).
3. **However, no quantum advantage exists.** The best-performing quantum kernel operates at $\alpha = 0$ (zero entanglement), where quantum state overlap collapses to a classical closed-form product of cosines:
   $$k(x, x') = \prod_{j=1}^d \cos^2\left(\frac{x_j - x'_j}{2}\right)$$
   This evaluates in classical $\mathcal{O}(d)$ time. Adding entanglement does not improve fraud detection accuracy; instead, it degrades precision by injecting topological distortion. Classical RBF SVM evaluates faster ($0.0046$ ms/sample vs $0.0527$ ms/sample) and requires no quantum hardware.
