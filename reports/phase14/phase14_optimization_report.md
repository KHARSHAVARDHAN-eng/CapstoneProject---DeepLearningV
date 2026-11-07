# Phase 14: Systematic Performance Optimization and Dimensionality Scaling Report

**Capstone Project:** Quantum Machine Learning for Financial Fraud Detection using Quantum Kernel Methods  
**Date:** October 2026  
**Phase:** Phase 14 — Performance Optimization, Feature Space Expansion ($K \in \{6, 10, 16\}$), Bandwidth Compensation, and Full 15-Fold Multi-Seed Evaluation  
**Status:** Completed & Empirically Validated (15/15 Folds across Seeds 42, 101, 2024; Sample Sizes $N \in \{2000, 3000\}$)  
**Primary Artifacts:**
- Implementation Module: [`src/experiments/phase14_optimization.py`](file:///Users/harsha/Desktop/Projects-files/quantum-fraud-detection(Capstone)/src/experiments/phase14_optimization.py)
- Execution Script: [`scripts/run_phase14_experiments.py`](file:///Users/harsha/Desktop/Projects-files/quantum-fraud-detection(Capstone)/scripts/run_phase14_experiments.py)
- Test Suite: [`tests/test_phase14_optimization.py`](file:///Users/harsha/Desktop/Projects-files/quantum-fraud-detection(Capstone)/tests/test_phase14_optimization.py) (96/96 passing suite-wide)
- Full Metrics JSON: [`reports/phase14/full_metrics.json`](file:///Users/harsha/Desktop/Projects-files/quantum-fraud-detection(Capstone)/reports/phase14/full_metrics.json)
- Summary CSV: [`reports/phase14/full_summary_table.csv`](file:///Users/harsha/Desktop/Projects-files/quantum-fraud-detection(Capstone)/reports/phase14/full_summary_table.csv)
- Execution Log: [`reports/phase14/phase14_execution.log`](file:///Users/harsha/Desktop/Projects-files/quantum-fraud-detection(Capstone)/reports/phase14/phase14_execution.log)

---

## 1. Executive Summary & Objective Assessment

Phase 14 evaluated systematic performance optimization strategies to advance beyond the Phase 13 Out-of-Fold (OOF) benchmark ($\text{PR-AUC} \approx 0.7287$ achieved by Classical RBF SVM at $N=500$, and $0.7215$ at $N=2,000$), testing an ambitious target of $\text{PR-AUC} \ge 0.85$.

### Key Empirical Findings:
1. **Measured Performance Advance**:  
   Across 15 stratified folds (3 random seeds: 42, 101, 2024 $\times$ 5 folds, $N_{\text{val}} = 45,396$ per fold, total pooled evaluation $N = 226,980$, natural prevalence $\pi \approx 0.165\%$):
   - At $N=2,000$, expanding feature space from $K=6$ to $K=16$ with dimensional bandwidth compensation lifted mean CV PR-AUC from **$0.7326 \pm 0.0308$** (Classical RBF $K=6$) to **$0.7430 \pm 0.0336$** (Classical RBF $K=16$, $p=0.0147$), matched by **$0.7432 \pm 0.0348$** (OptQSVM $K=16$) and **$0.7435 \pm 0.0342$** (Hybrid RBF-QSVM $K=16$).
   - Across pooled OOF predictions (226,980 instances per seed), aggregated PR-AUC reached **$0.7369 \pm 0.0019$** (OptQSVM $K=16, N=3,000$) and **$0.7368 \pm 0.0028$** (Hybrid Ensemble $K=16, N=2,000$).
   - The Macro Full-Data Control ($N=181,584$ training instances on $K=16$) achieved an OOF PR-AUC of **$0.7705 \pm 0.0065$** (individual fold mean $0.7718 \pm 0.0439$).
2. **Quantum vs. Classical Comparison**:
   - At $K=6$, Classical RBF SVM significantly outperformed OptQSVM ($\Delta = +0.0271$, $p < 0.0001$).
   - At $K=10$ and $K=16$, Optimized Separable QSVM achieved parity with Classical RBF:
     - $N=2,000, K=16$: Difference $= +0.0002$ ($t = 0.340, p = 0.7388$; Wilcoxon $p = 0.6788$).
     - $N=3,000, K=16$: Difference $= -0.0012$ ($t = -1.287, p = 0.2188$; Wilcoxon $p = 0.3028$).
   - Canonical ZZ QSVM suffered from severe exponential feature collapse as dimensions increased ($\text{PR-AUC} \approx 0.4362$ at $N=2,000$).
   - **Conclusion on Quantum Advantage**: **No quantum advantage is observed.** Optimized separable quantum kernels achieve statistical parity with classical RBF kernels under matched feature spaces, but do not provide a statistically significant benefit.
3. **Defensible Verdict on the 0.85 PR-AUC Target**:
   - **The 0.85 PR-AUC target was NOT achieved.** 
   - Kernel machines trained on $10\%$ fraud-enriched subsets ($N \in \{2000, 3000\}$) evaluated under extreme natural class imbalance ($\pi \approx 0.165\%$) exhibit an empirical ceiling at **$0.743$** (fold-level mean) and **$0.737$** (pooled OOF). Even training a classical non-linear model on the entire training dataset ($N=181,584$) reaches **$0.7705$**. In accordance with scientific integrity guidelines, no claim of reaching 0.85 is made.

---

## 2. Investigation of Pilot Identical Metrics: Root Cause & Resolution

During the initial Stage 2 pilot run, multiple distinct models exhibited identical calibrated metrics ($\text{Precision} \approx 27.31\%$ or $28.19\%$, $\text{Recall} \approx 82.67\%$ or $85.33\%$, $\text{FP} = 165$ or $163$, $\text{Cost} = \$4,250$ or $\$3,830$). An in-depth audit confirmed that this was caused by an artificial search boundary constraint in the threshold evaluation function:

1. **Percentile Grid Truncation**:
   Candidate thresholds were generated via:
   ```python
   candidate_thresholds = np.percentile(scores, np.linspace(0.0, 99.5, n_thresholds))
   ```
   The maximum threshold evaluated was capped at the **99.5th percentile**.
2. **Prevalence & Mathematical Boundary Locking**:
   In validation Fold 0 ($N_{\text{val}} = 45,396$, Frauds $= 75$, $\pi \approx 0.165\%$), predicting positives at the 99.5th percentile corresponds to selecting the top $0.5\%$ of all validation scores:
   $$\text{Predicted Positives} = \text{TP} + \text{FP} = \lceil 45,396 \times (1 - 0.995) \rceil = 227 \text{ instances}$$
3. **Asymmetric Step-Loss Gradient**:
   Under $C = 10 \cdot \text{FP} + 200 \cdot \text{FN}$ ($R = 20$), each false negative costs $20\times$ more than a false positive. Because fraud is rare ($0.165\%$), the true cost minimum for well-calibrated SVM decision scores occurs in the **99.7th to 99.9th percentiles** (predicting roughly $50\text{--}100$ total positives, where $\text{FP} \approx 30\text{--}60$ rather than $165$).
   Because the search was capped at $99.5\%$, the cost curve was strictly decreasing at the upper boundary. The optimization slammed into the 99.5% boundary ceiling, algebraically freezing $\text{TP} + \text{FP} = 227$:
   - Models identifying $62$ frauds ($\text{Recall} = 62/75 = 82.67\%$) were forced to have $\text{FP} = 227 - 62 = 165$, $\text{FN} = 13$, $\text{Precision} = 62/227 = 27.31\%$, and $\text{Cost} = 10(165) + 200(13) = \$4,250$.
   - Models identifying $64$ frauds ($\text{Recall} = 64/75 = 85.33\%$) were forced to have $\text{FP} = 227 - 64 = 163$, $\text{FN} = 11$, $\text{Precision} = 64/227 = 28.19\%$, and $\text{Cost} = 10(163) + 200(11) = \$3,830$.
   - This boundary clamp also caused the reported "calibrated" cost to exceed the default threshold ($\tau = 0.0$) cost for several models because $\tau = 0.0$ sat above the 99.5th percentile.
4. **Resolution**:
   `find_cost_optimal_threshold` was realigned with Phase 13 by drawing candidate thresholds directly from `precision_recall_curve` across the entire support and always evaluating default $\tau = 0.0$. This eliminated the boundary clamping and restored distinct, model-specific operating points where calibrated cost is strictly $\le$ default cost.

---

## 3. Full 15-Fold Multi-Seed Benchmark Results

### Table A: Aggregated Out-of-Fold (OOF) Metrics Across 3 Seeds (Pooled $N=226,980$, Frauds $= 378$)
*Evaluated by pooling predictions across the 5 validation folds for each seed, then computing mean and sample standard deviation across seeds 42, 101, and 2024.*

| Training Size | Model Architecture | Features ($K$) | Mean OOF PR-AUC | Std OOF PR-AUC | Mean OOF ROC-AUC | Mean Def Prec | Mean Def Rec | Mean Cal Cost ($) | Std Cal Cost ($) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **2000** | `baseline_classical_linear_svm_k6` | 6 | 0.6974 | 0.0079 | 0.9675 | 0.0818 | 0.8748 | $15,546.7 | 405.3 |
| **2000** | `baseline_classical_rbf_svm_k6` | 6 | 0.7215 | 0.0093 | 0.9670 | 0.2766 | 0.8554 | $14,270.0 | 380.0 |
| **2000** | `baseline_opt_separable_qsvm_k6` | 6 | 0.6953 | 0.0066 | 0.9681 | 0.1304 | 0.8642 | $14,240.0 | 160.9 |
| **2000** | `baseline_canonical_zz_qsvm_k6` | 6 | 0.3671 | 0.0237 | 0.9637 | 0.1142 | 0.8642 | $16,313.3 | 563.7 |
| **2000** | `baseline_projected_quantum_kernel_k6` | 6 | 0.6965 | 0.0088 | 0.9674 | 0.1832 | 0.8571 | $14,360.0 | 433.1 |
| **2000** | `p14_classical_rbf_k10` | 10 | 0.7352 | 0.0014 | 0.9819 | 0.6438 | 0.8201 | $14,820.0 | 55.7 |
| **2000** | `p14_opt_separable_qsvm_k10` | 10 | 0.7364 | 0.0014 | 0.9815 | 0.6863 | 0.8192 | $14,903.3 | 90.2 |
| **2000** | `p14_classical_linear_svm_k16` | 16 | 0.6708 | 0.0155 | 0.9774 | 0.0670 | 0.8995 | $16,266.7 | 219.6 |
| **2000** | `p14_classical_rbf_k16` | 16 | 0.7350 | 0.0011 | 0.9820 | 0.6481 | 0.8245 | $14,813.3 | 107.9 |
| **2000** | `p14_opt_separable_qsvm_k16` | 16 | 0.7363 | 0.0028 | 0.9815 | 0.7720 | 0.8025 | $14,843.3 | 66.6 |
| **2000** | `p14_anisotropic_qsvm_k16` | 16 | 0.7208 | 0.0063 | 0.9696 | 0.4998 | 0.8333 | $15,116.7 | 352.2 |
| **2000** | `p14_hybrid_rbf_qsvm_k16` | 16 | **0.7368** | 0.0028 | 0.9807 | 0.7031 | 0.8166 | $14,773.3 | 57.7 |
| **3000** | `baseline_classical_linear_svm_k6` | 6 | 0.7024 | 0.0029 | 0.9673 | 0.0770 | 0.8757 | $15,000.0 | 230.7 |
| **3000** | `baseline_classical_rbf_svm_k6` | 6 | 0.7201 | 0.0047 | 0.9670 | 0.1847 | 0.8571 | $14,393.3 | 234.6 |
| **3000** | `baseline_opt_separable_qsvm_k6` | 6 | 0.6978 | 0.0034 | 0.9675 | 0.1197 | 0.8660 | $14,006.7 | 357.3 |
| **3000** | `baseline_canonical_zz_qsvm_k6` | 6 | 0.3742 | 0.0349 | 0.9637 | 0.1198 | 0.8660 | $15,626.7 | 896.9 |
| **3000** | `baseline_projected_quantum_kernel_k6` | 6 | 0.6928 | 0.0038 | 0.9674 | 0.1375 | 0.8607 | $13,953.3 | 458.3 |
| **3000** | `p14_classical_rbf_k10` | 10 | 0.7338 | 0.0020 | 0.9826 | 0.5770 | 0.8298 | $14,806.7 | 80.8 |
| **3000** | `p14_opt_separable_qsvm_k10` | 10 | 0.7354 | 0.0012 | 0.9822 | 0.6326 | 0.8228 | $14,813.3 | 100.2 |
| **3000** | `p14_classical_linear_svm_k16` | 16 | 0.6853 | 0.0060 | 0.9798 | 0.0707 | 0.8977 | $15,220.0 | 95.4 |
| **3000** | `p14_classical_rbf_k16` | 16 | 0.7361 | 0.0023 | 0.9829 | 0.5921 | 0.8298 | $14,636.7 | 35.1 |
| **3000** | `p14_opt_separable_qsvm_k16` | 16 | **0.7369** | 0.0019 | 0.9824 | 0.6890 | 0.8201 | $14,766.7 | 57.7 |
| **3000** | `p14_anisotropic_qsvm_k16` | 16 | 0.7221 | 0.0032 | 0.9710 | 0.4493 | 0.8448 | $14,786.7 | 308.3 |
| **3000** | `p14_hybrid_rbf_qsvm_k16` | 16 | 0.7367 | 0.0034 | 0.9823 | 0.6546 | 0.8254 | $14,676.7 | 81.5 |
| **Macro** | `macro_classical_linear_svm` | 16 | 0.7122 | 0.0042 | 0.9808 | 0.0735 | 0.9004 | $14,400.0 | 306.4 |
| **Macro** | `macro_nystroem_rbf_approx` | 16 | **0.7705** | 0.0065 | 0.9834 | 0.0764 | 0.9074 | **$13,716.7** | 438.2 |

---

### Table B: 15-Fold Individual Cross-Validation Distribution Metrics
*Computed across all 15 individual validation folds (15 distinct folds $\times$ $N_{\text{val}} = 45,396$).*

| Training Size | Model Architecture | Mean PR-AUC | Std PR-AUC | Min PR-AUC | Max PR-AUC | Mean ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **2000** | `baseline_classical_linear_svm_k6` | 0.7059 | 0.0256 | 0.6659 | 0.7520 | 0.9675 |
| **2000** | `baseline_classical_rbf_svm_k6` | 0.7326 | 0.0308 | 0.6863 | 0.7779 | 0.9670 |
| **2000** | `baseline_opt_separable_qsvm_k6` | 0.7055 | 0.0332 | 0.6602 | 0.7598 | 0.9681 |
| **2000** | `baseline_canonical_zz_qsvm_k6` | 0.4362 | 0.1330 | 0.1623 | 0.6259 | 0.9637 |
| **2000** | `baseline_projected_quantum_kernel_k6` | 0.7091 | 0.0438 | 0.6399 | 0.7806 | 0.9674 |
| **2000** | `p14_classical_rbf_k10` | 0.7413 | 0.0331 | 0.6721 | 0.7928 | 0.9819 |
| **2000** | `p14_opt_separable_qsvm_k10` | 0.7419 | 0.0334 | 0.6704 | 0.7926 | 0.9815 |
| **2000** | `p14_classical_linear_svm_k16` | 0.6736 | 0.0552 | 0.5886 | 0.7651 | 0.9774 |
| **2000** | `p14_classical_rbf_k16` | 0.7430 | 0.0336 | 0.6704 | 0.7986 | 0.9820 |
| **2000** | `p14_opt_separable_qsvm_k16` | 0.7432 | 0.0348 | 0.6637 | 0.7973 | 0.9815 |
| **2000** | `p14_anisotropic_qsvm_k16` | 0.7286 | 0.0316 | 0.6686 | 0.7768 | 0.9696 |
| **2000** | `p14_hybrid_rbf_qsvm_k16` | **0.7435** | 0.0342 | 0.6666 | 0.7976 | 0.9807 |
| **3000** | `baseline_classical_linear_svm_k6` | 0.7095 | 0.0258 | 0.6698 | 0.7471 | 0.9673 |
| **3000** | `baseline_classical_rbf_svm_k6` | 0.7290 | 0.0307 | 0.6874 | 0.7788 | 0.9670 |
| **3000** | `baseline_opt_separable_qsvm_k6` | 0.7066 | 0.0304 | 0.6649 | 0.7534 | 0.9675 |
| **3000** | `baseline_canonical_zz_qsvm_k6` | 0.4117 | 0.0913 | 0.1773 | 0.5161 | 0.9637 |
| **3000** | `baseline_projected_quantum_kernel_k6` | 0.7045 | 0.0393 | 0.6453 | 0.7592 | 0.9674 |
| **3000** | `p14_classical_rbf_k10` | 0.7392 | 0.0328 | 0.6743 | 0.7931 | 0.9826 |
| **3000** | `p14_opt_separable_qsvm_k10` | 0.7404 | 0.0334 | 0.6704 | 0.7925 | 0.9822 |
| **3000** | `p14_classical_linear_svm_k16` | 0.6934 | 0.0390 | 0.6308 | 0.7561 | 0.9798 |
| **3000** | `p14_classical_rbf_k16` | 0.7422 | 0.0345 | 0.6672 | 0.7985 | 0.9829 |
| **3000** | `p14_opt_separable_qsvm_k16` | 0.7410 | 0.0353 | 0.6587 | 0.7956 | 0.9824 |
| **3000** | `p14_anisotropic_qsvm_k16` | 0.7276 | 0.0311 | 0.6750 | 0.7721 | 0.9710 |
| **3000** | `p14_hybrid_rbf_qsvm_k16` | 0.7418 | 0.0349 | 0.6626 | 0.7961 | 0.9823 |
| **Macro** | `macro_classical_linear_svm` | 0.7176 | 0.0321 | 0.6559 | 0.7615 | 0.9808 |
| **Macro** | `macro_nystroem_rbf_approx` | **0.7718** | 0.0439 | 0.6957 | 0.8398 | 0.9834 |

---

## 4. Rigorous Statistical Significance Testing (15 Paired Folds)

To verify whether quantum models outperform matched classical controls, paired two-tailed $t$-tests and non-parametric Wilcoxon signed-rank tests were conducted across all 15 matched folds:

### Sample Size $N=2,000$:
* **$K=6$: OptQSVM vs. Classical RBF**:
  - Mean difference: $-0.0271$ (Classical RBF is superior)
  - Paired $t$-test: $t = -9.576, p = 1.1 \times 10^{-7}$
  - Wilcoxon signed-rank test: $W = 0.0, p = 0.0001$ (Statistically significant classical superiority)
* **$K=10$: OptQSVM vs. Classical RBF**:
  - Mean difference: $+0.0006$
  - Paired $t$-test: $t = 1.836, p = 0.0877$
  - Wilcoxon signed-rank test: $p = 0.1354$ (No statistically significant difference; parity)
* **$K=16$: OptQSVM vs. Classical RBF**:
  - Mean difference: $+0.0002$
  - Paired $t$-test: $t = 0.340, p = 0.7388$
  - Wilcoxon signed-rank test: $p = 0.6788$ (No statistically significant difference; parity)
* **$K=16$: Hybrid RBF-QSVM vs. Classical RBF**:
  - Mean difference: $+0.0005$
  - Paired $t$-test: $t = 1.235, p = 0.2371$
  - Wilcoxon signed-rank test: $p = 0.0479$ (Marginally significant on rank sum, but absolute effect size $+0.0005$ is negligible)
* **$K=16$ vs. $K=6$ Classical RBF Feature Expansion**:
  - Mean difference: $+0.0104$
  - Paired $t$-test: $t = 2.781, p = 0.0147$ (Statistically significant improvement from expanding feature dimensions)

### Sample Size $N=3,000$:
* **$K=6$: OptQSVM vs. Classical RBF**:
  - Mean difference: $-0.0224$
  - Paired $t$-test: $t = -10.155, p = 5.9 \times 10^{-8}$ (Statistically significant classical superiority)
* **$K=10$: OptQSVM vs. Classical RBF**:
  - Mean difference: $+0.0012$
  - Paired $t$-test: $t = 2.053, p = 0.0593$
  - Wilcoxon signed-rank test: $p = 0.0554$ (Marginal trend toward parity, $p > 0.05$)
* **$K=16$: OptQSVM vs. Classical RBF**:
  - Mean difference: $-0.0012$
  - Paired $t$-test: $t = -1.287, p = 0.2188$
  - Wilcoxon signed-rank test: $p = 0.3028$ (No statistically significant difference; parity)
* **$K=16$: Hybrid RBF-QSVM vs. Classical RBF**:
  - Mean difference: $-0.0004$
  - Paired $t$-test: $t = -0.600, p = 0.5579$ (No difference)
* **$K=16$ vs. $K=6$ Classical RBF Feature Expansion**:
  - Mean difference: $+0.0132$
  - Paired $t$-test: $t = 3.622, p = 0.0028$ (Statistically significant feature dimension benefit)

---

## 5. Comparison with Phase 13 Baselines

| Evaluation Dimension | Phase 13 Protocol | Phase 14 Optimization Protocol | Key Observation |
| :--- | :--- | :--- | :--- |
| **Feature Dimension** | Fixed at $K=6$ | Expanded to $K \in \{6, 10, 16\}$ | Adding features from $K=6$ to $K=16$ boosted RBF PR-AUC from 0.7215 to 0.7350 (+0.0135). |
| **Quantum Bandwidth** | Fixed $\beta=0.50$ (separable) | Dimensionally scaled $\beta(K) = \beta_0 \sqrt{6/K}$ | Prevents exponential Gram matrix collapse as $K$ increases to 16. |
| **Sample Sizes** | $N \in \{500, 1000, 2000\}$ | $N \in \{2000, 3000\}$ | Scaling $N=2,000 \rightarrow 3,000$ provided negligible gain for kernel machines (+0.0006). |
| **Best Small-Sample Model** | Classical RBF ($N=500$, PR-AUC = 0.7287) | Hybrid / OptQSVM ($K=16$, PR-AUC = 0.7369) | Measurable +0.0082 OOF gain from feature expansion and hybrid blending. |
| **Best Overall Model** | Full Linear ($N=181k$, PR-AUC = 0.7055) | Full Nystroem RBF ($N=181k$, PR-AUC = 0.7705) | Macro Nystroem RBF on all data clearly outperforms subsampled kernel machines. |

---

## 6. Runtime, Memory, and System Profile

* **Total Benchmark Runtime**: 6,618.45 seconds (**110.31 minutes**, ~1.84 hours)
* **Execution Breakdown**:
  - Seed 42 (5 folds): 2,119.16s (~35.3 min)
  - Seed 101 (5 folds): 2,339.06s (~39.0 min)
  - Seed 2024 (5 folds): 2,132.76s (~35.5 min)
  - Average per-fold runtime: ~441 seconds (~7.3 minutes per fold across 2 sample sizes $\times$ 12 models + 2 macro models).
* **Peak Memory Allocation**: **8,571.93 MB** (~8.57 GB)
* **Stability & Reliability**: **15 / 15 folds completed successfully (0 failed folds)**.
* **Test Suite Verification**: 96 / 96 unit tests passing suite-wide across [`tests/test_phase14_optimization.py`](file:///Users/harsha/Desktop/Projects-files/quantum-fraud-detection(Capstone)/tests/test_phase14_optimization.py) and historical test modules.

---

## 7. Data Leakage and Validation Integrity Audit

1. **Historical Holdout Test Set**:
   `data/processed/X_test.csv` was **strictly isolated and never accessed** during Phase 14 training, feature selection, subset sampling, or metric evaluation.
2. **In-Fold Feature Selection**:
   ANOVA $F$-statistic ranking (`select_top_features_infold`) was executed strictly on $X_{\text{tr\_raw}}, y_{\text{tr\_raw}}$. Validation rows and labels were completely unseen during feature ranking.
3. **In-Fold Angle Scaling**:
   `MinMaxScaler(feature_range=(0.0, np.pi))` was fitted strictly on each fold's training subset ($X_{\text{tr\_sub}}$) and applied out-of-sample to validation instances ($X_{\text{val}}$).
4. **Natural Validation Prevalence**:
   All 15 validation folds maintained natural transaction prevalence ($\pi \approx 0.165\%$, ~75 frauds in 45,396 validation instances), preserving realistic real-world class imbalance.
5. **Threshold Calibration Integrity**:
   - Fold-level calibrated costs ($C = 10\text{FP} + 200\text{FN}$) are calculated using candidate thresholds derived from `precision_recall_curve(y_true, scores)`.
   - *Methodological Disclosure*: While useful to illustrate empirical cost-minimizing potential, picking $\tau^*$ directly on validation fold scores represents an optimistic bound. In true operational deployment, $\tau^*$ should be tuned on an inner cross-validation loop. Therefore, we emphasize **PR-AUC** as the primary rank-based, threshold-independent evaluation metric.

---

## 8. Strategic Recommendation for Next Optimization

Based on the empirical evidence across all 15 folds:
1. **Subsample Saturation**: Increasing subset size from $N=2,000$ to $N=3,000$ produced negligible improvement for kernel SVMs ($0.7368 \rightarrow 0.7369$). Subsampled kernel machines appear fundamentally bottlenecked around PR-AUC $\approx 0.74$.
2. **Data-Scale Advantage**: The Full-Data Nystroem RBF model achieved PR-AUC = **$0.7705$**, demonstrating that exposure to the full training distribution ($N=181,584$) is far more impactful than increasing quantum kernel complexity on a tiny subset ($N=3,000$).
3. **The Single Most Promising Next Optimization**:
   **Supervised Kernel Target Alignment (KTA) / Trainable Projected Quantum Kernel (PQK)**:
   Currently, the separable QSVM uses static isotropic bandwidths ($\beta_j = \text{const}$). Anisotropic weighting using raw ANOVA scores reached only $0.7221$, because linear ANOVA ignores non-linear feature interactions.
   By directly optimizing kernel parameters $\theta$ (or local measurement basis angles in PQK) using gradient ascent on the Kernel Target Alignment:
   $$A(K_\theta, Y) = \frac{\langle K_\theta, Y \rangle_F}{\|K_\theta\|_F \|Y\|_F}, \quad Y = y y^T$$
   the quantum feature map can be tailored to the fraudulent manifold, potentially bridging the gap toward the $0.80+$ regime.
