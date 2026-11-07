# Phase 15: Trainable Quantum Kernel Optimization via Centered Regularized KTA (Audited Pilot Report)

**Capstone Project:** Quantum Machine Learning for Financial Fraud Detection using Quantum Kernel Methods  
**Date:** October 2026  
**Phase:** Phase 15 — Stage 2 Pilot Execution (Fold 0, $N_{\text{val}} = 45,396$, 75 fraud cases, natural transaction prevalence $\pi = 0.165\%$)  
**Status:** Audited, Leakage-Free, and Empirically Validated  
**Primary Artifacts:**
- Implementation Module: [`src/experiments/phase15_trainable_kernels.py`](file:///Users/harsha/Desktop/Projects-files/quantum-fraud-detection(Capstone)/src/experiments/phase15_trainable_kernels.py)
- Execution Runner: [`scripts/run_phase15_experiments.py`](file:///Users/harsha/Desktop/Projects-files/quantum-fraud-detection(Capstone)/scripts/run_phase15_experiments.py)
- Test Suite: [`tests/test_phase15_trainable_kernels.py`](file:///Users/harsha/Desktop/Projects-files/quantum-fraud-detection(Capstone)/tests/test_phase15_trainable_kernels.py) (8/8 passing; 104/104 project-wide passing)
- Pilot Metrics JSON: [`reports/phase15/pilot_metrics.json`](file:///Users/harsha/Desktop/Projects-files/quantum-fraud-detection(Capstone)/reports/phase15/pilot_metrics.json)
- Pilot Execution Log: [`reports/phase15/phase15_execution.log`](file:///Users/harsha/Desktop/Projects-files/quantum-fraud-detection(Capstone)/reports/phase15/phase15_execution.log)

---

## 1. Mathematical Audit of Centered Regularized KTA

### 1.1 The Mathematical Challenge of KTA under Severe Class Imbalance
In standard (uncentered) Kernel Target Alignment:
$$\text{KTA}(K_\theta, Y) = \frac{\langle K_\theta, Y \rangle_F}{\|K_\theta\|_F \|Y\|_F}, \quad Y = y^* (y^*)^T, \quad y^* \in \{-1, +1\}^N$$

In credit card fraud detection where legitimate transactions constitute $99.835\%$ of transactions ($\pi \approx 0.165\%$):
Even in a 10% fraud-enriched training subset ($p = 0.10, q = 0.90$):
- $q^2 = 81\%$ of all instance pairs are legitimate–legitimate pairs ($y^*_i y^*_j = +1$).
- $p^2 = 1\%$ of all pairs are fraud–fraud pairs ($y^*_i y^*_j = +1$).
- $2pq = 18\%$ of all pairs are cross-class pairs ($y^*_i y^*_j = -1$).

Because uncentered KTA treats all matching pairs equally ($Y_{ij} = +1$ for both legitimate–legitimate and fraud–fraud), the optimizer maximizes alignment on the overwhelming majority legitimate pairs by driving feature bandwidths $\theta \rightarrow \infty$ (forcing $K_{ij} \rightarrow 1$ everywhere). In our preliminary empirical tests, uncentered KTA caused catastrophic collapse of PR-AUC from 0.7688 to **0.5870**.

### 1.2 Mathematical Derivation of Centered KTA (CKTA)
To eliminate majority-pair dominance, we apply the centering projection matrix:
$$H = \mathbf{I}_N - \frac{1}{N} \mathbf{1} \mathbf{1}^T$$

The centered target matrix is:
$$Y_c = H Y H = H y^* (y^*)^T H = (H y^*) (H y^*)^T = u u^T$$
where $u = y^* - \bar{y}^* \mathbf{1}$.

#### Analytical Prevalence Compensation Proof:
Let $p$ be the fraction of fraud instances and $q = 1 - p$ be the fraction of legitimate instances.
The mean bipolar label is $\bar{y}^* = p(+1) + q(-1) = p - q = 2p - 1$.
- For a fraud instance $i$ ($y^*_i = +1$):
  $$u_i = +1 - (2p - 1) = 2(1 - p) = 2q$$
- For a legitimate instance $j$ ($y^*_j = -1$):
  $$u_j = -1 - (2p - 1) = -2p$$

Now examine the pair weights in $Y_c = u u^T$:
- **Fraud–Fraud pair:** $(Y_c)_{i, i'} = u_i u_{i'} = (2q)^2 = 4q^2$. For $p = 0.10, q = 0.90 \implies (Y_c)_{i, i'} = 3.24$.
- **Legitimate–Legitimate pair:** $(Y_c)_{j, j'} = u_j u_{j'} = (-2p)^2 = 4p^2$. For $p = 0.10, q = 0.90 \implies (Y_c)_{j, j'} = 0.04$.
- **Cross-class pair:** $(Y_c)_{i, j} = u_i u_j = (2q)(-2p) = -4pq = -0.36$.

**Analytical Weighting Ratio:**
$$\frac{(Y_c)_{\text{fraud-fraud}}}{(Y_c)_{\text{legit-legit}}} = \left(\frac{q}{p}\right)^2 = \left(\frac{0.90}{0.10}\right)^2 = 81$$

The $81\times$ analytical weighting ratio under centering **exactly offsets** the $(0.90 / 0.10)^2 = 81$-fold greater frequency of legitimate–legitimate pairs.

#### Rank-1 Quadratic Equivalence Proof:
Because $H$ is a projection matrix ($H^2 = H$) and $H u = u$:
$$\langle K_c, Y_c \rangle_F = \text{Tr}(K_c Y_c) = \text{Tr}(K_c u u^T) = u^T K_c u = u^T (H K H) u = (H u)^T K (H u) = u^T K u$$
Furthermore, because $Y_c = u u^T$ is rank 1:
$$\|Y_c\|_F = \sqrt{\text{Tr}((u u^T)(u u^T))} = \sqrt{(u^T u)^2} = u^T u = \sum_{i=1}^N (y^*_i - \bar{y}^*)^2 = N \cdot \text{Var}(y^*)$$
And the centered kernel Frobenius norm is computed in $O(N^2)$ via:
$$(K_c)_{ij} = K_{ij} - \bar{K}_{i\cdot} - \bar{K}_{\cdot j} + \bar{K}_{\cdot\cdot}$$
$$\|K_c\|_F = \sqrt{\sum_{i,j=1}^N (K_c)_{ij}^2}$$

Both Frobenius inner products and quadratic forms match exactly to machine precision ($< 10^{-14}$), as validated in unit test `test_centered_kta_exact_math_equivalence`.

### 1.3 Prior Regularization
Unregularized CKTA allows individual feature bandwidths to drop toward zero on noisy features, causing feature suppression. We anchor the bandwidth vector $\theta \in \mathbb{R}^K$ to the geometric scaling baseline prior $\beta_0 = 0.05 \sqrt{6/K} \approx 0.0306$ (for $K=16$):
$$\mathcal{L}(\theta) = - \frac{u^T K_\theta u}{\|K_c\|_F (u^T u)} + \lambda \sum_{j=1}^K (\theta_j - \beta_0)^2$$
where $\lambda = 200.0$ and box constraints $\theta_j \in [0.005, 0.15]$ ensure numerical stability and prevent dimensional collapse.

---

## 2. In-Training Leakage-Free Protocol Audit

All four stages of the learning pipeline are strictly isolated inside each training fold:
1. **Feature Selection ($K=16$):** Evaluated strictly on `X_train_fold, y_train_fold` using ANOVA F-statistic. Outer validation fold features are never ranked or inspected.
2. **Preprocessing:** `MinMaxScaler(feature_range=(0, np.pi))` is fit strictly on the training subset $X_{\text{sub}}$ and then applies `.transform()` to outer validation fold $X_{\text{val}}$.
3. **Kernel Optimization:** KTA optimization of bandwidths $\theta$ or $\gamma$ runs strictly on $X_{\text{sub}}, y_{\text{sub}}$ inside the training fold.
4. **Threshold Calibration:**
   - Previous phases (Phases 13 and 14) calibrated decision thresholds by scanning validation fold scores directly, introducing validation leakage into threshold selection.
   - **Phase 15 Strict Fix:** Threshold calibration is executed strictly on **in-training out-of-fold decision scores** generated via an inner 3-fold `StratifiedKFold` on the training subset $(X_{\text{sub}}, y_{\text{sub}})$.
   - To account for subset fraud enrichment ($10\%$) vs natural population prevalence ($\pi_{\text{tr}} = \text{mean}(y_{\text{train\_fold}}) \approx 0.165\%$), expected cost is minimized analytically:
     $$\mathbb{E}[\text{Cost}](\tau) = c_{\text{FP}} (1 - \pi_{\text{tr}}) \text{FPR}(\tau) + c_{\text{FN}} \pi_{\text{tr}} \text{FNR}(\tau)$$
     where $\text{FPR} = \text{FP} / N_{\text{sub},-}$ and $\text{FNR} = \text{FN} / N_{\text{sub},+}$.
   - The calibrated threshold $\tau^*$ is frozen and applied to the outer validation fold. **Validation fold labels are never used to select, tune, or calibrate thresholds.**
5. **Historical Test Isolation:** `data/processed/X_test.csv` is completely isolated and was never accessed.

---

## 3. Matched Comparison Benchmark Results (Stage 2 Pilot Fold 0)

All five subset models are trained on the **exact same training subset** ($N=2,000$, 200 frauds, 1,800 legitimate) and evaluated on the **exact same validation fold** ($N_{\text{val}} = 45,396$, 75 frauds, natural transaction prevalence $\pi = 0.165\%$):

### 3.1 Primary Performance Table

| Model Architecture | Kernel Type | Training / Tuning Protocol | PR-AUC | ROC-AUC | Calibrated Threshold $\tau^*$ | Calibrated Prec (%) | Calibrated Rec (%) | Calibrated FP | Cost ($R=20$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `baseline_classical_linear_k6` | Linear ($K=6$) | In-fold CV QP ($C=5.0$) | 0.7025 | 0.9741 | $+0.9150$ | 69.23% | 72.00% | 24 | $4,440 |
| `p14_classical_rbf_k16` | Classical RBF ($K=16$) | Fixed wide-margin ($\gamma=0.0005$) | 0.7640 | 0.9754 | $-0.1427$ | 56.36% | 82.67% | 48 | $3,080 |
| `p14_opt_separable_qsvm_k16` | Separable QSVM ($K=16$) | Fixed geometric prior ($\beta=0.0306$) | **0.7686** | 0.9759 | $-0.3271$ | 60.19% | 82.67% | 41 | $3,010 |
| `p15_kta_trainable_qsvm_k16` | **Trainable QSVM ($K=16$)** | **Trainable CKTA ($\theta \in [0.0287, 0.0342]$)** | **0.7644** | **0.9762** | **$-0.2766$** | **62.63%** | **82.67%** | **37** | **$2,970** |
| `p15_kta_trainable_rbf_k16` | Trainable RBF ($K=16$) | Trainable CKTA ($\gamma \in [0.00005, 0.0021]$) | 0.7229 | 0.9789 | $+0.3869$ | 63.54% | 81.33% | 35 | $3,150 |
| `macro_classical_linear_svm` | Linear ($K=16$) | Full-data ($N=181,584$) | 0.7077 | 0.9781 | $+0.8546$ | 47.29% | 81.33% | 68 | $3,480 |
| `macro_nystroem_rbf_approx` | Nystroem RBF ($K=16$)| Full-data ($N=181,584, M=300$) | 0.7251 | 0.9838 | $+0.8895$ | 56.60% | 80.00% | 46 | $3,460 |

### 3.2 Default-Threshold ($\tau = 0.0$) Reference Table

| Model Architecture | Default Precision (%) | Default Recall (%) | Default False Positives | Default False Negatives | Default Cost ($R=20$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `baseline_classical_linear_k6` | 10.81% | 85.33% | 528 | 11 | $7,480 |
| `p14_classical_rbf_k16` | 65.59% | 81.33% | 32 | 14 | $3,120 |
| `p14_opt_separable_qsvm_k16` | 78.67% | 78.67% | 16 | 16 | $3,360 |
| `p15_kta_trainable_qsvm_k16` | 74.07% | 80.00% | 21 | 15 | $3,210 |
| `p15_kta_trainable_rbf_k16` | 21.77% | 85.33% | 230 | 11 | $4,500 |
| `macro_classical_linear_svm` | 7.24% | 89.33% | 858 | 8 | $10,180 |
| `macro_nystroem_rbf_approx` | 7.54% | 89.33% | 822 | 8 | $9,820 |

---

## 4. Scientific Audit & Correction of Pilot Claims

### 4.1 Strict Removal of Single-Fold Quantum Advantage Inferences
In earlier exploratory summaries, it was suggested that Trainable QSVM demonstrated "superior inductive bias" or "quantum advantage" over Classical RBF based on Fold 0 metrics. **Such claims are scientifically invalid and are formally retracted.**

Evidence:
1. On Fold 0:
   - Fixed QSVM achieves PR-AUC $= 0.7686$.
   - Trainable QSVM achieves PR-AUC $= 0.7644$.
   - Fixed Classical RBF achieves PR-AUC $= 0.7640$.
   The difference between Fixed Classical RBF (0.7640) and Trainable QSVM (0.7644) is **only $+0.0004$ PR-AUC**, which is indistinguishable from numerical noise.
2. The difference between Fixed QSVM (0.7686) and Fixed Classical RBF (0.7640) is $+0.0046$, which is well within the typical fold-to-fold standard deviation observed in Phase 14 ($\sigma \approx 0.030$).
3. **No claim of quantum advantage can or should be made from a single pilot fold.** Definitive scientific conclusions require the full 15-fold benchmark across 3 random seeds with paired Wilcoxon signed-rank and paired $t$-tests.

### 4.2 Observations on KTA Adaptation
- **Quantum Kernel Adaptation:** Regularized CKTA successfully tuned the 16 bandwidth parameters $\theta_j$ from initial $\beta_0 = 0.0306$ into the range $[0.0287, 0.0342]$ in 9.94 seconds, reaching $\text{CKTA} = 0.5785$. At calibrated threshold $\tau^* = -0.2766$, Trainable QSVM achieved the lowest financial cost ($\$2,970$) and lowest false positives ($37$ FP) among models with $\ge 82.67\%$ recall on this fold.
- **Classical RBF KTA Adaptation:** Anisotropic classical RBF KTA adapted $\gamma_j \in [0.00005, 0.00206]$ in 15.63 seconds, reaching $\text{CKTA} = 0.6222$. However, its PR-AUC on validation was 0.7229, demonstrating that higher training KTA does not automatically translate into higher test PR-AUC when the model capacity overfits anisotropic feature scaling.

---

## 5. Computational Resource Profile & Benchmark Projections

| Metric | Measured Stage 2 Pilot (Fold 0, N=2,000) | Full Benchmark Projection ($N \in \{2000, 3000\}$, 15 Folds) |
| :--- | :---: | :---: |
| **Total Runtime** | **81.52 seconds (1.36 minutes)** | **~40–45 minutes** |
| **Peak Memory Allocation** | **2,538.87 MB (~2.54 GB)** | **$\le 3.2$ GB** |
| **RAM Safety Margin** | 5.46 GB free headroom on 8 GB system | Safe for single-process background execution |
| **Test Suite Execution** | 104/104 passing in 11.91s | Verified |

---

## 6. Pre-Benchmark Checklist Status

- [x] **Audit CKTA Mathematics:** Centered kernel matrix and centered label matrix formulas verified; rank-1 quadratic form $\langle K_c, Y_c \rangle_F = u^T K u$ and $\|Y_c\|_F = u^T u$ proven and verified to machine precision ($< 10^{-14}$).
- [x] **Prevalence Compensation:** $81\times$ analytical weighting ratio verified to exactly offset majority legitimate pair dominance under class imbalance.
- [x] **In-Fold Isolation:** Feature selection, scaling, KTA kernel training, and threshold calibration occur strictly inside training folds.
- [x] **Threshold Calibration Leakage Removed:** Financial-cost thresholds are calibrated strictly on training-only inner out-of-fold predictions with analytical population prevalence adjustment. Zero validation labels used during calibration.
- [x] **Matched Comparison:** Fixed QSVM, trainable QSVM, fixed classical RBF, and trainable classical RBF evaluated on identical training subsets and validation fold.
- [x] **PR-AUC Primary Metric:** PR-AUC and ROC-AUC reported alongside default and calibrated cost metrics.
- [x] **Claims Corrected:** Single-fold quantum advantage inferences completely removed and retracted.
- [x] **All Tests Passing:** 8/8 Phase 15 unit tests and 104/104 full suite unit tests pass.
- [x] **Historical Test Isolation:** `data/processed/X_test.csv` strictly untouched.
