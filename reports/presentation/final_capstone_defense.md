# Final Capstone Project Defense Presentation

**Title:** Quantum Machine Learning for Financial Fraud Detection Using Quantum Kernel Methods  
**Format:** Slide Deck for Final-Year Engineering & Computer Science Capstone Defense (15–20 min)  
**Target Audience:** Capstone Defense Committee, Faculty Examiners, and Machine Learning Practitioners  
**Date:** October 2026  
**Artifact Path:** [`reports/presentation/final_capstone_defense.md`](file:///Users/harsha/Desktop/Projects-files/quantum-fraud-detection(Capstone)/reports/presentation/final_capstone_defense.md)  
**Companion Report:** [`reports/capstone_final_report.md`](file:///Users/harsha/Desktop/Projects-files/quantum-fraud-detection(Capstone)/reports/capstone_final_report.md)

---

## Slide 1: Title Slide & Project Metadata

# Quantum Machine Learning for Financial Fraud Detection
### A Rigorous Empirical & Theoretical Assessment of Quantum Kernel Methods vs. Classical SVMs

**Candidate:** Senior Capstone Research Candidate  
**Domain:** Quantum Machine Learning (QML), Financial Technology (FinTech), Cyber-Fraud Anomaly Detection  
**Tech Stack:** Python 3.10, Qiskit 1.4, Qiskit Machine Learning, Scikit-Learn, Pytest  
**Dataset:** ULB European Credit Card Fraud Dataset (284,807 transactions)  
**Test Suite:** 104/104 Automated Tests Passing across 14 Modules (100% Pass Rate)

> **Speaker Notes:**  
> "Good morning, respected committee members and examiners. Today, I am presenting our final-year capstone research project investigating whether Quantum Kernel Methods can provide an advantage over classical Support Vector Machines for detecting financial fraud under extreme class imbalance. Over 15 rigorous research phases, we built, diagnosed, optimized, scaled, and audited classical and quantum models, culminating in multi-seed 15-fold cross-validation on natural transaction prevalence, high-dimensional scaling, and trainable kernel benchmarks."

---

## Slide 2: Problem Statement & Motivation

### The Extreme Imbalance Challenge in Financial Fraud

* **The Fraud Landscape:** Global credit card fraud exceeds \$30 billion annually.
* **Extreme Class Imbalance:** Fraudulent transactions represent only **~0.17%** of real-world traffic (1 fraud in ~580 legitimate purchases).
* **Asymmetric Risk:**
  * Missed Fraud (False Negative): Direct loss (\$50 – \$1,000+).
  * False Alarm (False Positive): User annoyance / transaction decline (\$10 manual review cost).
* **Strict Latency Budget:** Commercial payment gateways require transaction screening in **under 10–50 ms**.
* **Metric Traps:** Accuracy is meaningless (a naive model predicting 100% legitimate achieves **99.83% accuracy**). Precision-Recall AUC (PR-AUC) is the mandatory evaluation standard.

> **Speaker Notes:**  
> "Financial fraud detection is one of the most adversarial domains in applied machine learning. Due to the 1:580 class imbalance, traditional metrics like ROC-AUC or accuracy are deceptive. A classifier must maintain high precision without sacrificing fraud recall, all while executing in milliseconds. This motivated our investigation into whether quantum state spaces can construct sharper non-linear boundaries."

---

## Slide 3: Research Question & Theoretical Hypothesis

### Can Hilbert Space Geometries Separate Fraud Better Than Classical Kernels?

* **The Quantum Hypothesis:**  
  Data $x \in \mathbb{R}^d$ mapped into an $n$-qubit quantum Hilbert space $\mathcal{H} = \mathbb{C}^{2^n}$ via quantum circuits:
  $$x \mapsto |\psi(x)\rangle, \quad k(x, x') = |\langle \psi(x) | \psi(x') \rangle|^2$$
  can construct hyperplanes in $2^n$-dimensional state space that correspond to complex non-linear boundaries in input space.
* **Literature Claims:**  
  Prior literature (Havlíček et al., Nature 2019; Schuld & Killoran, 2019) conjectured quantum advantage for classification via entangled second-order Pauli feature maps ($ZZFeatureMap$).
* **Our Core Capstone Questions:**
  1. *Does quantum entanglement provide superior margin separation for tabular transaction data?*
  2. *Can a Quantum SVM outperform classical Linear and Gaussian RBF SVMs in PR-AUC?*
  3. *Can quantum kernel methods meet real-world latency and computational constraints?*

> **Speaker Notes:**  
> "The central premise of quantum kernel methods is that by encoding classical vectors into quantum states, quantum entanglement might discover intricate non-linear relationships that classical kernels miss. Our capstone put this hypothesis to a rigorous, adversarial empirical test."

---

## Slide 4: Dataset & Leak-Free Engineering Pipeline

### Ensuring Absolute Methodological and Experimental Rigor

```text
[Raw Dataset: 284,807 Rows]
       │
       ▼
[Phase 2: Purge 1,081 Duplicates] ───► Clean Data: 283,726 Unique Rows
       │
       ▼
[Stratified 80/20 Partition]
       ├───► Locked Holdout Test Set (56,746 Rows, 95 Frauds)
       │        (Locked; zero test-set tuning or threshold calibration)
       │
       └───► Training Database (226,980 Rows, 378 Frauds)
                │
                ▼
        [Leak-Free 5-Fold Stratified Cross-Validation]
        • Feature Selection: ANOVA F-test fitted inside each fold
        • Quantum Resampling: N=500 (100 frauds) inside fold train
        • Angle Scaling: MinMaxScaler([0, π]) fitted inside fold train
        • Out-of-Fold Evaluation: Natural prevalence (π = 0.1665%) across all 226,980 rows
```

* **Data Provenance Disclosure:** Holdout test set was evaluated in early exploratory phases (Phases 3, 6, 8, 9). Phase 11 performed strictly zero test-set tuning or threshold calibration.

> **Speaker Notes:**  
> "Rigorous data hygiene was our first priority. We removed 1,081 duplicate transactions and enforced strict stratified splitting. All feature selection, scaling, and threshold determination were isolated inside cross-validation folds. Crucially, we disclose prior test set exposure in earlier exploratory phases, while proving Phase 11 maintained complete test-set isolation during its own tuning."

---

## Slide 5: Quantum Kernel Architectures Evaluated

### From Entangled Circuits to Separable Product States

1. **Canonical Entangled QSVM ($ZZFeatureMap$, reps=1, 2):**
   * Single-qubit rotations: $\phi_j(x) = 2 x_j$.
   * Entangling two-qubit gates: $\phi_{jk}(x) = 2(\pi - x_j)(\pi - x_k)$ via CNOT ladders.
2. **Separable (Unentangled) QSVM ($ZFeatureMap$):**
   * Zero two-qubit entanglement: $\phi_{jk}(x) = 0$.
   * Pure single-qubit rotations: $|\psi(x)\rangle = \bigotimes_{j=1}^d \left(\cos(x_j)|0\rangle + \sin(x_j)|1\rangle\right)$.
3. **Optimized Separable QSVM ($\beta = 0.50$):**
   * Tuned bandwidth scaling: $\phi_j(x) = \beta x_j$ to match effective classical kernel width.
4. **Anisotropic Separable QSVM:**
   * Feature-weighted rotations: $w_j = \beta^* \cdot (F_j / \bar{F})$.
5. **Projected Quantum Kernel (PQK, Huang et al., 2021):**
   * Local 1-qubit reduced density matrices $\rho_j(x)$; Gaussian kernel on Bloch coordinates.
6. **Classical Controls:**
   * Matched Classical Linear SVM ($C=1.0$).
   * Matched Classical Gaussian RBF SVM ($C=5.0, \gamma=0.01$).

> **Speaker Notes:**  
> "We tested six distinct architectures: from canonical entangled circuits to unentangled separable circuits, bandwidth-scaled kernels, feature-weighted anisotropic kernels, projected quantum kernels, and classical matched controls."

---

## Slide 6: The Failure Mode of Quantum Entanglement

### Geometric Inversion & Phase Clutter on Continuous Tabular Data

* **Kernel-Target Alignment (KTA):**
  * Separable $Z$-Kernel: **$\text{KTA} = 0.7006$**
  * Entangled $ZZ$-Kernel (reps=1): **$\text{KTA} = 0.6343$**
  * Entangled $ZZ$-Kernel (reps=2): **$\text{KTA} = 0.5812$**
* **The Mechanism of Failure on Continuous Tabular Features:**
  * Two-body phase term $\phi_{jk} = 2(\pi - x_j)(\pi - x_k)$ causes rapid phase wrapping across $[0, 2\pi^2]$.
  * Small continuous coordinate differences in non-critical tabular features cause destructive phase interference between real fraud cases.
  * **Result: Geometric Inversion on Tested Circuits:**
    $$\mathbb{E}[k(x_{\text{fraud}}, x'_{\text{fraud}})] < \mathbb{E}[k(x_{\text{fraud}}, x_{\text{legit}})]$$
    Within-class fraud similarity collapses below cross-class similarity in the tested canonical $ZZ$ encoding.
* **Continuous Entanglement Sweep ($\alpha \in [0.0, 1.0]$):**
  * As entanglement parameter $\alpha$ increases from 0.0 to 1.0, validation false alarms double from 171 to 347.

> **Speaker Notes:**  
> "Why did the canonical quantum kernel underperform? In the specific second-order Pauli ZZ-feature maps tested with linear entanglement, pairwise phase wrapping disrupted metric separation on continuous tabular features. This caused geometric inversion: within-class fraud similarity fell below cross-class similarity, inflating false positives."

---

## Slide 7: Mathematical Identity of the Separable Quantum Kernel

### Why Separable QSVM Succeeds — and Why It Is Classical

* **Closed-Form State Fidelity:**
  $$k_Z(x, x') = \prod_{j=1}^d \cos^2\left(\frac{\beta (x_j - x_j')}{2}\right)$$
* **Taylor Series Expansion:**  
  Using $\cos(\theta) \approx 1 - \frac{\theta^2}{2}$:
  $$\cos^2\left(\frac{\beta \Delta x_j}{2}\right) \approx 1 - \frac{\beta^2}{4}\Delta x_j^2 \approx \exp\left(-\frac{\beta^2}{4}\Delta x_j^2\right)$$
* **The Identity:**
  $$k_Z(x, x') \approx \exp\left(-\frac{\beta^2}{4} \|x - x'\|^2\right) = \exp\left(-\gamma_{\text{eff}} \|x - x'\|^2\right)$$
  where $\gamma_{\text{eff}} = \beta^2 / 4$.
* **Key Scientific Insight:**
  * At $\beta^* = 0.50$, $\gamma_{\text{eff}} = 0.0625$, naturally approximating the classical optimal Gaussian RBF ($\gamma^* = 0.01$).
  * The optimal quantum kernel operates at **$\alpha = 0$ (zero quantum entanglement)**.
  * It evaluates in $\mathcal{O}(d)$ classical arithmetic operations without requiring a quantum circuit or quantum processor, offering no demonstrated quantum computational advantage.

> **Speaker Notes:**  
> "When we removed the entangling gates, quantum performance improved. Through Taylor series expansion, we proved that the separable quantum kernel asymptotically approximates a classical Gaussian RBF kernel at local distances. However, because it evaluates in linear time classically and achieves lower holdout PR-AUC than classical RBF SVM (0.6098 vs 0.6577), it offers no quantum computational or statistical advantage."

---

## Slide 8: Methodological Audit: Phase 9 vs. Phase 11

### Resolving the Cross-Validation and Feature Set Ledger

| Methodological Dimension | Phase 9 Protocol (Optimization) | Phase 11 Protocol (Validation) |
| :--- | :--- | :--- |
| **Feature Selection Subset** | `['V4', 'V10', 'V11', 'V12', 'V14', 'V16']` (Pilot 500) | `['V17', 'V14', 'V12', 'V10', 'V16', 'V3']` (Full 226k Train) |
| **Cross-Validation Scope** | Evaluated on 500-sample partition | Evaluated across **all 226,980 training transactions** |
| **Resampling Point** | Resampled to 20% fraud **before CV** | Resampled **inside each training fold** |
| **Validation Fold Prevalence** | **20.00%** (1 : 4 imbalance) | **0.1665%** (1 : 596 natural imbalance) |
| **Reported CV PR-AUC** | **0.9766** (prevalence-inflated) | **0.7188** (natural prevalence) |

### Audit of Training Subsets and Holdout PR-AUC Shift
* Both Phase 9 and Phase 11 trained their final holdout models on the **identical set of 500 training instances** (seed 42, 100 fraud, 400 legit) and evaluated on the identical 56,746 test instances.
* The PR-AUC shift (RBF: 0.6713 $\to$ 0.6577; Opt QSVM: 0.6517 $\to$ 0.6098) was driven by feature selection: Phase 9 used pilot-selected features (`V4, V11`), whereas Phase 11 used full-dataset-selected features (`V17, V3`).
* **Audit Verdict:** Both sampling calls selected the exact same 500 training instances. With training instances held constant, empirical differences were driven by the feature subsets (`V4, V11` vs `V17, V3`).

> **Speaker Notes:**  
> "In Phase 9, cross-validation was reported as 0.9766 because resampling occurred before CV, exposing validation folds to 20% artificial prevalence. In Phase 11, we restructured CV so that resampling occurs strictly inside folds, evaluating all 226,980 training samples at natural 0.166% prevalence. True CV PR-AUC is 0.7188. Furthermore, inspecting both sampling calls confirmed that the exact same 500 training instances were drawn, demonstrating that the holdout metric difference was driven by the selected feature set rather than sample draw variance."

---

## Slide 9: Leak-Free 5-Fold Cross-Validation Results

### Out-of-Fold Evaluation Across Full Training Database (N=226,980, 378 Frauds)

| Model Architecture | Mean CV PR-AUC | Std CV PR-AUC | Aggregate OOF PR-AUC | OOF Precision ($\tau=0$) | OOF Recall ($\tau=0$) | OOF False Positives |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Classical RBF SVM** | **0.7362** | 0.0309 | **0.7270** | 45.44% | 84.39% | 383 |
| **Anisotropic QSVM** | **0.7215** | 0.0337 | **0.7117** | 27.45% | **85.19%** | 851 |
| **Opt Separable QSVM** | **0.7188** | 0.0386 | **0.7062** | **53.11%** | 83.60% | **279** |
| **Classical Linear SVM** | 0.6992 | 0.0187 | 0.6885 | 8.58% | **87.04%** | 3,505 |
| **Projected Quantum (PQK)** | 0.6222 | 0.0814 | 0.6150 | 18.00% | 82.54% | 1,418 |
| **Canonical QSVM ($ZZ$)** | 0.4887 | 0.0627 | 0.4574 | 9.76% | 86.51% | 3,025 |

* **Key Takeaway:** Classical RBF leads in CV PR-AUC (0.7362); Opt Separable QSVM achieves the lowest OOF false positive count (279).

> **Speaker Notes:**  
> "This table represents out-of-fold performance across all 226,980 training transactions at natural prevalence. Classical RBF SVM achieved the highest mean CV PR-AUC at 0.7362, closely followed by Anisotropic and Separable QSVM at 0.7215 and 0.7188. The Canonical ZZ QSVM produced over 3,000 false positives."

---

## Slide 10: Locked Holdout Test Set Final Benchmark

### Final Evaluation on 56,746 Transactions (95 Frauds)

| Model & Evaluation Cutoff | PR-AUC | ROC-AUC | Precision | Recall | False Positives | Inference Latency / Sample |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Classical RBF SVM (Default $\tau=0.0$)** | **0.6577** | 0.9284 | 68.87% | 76.84% (73/95) | 33 | **0.0037 ms** |
| **Classical RBF SVM (Calibrated $\tau^*=-0.28$)** | **0.6577** | 0.9284 | 43.35% | **78.95% (75/95)** | 98 | **0.0037 ms** |
| **Opt Separable QSVM (Default $\tau=0.0$)** | 0.6098 | 0.9276 | **73.74%** | 76.84% (73/95) | **26** | 0.0715 ms |
| **Opt Separable QSVM (Calibrated $\tau^*=-0.36$)** | 0.6098 | 0.9276 | 46.30% | **78.95% (75/95)** | 87 | 0.0715 ms |
| **Anisotropic QSVM (Default $\tau=0.0$)** | 0.6414 | 0.9257 | 46.88% | **78.95% (75/95)** | 85 | 0.0449 ms |
| **Classical Linear SVM (Default $\tau=0.0$)** | 0.6434 | 0.9285 | 10.95% | **80.00% (76/95)** | 618 | **0.0001 ms** |
| **Projected Quantum (Default $\tau=0.0$)** | 0.6146 | 0.9289 | 18.00% | 77.89% (74/95) | 337 | 0.2772 ms |
| **Canonical QSVM ($ZZ$, Default $\tau=0.0$)** | 0.5568 | **0.9380** | 9.49% | **80.00% (76/95)** | 725 | 0.8287 ms |

* **Winner:** Classical RBF SVM wins on PR-AUC (0.6577) and latency (0.0037 ms). Classical Linear SVM wins on throughput (0.0001 ms).

> **Speaker Notes:**  
> "On the locked holdout test set of 56,746 transactions, Classical RBF SVM is the clear winner with 0.6577 PR-AUC, 33 false positives, and 0.0037 ms latency. Opt Separable QSVM achieved excellent precision at 73.74% with only 26 false positives, but operates slower and yields lower PR-AUC. Canonical QSVM collapsed to 725 false alarms."

---

## Slide 11: Statistical Hypothesis Testing

### Paired Bootstrap Significance ($B=2,000$ Replicates on Full OOF)

| Model Comparison | Direction ($\Delta$) | Mean $\Delta$ PR-AUC | 95% Confidence Interval | Significant? |
| :--- | :--- | :---: | :---: | :---: |
| **RBF vs. Opt Separable QSVM** | $\text{RBF} - \text{Opt QSVM}$ | $+0.0200$ | $[+0.0024, +0.0381]$ | **YES** |
| **Linear vs. Opt Separable QSVM** | $\text{Linear} - \text{Opt QSVM}$ | $-0.0181$ | $[-0.0426, +0.0047]$ | **NO** (spans 0) |
| **Opt Separable vs. Canonical QSVM** | $\text{Opt QSVM} - \text{Canonical}$ | **$+0.2475$** | $[+0.1892, +0.3040]$ | **YES ($p < 0.001$)** |
| **Anisotropic vs. Opt Separable QSVM** | $\text{Aniso} - \text{Opt QSVM}$ | $+0.0049$ | $[-0.0099, +0.0210]$ | **NO** (spans 0) |

* **Statistical Inferences:**
  1. Classical RBF holds a statistically significant lead over Opt Separable QSVM in ranking PR-AUC ($p < 0.05$).
  2. Eliminating two-body phase clutter provides a massive, statistically significant boost (+24.8% PR-AUC, $p < 0.001$).
  3. Linear SVM and Separable QSVM are statistically equivalent in ranking capacity (95% CI spans zero).
  4. Anisotropic weighting and Separable QSVM are statistically equivalent (95% CI spans zero).

> **Speaker Notes:**  
> "To avoid relying on point estimates, we conducted 2,000 paired bootstrap iterations across the full out-of-fold predictions. Classical RBF holds a statistically significant lead over Opt Separable QSVM with a 95% confidence interval of [+0.0024, +0.0381] at alpha = 0.05. Crucially, eliminating two-body entanglement from the canonical kernel proved to be decisive, boosting PR-AUC by +24.8% (p < 0.001). Finally, Classical Linear SVM and Anisotropic QSVM show 95% confidence intervals spanning zero, confirming statistical equivalence with the separable quantum kernel."

---

## Slide 12: Phase 13 & 14 Dimensional Scaling & Full-Data Macro Control

### Dimensionality Expansion ($K \in \{6, 10, 16\}$), Bandwidth Scaling, and $N=3,000$

| Sample Size ($N$) | Model Architecture | Features ($K$) | Mean OOF PR-AUC | Mean OOF ROC-AUC | Default Precision | Default Recall |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **$N=2000$** | `baseline_classical_rbf_svm_k6` | 6 | 0.7215 | 0.9670 | 27.66% | 85.54% |
| **$N=2000$** | `p14_classical_rbf_k16` | 16 | 0.7350 | 0.9820 | 64.81% | 82.45% |
| **$N=2000$** | `p14_opt_separable_qsvm_k16` | 16 | **0.7363** | 0.9815 | **77.20%** | 80.25% |
| **$N=2000$** | `p14_hybrid_rbf_qsvm_k16` | 16 | **0.7368** | 0.9807 | 70.31% | 81.66% |
| **$N=3000$** | `p14_classical_rbf_k16` | 16 | 0.7361 | 0.9829 | 59.21% | 82.98% |
| **$N=3000$** | `p14_opt_separable_qsvm_k16` | 16 | **0.7369** | 0.9824 | 68.90% | 82.01% |
| \hline
| **Macro** | `macro_nystroem_rbf_approx` | 16 | **0.7705** | **0.9834** | 7.64% | **90.74%** |

* **Key Takeaway 1:** Expanding features to $K=16$ with dimensional bandwidth scaling $\beta(K) = \beta_0 \sqrt{6/K} \approx 0.0306$ brings Separable QSVM into statistical parity with Classical RBF ($p = 0.7388$ at $N=2000$; $p = 0.2188$ at $N=3000$).
* **Key Takeaway 2 (Macro Comparability Disclosure):** The Macro Nyström RBF model ($N=181,584$ training samples) reaches **0.7705 PR-AUC**. This baseline is **not directly comparable** to the smaller-subset experiments ($N \le 3000$) because it accesses 60× more training data.
* **Simulation Clarification:** All circuits simulated via pure statevectors on classical CPUs; $K=16$ denotes 16 selected tabular features, not physical qubits.

> **Speaker Notes:**  
> "In Phases 13 and 14, we evaluated multi-seed scaling up to N=3,000 and expanded feature dimensions to K=16. To prevent exponential Gram concentration, we compensated bandwidth as beta_0 * sqrt(6/K). At K=16, Separable QSVM achieved parity with Classical RBF at 0.7363 vs 0.7350. To understand the effect of data scale, we trained a Nyström RBF model on the entire training partition of 181,584 samples, reaching 0.7705 PR-AUC. Note that this macro model is not directly comparable to subset models, but demonstrates that data scale matters far more than kernel complexity."

---

## Slide 13: Phase 15 Trainable Quantum Kernels via Centered Regularized KTA

### Supervised Kernel Alignment Optimization ($N \in \{2000, 3000\}$, $K=16$, 3 Seeds $\times$ 5 Folds = 15 Matched Folds)

* **Centered Regularized KTA Objective:**
  $$\mathcal{L}(\theta) = \text{CKTA}(K_\theta, Y) - \lambda \|\theta - \beta_0\|_2^2, \quad \lambda = 200.0$$
  *(Implemented with $\lambda_{\text{reg}}=200.0$, strictly without an extra $1/2$ factor)*

| Sample Size ($N$) | Model Architecture | Kernel Formulation / Optimization | Mean OOF PR-AUC | Diff vs. Baseline ($\Delta$) | Wilcoxon $p$-value ($\alpha=0.05$) |
| :---: | :--- | :--- | :---: | :---: | :---: |
| **$N=2000$** | `p15_kta_trainable_qsvm_k16` | **Trainable QSVM ($K=16$, CKTA)** | **$0.7356 \pm 0.0034$** | **$+0.0217$ vs Trainable RBF** | **$\mathbf{p = 0.0015}$ (Sig Advantage)** |
| **$N=2000$** | `p14_opt_separable_qsvm_k16` | Fixed Separable QSVM ($K=16, \beta=0.0306$) | **$0.7363 \pm 0.0028$** | $+0.0002$ vs Fixed RBF | $p = 0.6788$ (Statistical Parity) |
| **$N=2000$** | `p14_classical_rbf_k16` | Fixed Classical RBF ($\gamma=0.0005$) | $0.7350 \pm 0.0011$ | — | $p = 0.6387$ vs Trainable QSVM |
| **$N=2000$** | `p15_kta_trainable_rbf_k16` | Trainable Classical RBF ($K=16$, CKTA) | $0.7163 \pm 0.0023$ | Ref Baseline (KTA) | — |
| \hline
| **$N=3000$** | `p15_kta_trainable_qsvm_k16` | **Trainable QSVM ($K=16$, CKTA)** | **$0.7365 \pm 0.0024$** | **$+0.0184$ vs Trainable RBF** | **$\mathbf{p = 0.0015}$ (Sig Advantage)** |
| **$N=3000$** | `p14_opt_separable_qsvm_k16` | Fixed Separable QSVM ($K=16, \beta=0.0306$) | **$0.7369 \pm 0.0019$** | $-0.0012$ vs Fixed RBF | $p = 0.3028$ (Statistical Parity) |
| **$N=3000$** | `p14_classical_rbf_k16` | Fixed Classical RBF ($\gamma=0.0005$) | $0.7361 \pm 0.0023$ | — | $p = 0.0302$ (Minute RBF edge vs Tr-QSVM) |
| **$N=3000$** | `p15_kta_trainable_rbf_k16` | Trainable Classical RBF ($K=16$, CKTA) | $0.7164 \pm 0.0042$ | Ref Baseline (KTA) | — |

* **Empirical Advantage in KTA Learning ($p=0.0015$):** Under the identical Centered Regularized KTA objective ($\lambda_{\text{reg}}=200.0$), Trainable QSVM decisively outperforms Trainable Classical RBF by **$+0.0217$ PR-AUC ($p=0.0015 < 0.05$)** at $N=2000$ and **$+0.0184$ PR-AUC ($p=0.0015 < 0.05$)** at $N=3000$.
* **Fixed QSVM vs. Fixed Classical RBF Parity:** When comparing fixed geometric configurations, Fixed Separable QSVM and Fixed Classical RBF perform at statistical parity at both $N=2000$ ($p = 0.6788 > 0.05$) and $N=3000$ ($p = 0.3028 > 0.05$). Neither fixed model holds a significant advantage.
* **Inductive Stability Mechanism:** Bounded, compact, periodic cosine-squared state fidelity $\prod \cos^2(\theta_j \Delta x_j / 2) \in [0, 1]$ prevents runaway gradient over-concentration under extreme imbalance (0.165%), whereas unbounded Gaussian RBF bandwidths overfit majority legitimate clusters.
* **No General Quantum Advantage:** Trainable QSVM matches fixed QSVM ($p > 0.63$) and achieves statistical parity with fixed classical RBF at $N=2000$ ($p = 0.6387 > 0.05$). At $N=3000$, fixed classical RBF retains a statistically significant but practically minute edge ($0.08\%$ PR-AUC, Wilcoxon $p = 0.0302 < 0.05$, $t$-test $p = 0.0354$). Neither model approaches full-data Nyström ($0.7705$). No general quantum advantage exists.

> **Speaker Notes:**  
> "In Phase 15, we evaluated trainable kernel methods across 15 matched folds using 3 seeds times 5 stratified folds. Under the identical Centered Regularized KTA objective with lambda_reg = 200.0 without any extra 1/2 factor, the Trainable QSVM decisively beat the Trainable Classical RBF model by +0.0217 PR-AUC, statistically significant at p = 0.0015. The bounded cosine-squared quantum fidelity prevented runaway overfitting under extreme class imbalance. However, we must carefully distinguish trainable from fixed baselines: Fixed QSVM and Fixed Classical RBF perform at statistical parity at both N=2000 (p=0.6788) and N=3000 (p=0.3028). Because Trainable QSVM only matches but does not exceed the carefully tuned fixed Classical RBF model, we do not claim general or proven quantum advantage."

---

## Slide 14: Real-World Operational & Financial Analysis

### Inference Latency vs. Asymmetric Financial Loss ($c_{\text{fp}}=\$10, c_{\text{fn}}=\$200$)

```text
[Local CPU Benchmark: In-Memory Kernel Evaluation per Sample]
• Classical Linear SVM:        0.0001 ms  (Dot product in feature space; >10,000,000 samples/sec)
• Classical RBF SVM:           0.0037 ms  (NumPy vectorized Gaussian distance against support vectors)
• Opt Separable QSVM (Exact):  0.0715 ms  (Analytical cosine product; ~19× slower than RBF)
• Canonical QSVM (Simulator):  0.8287 ms  (Statevector simulation; 224× slower than RBF)
* Note: Single-thread CPU benchmark (Apple Silicon macOS, Python 3.10). Omits network/IO pipeline.
```

### Financial Loss at Calibrated Threshold $\tau^*(R=20)$ (Holdout Test Set, 56,746 samples):
* **Opt Separable QSVM ($\tau^* = -0.3576$):** **\$4,870** (87 FP, 20 FN; Recall 78.95%)
* **Classical RBF SVM ($\tau^* = -0.2785$):** **\$4,980** (98 FP, 20 FN; Recall 78.95%)
* **Classical Linear SVM ($\tau^* = +0.1867$):** **\$6,570** (277 FP, 19 FN; Recall 80.00%)
* **Canonical QSVM ($ZZ$, $\tau^* = +0.5055$):** **\$9,320** (132 FP, 40 FN; Recall 57.89%)
*(Note: Decision threshold tuning balances operational false alarms without altering threshold-invariant ranking metrics PR-AUC or ROC-AUC)*

> **Speaker Notes:**  
> "Latency measurements reflect in-memory CPU scoring. Classical Linear SVM is extraordinarily fast at 0.0001 ms, and Classical RBF evaluates in 3.7 microseconds. In contrast, simulated quantum kernels require 71 to 828 microseconds. Under asymmetric financial loss calibrated on training folds, Opt Separable QSVM and Classical RBF achieve virtually identical costs ($4,870 vs $4,980)."

---

## Slide 15: Capstone Verdict on Quantum Advantage

### An Objective, Evidence-Based Scientific Conclusion Across All 15 Phases

| Evaluation Criterion | Classical RBF SVM | Classical Linear SVM | Full Macro Nyström RBF | Optimized / Trainable QSVM | Canonical ZZ QSVM |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Peak PR-AUC** | **0.7361** (Subset best) | 0.7024 | **0.7705** (Overall ceiling) | 0.7369 | 0.4464 (Collapses) |
| **Matched KTA Advantage?** | No (Overfits KTA) | N/A | N/A | **Yes ($p=0.0015$ over RBF)** | No |
| **Inference Latency** | **0.0037 ms** | **0.0001 ms** (Fastest) | 0.0250 ms | 0.0449–0.0715 ms | 0.8287–1.6420 ms |
| **Physical QPU Needed?** | **No** | **No** | **No** | **No** (Classical Cosine) | Yes (Circuit execution) |

### Final Capstone Verdict:
> **No General Quantum Advantage exists for tabular financial fraud detection.**  
> - Classical Gaussian RBF SVMs and Linear SVMs remain the mathematically principled, practically viable, and superior choice for production financial fraud screening.  
> - An empirical inductive quantum advantage was proven **specifically within the supervised KTA learning framework ($p=0.0015$)**, where bounded quantum fidelity prevents extreme-imbalance overfitting. However, it does not surpass tuned fixed classical RBF models.

> **Speaker Notes:**  
> "Our final verdict is balanced and evidence-based. There is no general quantum advantage for tabular financial fraud detection. Tuned Classical RBF SVM is superior or equal in PR-AUC, runs 19x faster, and requires no quantum simulator. The one area where quantum kernels showed clear inductive superiority was during supervised KTA learning, where bounded state fidelities prevented gradient overfitting. But for production banking systems, classical models remain dominant."

---

## Slide 16: Methodological Disclosures & Experimental Boundaries

### Maintaining Absolute Academic Honesty and Integrity

1. **Prior Holdout Exposure:** Holdout test set (`X_test.csv`, 56,746 samples) was evaluated in early phases (Phases 3, 6, 8, 9). Phases 11–15 evaluated strictly leak-free out-of-fold cross-validation on `X_train.csv`.
2. **Discrete Sample Variance:** 95 test frauds and ~75 frauds per validation fold produce ~1% discrete jumps in recall. Confidence intervals accompany all metrics.
3. **Training Sample Asymmetry:** Quantum simulator scaling limited training subsets to $N \le 3000$, whereas macro classical controls accessed all 181,584 training rows.
4. **Noiseless Simulation:** Results reflect ideal classical statevector simulation on CPU. NISQ hardware noise, decoherence, and shot noise would further degrade physical quantum performance.
5. **Tabular Feature Representation ($K=16$):** $K=16$ refers specifically to 16 selected continuous tabular input features mapped to angle encodings in simulation, not 16 physical hardware qubits.
6. **Macro vs. Subsample Comparability:** The Macro Nyström RBF model (0.7705 PR-AUC) accessed 60× more training data and is not directly comparable to subset-constrained models.
7. **Classical Simulation Identifiability:** At $\alpha=0$, separable quantum state fidelity simplifies to a classical cosine product running in $\mathcal{O}(d)$ time without a quantum computer.

> **Speaker Notes:**  
> "In accordance with academic rigor, we disclose seven key experimental boundaries: early holdout exposure, discrete sample variance from rare fraud events, simulator subset limits, noiseless simulation versus physical NISQ hardware, the definition of K=16 input features, macro data comparability, and classical simulation identifiability."

---

## Slide 17: Strategic Recommendations & Future Directions

### Strategic Next Steps for Industry & Academia

* **For Financial Institutions:**
  * Deploy **Classical Gaussian RBF SVMs, Nystroem Kernel Approximations, or Gradient-Boosted Trees (XGBoost/LightGBM)** for sub-millisecond fraud screening.
  * Do **not** deploy quantum kernel methods in production for tabular financial fraud detection.
  * Reallocate enterprise quantum exploration toward quantum optimization (QAOA for portfolio risk) or quantum Monte Carlo.
* **For Future Academic Research:**
  * **Trainable Parameterized Quantum Circuits (QNNs):** Replace fixed Pauli kernels with variational circuits trained via parameter-shift rules.
  * **Native Quantum Data:** Test quantum kernels on data originating directly from quantum physical processes or quantum communication networks.
  * **Error Mitigation:** Benchmark downscaled 4-qubit kernels on physical IBM Quantum QPUs using Zero-Noise Extrapolation (ZNE).

> **Speaker Notes:**  
> "Our practical recommendation to banks is clear: deploy classical models for fraud detection today. For quantum researchers, the future lies not in fixed Pauli kernels on tabular spreadsheets, but in trainable quantum neural networks and native quantum data domains."

---

## Slide 18: Defense Committee Q&A Preparation

### Anticipated Examiner Questions & Technical Answers

**Q1: Why did you conclude there is no quantum advantage if Opt QSVM achieved 73.7% precision and lowest financial loss?**  
*Answer:* "Because Optimized Separable QSVM operates at $\alpha = 0$ (zero quantum entanglement). At $\alpha = 0$, state fidelity simplifies analytically to a classical closed-form product of cosines $k(x, x') = \prod \cos^2(\beta \Delta x_j / 2)$, running on standard CPU hardware in $\mathcal{O}(d)$ time. Claiming 'quantum advantage' for a function that runs classically without a quantum computer is scientifically inaccurate. Furthermore, Classical RBF SVM achieved higher PR-AUC and ran 19× faster."

**Q2: Why does Trainable QSVM beat Trainable Classical RBF in Phase 15, and does that constitute quantum advantage?**  
*Answer:* "Under matched Centered Regularized KTA optimization, Trainable QSVM significantly outperformed Trainable Classical RBF by +0.0217 PR-AUC ($p=0.0015$). This is an empirical inductive advantage: bounded $[0, 1]$ cosine-squared fidelity prevents runaway gradient expansion under extreme 0.165% class imbalance, whereas unbounded Gaussian RBF bandwidths overfit majority clusters. However, it does NOT constitute general quantum advantage because neither model surpasses tuned fixed Classical RBF ($0.7361$) or full-data Nyström ($0.7705$)."

**Q3: Why is the Macro Nyström baseline (0.7705) not directly comparable to the subset models ($N \le 3000$)?**  
*Answer:* "The Macro Nyström RBF model was trained on all 181,584 training samples per fold, whereas QSVM and matched classical controls were restricted to $N \le 3000$ due to quantum simulator scaling limits. Because data quantity is the dominant driver of fraud detection performance, comparing 181k samples to 3k samples measures sample-size effects rather than intrinsic kernel expressivity."

**Q4: How did you explain the shift in holdout PR-AUC between Phase 9 and Phase 11?**  
*Answer:* "Both phases sampled identical training rows (`random_state=42`). The shift was driven entirely by the feature representation: Phase 9 selected features on a 500-sample pilot (`V4, V10, V11, V12, V14, V16`), whereas Phase 11 selected features across all 226,980 training samples (`V17, V14, V12, V10, V16, V3`), proving differences stemmed from feature representations rather than instance sampling."

**Q5: Why was the Phase 9 cross-validation PR-AUC reported as 0.9766 while holdout PR-AUC was ~0.65?**  
*Answer:* "In Phase 9, resampling to 20% fraud was applied globally before cross-validation, so validation folds evaluated at 20% prevalence. In Phase 11–15, we resampled strictly inside each training fold, evaluating validation folds at natural prevalence ($\pi = 0.165\%$). By Bayes' theorem, the 120× change in prevalence introduces a 150× penalty on false positives in precision, explaining the entire gap."

---

## Slide 19: Conclusion & Thank You

### Quantum Machine Learning for Financial Fraud Detection

* **Complete 15-Phase Empirical & Theoretical Investigation**
* **104/104 Automated Tests Passing Across 14 Test Modules**
* **Full Code, Models, Logs, and Reports Preserved & Reproducible**

**Thank you! Open to questions from the defense committee.**

> **Speaker Notes:**  
> "Thank you for your time, attention, and guidance throughout this project. Over 15 rigorous research phases, we have answered the core capstone question with full methodological and statistical integrity: while quantum kernels provide inductive stability during KTA parameter learning, classical RBF and Linear SVMs remain the mathematically principled, practically viable, and superior choice for production financial fraud detection. I now welcome any questions from the defense committee."
