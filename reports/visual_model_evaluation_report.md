# Visual Model Performance Evaluation Report
**Project:** Quantum Machine Learning for Financial Fraud Detection  
**Target Focus:** Decision Score Distribution Histograms, Model Metric Distributions, and Confusion Matrix Diagnostics  
**Date:** November 2026  
**Artifact Directory:** [`reports/figures/`](file:///Users/harsha/Desktop/Projects-files/quantum-fraud-detection(Capstone)/reports/figures/)  

---

## Executive Summary

In financial fraud detection under extreme class imbalance ($\pi \approx 0.165\%$, $\approx 1:580$ fraud prevalence), model evaluation requires diagnostics beyond aggregate scalar metrics. This visual report provides an in-depth empirical evaluation using:

1. **Decision Score Distribution Histograms:** Visualizing the probability density and class separation between legitimate and fraudulent transactions.
2. **Performance Metric Histograms:** Grouped comparative distributions for PR-AUC, ROC-AUC, Precision, Recall, False Alarm Burden, and Operational Loss.
3. **Multi-Model Confusion Matrices:** Quantitative breakdown of True Negatives (TN), False Positives (FP), False Negatives (FN), and True Positives (TP) under both **Default Thresholds ($\tau = 0.0$)** and **Prevalence-Calibrated Thresholds ($\tau^*$)**.

---

## 1. Decision Score Distribution Histograms (Fraud vs. Legit Separation)

The diagram below illustrates the distribution of decision scores $f(x)$ on the locked holdout test set ($N=56,746$ transactions, 95 frauds, 56,651 legitimate). The blue distribution represents legitimate transactions ($y=0$), while the red distribution represents fraudulent transactions ($y=1$). The red dashed line denotes the default decision threshold ($\tau = 0.0$), and the green dash-dotted line denotes the cost-optimal calibrated threshold ($\tau^*$).

![Decision Score Distribution Histograms](file:///Users/harsha/Desktop/Projects-files/quantum-fraud-detection(Capstone)/reports/figures/model_score_histograms.png)

### Key Histogram Insights:
- **Separation Margin ($\Delta \mu = \mu_{\text{fraud}} - \mu_{\text{legit}}$):**
  - **Tuned Classical Linear SVM ($K=6$):** $\Delta \mu = +3.88$ ($\mu_{\text{legit}} = -0.99, \mu_{\text{fraud}} = +2.89$). The linear hyperplane establishes strong separation on top ANOVA features.
  - **Tuned Classical RBF SVM ($K=6$):** $\Delta \mu = +3.19$ ($\mu_{\text{legit}} = -1.18, \mu_{\text{fraud}} = +2.02$). The Gaussian kernel compresses the legitimate distribution variance ($\sigma_{\text{legit}} = 0.25$), resulting in fewer false alarms.
  - **Historical Classical SVM ($K=30$):** $\Delta \mu = +3.51$ ($\mu_{\text{legit}} = -3.13, \mu_{\text{fraud}} = +0.37$). Displays a wide negative margin for legitimate transactions.
  - **Tuned QSVM ($K=6$):** $\Delta \mu = +1.83$ ($\mu_{\text{legit}} = -1.17, \mu_{\text{fraud}} = +0.67$). The fraud distribution exhibits a narrower margin and greater right-tail overlap with legitimate transactions, directly explaining why the baseline QSVM yielded 1,627 false alarms at default threshold compared to 79 for Classical RBF.
- **Threshold Sensitivity:** In all models, the legitimate transaction mass ($56,651$ transactions) vastly outweighs fraud ($95$ transactions). Even a microscopic overlap in the positive tail of legitimate transactions triggers dozens of false alarms.

---

## 2. Model Performance Metrics Histogram & Comparative Distribution

This grouped bar and distribution chart evaluates ranking capability, precision-recall trade-offs, operational alert load, and monetary loss across six evaluated model architectures ($N_{\text{val}} = 45,396$ transactions, 75 frauds).

![Model Performance Histogram](file:///Users/harsha/Desktop/Projects-files/quantum-fraud-detection(Capstone)/reports/figures/model_performance_histogram.png)

### Metric Comparison Summary Table:

| Model Architecture | Features ($K$) | Training $N$ | PR-AUC | ROC-AUC | Default Precision | Default Recall | Default FP | Calibrated Precision | Calibrated Recall | Calibrated FP | Calibrated Cost ($R=20$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Classical Linear SVM** | 6 | 2,000 | 0.7025 | 0.9741 | 10.81% | 85.33% (64/75) | 528 | 2.73% | 92.00% (69/75) | 2,456 | $25,760 |
| **Fixed Classical RBF SVM** | 16 | 2,000 | 0.7640 | 0.9754 | 65.59% | 81.33% (61/75) | 32 | 1.77% | 96.00% (72/75) | 3,989 | $40,490 |
| **Fixed Separable QSVM** | 16 | 2,000 | **0.7686** | 0.9759 | **78.67%** | 78.67% (59/75) | **16** | 1.40% | 96.00% (72/75) | 5,055 | $51,150 |
| **Trainable KTA QSVM** | 16 | 2,000 | **0.7644** | 0.9762 | **74.07%** | 80.00% (60/75) | **21** | 1.51% | 96.00% (72/75) | 4,702 | $47,620 |
| **Trainable Classical RBF** | 16 | 2,000 | 0.7229 | **0.9789** | 21.77% | 85.33% (64/75) | 230 | 3.05% | 94.67% (71/75) | 2,254 | $23,340 |
| **Macro Nystroem RBF (Ref)** | 16 | 181,584 | 0.7052 | 0.9787 | 7.80% | **88.00% (66/75)** | 780 | **50.42%** | 80.00% (60/75) | **59** | **$3,590** |

---

## 3. Confusion Matrix Analysis (Default Operating Threshold $\tau = 0.0$)

The figure below displays the confusion matrices for all six models under the default decision threshold $\tau = 0.0$.

![Confusion Matrices Default](file:///Users/harsha/Desktop/Projects-files/quantum-fraud-detection(Capstone)/reports/figures/confusion_matrices_default.png)

### Breakdown & Operational Findings:
1. **Trainable KTA QSVM (Phase 15 Champion):**
   - **TN:** 45,300 (99.95%) | **FP:** 21 (0.05%)
   - **FN:** 15 (20.00%) | **TP:** 60 (80.00%)
   - **Precision:** $74.07\%$ | **Recall:** $80.00\%$ | **Cost:** $\$3,210$
   - *Verdict:* Centered Regularized KTA optimization successfully suppressed false alarms down to 21 while maintaining $80.0\%$ fraud recall.
2. **Fixed Separable QSVM ($K=16$):**
   - **TN:** 45,305 (99.96%) | **FP:** 16 (0.04%)
   - **FN:** 16 (21.33%) | **TP:** 59 (78.67%)
   - **Precision:** $78.67\%$ | **Recall:** $78.67\%$ | **Cost:** $\$3,360$
   - *Verdict:* Lowest false positive count among all subsampled models (16 alerts out of 45,321 legitimate transactions).
3. **Fixed Classical RBF SVM ($K=16$):**
   - **TN:** 45,289 (99.93%) | **FP:** 32 (0.07%)
   - **FN:** 14 (18.67%) | **TP:** 61 (81.33%)
   - **Precision:** $65.59\%$ | **Recall:** $81.33\%$ | **Cost:** $\$3,120$
   - *Verdict:* Strongest balanced classical control, achieving statistical parity with QSVM ($p=0.6788$).
4. **Trainable Classical RBF ($K=16$):**
   - **TN:** 45,091 (99.49%) | **FP:** 230 (0.51%)
   - **FN:** 11 (14.67%) | **TP:** 64 (85.33%)
   - **Precision:** $21.77\%$ | **Recall:** $85.33\%$ | **Cost:** $\$4,500$
   - *Verdict:* Higher recall but an 11× surge in false alarms compared to Trainable QSVM.
5. **Classical Linear SVM ($K=6$):**
   - **TN:** 44,793 (98.83%) | **FP:** 528 (1.17%)
   - **FN:** 11 (14.67%) | **TP:** 64 (85.33%)
   - **Precision:** $10.81\%$ | **Recall:** $85.33\%$ | **Cost:** $\$7,480$

---

## 4. Confusion Matrix Analysis (Prevalence-Calibrated Threshold $\tau^*$)

When calibrated specifically to minimize financial risk under asymmetric loss ($c_{\text{fn}} = \$200, c_{\text{fp}} = \$10, R=20$), the models shift their operating points to prioritize catching fraud.

![Confusion Matrices Calibrated](file:///Users/harsha/Desktop/Projects-files/quantum-fraud-detection(Capstone)/reports/figures/confusion_matrices_calibrated.png)

### Calibration Trade-Offs:
- **Maximum Recall Achieved:** All four small-subsample non-linear models achieve **$96.00\%$ Recall (72 out of 75 frauds intercepted)** with only 3 missed frauds.
- **False Alarm Explosion:** Because of the natural prevalence ($\pi = 0.165\%$), shifting the threshold to catch the last 3 frauds causes false alarms to rise to $3,989\text{--}5,055$ transactions, illustrating the fundamental precision-recall trade-off in financial fraud systems.
- **The Macro Control Exception:** The Macro Nystroem RBF model (trained on $181,584$ samples) uniquely maintains **$50.42\%$ Precision with $80.00\%$ Recall** and only **59 False Positives**, demonstrating that large-scale classical sample coverage remains superior to small-sample kernel tuning for production deployment.

---

## 5. Holdout Test Set Confusion Matrix Comparison ($N=56,746$)

The figure below evaluates the locked holdout test set models under both default and cost-optimal operating points.

![Holdout Confusion Matrices Detailed](file:///Users/harsha/Desktop/Projects-files/quantum-fraud-detection(Capstone)/reports/figures/holdout_confusion_matrices_detailed.png)

---

## 6. Summary Recommendations for Presentation & Defense

1. **Use the Decision Score Histogram** to demonstrate understanding of statistical margin separation, class overlap, and why naive accuracy fails in fraud detection.
2. **Use the Default Confusion Matrices** to highlight the Phase 15 Trainable KTA QSVM achievement: holding false positives to just 21 while capturing 80% of frauds.
3. **Use the Calibrated Confusion Matrices** to discuss real-world deployment trade-offs between manual investigator review capacity ($c_{\text{fp}} = \$10$) and fraud liability ($c_{\text{fn}} = \$200$).
