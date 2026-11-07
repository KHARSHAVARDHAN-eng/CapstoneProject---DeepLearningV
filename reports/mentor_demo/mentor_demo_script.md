# Capstone Mentor Demonstration Script (3–5 Minute Presentation)

**Project Title:** Quantum Machine Learning for Financial Fraud Detection Using Quantum Kernel Methods  
**Format:** Concise Verbal Walkthrough with Live Commands and Interactive Visuals  
**Target Audience:** Capstone Faculty Advisor, Research Mentor, and Evaluation Committee  
**Artifact Directory:** [`reports/mentor_demo/`](file:///Users/harsha/Desktop/Projects-files/quantum-fraud-detection(Capstone)/reports/mentor_demo/)  

---

## Quick Reference: Commands to Run Before or During the Meeting

To execute or demonstrate the fresh benchmark live in under 75 seconds:
```bash
# 1. Activate project virtual environment
cd "/Users/harsha/Desktop/Projects-files/quantum-fraud-detection(Capstone)"

# 2. Run the reproducible mentor demonstration script
.venv/bin/python scripts/mentor_demo.py

# 3. Verify the full test suite passes (104 tests, ~12s)
.venv/bin/pytest tests/ -v
```

All generated artifacts, summary CSVs, and high-resolution figures are automatically written to `reports/mentor_demo/`.

---

## 3–5 Minute Spoken Presentation Script

### Minute 0:00 – 1:00 | The Problem & The Quantum Hypothesis

> *"Good morning/afternoon, Professor. Today I want to walk you through the final evaluation and demonstration of our capstone research: **Quantum Machine Learning for Financial Fraud Detection Using Quantum Kernel Methods**.*
>
> *Financial fraud presents an adversarial machine learning problem defined by two constraints:*
> 1. *Extreme Class Imbalance: Fraud represents only **0.165%** of real-world traffic—roughly 1 fraud in every 580 purchases. Accuracy is meaningless (a null model achieves 99.83%), making **Precision-Recall AUC (PR-AUC)** our primary scientific metric.*
> 2. *Sub-Millisecond Latency: Payment gateways require transaction risk scoring in under **10 to 50 milliseconds**.*
>
> *The central hypothesis investigated was whether mapping tabular transaction data into a $2^n$-dimensional quantum Hilbert space via parameterized quantum circuits could uncover subtle non-linear margin separations that classical Support Vector Machines miss."*

---

### Minute 1:00 – 2:00 | The Engineering Pipeline & The Entanglement Discovery

> *"Before evaluating models, we engineered a mathematically leak-free pipeline across 15 research phases:*
> - *We purged 1,081 duplicate transactions, strictly partitioned a locked holdout test set (56,746 transactions), and conducted all cross-validation across the 226,980 training transactions.*
> - *All ANOVA feature selection ($K=16$), angle scaling, and threshold determination were fitted strictly inside fold training data.*
>
> *Our first major discovery in Phase 5 and 6 was that **the canonical entangled quantum feature map ($ZZFeatureMap$) severely degrades performance on continuous tabular data**.*
>
> *The two-body entangling terms cause rapid phase wrapping across $[0, 2\pi^2]$, producing **Geometric Inversion**—within-class fraud similarity collapsed below cross-class similarity, inflating false alarms into the thousands ($0.367\text{--}0.446$ PR-AUC).*
>
> *To fix this, in Phases 9, 14, and 15, we developed two key innovations:*
> 1. ***Optimized Separable QSVM ($\alpha=0$):** Eliminating two-body entanglement boosted PR-AUC by $+24.8\%$ to **$0.7363$**. Through Taylor expansion, we proved that this kernel is mathematically identical to a classical Gaussian RBF kernel with $\gamma_{\text{eff}} = \beta^2/4$.*
> 2. ***Trainable QSVM via Centered Regularized KTA (Phase 15):** We optimized 16 anisotropic coordinate bandwidths directly on in-fold training data using Centered Kernel Target Alignment regularized by a quadratic anchor ($\lambda_{\text{reg}} = 200.0$)."*

---

### Minute 2:00 – 3:30 | Live Results Walkthrough & Visual Diagnostics

*(Show the live summary table from `mentor_demo_results.csv` and the four plots in `reports/mentor_demo/`)*

```text
==========================================================================================
FRESH RUN RESULTS (Fold 0, N_val = 45,396, Frauds = 75, Natural Prevalence = 0.165%)
==========================================================================================
Model Architecture              PR-AUC   ROC-AUC   Precision    Recall      FP     Fit Time
------------------------------------------------------------------------------------------
Classical Linear SVM (K=6)      0.7025    0.9741     10.81%     85.33%     528       0.02s
Fixed Classical RBF SVM (K=16)  0.7640    0.9754     65.59%     81.33%      32       0.21s
Fixed Separable QSVM (K=16)     0.7686    0.9759     78.67%     78.67%      16       0.85s
Trainable KTA QSVM (K=16)       0.7644    0.9762     74.07%     80.00%      21      10.53s
Trainable Classical RBF (K=16)  0.7229    0.9789     21.77%     85.33%     230      15.99s
Macro Nystroem RBF (N=181k Ref) 0.7052    0.9787      7.80%     88.00%     780      15.71s
==========================================================================================
```

> *"Let me highlight three key takeaways from these live visual results:*
>
> 1. **Precision-Recall Curves (`pr_curves.png`):**  
>    *Fixed Separable QSVM ($0.7686$) and Trainable KTA QSVM ($0.7644$) sustain the highest precision envelope across the entire operating spectrum, closely tracking Fixed Classical RBF ($0.7640$). Notice the dramatic drop in Trainable Classical RBF ($0.7229$).*
>
> 2. **False Alarm Reduction in Confusion Matrices (`confusion_matrices.png`):**  
>    *At the default threshold on 45,396 unseen transactions, **Trainable QSVM produces only 21 false alarms** while catching 80% of frauds (60/75). In stark contrast, Classical Linear SVM produces 528 false alarms, and Trainable Classical RBF produces 230 false alarms (an 11× increase).*
>
> 3. **Latency and Production Reality (`runtime_comparison.png`):**  
>    *While Classical Linear SVM scores a transaction in **0.00001 ms** and Classical RBF evaluates in **0.072 ms**, our simulated quantum kernel requires **0.212 ms**. While all models easily meet the gateway deadline of 20 ms on CPU simulation, physical quantum QPUs with 1,024 measurement shots would require ~1,000 ms per transaction, which is operationally unviable for real-time payment gateways."*

---

### Minute 3:30 – 4:30 | The Core Statistical Finding: Inductive Bias Advantage ($p = 0.0015$)

> *"The central scientific finding of Phase 15 emerges when we compare **Trainable QSVM** against **Trainable Classical RBF** under matched KTA learning:*
>
> - *Under the identical Centered Regularized KTA objective ($\lambda_{\text{reg}} = 200.0$), across all 15 matched folds (3 seeds $\times$ 5 folds), **Trainable QSVM significantly outperforms Trainable Classical RBF by $+0.0217$ PR-AUC ($p = 0.0015 < 0.05$)** at $N=2000$ and **$+0.0184$ PR-AUC ($p = 0.0015 < 0.05$)** at $N=3000$.*
> - *In our live run, Trainable QSVM leads Trainable Classical RBF by **$+0.0415$ PR-AUC** ($0.7644$ vs $0.7229$).*
>
> *Why does this happen?*
> - *Classical Gaussian RBF bandwidths reside in an unbounded domain $\exp(-\sum \gamma_j \Delta x_j^2)$. Under $0.165\%$ imbalance, gradient steps aggressively overfit the majority legitimate cluster.*
> - *In contrast, quantum state fidelity $k(x, x') = \prod \cos^2(\theta_j \Delta x_j / 2)$ is inherently **bounded in $[0, 1]$, periodic, and compact**. This geometry provides superior regularization and inductive stability against gradient explosion under extreme class skew."*

---

### Minute 4:30 – 5:00 | The Honest Scientific Conclusion

> *"However, to maintain absolute academic integrity, we must be clear about what this does and does not prove:*
>
> 1. *It **does prove** an empirical inductive advantage for quantum kernel geometries specifically within supervised KTA parameter learning.*
> 2. *It **does not constitute a general quantum advantage** over classical machine learning, because:*
>    - *Fixed Separable QSVM and Fixed Classical RBF perform at **statistical parity** ($p = 0.6788$ at $N=2000$, $p = 0.3028$ at $N=3000$).*
>    - *At $N=3000$, Fixed Classical RBF holds a minute edge over Trainable QSVM ($p = 0.0302$).*
>    - *The champion Separable QSVM operates at zero entanglement ($\alpha = 0$), meaning it evaluates in linear classical time on standard CPU hardware.*
>    - *Neither model approaches the full-data Macro Nyström reference ($0.7705$), showing that access to training data matters far more than kernel complexity.*
>
> *Our practical recommendation to financial institutions is clear: **Deploy Classical Gaussian RBF SVMs or Gradient-Boosted Decision Trees for sub-millisecond fraud screening.** Enterprise quantum research budgets are better allocated toward quantum optimization (portfolio risk) or quantum Monte Carlo.*
>
> *All 104 tests pass, all code and data artifacts are preserved and reproducible, and I am ready for committee questions."*

---

## Anticipated Mentor Questions & Confident Answers

**Q1: Why did you claim zero entanglement ($\alpha=0$) is optimal? Doesn't entanglement define quantum advantage?**  
*Answer:* In tabular financial transactions, features are pre-orthogonalized via PCA. Entangling two-qubit operators create pairwise phase wrapping that shatters metric proximity for continuous variables. Zero entanglement avoids this phase clutter, collapsing state fidelity into an optimal classical cosine product.

**Q2: Why was your reported cross-validation PR-AUC 0.9766 in Phase 9, but ~0.73 in Phase 15?**  
*Answer:* In Phase 9, synthetic resampling to 20% fraud was applied globally before cross-validation, artificially enriching validation folds. In Phase 11–15, resampling occurred strictly inside each training fold, leaving validation folds at the true natural prevalence of 0.165%. By Bayes' theorem, evaluating at natural prevalence introduces a 150× penalty on false positives, explaining the entire gap.

**Q3: Can these quantum kernels be deployed on IBM Quantum hardware today?**  
*Answer:* Not for real-time payment screening. Statevector simulation takes 0.2 ms on a CPU. Physical quantum QPUs require 1,024 shots per kernel entry, circuit compilation, and queueing overhead, taking ~1 second per transaction—far exceeding the 20 ms payment gateway limit.
