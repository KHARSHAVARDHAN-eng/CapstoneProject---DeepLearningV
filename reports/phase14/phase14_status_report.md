# Phase 14 Systematic Optimization: Status Report & Leakage Audit

**Date**: 2026-10-14  
**Pipeline**: `src/experiments/phase14_optimization.py`  
**Execution Script**: `scripts/run_phase14_experiments.py`  
**Target Metric**: Out-of-Fold PR-AUC $\ge 0.85$ (Benchmark Baseline: Phase 13 Classical RBF SVM = $0.7287$)

---

## 1. Executive Status

1. **Benchmark Execution Status**:
   - The full multi-seed, 5-fold benchmark is **actively executing** in the background (Task ID `105608d5-4368-4dca-9abc-cf310b1efe67/task-1857`, PID 64517).
   - **No duplicate runs** have been launched.
   - **Progress**:
     - Seed 42, Fold 1/5: Completed in 355.42s (~5.9 min)
     - Seed 42, Fold 2/5: Completed in 394.71s (~6.5 min)
     - Seed 42, Fold 3/5: In progress
     - Total Planned Scope: 3 seeds (42, 101, 2024) $\times$ 5 stratified folds = 15 folds total, evaluating sample sizes $N=2,000$ and $N=3,000$ across 12 distinct model configurations plus 2 macro controls.
     - Process Health: Active CPU consumption, memory footprint stable at ~1.2 GB resident memory.

2. **Stage 2 Pilot Threshold Calibration Audit**:
   - Multiple distinct models in the initial pilot run displayed identical calibrated metrics ($\text{Precision} \approx 27.31\%$ or $28.19\%$, $\text{Recall} \approx 82.67\%$ or $85.33\%$, $\text{FP} = 165$ or $163$, $\text{Cost} = \$4,250$ or $\$3,830$).
   - **Root Cause**: Candidate thresholds were previously generated using `np.percentile(scores, np.linspace(0.0, 99.5, n_thresholds))`, which artificially capped the search at the 99.5th percentile.
   - **Mathematical Proof**: On validation Fold 0 ($N_{\text{val}} = 45,396$), the 99.5th percentile threshold enforces $\text{TP} + \text{FP} = \lceil 45,396 \times 0.005 \rceil = 227$. Because the cost-optimal operating point for rare fraud ($\pi \approx 0.165\%$) under $C = 10\text{FP} + 200\text{FN}$ occurs between the 99.7th and 99.9th percentiles (where $\text{FP} \approx 30\text{--}60$), the optimization slammed into the 99.5% boundary ceiling, algebraically freezing $\text{TP} + \text{FP} = 227$ for all models.
   - **Resolution**: Realigned `find_cost_optimal_threshold` with the Phase 13 protocol by extracting candidate thresholds from `precision_recall_curve` across the entire support and evaluating default threshold $\tau = 0.0$. Re-evaluation confirmed distinct, model-specific operating points:
     - Classical RBF ($K=6$): $\text{PR-AUC} = 0.7286$, $\text{Prec} = 64.21\%$, $\text{Recall} = 81.33\%$, $\text{FP} = 34$, $\text{Cost} = \$3,140$
     - OptQSVM ($K=6$): $\text{PR-AUC} = 0.7082$, $\text{Prec} = 54.39\%$, $\text{Recall} = 82.67\%$, $\text{FP} = 52$, $\text{Cost} = \$3,120$
     - OptQSVM ($K=16$): $\text{PR-AUC} = 0.7686$, $\text{Prec} = 62.63\%$, $\text{Recall} = 82.67\%$, $\text{FP} = 37$, $\text{Cost} = \$2,970$
     - Classical RBF ($K=16$): $\text{PR-AUC} = 0.7640$, $\text{Prec} = 50.79\%$, $\text{Recall} = 85.33\%$, $\text{FP} = 62$, $\text{Cost} = \$2,820$
     - Hybrid Ensemble ($K=16$): $\text{PR-AUC} = 0.7667$, $\text{Prec} = 64.58\%$, $\text{Recall} = 82.67\%$, $\text{FP} = 34$, $\text{Cost} = \$2,940$

---

## 2. Rigorous Data Leakage & Isolation Audit

| Pipeline Stage | Implementation Detail | Leakage Verification Status |
| :--- | :--- | :--- |
| **Historical Test Set Isolation** | `data/processed/X_test.csv` is not read, referenced, or loaded in Phase 14 scripts. | **PASSED** (Strict isolation preserved) |
| **Cross-Validation Split** | `StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)` partitions exclusively `X_train.csv` (226,980 rows). | **PASSED** (Zero leakage between folds) |
| **In-Fold Feature Selection** | ANOVA $F$-statistic ranking (`select_top_features_infold`) is fitted strictly on $X_{\text{tr\_raw}}, y_{\text{tr\_raw}}$. | **PASSED** (Validation features/labels unseen) |
| **Subset Sampling** | Nested subsets ($N \in \{2000, 3000\}$, 10% fraud ratio) are drawn exclusively from within the training partition. | **PASSED** (Strict nesting $D_{2000} \subset D_{3000}$) |
| **Angle Scaling** | `MinMaxScaler(feature_range=(0, np.pi))` is fitted strictly on $X_{\text{tr\_sub}}$ and applied to $X_{\text{val}}$. | **PASSED** (No validation distribution leakage) |
| **Natural Validation Prevalence** | Folds retain natural prevalence ($\pi \approx 0.165\%$, ~75 frauds in 45,396 validation instances). | **PASSED** (Realistic evaluation conditions) |

### Potential Validation Leakage Clarification: Threshold Selection
- In reporting fold-level "calibrated cost", the threshold $\tau^*$ is found by minimizing empirical cost on the validation partition scores.
- **Methodological Disclosure**: While standard in diagnostic capstone reporting to illustrate the theoretical minimum financial cost on that fold, this represents an optimistic bound. In true out-of-sample deployment, $\tau^*$ must be calibrated on training/inner-validation splits. In our reporting, we clearly distinguish between **default threshold ($\tau=0.0$)** and **calibrated threshold ($\tau^*$)**, and emphasize **PR-AUC** as the primary rank-based, threshold-independent evaluation metric.

---

## 3. Preliminary Stage 2 Pilot Findings ($N=2,000$, Fold 0)

| Model Architecture | Features ($K$) | PR-AUC | ROC-AUC | Calibrated Prec (%) | Calibrated Recall (%) | FP | Calibrated Cost ($) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline Linear SVM** | 6 | 0.7025 | 0.9741 | 52.14% | 81.33% | 56 | $3,360 |
| **Baseline Classical RBF** | 6 | 0.7286 | 0.9763 | 64.21% | 81.33% | 34 | $3,140 |
| **Baseline Opt Separable QSVM** | 6 | 0.7082 | 0.9759 | 54.39% | 82.67% | 52 | $3,120 |
| **Baseline Canonical ZZ QSVM** | 6 | 0.5156 | 0.9744 | 62.63% | 82.67% | 37 | $2,970 |
| **Baseline PQK** | 6 | 0.6717 | 0.9755 | 67.78% | 81.33% | 29 | $3,090 |
| **Phase 14 Classical RBF** | 10 | 0.7605 | 0.9814 | 59.62% | 82.67% | 42 | $3,020 |
| **Phase 14 Opt Separable QSVM** | 10 | 0.7636 | 0.9807 | 59.05% | 82.67% | 43 | $3,030 |
| **Phase 14 Classical Linear SVM** | 16 | 0.6444 | 0.9757 | 36.69% | 82.67% | 107 | $3,670 |
| **Phase 14 Classical RBF** | 16 | 0.7640 | 0.9754 | 50.79% | 85.33% | 62 | $2,820 |
| **Phase 14 Opt Separable QSVM** | 16 | **0.7686** | 0.9759 | 62.63% | 82.67% | 37 | $2,970 |
| **Phase 14 Anisotropic QSVM** | 16 | 0.7279 | 0.9731 | 54.87% | 82.67% | 51 | $3,110 |
| **Phase 14 Hybrid RBF-QSVM** | 16 | 0.7667 | 0.9756 | 64.58% | 82.67% | 34 | $2,940 |
| **Macro Classical Linear (Full Data)**| 16 | 0.7077 | 0.9781 | 38.69% | 86.67% | 103 | $3,030 |
| **Macro Nystroem RBF (Full Data)** | 16 | 0.7251 | 0.9838 | 42.67% | 85.33% | 86 | $3,060 |

---

## 4. Next Steps
1. Maintain execution of `task-1857` without interruption until all 15 folds (3 seeds $\times$ 5 folds) across $N=2,000$ and $N=3,000$ conclude.
2. Verify all 15 fold metrics in `reports/phase14/full_metrics.json` and generate `full_summary_table.csv`.
3. Provide the full multi-seed aggregate comparative analysis, runtime/memory profile, and a transparent evaluation against the 0.85 PR-AUC target.
