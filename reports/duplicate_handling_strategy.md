# Phase 2: Duplicate Row Investigation and Handling Strategy

**Project Title**: Quantum Machine Learning for Financial Fraud Detection using Quantum Kernel Methods  
**Dataset**: ULB Credit Card Fraud Detection (`data/raw/creditcard.csv`)  
**Phase**: Phase 2 — Data Preprocessing & Classical Baseline Preparation  

---

## 1. Executive Summary

During Phase 1 exploratory data inspection, **1,081 duplicate rows** (0.3796% of the 284,807 total rows) were detected across all 31 columns (`Time`, `Amount`, `V1`–`V28`, and `Class`).

This report presents an empirical investigation into these duplicate records and provides a mathematically and methodologically justified strategy for handling them during Phase 2 preprocessing.

**Core Decision**: **Remove all 1,081 redundant duplicate instances (`keep='first'`) prior to train/test partitioning.**

---

## 2. Empirical Investigation of Duplicate Records

### 2.1 Breakdown by Target Class
- **Total rows involved in duplicate clusters (`keep=False`)**: 1,854 rows (representing 773 unique transaction signatures).
- **Redundant duplicate rows to drop (`keep='first'`)**: **1,081 rows**
  - **Legitimate Transactions (`Class = 0`)**: 1,062 duplicate instances
  - **Fraudulent Transactions (`Class = 1`)**: 19 duplicate instances
- **Conflict Analysis**: Exactly **0** conflicting records exist. Every duplicate record is identical across all 30 features *and* shares the identical class label.

### 2.2 Consecutive Index Repetition (Logging Artifacts)
Detailed examination of the fraudulent duplicate clusters reveals that duplicates occur at adjacent indices with identical timestamps, identical amounts, and identical PCA coordinates down to 14 decimal places:
- Example: Indices `102441` to `102446` are 6 consecutive rows with `Time = 68207.0`, `Amount = $1.00`, and identical PCA feature vectors.
- Example: Indices `141257` and `141258` are identical consecutive rows with `Time = 84204.0`, `Amount = $0.00`.
- Example: Indices `141259` and `141260` are identical consecutive rows with `Time = 84204.0`, `Amount = $512.25`.

These consecutive identical rows represent transaction logging retries, network retransmissions, or batch recording artifacts in financial processing systems, rather than independent real-world financial fraud events.

---

## 3. Methodological Justification for Removal

### 3.1 Prevention of Train-Test Data Leakage
If duplicates are retained and an 80/20 train/test split is performed:
- Identical records will inevitably be partitioned into both the training and test sets.
- A classifier (whether classical SVM, Random Forest, or Quantum Kernel Estimator) would be evaluated on test samples that are verbatim copies of training samples.
- This creates **train-test contamination**, artificially inflating evaluation metrics (AUC-ROC, AUC-PR, Precision, Recall).

### 3.2 Impact on Fraud Representation
- Total unique fraudulent signatures involved in duplicate clusters: 13.
- By using `keep='first'`, all 13 unique fraudulent transaction signatures are retained.
- The fraud count shifts from 492 to 473 (a reduction of 19 redundant rows).
- The class ratio shifts minimally from **0.1727%** (1 in 579) to **0.1667%** (1 in 600).
- No unique fraud behavior or pattern is lost.

### 3.3 Computational Implications for Quantum Kernel Methods
- Quantum kernel matrix estimation requires computing pairwise inner products $|\langle \phi(x_i) | \phi(x_j) \rangle|^2$ for all training points $i, j \in \{1, \dots, N\}$.
- Kernel matrix computation scales quadratically $O(N^2)$ in quantum circuit evaluations.
- Retaining identical samples causes redundant quantum circuit evaluations where $K(x_i, x_i) = 1$ and introduces rank deficiency (linearly dependent rows) into the Gram matrix, potentially destabilizing dual SVM quadratic programming solvers.

---

## 4. Preprocessing Implementation Specification

1. **Deduplication Step**:
   - `df_dedup = df.drop_duplicates(keep='first')`
   - Log exact counts removed: 1,062 legitimate and 19 fraudulent records.
2. **Post-Deduplication Dimensions**:
   - Total rows: **283,726**
   - Legitimate rows (`Class = 0`): **283,253** (99.8333%)
   - Fraudulent rows (`Class = 1`): **473** (0.1667%)
3. **Partitioning**:
   - Stratified 80/20 train/test split on deduplicated data:
     - **Training Set (80%)**: 226,980 samples (226,602 legitimate, 378 fraudulent)
     - **Testing Set (20%)**: 56,746 samples (56,651 legitimate, 95 fraudulent)
