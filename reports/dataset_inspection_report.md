# Phase 1: Dataset Inspection Report

**Project Title**: Quantum Machine Learning for Financial Fraud Detection using Quantum Kernel Methods  
**Dataset**: ULB Credit Card Fraud Detection (`data/raw/creditcard.csv`)  

---

## 1. Dataset Dimensions & Memory Footprint

- **Total Rows (Observations)**: 284,807
- **Total Columns (Features + Target)**: 31
- **In-Memory Size**: ~67 MB

## 2. Missing Values & Duplicate Records

- **Total Missing (Null) Values**: 0
- **Columns with Missing Values**: 0
- **Duplicate Rows**: 1,081 (0.3796%)

## 3. Class Imbalance Analysis

- **Total Transactions**: 284,807
- **Legitimate Transactions (`Class = 0`)**: 284,315 (99.8273%)
- **Fraudulent Transactions (`Class = 1`)**: 492 (0.1727%)
- **Imbalance Ratio**: 577.88:1 (Legitimate to Fraud)

## 4. Key Feature Statistics

### Transaction Amount (`Amount`)
- **Mean**: $88.35
- **Standard Deviation**: $250.12
- **Min / Max**: $0.00 / $25691.16
- **25% / 50% (Median) / 75%**: $5.60 / $22.00 / $77.16
- **Skewness**: 16.98 (High right-skewed tail)

### Transaction Elapsed Time (`Time`)
- **Range**: 0.0s to 172792.0s (~48.00 hours / 2 days)

### Sample Numerical Feature Statistics (Subset)

| Feature | mean       | std       | min      | 50%     | max      | skewness |
| ------- | ---------- | --------- | -------- | ------- | -------- | -------- |
| Time    | 94813.8596 | 47488.146 | 0.0      | 84692.0 | 172792.0 | -0.0356  |
| Amount  | 88.3496    | 250.1201  | 0.0      | 22.0    | 25691.16 | 16.9777  |
| V1      | 0.0        | 1.9587    | -56.4075 | 0.0181  | 2.4549   | -3.2807  |
| V2      | 0.0        | 1.6513    | -72.7157 | 0.0655  | 22.0577  | -4.6249  |
| V3      | -0.0       | 1.5163    | -48.3256 | 0.1798  | 9.3826   | -2.2402  |
| V4      | 0.0        | 1.4159    | -5.6832  | -0.0198 | 16.8753  | 0.6763   |
| V17     | -0.0       | 0.8493    | -25.1628 | -0.0657 | 9.2535   | -3.8449  |
| V28     | -0.0       | 0.3301    | -15.4301 | 0.0112  | 33.8478  | 11.1921  |

## 5. Column Schema & Data Types

| Column | Dtype   | Non-Null Count | Null Count | Null % |
| ------ | ------- | -------------- | ---------- | ------ |
| Time   | float64 | 284807         | 0          | 0.0    |
| V1     | float64 | 284807         | 0          | 0.0    |
| V2     | float64 | 284807         | 0          | 0.0    |
| V3     | float64 | 284807         | 0          | 0.0    |
| V4     | float64 | 284807         | 0          | 0.0    |
| V5     | float64 | 284807         | 0          | 0.0    |
| V6     | float64 | 284807         | 0          | 0.0    |
| V7     | float64 | 284807         | 0          | 0.0    |
| V8     | float64 | 284807         | 0          | 0.0    |
| V9     | float64 | 284807         | 0          | 0.0    |
| V10    | float64 | 284807         | 0          | 0.0    |
| V11    | float64 | 284807         | 0          | 0.0    |
| V12    | float64 | 284807         | 0          | 0.0    |
| V13    | float64 | 284807         | 0          | 0.0    |
| V14    | float64 | 284807         | 0          | 0.0    |
| V15    | float64 | 284807         | 0          | 0.0    |
| V16    | float64 | 284807         | 0          | 0.0    |
| V17    | float64 | 284807         | 0          | 0.0    |
| V18    | float64 | 284807         | 0          | 0.0    |
| V19    | float64 | 284807         | 0          | 0.0    |
| V20    | float64 | 284807         | 0          | 0.0    |
| V21    | float64 | 284807         | 0          | 0.0    |
| V22    | float64 | 284807         | 0          | 0.0    |
| V23    | float64 | 284807         | 0          | 0.0    |
| V24    | float64 | 284807         | 0          | 0.0    |
| V25    | float64 | 284807         | 0          | 0.0    |
| V26    | float64 | 284807         | 0          | 0.0    |
| V27    | float64 | 284807         | 0          | 0.0    |
| V28    | float64 | 284807         | 0          | 0.0    |
| Amount | float64 | 284807         | 0          | 0.0    |
| Class  | int64   | 284807         | 0          | 0.0    |

## 6. Visualizations Saved

- `reports/figures/class_imbalance.png`
- `reports/figures/amount_distribution.png`
- `reports/figures/amount_by_class_boxplot.png`

## 7. Implications for Quantum Kernel Methods (Next Phases)

1. **Dimensionality Reduction Requirement**:
   - The dataset contains 30 input features (`Time`, `Amount`, `V1`–`V28`).
   - Current Noisy Intermediate-Scale Quantum (NISQ) simulators and hardware have limited qubit capacity (typically 8–16 qubits for practical quantum kernel simulation).
   - Feature selection or further dimensionality reduction (e.g., PCA, autoencoders, or feature importance ranking) will be essential prior to quantum feature mapping.

2. **Severe Class Imbalance Strategy**:
   - With only 0.17% fraud prevalence, training quantum kernel classifiers directly on the full dataset is computationally prohibitive ($O(N^2)$ kernel matrix complexity) and susceptible to majority class bias.
   - Balanced sub-sampling (e.g., all frauds + sampled legitimate cases) or undersampling/SMOTE techniques will be needed for quantum kernel matrix calculation.

3. **Data Scaling for Quantum State Encoding**:
   - Quantum feature maps (e.g., angle embedding, ZZ-feature maps) require features bounded in $[0, \pi]$ or $[-\pi, \pi]$ to prevent phase ambiguity.
   - Power-transforming `Amount` (to handle extreme skewness) followed by `MinMaxScaler` or `RobustScaler` will be crucial in Phase 2.
