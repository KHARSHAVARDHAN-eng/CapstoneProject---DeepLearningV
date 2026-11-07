# Phase 4A: Quantum Kernel SVM Experiment Design & Specification

**Project Title**: Quantum Machine Learning for Financial Fraud Detection using Quantum Kernel Methods  
**Phase**: Phase 4A — Quantum Kernel Experiment Design, Framework Evaluation & Protocol Preparation  
**Status**: Architecture & Design Phase (No quantum models or kernels trained yet)  

---

## 1. Executive Summary & Objective

Phase 4 introduces Quantum Machine Learning (QML) to the financial fraud detection pipeline. The core objective is to evaluate whether quantum Hilbert-space embeddings via parameterized quantum circuits—specifically **Quantum Kernel Methods (Quantum Support Vector Machines / QSVM)**—offer distinct decision boundaries, representational geometry, or inductive biases compared to classical linear and non-linear (RBF) SVM baselines.

In accordance with Phase 4A requirements:
- No quantum circuits are executed or trained in this phase.
- An empirical, mathematically rigorous experimental protocol is designed.
- Framework dependencies, computational complexity bottlenecks, subset selection protocols, and fair classical benchmarking controls are established.
- No claims of quantum advantage or superiority are made without empirical verification.

---

## 2. QML Framework Recommendation & Compatibility Analysis

A critical technical constraint in our environment is **Python 3.10.0** running on **macOS Apple Silicon (arm64)** with **NumPy 2.2.6**, **scikit-learn 1.7.2**, and **SciPy 1.15.3**.

We empirically evaluated the two leading open-source QML frameworks:

| Framework & Version | NumPy 2.x Compatibility | Native Scikit-Learn Integration | Apple Silicon (arm64) Hardware Acceleration | Primary Strengths |
| :--- | :---: | :---: | :---: | :--- |
| **Qiskit Machine Learning** (`qiskit>=2.5`, `qiskit-machine-learning>=0.9`) | **Full (Native)** | **Direct** (`FidelityQuantumKernel`, `QSVC`) | Accelerate / OpenBLAS statevector simulation | Standard `ZZFeatureMap`, industry-standard circuit primitives (`SamplerV2`, `StatevectorSampler`), exact alignment with Havlíček et al. (Nature 2019). |
| **PennyLane** (`pennylane>=0.42`) | **Full (Native)** | **Custom wrapper required** | `lightning.qubit` (C++ OpenMP / AVX / Accelerate) | Extreme flexibility, automatic differentiation, easy transition to Projected Quantum Kernels (PQK) and variational circuits. |

### Recommendation: **Qiskit + Qiskit Machine Learning** (with PennyLane as an alternative)
1. **Direct Scikit-Learn Ecosystem Parity**: `FidelityQuantumKernel` integrates directly into `scikit-learn`'s `SVC(kernel=kernel.evaluate)` and `QSVC`, reusing the exact same quadratic programming solver and evaluation harness established in Phase 3.
2. **Canonical Feature Maps**: Qiskit natively provides the canonical `ZZFeatureMap` and `PauliFeatureMap` architectures standard in quantum kernel research.
3. **Dependency Resolution**: Dry-run verification confirms `qiskit==2.5.2` and `qiskit-machine-learning==0.9.1` install cleanly into `.venv` without downgrading NumPy 2.2.6 or scikit-learn 1.7.2.

---

## 3. Computational Bottleneck & Complexity Analysis

In Phase 3, training a classical RBF SVM on all $N = 226,980$ samples took 59.83 seconds using optimized C/C++ LIBSVM routines. However, **scaling Quantum Kernel estimation to 226,980 samples is strictly mathematically and computationally infeasible**:

### The $O(N^2)$ Kernel Matrix Barrier:
1. **Circuit Pairs Calculation**:
   Computing the symmetric quantum kernel Gram matrix $K \in \mathbb{R}^{N \times N}$ requires evaluating:
   $$\frac{N(N - 1)}{2} \text{ unique off-diagonal inner products } |\langle \phi(x_i) | \phi(x_j) \rangle|^2$$
   For $N = 226,980$:
   $$\frac{226,980 \times 226,979}{2} \approx 25,759,850,710 \quad (\mathbf{25.76\text{ billion circuit evaluations}})$$
2. **Simulator Wall-Clock Time**:
   At an optimistic simulation speed of 1,000 statevector inner products per second on a high-end CPU/GPU:
   $$25.76 \times 10^9 \div 1,000 \text{ s} \approx 25,759,850 \text{ seconds} \approx \mathbf{298.1\text{ days}}$$
3. **Physical NISQ Quantum Hardware**:
   On physical quantum processors (e.g., IBM Quantum), executing 25.76 billion circuit pairs at 1,024 shots per pair would require $\sim 2.6 \times 10^{13}$ shots, costing millions of dollars and years of QPU queue time.

**Conclusion**: Quantum kernel matrix construction must operate on a defensible, controlled subset of the data.

---

## 4. Defensible Subset Selection Strategy

Because the fraud prevalence is only 0.1667%, a naive uniform random sample of 500 records from `X_train` would yield $500 \times 0.001667 \approx 0.83$ fraud cases (meaning most draws would contain 0 or 1 fraud sample), rendering model training impossible.

We propose a **two-tier subset partitioning strategy** derived strictly from the training split `X_train` (keeping the holdout test set `X_test` isolated):

```text
Full Preprocessed Training Split: X_train (226,980 samples: 226,602 Legit, 378 Fraud)
   │
   ├──> Tier 1: Quantum Training Subset (D_train_q)
   │      • Size: N_train = 500 samples
   │      • Composition: 100 Fraudulent (sampled from 378 frauds) + 400 Legitimate (sampled from 226k legit)
   │      • Imbalance Ratio: 4:1 (20% fraud, providing sufficient minority signal for support vector identification)
   │      • Computational Cost: (500 × 499) / 2 = 124,750 circuit evaluations (~30–60 seconds on statevector simulator)
   │
   └──> Tier 2: Quantum Validation / Evaluation Subset (D_eval_q)
          • Size: N_eval = 1,000 samples
          • Drawn from: Remaining training data (X_train \ D_train_q)
          • Composition: Realistic Class Prevalence (e.g., 2 Fraud, 998 Legit, ~0.20% fraud rate) OR 
            controlled prevalence for high-resolution PR-curve mapping.
          • Inference Cost: 1,000 × 500 = 500,000 circuit evaluations (~2 minutes).
```

---

## 5. Feature Reduction Strategy (Qubit Budget)

The dataset contains 30 features. Simulating a 30-qubit quantum statevector requires tracking a Hilbert space of dimension $2^{30} \approx 1.07 \times 10^9$ complex amplitudes ($\sim 8\text{ GB}$ of RAM per statevector), which cannot be repeatedly evaluated for hundreds of thousands of kernel entries.

Realistic NISQ simulation on local hardware requires a **qubit budget of $n \in \{4, 6, 8\}$ qubits**.

### Feature Reduction Approach:
To compress 30 features into $n$ quantum dimensions:
1. **Method A — Supervised Feature Selection (`SelectKBest` with ANOVA $F$-statistic)**:
   - Empirical analysis on `X_train` revealed extreme signal concentration:
     - Top features: `V17` ($F = 26,015$), `V14` ($F = 22,720$), `V12` ($F = 15,744$), `V10` ($F = 10,644$), `V16` ($F = 8,531$), `V3` ($F = 8,396$), `V7` ($F = 7,686$), `V11` ($F = 5,376$).
     - Features `V18`–`V28`, `Amount`, and `Time` have orders of magnitude lower direct linear/ANOVA separation.
   - Selecting the top $n = 6$ or $8$ features directly preserves the highest signal-to-noise fraud dimensions.
2. **Method B — Principal Component Analysis (PCA)**:
   - Unsupervised orthogonal projection capturing the top $n$ directions of maximal variance across all 30 features.
   - Standard in classical QML literature for unbiased dimensionality reduction.
3. **Data Leakage Safeguard**:
   - The feature reduction transformer (`PCA` or `SelectKBest`) **must be fit strictly on the training partition `X_train`** (or `D_train_q`), and subsequently applied to transform evaluation sets.
4. **Quantum Angle Bounding**:
   - Quantum rotation gates ($R_z, R_{zz}$) are periodic with period $2\pi$.
   - Features scaled arbitrarily can wrap around $2\pi$, causing artificial phase collisions (aliasing).
   - After dimensionality reduction to $n$ dimensions, features must be passed through a `MinMaxScaler(feature_range=(0, np.pi))` fit strictly on the training subset.

---

## 6. Quantum Feature Map & Simulator Specification

### 6.1 Quantum Feature Map Architecture
We specify the **Second-Order Pauli-Z Feature Map (`ZZFeatureMap`)** (Havlíček et al., 2019):
$$\mathcal{U}_{\Phi}(x) = \left( U_{\Phi}(x) H^{\otimes n} \right)^d$$
Where $d = 2$ repetitions (depth), and the data-encoding unitary is:
$$U_{\Phi}(x) = \exp \left( i \sum_{j=1}^n \phi_j(x) Z_j + i \sum_{j < k}^n \phi_{jk}(x) Z_j Z_k \right)$$
- Non-linear interactions: $\phi_j(x) = x_j$ (single-qubit rotations) and $\phi_{jk}(x) = (\pi - x_j)(\pi - x_k)$ (two-qubit $ZZ$ entangling gates).
- Entanglement topology: **Linear entanglement** (nearest-neighbor CNOT gates, optimal for planar NISQ architectures) or **Full entanglement**.

### 6.2 Quantum Kernel Definition
The state fidelity kernel measures the transition probability between encoded quantum states $|\phi(x)\rangle$ and $|\phi(x')\rangle$:
$$K(x, x') = |\langle \phi(x') | \phi(x) \rangle|^2 = |\langle 0^{\otimes n} | \mathcal{U}_{\Phi}^\dagger(x') \mathcal{U}_{\Phi}(x) | 0^{\otimes n} \rangle|^2$$

### 6.3 Simulation Engine & Reproducibility
- **Simulator**: Qiskit `StatevectorSampler` / `AerSimulator(method='statevector')` (analytic statevector simulation to eliminate shot noise during foundational kernel comparison).
- **Shot-Based Comparison (Secondary)**: 1,024 shots to evaluate tolerance to quantum measurement noise.
- **Random Seed Control**: `seed=42` fixed across NumPy, Scikit-learn, and the Qiskit simulator RNG.

---

## 7. Fair Classical Benchmarking Protocol

To prevent misleading claims of "quantum advantage" or "quantum inferiority":

### The "Matched-Capacity" Control Principle:
Any quantum model trained on a reduced dataset ($N_{\text{train\_q}} = 500$, $n = 6$ features) **must be directly compared against classical SVMs trained on the exact same $N_{\text{train\_q}}$ samples and $k = 6$ features**:

| Benchmark Model | Training Set | Feature Set | Purpose |
| :--- | :--- | :--- | :--- |
| **Classical Linear SVM (Matched)** | $D_{\text{train\_q}}$ ($N = 500$) | Top $n$ features | Measures linear separability in reduced space |
| **Classical RBF SVM (Matched)** | $D_{\text{train\_q}}$ ($N = 500$) | Top $n$ features | Standard non-linear kernel baseline in identical space |
| **Quantum Kernel SVM (QSVM)** | $D_{\text{train\_q}}$ ($N = 500$) | Top $n$ features | Evaluates quantum Hilbert space mapping geometry |
| **Phase 3 Macro Classical Baseline** | Full $X_{\text{train}}$ ($N = 226,980$) | All 30 features | Full-capacity macroeconomic baseline |

All models will be evaluated on the **exact same evaluation set** using identical metrics:
- Precision, Recall, F1-Score
- PR-AUC (Average Precision)
- ROC-AUC
- Training Time & Inference Latency

---

## 8. Protection of the Holdout Test Set

- The Phase 2 holdout test set ([data/processed/X_test.csv](file:///Users/harsha/Desktop/quantum-fraud-detection(Capstone)/data/processed/X_test.csv), 56,746 samples) was used once in Phase 3 to establish the macro benchmark.
- **Rule for Phase 4**: To prevent test set contamination through iterative parameter tuning, the holdout test set will **not** be queried during quantum feature map design, qubit tuning, or subset balancing.
- An internal validation split $D_{\text{val\_q}}$ will be carved strictly from `X_train` for all exploratory comparisons.
- The holdout test set will be evaluated only once for the final, frozen quantum model configuration.

---

## 9. Dependency Recommendations & Next Steps

### Recommended Additions to `requirements.txt`:
```text
# Phase 4: Quantum Machine Learning Dependencies
qiskit>=2.5.0
qiskit-machine-learning>=0.9.0
```
*(Verified fully compatible with existing NumPy 2.2.6 and Python 3.10 on macOS arm64).*

---

## 10. Summary of Proposed Configuration & Unresolved Decisions

### Proposed Baseline Configuration:
- **Framework**: Qiskit Machine Learning (`FidelityQuantumKernel`, `StatevectorSampler`).
- **Qubit Count**: $n = 6$ qubits (6 features).
- **Feature Selection**: Supervised `SelectKBest(f_classif, k=6)` on `X_train`, followed by `MinMaxScaler(feature_range=(0, np.pi))`.
- **Training Subset Size**: $N_{\text{train\_q}} = 500$ (100 Fraud, 400 Legit).
- **Validation Subset Size**: $N_{\text{val\_q}} = 1,000$ (realistic prevalence).
- **Feature Map**: `ZZFeatureMap(feature_dimension=6, reps=2, entanglement='linear')`.
- **Comparison Models**: Classical Linear SVM and Classical RBF SVM trained on the exact same 500 samples $\times$ 6 features.

### Unresolved Decisions for User Review:
1. **Framework Preference**: Qiskit Machine Learning (recommended for standard fidelity kernels) vs. PennyLane (recommended if exploring Projected Quantum Kernels or variational circuits).
2. **Qubit Budget**: 6 qubits (faster simulation, ~125k circuit evaluations in 30s) vs. 8 qubits (higher feature fidelity, ~3x higher simulation cost).
3. **Dimensionality Reduction**: Supervised `SelectKBest` (retains top fraud discriminators like `V17, V14, V12`) vs. Unsupervised `PCA` (standard in QML literature).
