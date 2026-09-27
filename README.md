# Kaggle Digit Recognizer (MNIST) — Pocket Data Science IV

[![Python 3.14](https://img.shields.io/badge/Python-3.14-3776AB?logo=python&logoColor=white)](https://python.org)
[![Platform: Android Termux](https://img.shields.io/badge/Platform-Android%20Termux-black?logo=android&logoColor=white)](https://termux.dev)
[![Agent: Google Antigravity CLI](https://img.shields.io/badge/AI%20Agent-Google%20Antigravity%20CLI-4285F4?logo=google&logoColor=white)](https://antigravity.google)
[![Kaggle Rank](https://img.shields.io/badge/Kaggle%20Rank-%23406%20%2F%20863%20(Top%2047.05%25)-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/competitions/digit-recognizer)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A complete, systematic study climbing the [Kaggle Digit Recognizer (MNIST)](https://www.kaggle.com/competitions/digit-recognizer) leaderboard from **zero-parameter baselines** (`0.10139`) to **Top 47.05%** (`0.98792`, Rank #406 of 863 teams, beating 457 teams) executed entirely on a consumer **Android smartphone CPU** using **Termux PRoot** and **Google Antigravity CLI (`agy`)**.

---

## 📖 Published Technical Articles

* **Canonical Technical Report:** [malcolmlow.com &mdash; Pocket Data Science IV: Tackling Kaggle MNIST on Android with Antigravity CLI](https://malcolmlow.com/2026/09/27/pocket-data-science-4-mnist-digit-recognizer-subspace-svm-android-termux/)
* **DEV Community Article:** [dev.to/malcolmlow &mdash; Pocket Data Science IV: Tackling Kaggle MNIST on Android](https://dev.to/malcolmlow/pocket-data-science-iv-tackling-kaggle-mnist-on-android-with-antigravity-cli-1nfk)

---

## 🏆 The 10-Tier Experiment Progression

Every submission was generated, validated, and evaluated against Kaggle's public test set (28,000 unseen images):

| Exp | Script | Model / Strategy | CV Acc | Kaggle Score | LB Rank | Percentile | Teams Beaten |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **01** | [`exp01_random_baseline.py`](exp01_random_baseline.py) | Empirical Random Prior | N/A | `0.10139` (10.14%) | #847 | Bottom 1.9% | 16 |
| **02** | [`exp02_majority_baseline.py`](exp02_majority_baseline.py) | Constant Majority Class (Digit 1) | 11.35% | `0.11403` (11.40%) | #846 | Bottom 2.0% | 17 |
| **03** | [`exp03_nearest_centroid.py`](exp03_nearest_centroid.py) | Nearest Centroid ("Ghost Templates") | 82.04% | `0.81446` (81.45%) | #835 | Bottom 3.2% | 28 |
| **04** | [`exp04_softmax_regression.py`](exp04_softmax_regression.py) | Multinomial Softmax Logistic Regression | 92.56% | `0.92089` (92.09%) | #806 | Bottom 6.6% | 57 |
| **05** | [`exp05_classical_ml.py`](exp05_classical_ml.py) | Latent Ensemble: PCA(55) + k-NN + ExtraTrees | 97.46% | `0.97578` (97.58%) | #568 | Top 65.9% | 294 |
| **06** | [`exp06_neural_network_ensemble.py`](exp06_neural_network_ensemble.py) | Neural Tri-Blend: Deep MLP + k-NN + ExtraTrees | 97.98% | `0.98085` (98.09%) | #523 | Top 60.7% | 339 |
| **07** | [`exp07_rbf_svm.py`](exp07_rbf_svm.py) | Orthogonal Subspace SVM: PCA(55) + RBF-SVM | 98.60% | `0.98453` (98.45%) | #487 | Top 56.4% | 376 |
| **08** | [`exp08_meta_ensemble.py`](exp08_meta_ensemble.py) | Dual Meta-Ensemble: 90% Subspace SVM + 10% MLP | 98.66% | `0.98514` (98.51%) | #481 | Top 55.7% | 382 |
| **09** | [`exp09_augmented_manifold_svm.py`](exp09_augmented_manifold_svm.py) | 2× Translation-Augmented Subspace SVM (84k train) | 98.68% | `0.98696` (98.70%) | #438 | Top 50.8% | 425 |
| **10** | [`exp10_dual_axis_augmented_svm.py`](exp10_dual_axis_augmented_svm.py) | **3× Dual-Axis Augmented Subspace SVM (126k train)** | **98.78%** | **`0.98792` (98.79%)** | **#406** | **Top 47.05%** | **457** |

---

## ⚡ Mobile Compute & Throughput Benchmark

All benchmarks measured directly on an ARM64 8-core mobile CPU under standard thermal constraints:

| Pipeline | Features | 5k CV Fit | Val Acc | Full Train Fit | Test Inference (28k) | Throughput |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Deep MLP Baseline** | Raw 784px | 27.8s | 98.18% | 344.4s (5.7 min) | 2.5s | ~11,200 img/s |
| **Raw RBF-SVM** | Raw 784px | 148.6s | 98.24% | ~25 min (est.) | N/A | High latency |
| **Subspace RBF-SVM (Exp 07)** | PCA(55) | 7.2s | 98.60% | **22.5s (42k)** | **19.6s** | **~1,425 img/s** |
| **Dual Meta-Ensemble (Exp 08)** | Subspace + Raw | 364.6s | 98.66% | 544.1s (9.1 min) | 36.7s | ~763 img/s |
| **Augmented Subspace SVM (Exp 09)** | PCA(55) (84k) | 50.8s | 98.68% | 51.4s (84k) | 52.9s | ~529 img/s |
| **Dual-Axis Aug SVM (Exp 10)** | **PCA(55) (126k)** | **91.8s** | **98.78%** | **205.6s (3.4 min)** | **168.9s (2.8 min)** | **~166 img/s** |

---

## 🔬 Key Engineering & Mathematical Findings

### 1. Zero-Parameter "Ghost Templates" (81.45%)
Computing the element-wise arithmetic mean vector $\bar{\mu}_c = \frac{1}{N_c} \sum_{i \in c} x_i$ for each digit class $c \in \{0..9\}$ produces 10 average prototype templates. Assigning test samples by maximum cosine similarity $\arg\max_c \frac{x \cdot \bar{\mu}_c}{\|x\|\|\bar{\mu}_c\|}$ requires zero gradient updates or iterations, yet achieves **81.45% accuracy**, demonstrating the strong geometric separation already present in raw pixel space.

### 2. Latent Space Regularization & Noise Filtering
Projecting 784-pixel images onto an orthogonal 55-dimensional PCA subspace (preserving 83.2% variance) does more than compress the data $14.25\times$:
* It acts as an optimal **low-pass filter against boundary pixel noise**.
* Subspace RBF-SVM achieved **98.60% validation accuracy** compared to **98.24%** on raw pixels, while training **17× faster** (7.2s vs 148.6s).

### 3. Dual-Axis Translation Invariance (126,000 Samples)
Radial Basis Function (RBF) kernels compute Euclidean distances:
$$K(x, z) = \exp\left(-\gamma \|x - z\|^2\right)$$
Shifting a digit by just 1 pixel drastically alters $\|x - z\|^2$ in pixel space. By systematically tripling the training set to **126,000 samples** via dual-axis translational shifts:
* Horizontal jitter: $\Delta x \in \{+1, -1\}$ (&plusmn;1px on X-axis, 42,000 samples)
* Vertical jitter: $\Delta y \in \{+1, -1\}$ (&plusmn;1px on Y-axis, 42,000 samples)
* Base raw images (42,000 samples)

The learned support vectors internalize 2D positional invariance, driving our Kaggle test score to **`0.98792` (Rank #406, Top 47.05%)** in just 3.4 minutes of training time on mobile CPU!

---

## 🚀 How to Reproduce

### 1. Requirements
```bash
pip install -r requirements.txt
```

### 2. Download Kaggle Dataset
```bash
kaggle competitions download -c digit-recognizer
unzip digit-recognizer.zip
```

### 3. Run Experiments
Each script is completely self-contained with no external dependencies beyond scikit-learn and standard scientific Python:

```bash
# Baseline Ghost Templates (Exp 03)
python3 exp03_nearest_centroid.py

# Multinomial Softmax Regression (Exp 04)
python3 exp04_softmax_regression.py

# Classical Latent Ensemble (Exp 05)
python3 exp05_classical_ml.py

# Deep Neural Network Tri-Blend (Exp 06)
python3 exp06_neural_network_ensemble.py

# Orthogonal Subspace SVM (Exp 07)
python3 exp07_rbf_svm.py

# Dual Meta-Ensemble (Exp 08)
python3 exp08_meta_ensemble.py

# Translation-Augmented SVM (Exp 09)
python3 exp09_augmented_manifold_svm.py

# Top 47% Dual-Axis Augmented Subspace SVM (Exp 10)
python3 exp10_dual_axis_augmented_svm.py
```

---

## 📜 License

MIT License &copy; 2026 Malcolm Low.
