# Multi-Domain ECG Feature Extraction for Automated Psychiatric Disorder Recognition

---

## Overview

This repository contains the full pipeline for automated classification of major psychiatric disorders — **Bipolar Disorder (BD)**, **Major Depressive Disorder (MDD)**, and **Schizophrenia (SCZ)** — from standard 12-lead ECG recordings using multi-domain feature extraction and ensemble machine learning.

The system achieves **94.8% overall accuracy** (95% CI: 92.1–96.8%) across four diagnostic categories (BD / MDD / SCZ / Healthy Control), with a mean processing time of **12.7 seconds per participant** on standard clinical hardware.

---

## Key Results

| Diagnostic Category | Accuracy | Sensitivity | Specificity | AUC |
|---|---|---|---|---|
| Bipolar Disorder | 93.5% | 92.3% | 94.1% | 0.961 |
| Major Depression | 91.8% | 89.7% | 92.4% | 0.943 |
| Schizophrenia | 96.4% | 95.1% | 96.8% | 0.973 |
| **Overall (macro)** | **94.8%** | **93.2%** | **95.3%** | **0.957** |

*All metrics from stratified 10-fold cross-validation with 1,000-iteration bootstrap confidence intervals.*

---

## Repository Structure

```
.
├── ablation/                  # Ablation study scripts and results
│   └── ...                    # Feature group removal, lead subset, model component ablations
│
├── baselines/                 # Baseline model implementations for comparison
│   └── ...                    # SVM, XGBoost, Random Forest, DNN, 1D-CNN baselines
│
├── experiments/               # Experiment entry points and configurations
│   └── ...                    # Cross-validation, temporal split, subgroup analysis experiments
│
├── preprocess/                # ECG signal preprocessing pipeline
│   └── ...                    # Bandpass filtering, baseline correction, artifact detection, R-peak detection
│
├── results/                   # Saved outputs, metrics, and figures
│   └── ...                    # Confusion matrices, ROC curves, feature importance plots
│
├── runs/                      # Training run logs and checkpoints
│   └── ...                    # Per-fold model checkpoints, TensorBoard / logging outputs
│
├── statistic_analysis/        # Statistical analysis and reporting
│   └── ...                    # HRV group comparisons, ANOVA, post-hoc tests, effect sizes
│
├── config.py                  # Global configuration: paths, hyperparameters, feature settings
├── main.py                    # Main entry point: training, evaluation, and inference
└── train_valid_and_test.py    # Core training / validation / test loop with nested CV and SMOTE
```

---

## Method Summary

### 1. Signal Preprocessing (`preprocess/`)
- 4th-order Butterworth bandpass filter (0.5–100 Hz)
- Adaptive notch filter for 50/60 Hz power line interference
- Polynomial + median-filter baseline correction
- Statistical artifact detection and removal

### 2. Multi-Domain Feature Extraction
Features are extracted independently from all **12 ECG leads**, yielding **1,248 initial features** across five domains:

| Domain | Key Features |
|---|---|
| **Time-Domain HRV** | SDNN, RMSSD, pNN50, SDANN, skewness, kurtosis, TINN |
| **Frequency-Domain** | VLF / LF / HF power, LF/HF ratio, spectral entropy (multitaper method) |
| **Wavelet Analysis** | Wavelet packet energy, entropy per subband, cross-packet correlations (MODWPT, Daubechies-4) |
| **Nonlinear Dynamics** | Sample entropy, multiscale entropy, DFA scaling exponent, Poincaré SD1/SD2 |
| **Morphological** | QTc, QT dispersion, P-wave duration, T-wave amplitude, PR/QRS intervals |

### 3. Feature Selection
1. ANOVA F-test with Bonferroni correction → 387 significant features
2. Recursive Feature Elimination (RFE) with cross-validation → **84 optimal features**
3. Elastic net regularisation (α=0.5) + bootstrap stability selection (Π > 0.75 over 100 iterations)

**Top discriminative features:** RMSSD in lead V4 (importance: 0.142) · QT dispersion (0.138) · LF/HF ratio in lead II (0.126)

### 4. Classification & Ensemble (`train_valid_and_test.py`)
- Individual classifiers: SVM (RBF), XGBoost, Random Forest, Deep Neural Network
- **Ensemble**: confidence-weighted voting across top-3 classifiers
- Class imbalance: SMOTE + Tomek links, applied **within training folds only**
- Validation: stratified 10-fold outer CV with 5-fold inner CV for hyperparameter tuning

---

## Getting Started

### Requirements

```bash
pip install numpy scipy scikit-learn xgboost imbalanced-learn neurokit2 mne wfdb matplotlib seaborn
```

> Python ≥ 3.8 recommended.

### Configuration

Edit `config.py` to set dataset paths, output directories, and hyperparameters:

```python
DATA_DIR      = "path/to/ecg/data"
RESULTS_DIR   = "results/"
N_LEADS       = 12
SAMPLING_RATE = 500          # Hz
N_FOLDS_OUTER = 10
N_FOLDS_INNER = 5
SMOTE_K       = 5
ELASTIC_NET_ALPHA = 0.5
STABILITY_THRESHOLD = 0.75
```

### Run Full Pipeline

```bash
# Full training + evaluation (nested CV)
python main.py --mode train

# Inference on new ECG recordings
python main.py --mode infer --input path/to/ecg.csv

# Run ablation studies
python ablation/run_ablation.py

# Run statistical analysis
python statistic_analysis/hrv_group_comparison.py
```

---

## Data

The study enrolled **233 participants** from a single tertiary psychiatric centre:

| Group | N | Mean Age |
|---|---|---|
| Bipolar Disorder | 62 | 32.2 ± 8.3 yr |
| Major Depression | 17 | 34.1 ± 9.8 yr |
| Schizophrenia | 119 | 35.3 ± 11.2 yr |
| Healthy Controls | 35 | 33.8 ± 9.5 yr |

ECG recordings: 12-lead, 500 Hz, mean duration 598.7 ± 45.2 seconds, acquired under controlled resting conditions.

> **Note:** The raw clinical dataset is not publicly released due to patient privacy regulations. To reproduce results, please contact the corresponding authors for data access agreements.

---

## Ablation Studies (`ablation/`)

The following ablation experiments are provided:

- **Feature domain removal** — evaluate contribution of each of the 5 feature domains
- **Lead subset analysis** — single-lead vs. multi-lead performance (V4 best single: 78.3%; all 12 leads: 94.8%)
- **Classifier component** — individual vs. ensemble model
- **SMOTE impact** — with vs. without synthetic oversampling (+3.2% accuracy)

---


## License

This project is released for **academic and research use only**. Please refer to the paper for full methodological details and limitations, including the small MDD cohort (n=17) and the single-centre design, which necessitate external multi-centre validation before any clinical deployment.
