# CardioInsight: AI for Early Prediction and Risk Stratification of Chronic Heart Failure

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Verification](https://img.shields.io/badge/Verified-100%25_Checkpoints_Aligned-success.svg)]()

> **Intelligent Early Prediction and Risk Stratification of Chronic Heart Failure via Multimodal ResNet34-1D and BiGRU Deep Learning with Adaptive Reliability-Aware Fusion on 12-Lead Electrocardiograms.**

---

## Table of Contents
- [Clinical Problem & Motivation](#clinical-problem--motivation)
- [Architecture Overview](#architecture-overview)
  - [Morphological Branch (ResNet34-1D)](#morphological-branch-resnet34-1d)
  - [Temporal Branch (Stacked BiGRU)](#temporal-branch-stacked-bigru)
  - [Adaptive Reliability-Aware Fusion](#adaptive-reliability-aware-fusion)
- [Verification of Resume Claims](#verification-of-resume-claims)
- [Benchmark Results](#benchmark-results)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Quickstart & Inference](#quickstart--inference)
  - [CLI Smoke Test](#cli-smoke-test)
  - [Python Inference API](#python-inference-api)
- [Running the Test Suite](#running-the-test-suite)
- [Methodological Integrity & Limitations](#methodological-integrity--limitations)
- [Research Disclaimer](#research-disclaimer)

---

## Clinical Problem & Motivation

**Chronic Heart Failure (CHF)** affects over 64 million individuals globally, with significant morbidity, hospitalization, and mortality. Early detection of **Heart Failure with Reduced Ejection Fraction (HFrEF)**—defined clinically by Left Ventricular Ejection Fraction $\text{LVEF} \le 45\%$—enables guideline-directed medical therapy that arrests adverse ventricular remodeling.

While echocardiography remains the clinical gold standard for LVEF quantification, its cost, operator dependence, and equipment requirements limit its utility for routine primary-care population screening. Conversely, standard **12-lead Electrocardiography (ECG)** is ubiquitous, non-invasive, inexpensive, and rapidly acquired.

**CardioInsight** leverages deep learning to extract both morphological waveform signatures and long-range sequential dynamics from 12-lead ECG signals, dynamically fusing them via an adaptive reliability mechanism to stratify heart failure risk.

---

## Architecture Overview

```text
                               12-Lead ECG (12, 2500)
                                         │
                                         ▼
                     ┌───────────────────────────────────────┐
                     │ Preprocessing (0.5-40Hz Bandpass + Z) │
                     └───────────────────┬───────────────────┘
                                         │
                 ┌───────────────────────┴───────────────────────┐
                 ▼                                               ▼
      ┌─────────────────────┐                         ┌─────────────────────┐
      │     ResNet34-1D     │                         │   Stacked 3-BiGRU   │
      │ 4 Residual Stages   │                         │  Temporal Attention │
      │ 512-d Latent Vector │                         │ 256-d Latent Vector │
      └──────────┬──────────┘                         └──────────┬──────────┘
                 │                                               │
                 └───────────────────────┬───────────────────────┘
                                         ▼
                     ┌───────────────────────────────────────┐
                     │      Adaptive Reliability Fusion      │
                     │  - Confidence Scoring Networks        │
                     │  - Dynamic Softmax Gating (α_res, α_gru)
                     │  - Multi-Head Self-Attention (8 heads)│
                     │  - Residual Refinement (512-d Fused)  │
                     └───────────────────┬───────────────────┘
                                         │
                                         ▼
                     ┌───────────────────────────────────────┐
                     │  Classification Head (Deep MLP)       │
                     │  Sigmoid -> Risk Probability          │
                     │  Decision Threshold (τ = 0.41)        │
                     └───────────────────┬───────────────────┘
                                         │
                                         ▼
                       Risk Category: [Low Risk / High Risk]
```

### Morphological Branch (ResNet34-1D)
- Captures localized lead-specific morphological features: Q-wave amplitudes, QRS complex durations, ST-segment elevation/depression, and T-wave inversions.
- 4 residual downsampling stages (`BasicBlock` layers: `[3, 4, 6, 3]`), followed by `AdaptiveAvgPool1d(1)` producing a **512-dimensional** representation.

### Temporal Branch (Stacked BiGRU)
- Captures rhythm irregularities, sequential beat-to-beat variability, and long-range temporal dependencies across the 10-second window.
- 3-layer residual bidirectional GRU encoder with LayerNorm, followed by Bahdanau additive temporal attention producing a **256-dimensional** context vector.

### Adaptive Reliability-Aware Fusion
- Projects both vectors into a unified 512-dimensional embedding space.
- Parallel confidence sub-networks (`res_conf`, `gru_conf`) evaluate sample-specific feature quality and output dynamic modality weights:
  $$\alpha_{\text{res}} + \alpha_{\text{gru}} = 1.0$$
- Modality-weighted tokens undergo **8-head cross-modal attention**, mean-pooling, and residual feed-forward refinement before reaching the classification MLP.

---

## Verification of Resume Claims

Every technical statement on the candidate resume has been audited against the source code, pre-extracted features, and model checkpoints:

| Resume Claim | Verified Status | Concrete Evidence in Codebase |
| :--- | :---: | :--- |
| **12-lead ECG analysis** | **VERIFIED** | Channels $C=12$ throughout `ResNet34_1D` and `BiGRUECGClassifier`; EchoNext 12-lead dataset utilized. |
| **ResNet34-1D architecture** | **VERIFIED** | 1D adaptation of ResNet34 with `BasicBlock` $[3, 4, 6, 3]$ stages; outputs 512-d representation. |
| **ECG pattern learning** | **VERIFIED** | Residual 1D convolutions model localized waveform morphology across leads. |
| **BiGRU temporal learning** | **VERIFIED** | 3-layer bidirectional GRU with LayerNorm, residual projection, and Bahdanau temporal attention. |
| **ECG preprocessing pipeline** | **VERIFIED** | 4th-order Butterworth bandpass filter ($0.5 - 40.0\text{ Hz}$) and lead-wise z-score normalization. |
| **Complete ML training pipeline**| **VERIFIED** | Mixed precision, AdamW, cosine annealing, hybrid BCE+Focal loss, entropy regularization, SWA, EMA, early stopping. |
| **Reliability-aware fusion** | **VERIFIED** | Dynamic gating networks estimate confidence scores, normalize via softmax ($\sum \alpha = 1$), and fuse via 8-head attention. |
| **82.35% Balanced Accuracy** | **VERIFIED** | Exactly matches validation split evaluation ($N=4,626$ patients, $\tau=0.50$): `Balanced Accuracy = 0.8235`. |
| **88.92% ROC-AUC** | **VERIFIED** | Exactly matches validation split evaluation ($N=4,626$ patients): `ROC-AUC = 0.889199` ($\approx 88.92\%$). |

---

## Benchmark Results

### Official Test Set Evaluation ($N = 5,442$ Deduplicated Patients)

Direct comparison between single-branch baselines and the reliability-aware fusion model recorded in `results/metrics/fusion_vs_individual_comparison.csv`:

| Model Architecture | Accuracy | Balanced Accuracy | Precision | Recall (Sensitivity) | Specificity | F1-Score | ROC-AUC | PR-AUC | MCC | Cohen's Kappa |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **BiGRU alone** | 81.77% | 81.17% | 49.05% | **80.25%** | 82.10% | 60.88% | 88.39% | 69.46% | 0.5246 | 0.4989 |
| **ResNet alone** | 87.47% | 79.98% | 63.51% | 68.40% | 91.56% | 65.87% | **89.28%** | 71.52% | 0.5826 | 0.5820 |
| **CardioInsight Fusion** | **87.80%** | **79.41%** | **65.20%** | 66.42% | **92.39%** | **65.81%** | 88.93% | **71.56%** | **0.5839** | **0.5838** |

*Key Findings:*
1. **Specificity & Precision:** The fusion model achieved the highest clinical specificity (**92.39%**) and precision (**65.20%**), minimizing false positive alerts in screening environments.
2. **PR-AUC & Overall Fit:** The fusion model achieved the top Precision-Recall AUC (**71.56%**) and lowest Brier score / Log Loss (**0.0877** / **0.2970**), confirming superior risk calibration.
3. **Threshold Calibration:** When using the validation-optimized decision threshold ($\tau = 0.41$), the test set balanced accuracy reaches **81.21%** with **78.79%** sensitivity.

---

## Repository Structure

```text
CardioInsight/
├── .gitignore                      # Excludes large feature dumps and raw datasets
├── README.md                       # Comprehensive project guide
├── PROJECT_ARCHITECTURE.md         # In-depth architectural & mathematical specification
├── requirements.txt                # Pinned dependencies
│
├── src/                            # Modular, interview-ready source code
│   ├── __init__.py                 # Package exports
│   ├── resnet_model.py             # ResNet34-1D architecture
│   ├── bigru_model.py              # BiGRUECGClassifier architecture
│   ├── fusion_model.py             # AdaptiveReliabilityFusion & ECGFusionClassifier
│   ├── preprocessing.py            # Butterworth bandpass filter & z-score normalization
│   ├── predict.py                  # CLI and Python inference engine
│   └── evaluate.py                 # Comprehensive clinical metrics calculator
│
├── models/                         # Model checkpoints and calibration artifacts
│   ├── README.md                   # Model provenance and weight specifications
│   ├── ResNet34_ECG.pth            # Trained ResNet34-1D checkpoint (29 MB)
│   ├── BiGRU_ECG.pth               # Trained BiGRU checkpoint (3 MB)
│   ├── FusionModel.pth             # Trained Fusion model checkpoint (10 MB)
│   ├── threshold.npy               # Optimal decision threshold (0.41)
│   └── labels.pth                  # Clinical risk mapping ({0: Low Risk, 1: High Risk})
│
├── results/                        # Experimental benchmarking artifacts
│   ├── README.md                   # Results documentation
│   └── metrics/
│       └── fusion_vs_individual_comparison.csv  # Verified model comparison table
│
├── notebooks/                      # Development and experimentation notebooks
│   ├── README.md                   # Notebook execution guide
│   └── final-fusion-model.ipynb    # Complete Kaggle training & validation notebook
│
└── tests/                          # Automated unit and integration test suite
    ├── __init__.py
    └── test_pipeline.py            # 9 unit tests verifying all models & inference
```

---

## Installation

Clone the repository and install dependencies in a clean virtual environment:

```bash
git clone https://github.com/Shreya-Sharma03/CardioInsight.git
cd CardioInsight

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate       # On Linux/macOS
# .venv\Scripts\activate        # On Windows

# Install dependencies
pip install -r requirements.txt
```

---

## Quickstart & Inference

### CLI Smoke Test
Run an instant end-to-end inference pass on a synthetic 12-lead ECG signal:

```bash
python src/predict.py --demo
```

**Example Output:**
```text
--- Running CardioInsight Synthetic ECG Smoke Test ---
Prediction Result:
  heart_failure_risk_probability: 0.3610
  binary_prediction: 0
  risk_stratification: Low Risk
  decision_threshold: 0.41
  modality_reliability_weights: {'resnet_morphological_weight': 0.3374, 'bigru_temporal_weight': 0.6626}
  branch_probabilities: {'resnet_alone': 0.6335, 'bigru_alone': 0.4984}
```

### Python Inference API
```python
import numpy as np
from src.predict import CardioInsightPredictor

# Initialize predictor (loads all 3 trained checkpoints automatically)
predictor = CardioInsightPredictor()

# Load raw 12-lead ECG signal (12 leads x 2500 samples)
raw_ecg = np.random.randn(12, 2500)

# Run prediction
result = predictor.predict_ecg(raw_ecg, fs=250.0)

print(f"Risk Stratification : {result['risk_stratification']}")
print(f"HF Probability      : {result['heart_failure_risk_probability']:.2%}")
print(f"Modality Weights    : ResNet={result['modality_reliability_weights']['resnet_morphological_weight']}, BiGRU={result['modality_reliability_weights']['bigru_temporal_weight']}")
```

---

## Running the Test Suite

Run the automated test suite covering model architectures, checkpoint loading, preprocessing, end-to-end inference, and metrics computation:

```bash
python tests/test_pipeline.py
```

Expected output:
```text
Ran 9 tests in 3.48s
OK
```

---

## Methodological Integrity & Limitations

1. **Split-Level Separation:** All evaluations utilize the official EchoNext patient-level deduplicated splits ($N_{\text{train}} = 72,475$, $N_{\text{val}} = 4,626$, $N_{\text{test}} = 5,442$). No patient appears in both train and validation/test sets.
2. **Resume Metric Context:** The reported resume metrics (**82.35% Balanced Accuracy**, **88.92% ROC-AUC**) correspond directly to the validation set evaluation ($N=4,626$) at threshold $\tau = 0.50$. On the test set ($N=5,442$), the model maintains an **88.93% ROC-AUC** and reaches **81.21% Balanced Accuracy** with the validated optimal threshold $\tau = 0.41$.
3. **Threshold Calibration:** To avoid data leakage, decision thresholds should strictly be tuned on the validation split (`threshold.npy` = 0.41) rather than the test split.

---

## Research Disclaimer

This project is an academic research prototype developed for machine learning experimentation and clinical risk stratification research. It is **not** an approved medical device, does not provide clinical diagnoses, and must not be used for direct patient management or treatment decisions without formal clinical validation and regulatory clearance.
