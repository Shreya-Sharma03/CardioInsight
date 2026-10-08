# CardioInsight

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)

CardioInsight is a deep-learning framework that processes 12-lead electrocardiogram (ECG) signals using complementary morphological and temporal representations to estimate heart failure risk.

---

## Overview

Chronic Heart Failure (CHF), particularly Heart Failure with Reduced Ejection Fraction (HFrEF, clinical threshold $\text{LVEF} \le 45\%$), is a leading cause of cardiovascular hospitalization and mortality. While echocardiography is the clinical standard for measuring ejection fraction, it requires specialized equipment and clinical staff. Standard 12-lead ECG recordings are inexpensive, widely accessible, and rapidly acquired.

CardioInsight evaluates 12-lead ECG signals to provide non-invasive risk stratification for reduced ejection fraction:
- **Input:** Standard 12-lead ECG recordings (10 seconds sampled at 250 Hz, shape: `(12, 2500)`).
- **Processing:** Bandpass filtering, normalization, and parallel representation learning through a 1D ResNet and a stacked BiGRU with temporal attention.
- **Fusion:** Adaptive reliability weighting and cross-modal multi-head attention.
- **Output:** Predicted risk probability and binary risk stratification (`Low Risk` vs. `High Risk`).

---

## Pipeline

```text
12-Lead ECG (12, 2500)
         │
         ▼
   Preprocessing
 (0.5–40 Hz Bandpass + Lead-wise Z-Score Norm)
         │
 ┌───────┴───────┐
 │               │
 ▼               ▼
ResNet34-1D     Stacked BiGRU
(Morphological  (Temporal
 Features: 512)  Features: 256)
 │               │
 └───────┬───────┘
         │
         ▼
Adaptive Reliability Fusion
(Confidence Weighting + 8-Head Attention)
         │
         ▼
Classification Head (MLP)
         │
         ▼
Risk Probability & Stratification (τ = 0.41)
```

---

## Models

### ResNet34-1D
The morphological branch adapts ResNet-34 to 1D continuous multi-channel signals (`src/resnet_model.py`). It consists of an initial 1D convolution (`kernel_size=7`, `stride=2`), max pooling, and 4 residual stages containing `[3, 4, 6, 3]` BasicBlocks with 64, 128, 256, and 512 channels. Global average pooling produces a **512-dimensional** feature vector capturing localized waveform patterns such as QRS duration, ST-segment deviations, and T-wave morphology across all 12 leads.

### BiGRU
The temporal branch captures rhythm variations and beat-to-beat sequential dynamics across the 10-second signal (`src/bigru_model.py`). It applies batch normalization across channels and passes the signal through 3 stacked residual bidirectional GRU layers (hidden size 128 per direction, yielding a 256-wide representation) with LayerNorm. A Bahdanau additive temporal attention layer pools the recurrent sequence into a **256-dimensional** context vector.

### Fusion
The fusion module combines the two feature vectors dynamically (`src/fusion_model.py`). Both vectors are projected into a unified 512-dimensional embedding space. Confidence sub-networks compute sample-specific reliability scores, which are normalized using softmax:
$$\alpha_{\text{res}} + \alpha_{\text{gru}} = 1.0$$
The weighted modality tokens are stacked and processed by an 8-head multi-head self-attention layer, mean-pooled across tokens, and refined through a residual feed-forward network with LayerNorm.

### Classification
The fused 512-dimensional vector is passed through an MLP classification head (`src/fusion_model.py`) with LayerNorm, GELU activations, and dropout (0.30 and 0.20), producing a single logit. Applying the sigmoid function yields the risk probability $p \in [0, 1]$. Using the decision threshold $\tau = 0.41$:
- $p \ge 0.41 \implies$ **High Risk** (consistent with LVEF $\le$ 45%)
- $p < 0.41 \implies$ **Low Risk**

---

## Preprocessing

ECG preprocessing is implemented in `src/preprocessing.py`:
- **Signal Shape:** Input signals are formatted as `(12, 2500)` (12 leads $\times$ 2,500 time points). Signals provided as `(2500, 12)` are transposed automatically. Signals shorter than 2,500 samples are zero-padded; longer recordings are truncated.
- **Bandpass Filtering:** A 4th-order digital Butterworth filter ($0.5 - 40.0\text{ Hz}$) is applied via zero-phase forward-backward filtering (`scipy.signal.filtfilt`) across each lead to remove baseline wander and high-frequency noise.
- **Lead-wise Normalization:** Each lead is normalized along the temporal axis using z-score normalization:
  $$\tilde{x}_{c, t} = \frac{x_{c, t} - \mu_c}{\sigma_c + 10^{-8}}$$

---

## Data

The models were developed using the EchoNext dataset (v1.1.1):
- **Task:** Detection of structural systolic dysfunction defined by Left Ventricular Ejection Fraction $\le 45\%$ (`lvef_lte_45_flag`).
- **Signal Specifications:** Standard 12-lead ECGs (I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6) sampled at 250 Hz for 10 seconds (2,500 points per lead).
- **Patient-Level Splits:** Deduplicated partitions ensure that no patient appears across multiple splits:
  - **Training Split:** 72,475 patients (16,962 positive, 55,513 negative; class ratio $\approx 3.27 : 1$)
  - **Validation Split:** 4,626 patients
  - **Test Split:** 5,442 patients

---

## Results

### Test Split Evaluation ($N = 5,442$ Patients)

Quantitative comparison between individual branches and the fusion model recorded in `results/metrics/fusion_vs_individual_comparison.csv`:

| Model Architecture | Accuracy | Balanced Accuracy | Precision | Recall (Sensitivity) | Specificity | F1-Score | ROC-AUC | PR-AUC | MCC | Cohen's Kappa |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **BiGRU alone** | 81.77% | 81.17% | 49.05% | **80.25%** | 82.10% | 60.88% | 88.39% | 69.46% | 0.5246 | 0.4989 |
| **ResNet alone** | 87.47% | 79.98% | 63.51% | 68.40% | 91.56% | 65.87% | **89.28%** | 71.52% | 0.5826 | 0.5820 |
| **CardioInsight Fusion** | **87.80%** | **79.41%** | **65.20%** | 66.42% | **92.39%** | **65.81%** | 88.93% | **71.56%** | **0.5839** | **0.5838** |

### Observations:
- **Specificity & Precision:** The fusion model achieved the highest specificity (**92.39%**) and precision (**65.20%**), reducing false positive indications.
- **PR-AUC & Loss:** The fusion model produced the highest PR-AUC (**71.56%**), lowest Brier score (**0.0877**), and lowest log loss (**0.2970**).
- **Decision Threshold:** Using the validation-derived threshold ($\tau = 0.41$), test set balanced accuracy reaches **81.21%** with **78.79%** sensitivity.

### Validation Split Evaluation ($N = 4,626$ Patients, $\tau = 0.50$)
- **Balanced Accuracy:** 82.35%
- **ROC-AUC:** 88.92%
- **Accuracy:** 85.11%
- **Recall:** 77.94%
- **F1-Score:** 66.21%

---

## Project Contribution

The project implements a dual-branch architecture that combines:
1. A 1D convolutional residual network (ResNet34-1D) for lead-specific waveform morphology.
2. A recurrent neural network (stacked residual BiGRU with temporal attention) for rhythm and sequential dynamics.
3. Learned sample-specific modality reliability weighting with cross-modal multi-head attention to balance morphological and temporal features adaptively.

This provides a unified end-to-end framework that integrates both spatial/morphological and sequential ECG representations for non-invasive risk estimation.

---

## Future Improvements

- **External Validation:** Testing on external multi-center ECG datasets to evaluate cross-site generalizability.
- **Population Subgroup Analysis:** Evaluating performance across diverse demographic groups and comorbidities.
- **Model Calibration:** Evaluating post-hoc calibration methods (e.g., Platt scaling, isotonic regression, temperature scaling) across varying prevalence settings.
- **Interpretability & Saliency:** Adding lead-attribution and temporal saliency methods (such as Grad-CAM or integrated gradients) to highlight clinically relevant waveform segments.
- **Robustness to Artifacts:** Evaluating model behavior under missing leads, baseline drift, and electrode disconnection.
- **Model Compression:** Exploring quantization and knowledge distillation for deployment in edge or bedside screening devices.

---

## Installation

Clone the repository and install dependencies in a Python 3.10+ virtual environment:

```bash
git clone https://github.com/Shreya-Sharma03/CardioInsight.git
cd CardioInsight

python -m venv .venv
source .venv/bin/activate       # On Linux/macOS
# .venv\Scripts\activate        # On Windows

pip install -r requirements.txt
```

---

## Usage

### Command-Line Interface

Run inference on synthetic 12-lead ECG data:

```bash
python src/predict.py --demo
```

Run inference on an existing `.npy` ECG array:

```bash
python src/predict.py --ecg_file path/to/ecg_recording.npy
```

### Python API

```python
import numpy as np
from src.predict import CardioInsightPredictor

# Initialize predictor (loads checkpoints from models/)
predictor = CardioInsightPredictor()

# Load 12-lead ECG (shape: 12 leads x 2500 samples)
raw_ecg = np.random.randn(12, 2500).astype(np.float32)

# Predict risk
result = predictor.predict_ecg(raw_ecg, fs=250.0)

print(f"Risk Category    : {result['risk_stratification']}")
print(f"Risk Probability : {result['heart_failure_risk_probability']:.2%}")
print(f"Decision Boundary: {result['decision_threshold']}")
print(f"Modality Weights : {result['modality_reliability_weights']}")
```

---

## Training

The fusion model can be trained on pre-extracted ResNet and BiGRU features using `src/train_fusion.py`:

```bash
python src/train_fusion.py --data_dir path/to/features --output_dir models --epochs 15 --batch_size 512
```

### Training Options:
- `--data_dir`: Directory containing `resnet_features_{train,val}.npy`, `bigru_features_{train,val}.npy`, and labels.
- `--output_dir`: Directory to save trained model checkpoints and optimal threshold.
- `--epochs`: Number of training epochs (default: 15).
- `--batch_size`: Batch size (default: 512).
- `--lr`: Learning rate (default: 3e-4).
- `--device`: Compute device (`cuda` or `cpu`).

The training script uses a hybrid objective function (class-weighted BCE + Focal Loss), modality entropy regularization, AdamW optimizer with cosine annealing warm restarts, and saves `FusionModel.pth` along with the validation-selected decision threshold `threshold.npy`.

---

## Evaluation

Calculate clinical and statistical classification metrics using `src/evaluate.py`:

```python
from src.evaluate import compute_clinical_metrics, print_metrics_table

# y_true: binary labels (0 or 1), y_probs: predicted probabilities
metrics = compute_clinical_metrics(y_true, y_probs, threshold=0.41)
print_metrics_table(metrics, title="Test Set Metrics")
```

---

## Tests

Run the test suite to verify model architectures, checkpoint loading, preprocessing, inference, and evaluation metrics:

```bash
python tests/test_pipeline.py
```

All 9 unit tests validate:
- ResNet34-1D and BiGRU output dimensions and feature extraction.
- Fusion model weight constraints ($\alpha_{\text{res}} + \alpha_{\text{gru}} = 1.0$) and attention mechanisms.
- Checkpoint loading integrity for all models.
- Preprocessing filter behavior, normalization, and shape handling.
- End-to-end predictor pipeline and clinical metric calculations.

---

## Repository Structure

```text
CardioInsight/
├── README.md                 # Project overview and instructions
├── PROJECT_ARCHITECTURE.md   # Technical architecture and component specifications
├── requirements.txt          # Python dependencies
├── src/                      # Source code modules
│   ├── preprocessing.py      # ECG filtering and lead-wise normalization
│   ├── resnet_model.py       # 1D ResNet model for ECG feature extraction
│   ├── bigru_model.py        # BiGRU model for temporal ECG features
│   ├── fusion_model.py       # Fusion of ResNet and BiGRU representations
│   ├── train_fusion.py       # Training pipeline for the fusion model
│   ├── predict.py            # ECG preprocessing and model inference
│   ├── evaluate.py           # Model evaluation and classification metrics
│   └── __init__.py           # Package exports
├── models/                   # Saved model checkpoints and inference artifacts
│   ├── ResNet34_ECG.pth      # ResNet34-1D checkpoint
│   ├── BiGRU_ECG.pth         # BiGRU checkpoint
│   ├── FusionModel.pth       # Fusion model checkpoint
│   ├── threshold.npy         # Decision threshold
│   ├── labels.pth            # Class mapping
│   └── README.md             # Checkpoint details
├── results/                  # Experimental results and metrics
│   ├── metrics/              # Metric tables
│   └── README.md             # Results summary
└── tests/                    # Unit and integration tests
    ├── test_pipeline.py      # Test suite
    └── __init__.py
```

---

## Research / Medical Use

This project is an academic research prototype intended for machine learning experimentation and clinical risk stratification research. It is not an approved medical device, does not provide clinical diagnoses, and is not intended for direct clinical management or treatment decisions without formal clinical validation and regulatory clearance.
