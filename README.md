# CardioInsight

CardioInsight is a deep-learning project for estimating the risk of reduced left ventricular ejection fraction from standard 12-lead electrocardiogram (ECG) recordings. The system combines complementary convolutional and recurrent representations of ECG signals and uses adaptive feature fusion before classification.

---

## Overview

Reduced left ventricular ejection fraction (LVEF) is an important indicator of systolic cardiac dysfunction. CardioInsight explores whether information contained in a standard 12-lead ECG can be used to estimate this condition without directly using echocardiographic measurements as model input.

The pipeline processes a 10-second, 12-lead ECG recording and extracts two complementary types of representations:

* **Morphological features** from a 1D ResNet34 network
* **Temporal features** from a stacked bidirectional GRU with temporal attention

These representations are combined using learned modality weighting and multi-head attention before producing the final risk probability.

### Input

* 12-lead ECG
* Sampling frequency: **250 Hz**
* Recording duration: **10 seconds**
* Input shape: **`(12, 2500)`**

### Output

* Predicted probability
* Binary risk classification:

  * **Low Risk**
  * **High Risk**

The target is based on the `lvef_lte_45_flag` label used in the dataset.

---

## Pipeline

```text
                    12-Lead ECG
                    (12 × 2500)
                         │
                         ▼
                ┌─────────────────┐
                │  Preprocessing  │
                │                 │
                │ 0.5–40 Hz       │
                │ Bandpass Filter │
                │ + Z-score Norm  │
                └────────┬────────┘
                         │
               ┌─────────┴─────────┐
               │                   │
               ▼                   ▼
       ┌───────────────┐   ┌────────────────┐
       │ ResNet34-1D   │   │     BiGRU      │
       │               │   │                │
       │ Morphological │   │ Temporal       │
       │ Features      │   │ Features       │
       └───────┬───────┘   └───────┬────────┘
               │                   │
               │    512-d / 256-d  │
               └─────────┬─────────┘
                         ▼
              ┌─────────────────────┐
              │ Adaptive Reliability│
              │ Fusion              │
              │                     │
              │ Learned Weights     │
              │ + Multi-Head Attn.  │
              └──────────┬──────────┘
                         ▼
                Classification Head
                         │
                         ▼
                  Risk Probability
                         │
                         ▼
                Decision Threshold
                       τ = 0.41
                         │
                         ▼
              Low Risk / High Risk
```

---

## Model Architecture

### 1. ECG Preprocessing

Implemented in `src/preprocessing.py`.

The preprocessing pipeline standardizes ECG recordings before they are passed to the neural networks.

**Operations:**

1. Input shape validation and formatting
2. Zero-padding or truncation to 2,500 samples when required
3. 4th-order Butterworth bandpass filtering from **0.5 to 40 Hz**
4. Lead-wise z-score normalization

The normalization is applied independently to each ECG lead:

$$
\tilde{x}_{c,t}
=
\frac{x_{c,t}-\mu_c}
{\sigma_c+\epsilon}
$$

where \(c\) denotes the lead and \(t\) denotes the time sample.

---

### 2. ResNet34-1D Branch

Implemented in `src/resnet_model.py`.

The ResNet branch adapts the ResNet-34 architecture to one-dimensional ECG signals.

**Architecture:**

* Input channels: `12`
* Initial `Conv1d`
* Max pooling
* Four residual stages
* BasicBlock configuration: `[3, 4, 6, 3]`
* Channel progression: `64 → 128 → 256 → 512`
* Adaptive average pooling
* Output representation: **512 dimensions**

This branch focuses on local waveform and morphological information distributed across the ECG leads.

---

### 3. BiGRU Branch

Implemented in `src/bigru_model.py`.

The recurrent branch models sequential information in the ECG waveform.

**Architecture:**

* Channel-wise batch normalization
* Three stacked bidirectional GRU layers
* Hidden size: `128` per direction
* Output width: `256`
* Residual connections between recurrent layers
* Layer normalization
* Bahdanau additive temporal attention
* Output representation: **256 dimensions**

The attention layer weights different time steps before producing the final temporal representation.

---

### 4. Adaptive Feature Fusion

Implemented in `src/fusion_model.py`.

The ResNet and BiGRU representations have different dimensions, so they are first projected into a shared **512-dimensional feature space**.

The fusion module then:

1. Projects both feature vectors to the same dimension
2. Computes a reliability score for each branch
3. Converts the scores into normalized modality weights using softmax
4. Forms weighted modality tokens
5. Applies **8-head multi-head self-attention**
6. Mean-pools the resulting representations
7. Applies residual feed-forward refinement

The two learned modality weights satisfy:

$$
\alpha_{\text{ResNet}}+\alpha_{\text{BiGRU}}=1
$$

This allows the relative contribution of the two branches to vary between samples instead of relying on fixed weighting.

---

### 5. Classification Head

The fused 512-dimensional representation is passed through a multilayer perceptron containing:

* Linear layers
* Layer normalization
* GELU activations
* Dropout
* Final single-output logit

The logit is converted into a probability using the sigmoid function:

$$
p = \sigma(z)
$$

A decision threshold of **0.41**, selected from the validation set, is used for binary classification:

```text
p >= 0.41  →  High Risk
p <  0.41  →  Low Risk
```

---

## Data

The project uses the **EchoNext v1.1.1** dataset.

### Target

The prediction target is:

`lvef_lte_45_flag`

which identifies recordings associated with LVEF at or below 45%.

### ECG Format

Each recording contains:

* 12 standard ECG leads
* 250 Hz sampling frequency
* 10 seconds of signal
* 2,500 samples per lead

### Dataset Splits

The repository uses patient-level, deduplicated splits:

| Split      | Patients | Positive | Negative |
| ---------- | -------: | -------: | -------: |
| Training   |   72,475 |   16,962 |   55,513 |
| Validation |    4,626 |        — |        — |
| Test       |    5,442 |        — |        — |

The training split has a negative-to-positive ratio of approximately **3.27:1**, which is addressed during training through class-weighted loss.

---

## Training

The training pipeline uses a combination of class-imbalance handling and regularization techniques.

### Loss

The training objective combines:

* Class-weighted binary cross-entropy
* Focal loss
* Modality entropy regularization

The positive class weight is derived from the training-set class distribution.

### Optimization

* **Optimizer:** AdamW
* **Learning rate:** `3 × 10⁻⁴`
* **Weight decay:** `1 × 10⁻⁴`
* **Scheduler:** Cosine Annealing Warm Restarts
* **Gradient clipping:** maximum norm `1.0`
* **Early stopping:** validation-based

The fusion model is trained using the feature representations produced by the two branches.

### Training Script

The training workflow is available through:

```bash
python src/train_fusion.py
```

Refer to the script's command-line arguments for dataset and output-path configuration.

---

## Results

### Test Set

The test split contains **5,442 patients**.

The table below compares the two individual branches with the combined fusion model.

| Model                    |   Accuracy | Balanced Accuracy |  Precision |     Recall | Specificity |         F1 |    ROC-AUC |     PR-AUC |
| ------------------------ | ---------: | ----------------: | ---------: | ---------: | ----------: | ---------: | ---------: | ---------: |
| BiGRU                    |     81.77% |            81.17% |     49.05% | **80.25%** |      82.10% |     60.88% |     88.39% |     69.46% |
| ResNet34-1D              |     87.47% |            79.98% |     63.51% |     68.40% |      91.56% | **65.87%** | **89.28%** |     71.52% |
| **CardioInsight Fusion** | **87.80%** |            79.41% | **65.20%** |     66.42% |  **92.39%** |     65.81% |     88.93% | **71.56%** |

### Additional Test Metrics

For the fusion model:

| Metric        |      Value |
| ------------- | ---------: |
| MCC           | **0.5839** |
| Cohen's Kappa | **0.5838** |
| Brier Score   | **0.0877** |
| Log Loss      | **0.2970** |

Using the validation-derived threshold of **0.41**, the fusion model reaches:

* **81.21% Balanced Accuracy**
* **78.79% Sensitivity**

### Validation Set

On the validation split (`N = 4,626`) at threshold `0.50`:

| Metric            |      Value |
| ----------------- | ---------: |
| Balanced Accuracy | **82.35%** |
| ROC-AUC           | **88.92%** |
| Accuracy          | **85.11%** |
| Recall            | **77.94%** |
| F1-Score          | **66.21%** |

### Interpretation

The three configurations emphasize different aspects of the ECG signal:

* **BiGRU:** highest recall among the tested models
* **ResNet34-1D:** highest ROC-AUC and F1-score
* **Fusion model:** highest accuracy, precision, specificity, and PR-AUC

This comparison shows the effect of combining complementary morphological and temporal representations rather than relying on a single branch.

---

## Project Contribution

The main contribution of CardioInsight is the combination of two complementary ECG representations within a single prediction pipeline.

Instead of using only one representation:

* the **ResNet34-1D branch** captures waveform and morphological patterns,
* the **BiGRU branch** captures temporal and sequential information,
* the **adaptive fusion module** learns sample-specific weights for the two representations before attention-based feature integration.

The project therefore explores an end-to-end way of combining morphological and temporal ECG information for risk estimation.

This should be considered a **project-level architectural contribution**, rather than a claim of a fundamentally new neural-network architecture.

---

## Future Improvements

Several extensions could improve the current system:

* **External validation:** evaluate the trained models on independent ECG datasets from other sources.
* **Interpretability:** add lead-level and time-level attribution methods to identify influential ECG regions.
* **Robustness testing:** evaluate performance under noisy signals, missing leads, baseline drift, and other acquisition artifacts.
* **Calibration:** study probability calibration across datasets with different disease prevalence.
* **Subgroup analysis:** compare performance across demographic and clinical subgroups.
* **Additional baselines:** compare against other temporal, convolutional, and transformer-based ECG models.
* **Efficient inference:** explore model compression, quantization, and optimized inference for resource-constrained devices.

---

## Repository Structure

```text
CardioInsight/
│
├── README.md
├── PROJECT_ARCHITECTURE.md
├── requirements.txt
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── resnet_model.py
│   ├── bigru_model.py
│   ├── fusion_model.py
│   ├── train_fusion.py
│   ├── predict.py
│   └── evaluate.py
│
├── models/
│   ├── ResNet34_ECG.pth
│   ├── BiGRU_ECG.pth
│   ├── FusionModel.pth
│   ├── threshold.npy
│   ├── labels.pth
│   └── README.md
│
├── results/
│   ├── metrics/
│   │   └── fusion_vs_individual_comparison.csv
│   └── README.md
│
└── tests/
    ├── __init__.py
    └── test_pipeline.py
```

---

## Installation

Clone the repository and create a Python virtual environment:

```bash
git clone https://github.com/Shreya-Sharma03/CardioInsight.git
cd CardioInsight

python -m venv .venv
```

### Linux / macOS

```bash
source .venv/bin/activate
```

### Windows

```bash
.venv\Scripts\activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

## Usage

### Demo Inference

Run the built-in synthetic ECG example:

```bash
python src/predict.py --demo
```

### ECG File

For an ECG stored as a NumPy array:

```bash
python src/predict.py --ecg_file path/to/ecg_recording.npy
```

The input should contain a 12-lead ECG recording compatible with the preprocessing pipeline.

### Python API

```python
import numpy as np
from src.predict import CardioInsightPredictor

predictor = CardioInsightPredictor()

ecg = np.random.randn(12, 2500).astype(np.float32)

result = predictor.predict_ecg(ecg, fs=250.0)

print("Risk Category:", result["risk_stratification"])
print("Risk Probability:", result["heart_failure_risk_probability"])
print("Decision Threshold:", result["decision_threshold"])
print("Modality Weights:", result["modality_reliability_weights"])
```

---

## Evaluation

Evaluation utilities are provided in `src/evaluate.py`.

Example:

```python
from src.evaluate import compute_clinical_metrics, print_metrics_table

metrics = compute_clinical_metrics(
    y_true,
    y_probs,
    threshold=0.41
)

print_metrics_table(metrics, title="Test Set Metrics")
```

The evaluation utilities calculate classification and statistical metrics such as accuracy, balanced accuracy, precision, recall, specificity, F1-score, ROC-AUC, PR-AUC, MCC, and Cohen's Kappa.

---

## Tests

Run the test suite with:

```bash
python tests/test_pipeline.py
```

The tests cover the major parts of the pipeline, including:

* ResNet34-1D output dimensions
* BiGRU feature extraction
* Fusion and modality-weight constraints
* Checkpoint loading
* ECG preprocessing
* Inference
* Evaluation utilities

---

## Research / Medical Use

CardioInsight is an academic research prototype for machine-learning experimentation with ECG data. It is not a medical device and should not be used as a standalone system for diagnosis, treatment, or clinical decision-making without appropriate clinical validation and regulatory approval.
