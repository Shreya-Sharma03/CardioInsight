# CardioInsight

CardioInsight is a deep-learning project for estimating the risk of reduced left ventricular ejection fraction (LVEF) from standard 12-lead electrocardiogram (ECG) recordings. The system combines convolutional and recurrent representations of ECG signals and fuses them before binary classification.

---

## Overview

Reduced left ventricular ejection fraction is an indicator of systolic cardiac dysfunction. CardioInsight explores the use of standard 12-lead ECG recordings for estimating whether a recording belongs to the target class defined by `lvef_lte_45_flag`.

The system processes a 10-second ECG recording and extracts complementary information using two branches:

* **ResNet34-1D** for waveform and morphological features
* **Stacked BiGRU with temporal attention** for sequential features

The resulting representations are combined using adaptive modality weighting and multi-head attention before producing a binary prediction.

### Input

* Standard 12-lead ECG
* Sampling frequency: **250 Hz**
* Recording length: **10 seconds**
* Input shape: **`(12, 2500)`**

### Output

The classifier produces a probability for the positive class and converts it into a binary prediction using a decision threshold.

```text
Probability < 0.41  →  Low Risk  (class 0)
Probability ≥ 0.41  →  High Risk (class 1)
```

The two labels correspond to the binary target used by the project:

```text
0 → Low Risk / Normal
1 → High Risk / LVEF ≤ 45%
```

The probability is continuous, but the final categorical prediction is **binary**. There is no separate Moderate Risk class in the current implementation.

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
                │    0.5–40 Hz    │
                │ Bandpass Filter │
                │ + Z-Score Norm  │
                └────────┬────────┘
                         │
                ┌────────┴────────┐
                │                 │
                ▼                 ▼
        ┌───────────────┐  ┌────────────────┐
        │ ResNet34-1D   │  │     BiGRU      │
        │               │  │                │
        │ Morphological │  │    Temporal    │
        │   Features    │  │    Features    │
        │    512-d      │  │     256-d      │
        └───────┬───────┘  └───────┬────────┘
                │                  │
                └────────┬─────────┘
                         ▼
               Adaptive Reliability Fusion
                         │
                  8-Head Attention
                         │
                         ▼
                 Classification MLP
                         │
                         ▼
                 Sigmoid Probability
                         │
                         ▼
               Decision Threshold 0.41
                         │
                  ┌──────┴──────┐
                  ▼             ▼
               Low Risk      High Risk
               Class 0        Class 1
```

---

## Model Architecture

### ResNet34-1D

Implemented in `src/resnet_model.py`.

The first branch adapts the ResNet-34 architecture to one-dimensional multi-channel ECG signals.

The network contains:

* Input channels: `12`
* Initial `Conv1d`
* Max pooling
* Four residual stages
* BasicBlock configuration: `[3, 4, 6, 3]`
* Channel progression: `64 → 128 → 256 → 512`
* Adaptive average pooling
* 512-dimensional feature representation

This branch captures local waveform and morphological characteristics in the ECG signal.

---

### BiGRU

Implemented in `src/bigru_model.py`.

The second branch models sequential information across the ECG recording.

The network contains:

* Channel-wise batch normalization
* Three stacked bidirectional GRU layers
* Hidden size: `128` per direction
* Output width: `256`
* Residual connections between recurrent layers
* Layer normalization
* Bahdanau additive temporal attention
* 256-dimensional temporal representation

The attention mechanism assigns different weights to time steps before generating the final temporal feature vector.

---

### Adaptive Reliability Fusion

Implemented in `src/fusion_model.py`.

The ResNet and BiGRU representations have different dimensions, so they are first projected into a common 512-dimensional embedding space.

The fusion process is:

1. Project ResNet features from `512 → 512`
2. Project BiGRU features from `256 → 512`
3. Estimate a reliability score for each branch
4. Convert the two scores into normalized weights using softmax
5. Weight the two modality representations
6. Treat them as two feature tokens
7. Apply 8-head multi-head self-attention
8. Mean-pool the attention output
9. Apply residual feed-forward refinement and LayerNorm

The modality weights satisfy:

$$
\alpha_{\text{ResNet}} + \alpha_{\text{BiGRU}} = 1
$$

This allows the relative contribution of the two branches to vary between samples.

---

### Classification Head

The fused 512-dimensional representation is passed through a multilayer perceptron:

```text
512 → 512 → 256 → 64 → 1
```

The hidden layers use LayerNorm, GELU activations, and dropout.

The final layer produces a single logit:

$$
z \in \mathbb{R}
$$

The sigmoid function converts the logit into a probability:

$$
p = \frac{1}{1 + e^{-z}}
$$

The final prediction is then determined using the decision threshold:

$$
p \geq 0.41 \Rightarrow \text{High Risk}
$$

$$
p < 0.41 \Rightarrow \text{Low Risk}
$$

The threshold `0.41` is stored in `models/threshold.npy` and was selected using the validation data.

---

## Preprocessing

Implemented in `src/preprocessing.py`.

The preprocessing pipeline performs input formatting, filtering, and normalization.

### Input Formatting

The expected ECG shape is:

```text
(12, 2500)
```

Signals supplied as `(2500, 12)` are transposed automatically.

Recordings shorter than 2,500 samples are zero-padded, while longer recordings are truncated.

### Bandpass Filtering

A 4th-order Butterworth bandpass filter is applied between **0.5 Hz and 40 Hz** using zero-phase forward-backward filtering.

This reduces low-frequency baseline variation and higher-frequency noise.

### Lead-wise Z-Score Normalization

Each ECG lead is normalized independently along the time dimension:

$$
\tilde{x}_{c,t}
=
\frac{x_{c,t}-\mu_c}
{\sigma_c+\epsilon}
$$

where:

- \(x_{c,t}\) is the signal value at time \(t\) for lead \(c\)
- \(\mu_c\) is the mean of lead \(c\)
- \(\sigma_c\) is the standard deviation of lead \(c\)
- \(\epsilon = 10^{-8}\) prevents division by zero

---

## Data

The project uses the **EchoNext v1.1.1** dataset.

### Target

The target variable is:

```text
lvef_lte_45_flag
```

The project treats this as a binary classification problem corresponding to whether LVEF is at or below 45%.

### ECG Format

Each recording contains:

* 12 standard ECG leads
* Sampling frequency: **250 Hz**
* Duration: **10 seconds**
* 2,500 samples per lead

The standard leads are:

```text
I, II, III, aVR, aVL, aVF,
V1, V2, V3, V4, V5, V6
```

### Dataset Splits

Patient-level, deduplicated partitions are used.

| Split      | Patients | Positive | Negative |
| ---------- | -------: | -------: | -------: |
| Training   |   72,475 |   16,962 |   55,513 |
| Validation |    4,626 |        — |        — |
| Test       |    5,442 |        — |        — |

The training split contains approximately **3.27 negative samples for every positive sample**.

---

## Training

The fusion model uses class-imbalance handling and regularization during training.

### Loss Function

The training objective combines:

* Class-weighted Binary Cross-Entropy
* Focal Loss
* Modality entropy regularization

The positive-class weight is calculated from the training-set class distribution:

$$
w_{\text{pos}}
=
\frac{N_{\text{negative}}}
{N_{\text{positive}}}
=
\frac{55,513}{16,962}
\approx 3.2728
$$

### Optimization

* **Optimizer:** AdamW
* **Learning rate:** `3 × 10⁻⁴`
* **Weight decay:** `1 × 10⁻⁴`
* **Scheduler:** Cosine Annealing Warm Restarts
* **Gradient clipping:** `1.0`
* **Early stopping:** validation-based

The training workflow is available through `src/train_fusion.py`.

---

## Results

### Test Set

The test split contains **5,442 patients**.

The following table compares the individual branches with the combined fusion model.

| Model                    |   Accuracy | Balanced Accuracy |  Precision |     Recall | Specificity |         F1 |    ROC-AUC |     PR-AUC |
| ------------------------ | ---------: | ----------------: | ---------: | ---------: | ----------: | ---------: | ---------: | ---------: |
| BiGRU                    |     81.77% |        **81.17%** |     49.05% | **80.25%** |      82.10% |     60.88% |     88.39% |     69.46% |
| ResNet34-1D              |     87.47% |            79.98% |     63.51% |     68.40% |      91.56% | **65.87%** | **89.28%** |     71.52% |
| **CardioInsight Fusion** | **87.80%** |            79.41% | **65.20%** |     66.42% |  **92.39%** |     65.81% |     88.93% | **71.56%** |

### Additional Fusion Metrics

| Metric                           |      Value |
| -------------------------------- | ---------: |
| Matthews Correlation Coefficient | **0.5839** |
| Cohen's Kappa                    | **0.5838** |
| Brier Score                      | **0.0877** |
| Log Loss                         | **0.2970** |

Using the validation-derived threshold of `0.41`, the fusion model achieves:

* **81.21% Balanced Accuracy**
* **78.79% Sensitivity**

### Validation Set

The validation split contains **4,626 patients**.

At threshold `0.50`, the recorded metrics are:

| Metric            |      Value |
| ----------------- | ---------: |
| Balanced Accuracy | **82.35%** |
| ROC-AUC           | **88.92%** |
| Accuracy          | **85.11%** |
| Recall            | **77.94%** |
| F1-Score          | **66.21%** |

### Comparison

The individual branches show different strengths:

* **BiGRU** provides the highest recall.
* **ResNet34-1D** provides the highest ROC-AUC and F1-score.
* **Fusion** provides the highest accuracy, precision, specificity, and PR-AUC.

The comparison illustrates the effect of combining complementary morphological and temporal ECG representations.

---

## Project Contribution

The main contribution of CardioInsight is the integration of complementary morphological and temporal ECG representations within one prediction pipeline.

The approach combines:

1. **ResNet34-1D** for waveform and morphological features.
2. **BiGRU with temporal attention** for sequential ECG information.
3. **Learned sample-specific modality weighting** to control the relative contribution of the two branches.
4. **Multi-head attention-based fusion** to integrate the resulting representations.

The contribution is primarily architectural and experimental: it explores whether adaptive fusion of complementary ECG representations can improve binary risk estimation compared with individual branches.

---

## Future Improvements

Possible extensions of the project include:

* **External validation:** evaluate the models on independent ECG datasets.
* **Interpretability:** add lead-level and time-level attribution methods to identify influential ECG regions.
* **Robustness:** evaluate the system under noisy signals, missing leads, baseline drift, and acquisition artifacts.
* **Calibration:** assess probability calibration under different class prevalences and datasets.
* **Subgroup analysis:** evaluate performance across demographic and clinical subgroups.
* **Additional baselines:** compare with transformer-based and other ECG-specific architectures.
* **Efficient inference:** explore quantization, pruning, and knowledge distillation for resource-constrained environments.

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

Install the dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

### Demo

Run inference using a synthetic 12-lead ECG:

```bash
python src/predict.py --demo
```

### ECG File

For an ECG stored as a NumPy `.npy` file:

```bash
python src/predict.py --ecg_file path/to/ecg_recording.npy
```

The array should contain a 12-lead ECG compatible with the preprocessing pipeline.

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

Example output contains:

```text
Risk Category: Low Risk
Risk Probability: 0.36
Decision Threshold: 0.41
```

The exact probability depends on the input signal.

---

## Training

The fusion model training workflow is provided in:

```text
src/train_fusion.py
```

A typical invocation is:

```bash
python src/train_fusion.py \
    --data_dir path/to/features \
    --output_dir models \
    --epochs 15 \
    --batch_size 512
```

Available options include:

```text
--data_dir
--output_dir
--epochs
--batch_size
--lr
--device
```

The training script operates on the feature representations required by the fusion model and saves the resulting checkpoint and decision threshold.

---

## Evaluation

Evaluation utilities are provided in:

```text
src/evaluate.py
```

The module calculates:

* Accuracy
* Balanced Accuracy
* Precision
* Recall / Sensitivity
* Specificity
* F1-Score
* ROC-AUC
* PR-AUC
* Matthews Correlation Coefficient
* Cohen's Kappa
* Brier Score
* Log Loss

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

---

## Tests

Run the test suite using:

```bash
python tests/test_pipeline.py
```

The tests cover the main components of the pipeline, including:

* ResNet34-1D feature extraction
* BiGRU feature extraction
* Fusion output and modality weights
* Checkpoint loading
* ECG preprocessing
* End-to-end inference
* Evaluation utilities

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

## Research / Medical Use

CardioInsight is an academic research prototype for experimentation with ECG-based machine learning. It is not a medical device and should not be used as a standalone system for diagnosis, treatment, or clinical decision-making without appropriate clinical validation and regulatory approval.

