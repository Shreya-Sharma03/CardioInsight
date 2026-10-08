# CardioInsight: System Architecture & Technical Specification

## Overview

CardioInsight is a deep-learning framework for estimating chronic heart failure risk (specifically identifying left ventricular ejection fraction $\text{LVEF} \le 45\%$) from standard 12-lead electrocardiograms (ECG).

The system uses a dual-branch architecture:
- **ResNet34-1D** extracts localized morphological waveform features (QRS complexes, ST segments, T-wave patterns).
- **BiGRU with temporal attention** extracts rhythm variations and sequential dependencies across the 10-second recording.
- **Adaptive Reliability Fusion** estimates sample-specific confidence for each representation, applies dynamic softmax weighting, and fuses tokens using multi-head self-attention.
- **Classification Head** maps the fused representation to a calibrated risk probability and stratifies patients into risk categories.

---

## Architecture Flowchart

```text
                               12-Lead ECG Signal
                              Shape: (B, 12, 2500)
                                       │
                                       ▼
                       ┌───────────────────────────────┐
                       │        Preprocessing          │
                       │  - Butterworth Bandpass Filter │
                       │    (0.5 - 40.0 Hz, 4th Order)  │
                       │  - Lead-wise Z-Score Norm     │
                       └───────────────┬───────────────┘
                                       │
               ┌───────────────────────┴───────────────────────┐
               ▼                                               ▼
    ┌─────────────────────┐                         ┌─────────────────────┐
    │    ResNet34-1D      │                         │  Stacked BiGRU +    │
    │ Morphological Branch│                         │ Temporal Attention  │
    │  - Conv1d (k=7, s=2)│                         │  - Input BatchNorm1d│
    │  - 4 Residual Stages│                         │  - 3x Residual BiGRU│
    │    [3, 4, 6, 3]     │                         │  - Bahdanau Attn    │
    │  - AdaptiveAvgPool  │                         │                     │
    └──────────┬──────────┘                         └──────────┬──────────┘
               │                                               │
               ▼                                               ▼
        ResNet Features                                 BiGRU Features
         Shape: (B, 512)                                 Shape: (B, 256)
               │                                               │
               └───────────────────────┬───────────────────────┘
                                       ▼
                       ┌───────────────────────────────┐
                       │  Adaptive Reliability Fusion  │
                       │                               │
                       │ 1. Unified Linear Projection  │
                       │    ResNet: 512 -> 512         │
                       │    BiGRU:  256 -> 512         │
                       │                               │
                       │ 2. Reliability Estimators     │
                       │    res_conf: 512 -> 128 -> 1  │
                       │    gru_conf: 512 -> 128 -> 1  │
                       │                               │
                       │ 3. Softmax Gating             │
                       │    [α_res, α_gru] (sum = 1.0) │
                       │                               │
                       │ 4. Weighted Token Stacking    │
                       │    Tokens: (B, 2, 512)        │
                       │                               │
                       │ 5. Multi-Head Self-Attention  │
                       │    8 heads, embed_dim = 512   │
                       │                               │
                       │ 6. Mean Pool + Residual FFN   │
                       │    fused: (B, 512)            │
                       └───────────────┬───────────────┘
                                       │
                                       ▼
                       ┌───────────────────────────────┐
                       │      Classification Head      │
                       │  - Linear(512 -> 512) + LN    │
                       │  - GELU + Dropout(0.30)       │
                       │  - Linear(512 -> 256) + LN    │
                       │  - GELU + Dropout(0.20)       │
                       │  - Linear(256 -> 64) + GELU   │
                       │  - Linear(64 -> 1)            │
                       └───────────────┬───────────────┘
                                       │
                                       ▼
                                  Raw Logit
                                       │
                                       ▼ Sigmoid
                             Calibrated Probability
                                       │
                                       ▼ Threshold (τ = 0.41)
                           Binary Risk Stratification
                          [Low Risk  vs.  High Risk]
```

---

## Component Specifications

### 1. Input
- **Leads:** Standard 12-lead ECG (I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6).
- **Sampling Rate:** 250 Hz.
- **Duration:** 10 seconds per recording (2,500 samples per lead).
- **Input Tensor Shape:** `(B, 12, 2500)`.

---

### 2. Preprocessing (`src/preprocessing.py`)
- **Butterworth Bandpass Filter:**
  - 4th-order digital Butterworth filter.
  - Frequency band: 0.5 Hz to 40.0 Hz.
  - Zero-phase filtering using forward-backward application (`scipy.signal.filtfilt`).
  - Removes respiratory baseline drift ($<0.5\text{ Hz}$) and high-frequency muscular noise or powerline interference ($>40.0\text{ Hz}$).
- **Lead-wise Z-Score Normalization:**
  $$\tilde{x}_{c, t} = \frac{x_{c, t} - \mu_c}{\sigma_c + \epsilon}$$
  where $\mu_c$ and $\sigma_c$ are the per-lead mean and standard deviation across the temporal dimension $T$, with $\epsilon = 10^{-8}$.
- **Length Alignment:**
  Signals shorter than 2,500 samples are zero-padded; longer recordings are truncated to 2,500 samples.

---

### 3. ResNet34-1D (`src/resnet_model.py`)
Standard 2D ResNet convolutions are converted to 1D continuous operations across multi-channel time series:
- **Input Channels:** 12.
- **Stem:** `Conv1d(12, 64, kernel_size=7, stride=2, padding=3, bias=False)` followed by `BatchNorm1d` and `ReLU`.
- **Downsampling:** `MaxPool1d(kernel_size=3, stride=2, padding=1)`.
- **Residual Stages:**
  - Stage 1: 3 BasicBlocks, 64 channels, stride 1.
  - Stage 2: 4 BasicBlocks, 128 channels, stride 2.
  - Stage 3: 6 BasicBlocks, 256 channels, stride 2.
  - Stage 4: 3 BasicBlocks, 512 channels, stride 2.
- **Pooling:** `AdaptiveAvgPool1d(1)` collapsing temporal dimension $T \to 1$.
- **Feature Vector:** $\mathbf{h}_{\text{res}} \in \mathbb{R}^{512}$.
- **Single-Branch Head:** `Linear(512, 1)` producing an auxiliary morphological classification logit.

---

### 4. BiGRU (`src/bigru_model.py`)
Models sequential relationships, rhythm variations, and beat-to-beat dynamics:
- **Input Pre-conditioning:** `BatchNorm1d(12)` followed by channel transposition to `(B, T, 12)`.
- **Stacked BiGRU Layers:** 3 residual bidirectional GRU blocks with hidden dimension $H = 128$ per direction (256-dimensional total width).
- **Residual Connections & Normalization:**
  $$\mathbf{z}^{(l)} = \text{LayerNorm}\left(\text{BiGRU}^{(l)}(\mathbf{z}^{(l-1)}) + W_{\text{res}}^{(l)} \mathbf{z}^{(l-1)}\right)$$
- **Bahdanau Temporal Attention:**
  Over per-step recurrent hidden states $\mathbf{u}_t \in \mathbb{R}^{256}$:
  $$e_t = \mathbf{v}^\top \tanh(W_a \mathbf{u}_t + \mathbf{b}_a)$$
  $$\beta_t = \frac{\exp(e_t)}{\sum_{k=1}^T \exp(e_k)}$$
  $$\mathbf{h}_{\text{gru}} = \sum_{t=1}^T \beta_t \mathbf{u}_t \in \mathbb{R}^{256}$$
- **Auxiliary Head:** Two-layer MLP with residual linear block yielding an auxiliary temporal classification logit.

---

### 5. Fusion (`src/fusion_model.py`)
Combines morphological ($\mathbf{h}_{\text{res}} \in \mathbb{R}^{512}$) and temporal ($\mathbf{h}_{\text{gru}} \in \mathbb{R}^{256}$) representations with sample-specific weighting:
1. **Projection into Joint Space ($D = 512$):**
   $$\tilde{\mathbf{h}}_{\text{res}} = W_{p,\text{res}} \mathbf{h}_{\text{res}} \in \mathbb{R}^{512}, \quad \tilde{\mathbf{h}}_{\text{gru}} = W_{p,\text{gru}} \mathbf{h}_{\text{gru}} \in \mathbb{R}^{512}$$
2. **Reliability Scoring Networks:**
   $$s_{\text{res}} = W_{c,2}^{(\text{res})} \text{GELU}(W_{c,1}^{(\text{res})} \tilde{\mathbf{h}}_{\text{res}}) \in \mathbb{R}^1$$
   $$s_{\text{gru}} = W_{c,2}^{(\text{gru})} \text{GELU}(W_{c,1}^{(\text{gru})} \tilde{\mathbf{h}}_{\text{gru}}) \in \mathbb{R}^1$$
3. **Softmax Gating:**
   $$\alpha_{\text{res}} = \frac{\exp(s_{\text{res}})}{\exp(s_{\text{res}}) + \exp(s_{\text{gru}})}, \quad \alpha_{\text{gru}} = \frac{\exp(s_{\text{gru}})}{\exp(s_{\text{res}}) + \exp(s_{\text{gru}})}$$
   where $\alpha_{\text{res}} + \alpha_{\text{gru}} = 1.0$.
4. **Weighted Modality Tokens:**
   $$\mathbf{t}_1 = \alpha_{\text{res}} \cdot \tilde{\mathbf{h}}_{\text{res}}, \quad \mathbf{t}_2 = \alpha_{\text{gru}} \cdot \tilde{\mathbf{h}}_{\text{gru}}$$
   $$\mathbf{T} = [\mathbf{t}_1, \mathbf{t}_2] \in \mathbb{R}^{B \times 2 \times 512}$$
5. **Cross-Modal Multi-Head Self-Attention:**
   $$\mathbf{Z} = \text{MultiHeadAttention}(\mathbf{T}, \mathbf{T}, \mathbf{T}, \text{heads}=8, d_{\text{drop}}=0.30)$$
   $$\bar{\mathbf{z}} = \frac{1}{2}(\mathbf{Z}_1 + \mathbf{Z}_2)$$
6. **Residual Refinement:**
   $$\mathbf{f}_{\text{fused}} = \text{LayerNorm}(\bar{\mathbf{z}} + \text{FFN}(\bar{\mathbf{z}})) \in \mathbb{R}^{512}$$

---

### 6. Classification Head (`src/fusion_model.py`)
Maps the fused representation to the final logit:
- MLP architecture:
  - `Linear(512, 512)` + `LayerNorm` + `GELU` + `Dropout(0.30)`
  - `Linear(512, 256)` + `LayerNorm` + `GELU` + `Dropout(0.20)`
  - `Linear(256, 64)` + `GELU`
  - `Linear(64, 1)`
- **Output:** Raw logit $\hat{y}$, converted to probability via sigmoid:
  $$p = \frac{1}{1 + \exp(-\hat{y})}$$
- **Decision Boundary:** Calibrated threshold $\tau = 0.41$:
  - $p \ge 0.41 \implies$ High Risk (LVEF $\le$ 45%)
  - $p < 0.41 \implies$ Low Risk

---

### 7. Training Configuration (`src/train_fusion.py`)
- **Class Imbalance:**
  The training split contains 16,962 positive cases and 55,513 negative cases ($\approx 3.27 : 1$ ratio). Positive weight in BCE is set to $w_{\text{pos}} = 55,513 / 16,962 \approx 3.2728$.
- **Hybrid Objective Function:**
  $$\mathcal{L}_{\text{hybrid}} = 0.70 \cdot \mathcal{L}_{\text{BCE}}(y, \hat{y}; w_{\text{pos}}) + 0.30 \cdot \mathcal{L}_{\text{Focal}}(y, p; \alpha=0.25, \gamma=2.0)$$
  where:
  $$\mathcal{L}_{\text{Focal}} = -\alpha (1 - p_t)^\gamma \log(p_t + \epsilon), \quad p_t = y p + (1 - y)(1 - p)$$
- **Modality Entropy Regularization:**
  Prevents modality collapse (one branch dominating with weight $\approx 1.0$):
  $$\mathcal{H}(\alpha) = -(\alpha_{\text{res}} \log(\alpha_{\text{res}} + \epsilon) + \alpha_{\text{gru}} \log(\alpha_{\text{gru}} + \epsilon))$$
  $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{hybrid}} - 0.01 \cdot \mathcal{H}(\alpha)$$
- **Optimizer:** AdamW with learning rate $3 \times 10^{-4}$ and weight decay $10^{-4}$.
- **Scheduler:** `CosineAnnealingWarmRestarts` with $T_0 = 5, T_{\text{mult}} = 2, \eta_{\text{min}} = 10^{-6}$.
- **Gradient Clipping:** Maximum norm 1.0.
- **Model Selection Metric:**
  $$\text{Score} = 0.40 \cdot \text{PR-AUC} + 0.30 \cdot \text{ROC-AUC} + 0.20 \cdot \text{F1} + 0.10 \cdot \text{Recall}$$

---

### 8. Evaluation (`src/evaluate.py`)
- **Metrics Evaluated:**
  - Area Under the Receiver Operating Characteristic (ROC-AUC)
  - Precision-Recall AUC (PR-AUC)
  - Accuracy and Balanced Accuracy
  - Precision, Recall (Sensitivity), and Specificity
  - F1-Score
  - Matthews Correlation Coefficient (MCC)
  - Cohen's Kappa
  - Brier Score and Log Loss
- **Threshold Tuning:**
  Decision thresholds are evaluated across $[0.10, 0.90]$ on the validation split. The optimal threshold $\tau = 0.41$ balances precision and recall on unseen validation data without test-set exposure.
