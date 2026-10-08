# CardioInsight: System Architecture & Technical Specification

## Overview & Clinical Motivation

**CardioInsight** is an artificial intelligence framework designed for early detection and risk stratification of **Chronic Heart Failure with Reduced Ejection Fraction (HFrEF)** directly from standard **12-lead Electrocardiogram (ECG)** recordings.

In clinical cardiology, heart failure is diagnosed via echocardiography by assessing the Left Ventricular Ejection Fraction (LVEF). A threshold of $\text{LVEF} \le 45\%$ identifies structural systolic dysfunction requiring targeted pharmacotherapy (e.g., ACE inhibitors, ARBs, beta-blockers, SGLT2 inhibitors). However, echocardiography requires specialized equipment and clinical sonographers, whereas standard 12-lead ECG is cheap, non-invasive, and available at primary care triage.

CardioInsight extracts complementary clinical representations from 12-lead ECG signals:
1. **Morphological Waveform Patterns (ResNet34-1D):** Localized inter-lead voltage and morphology changes (e.g., pathological Q-waves, QRS widening, ST-segment elevation/depression, T-wave inversion).
2. **Sequential Temporal Dynamics (BiGRU):** Long-range rhythm variability, beat-to-beat temporal dependencies, and cardiac cycle periodicity.
3. **Adaptive Reliability Fusion:** Dynamic gating that estimates sample-level confidence for each modality, weights their representations, and models cross-modal interactions via multi-head attention.

---

## High-Level Architecture Flowchart

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

## Detailed Component Specifications

### 1. Preprocessing Pipeline (`src/preprocessing.py`)

- **Input Dimension:** 12 leads $\times$ 2,500 samples (10 seconds sampled at 250 Hz).
- **Butterworth Bandpass Filter:**
  - Cutoff frequencies: $[0.5\text{ Hz}, 40.0\text{ Hz}]$.
  - Filter order: 4th order.
  - Forward-backward filtering (`scipy.signal.filtfilt`) for zero phase distortion.
  - Suppresses baseline wander (respiratory artifact $<0.5\text{ Hz}$) and high-frequency noise (EMG, 50/60 Hz powerline interference $>40.0\text{ Hz}$).
- **Lead-wise Z-Score Normalization:**
  $$\tilde{x}_{c, t} = \frac{x_{c, t} - \mu_c}{\sigma_c + \epsilon}$$
  where $\mu_c$ and $\sigma_c$ denote the mean and standard deviation of channel $c$ computed across the temporal dimension $T$, and $\epsilon = 10^{-8}$.

---

### 2. Morphological Branch: ResNet34-1D (`src/resnet_model.py`)

Standard 2D ResNet architectures are modified for continuous multi-channel 1D biomedical time series:
- **Input Channels:** 12 channels.
- **Initial Convolution:** `Conv1d(12, 64, kernel_size=7, stride=2, padding=3, bias=False)`.
- **Downsampling:** `MaxPool1d(kernel_size=3, stride=2, padding=1)`.
- **Residual Stages:**
  - **Stage 1:** 3 BasicBlocks, 64 channels, stride 1.
  - **Stage 2:** 4 BasicBlocks, 128 channels, stride 2.
  - **Stage 3:** 6 BasicBlocks, 256 channels, stride 2.
  - **Stage 4:** 3 BasicBlocks, 512 channels, stride 2.
- **Global Pooling:** `AdaptiveAvgPool1d(1)` collapsing temporal dimension $T \to 1$.
- **Extracted Feature Vector:** $\mathbf{h}_{\text{res}} \in \mathbb{R}^{512}$.
- **Single-Branch Head:** `Linear(512, 1)` yielding auxiliary morphological logit.

---

### 3. Temporal Branch: Stacked BiGRU (`src/bigru_model.py`)

Captures rhythm patterns and sequential context:
- **Input Pre-conditioning:** `BatchNorm1d(12)` followed by channel transposition to $(B, T, 12)$.
- **Stacked BiGRU Layers:** 3 residual bidirectional GRU blocks with hidden size $H = 128$ per direction ($2H = 256$ total output width).
- **Residual Projection & Normalization:**
  $$\mathbf{z}^{(l)} = \text{LayerNorm}\left(\text{BiGRU}^{(l)}(\mathbf{z}^{(l-1)}) + W_{\text{res}}^{(l)} \mathbf{z}^{(l-1)}\right)$$
- **Bahdanau Additive Temporal Attention:**
  Over per-step recurrent representations $\mathbf{u}_t \in \mathbb{R}^{256}$:
  $$e_t = \mathbf{v}^\top \tanh(W_a \mathbf{u}_t + \mathbf{b}_a)$$
  $$\beta_t = \frac{\exp(e_t)}{\sum_{k=1}^T \exp(e_k)}$$
  $$\mathbf{h}_{\text{gru}} = \sum_{t=1}^T \beta_t \mathbf{u}_t \in \mathbb{R}^{256}$$
- **Single-Branch Head:** Two-layer MLP with residual FC block yielding auxiliary temporal logit.

---

### 4. Adaptive Reliability-Aware Fusion (`src/fusion_model.py`)

Rather than naive concatenation ($\mathbf{h}_{\text{cat}} = [\mathbf{h}_{\text{res}}, \mathbf{h}_{\text{gru}}]$), CardioInsight computes patient-specific reliability weights:

1. **Projection into Joint Space ($D = 512$):**
   $$\tilde{\mathbf{h}}_{\text{res}} = W_{p,\text{res}} \mathbf{h}_{\text{res}} \in \mathbb{R}^{512}, \quad \tilde{\mathbf{h}}_{\text{gru}} = W_{p,\text{gru}} \mathbf{h}_{\text{gru}} \in \mathbb{R}^{512}$$
2. **Confidence Estimation Networks:**
   $$s_{\text{res}} = W_{c,2}^{(\text{res})} \text{GELU}(W_{c,1}^{(\text{res})} \tilde{\mathbf{h}}_{\text{res}}) \in \mathbb{R}^1$$
   $$s_{\text{gru}} = W_{c,2}^{(\text{gru})} \text{GELU}(W_{c,1}^{(\text{gru})} \tilde{\mathbf{h}}_{\text{gru}}) \in \mathbb{R}^1$$
3. **Softmax Gating:**
   $$\alpha_{\text{res}} = \frac{\exp(s_{\text{res}})}{\exp(s_{\text{res}}) + \exp(s_{\text{gru}})}, \quad \alpha_{\text{gru}} = \frac{\exp(s_{\text{gru}})}{\exp(s_{\text{res}}) + \exp(s_{\text{gru}})}$$
   guaranteeing $\alpha_{\text{res}} + \alpha_{\text{gru}} = 1.0$ for all samples.
4. **Weighted Modality Tokens:**
   $$\mathbf{t}_1 = \alpha_{\text{res}} \cdot \tilde{\mathbf{h}}_{\text{res}}, \quad \mathbf{t}_2 = \alpha_{\text{gru}} \cdot \tilde{\mathbf{h}}_{\text{gru}}$$
   $$\mathbf{T} = [\mathbf{t}_1, \mathbf{t}_2] \in \mathbb{R}^{B \times 2 \times 512}$$
5. **Cross-Modal Multi-Head Self-Attention:**
   $$\mathbf{Z} = \text{MultiHeadAttention}(\mathbf{T}, \mathbf{T}, \mathbf{T}, \text{heads}=8, d_{\text{drop}}=0.30)$$
   $$\bar{\mathbf{z}} = \frac{1}{2}(\mathbf{Z}_1 + \mathbf{Z}_2)$$
6. **Residual Refinement & Classification:**
   $$\mathbf{f}_{\text{fused}} = \text{LayerNorm}(\bar{\mathbf{z}} + \text{FFN}(\bar{\mathbf{z}}))$$
   $$\hat{y}_{\text{logit}} = \text{MLP}(\mathbf{f}_{\text{fused}})$$

---

### 5. Training Loss & Optimization

#### A. Class Imbalance
The EchoNext dataset contains 16,962 positive records and 55,513 negative records in the training split ($\approx 3.27 : 1$ negative-to-positive ratio).
Positive weight for BCE is set dynamically:
$$w_{\text{pos}} = \frac{N_{\text{neg}}}{N_{\text{pos}}} = \frac{55513}{16962} \approx 3.2728$$

#### B. Hybrid Objective Function
$$\mathcal{L}_{\text{hybrid}} = 0.70 \cdot \mathcal{L}_{\text{BCE}}(y, \hat{y}; w_{\text{pos}}) + 0.30 \cdot \mathcal{L}_{\text{Focal}}(y, \hat{p}; \alpha=0.25, \gamma=2.0)$$
where:
$$\mathcal{L}_{\text{Focal}} = -\alpha (1 - p_t)^\gamma \log(p_t + \epsilon)$$
$$p_t = y \hat{p} + (1 - y)(1 - \hat{p})$$

#### C. Modality Entropy Regularization
To prevent modality collapse (where the network assigns $\approx 1.0$ to one modality and ignores the other):
$$\mathcal{H}(\alpha) = -\left(\alpha_{\text{res}} \log(\alpha_{\text{res}} + \epsilon) + \alpha_{\text{gru}} \log(\alpha_{\text{gru}} + \epsilon)\right)$$
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{hybrid}} - 0.01 \cdot \mathcal{H}(\alpha)$$

#### D. Optimization Strategy
- **Optimizer:** AdamW ($lr = 3 \times 10^{-4}$, weight decay $= 10^{-4}$).
- **Scheduler:** `CosineAnnealingWarmRestarts` ($T_0 = 5, T_{\text{mult}} = 2, \eta_{\text{min}} = 10^{-6}$).
- **Gradient Clipping:** Max norm = 1.0.
- **Model Selection:** Monitored composite validation score:
  $$\text{Score} = 0.40 \cdot \text{PR-AUC} + 0.30 \cdot \text{ROC-AUC} + 0.20 \cdot \text{F1} + 0.10 \cdot \text{Recall}$$
- **Early Stopping:** Patience = 8 epochs.
