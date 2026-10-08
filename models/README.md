# CardioInsight Model Checkpoints

This directory houses the trained model checkpoints, decision threshold, and class mapping artifacts for the CardioInsight dual-branch heart failure prediction framework.

---

## Checkpoint Inventory

| Checkpoint | File Size | Target Architecture | Input Dimension | Output Representation | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `ResNet34_ECG.pth` | 29.0 MB | `ResNet34_1D` | `(B, 12, T)` | `logit: (B,)`, `feat: (B, 512)` | Strict state-dict match (218 keys) |
| `BiGRU_ECG.pth` | 3.0 MB | `BiGRUECGClassifier` | `(B, 12, T)` | `logit: (B,)`, `feat: (B, 256)` | Strict state-dict match (50 keys) |
| `FusionModel.pth` | 10.1 MB | `ECGFusionClassifier` | `(B, 512), (B, 256)` | `logits: (B,)`, `fused: (B, 512)` | Includes reliability gating and cross-modal attention |
| `threshold.npy` | 136 B | NumPy scalar | — | Decision Threshold = `0.41` | Optimal decision threshold determined on validation set |
| `labels.pth` | 1.3 KB | Python dict | — | `{0: 'Low Risk', 1: 'High Risk'}` | Clinical risk stratification categories |

---

## Architectural Compatibility

All checkpoints load with `strict=True` into their respective module definitions in `src/`:

```python
import torch
from src.resnet_model import ResNet34_1D
from src.bigru_model import BiGRUECGClassifier
from src.fusion_model import ECGFusionClassifier

resnet = ResNet34_1D()
resnet.load_state_dict(torch.load("models/ResNet34_ECG.pth", map_location="cpu"), strict=True)

bigru = BiGRUECGClassifier()
bigru.load_state_dict(torch.load("models/BiGRU_ECG.pth", map_location="cpu"), strict=True)

fusion = ECGFusionClassifier()
fusion.load_state_dict(torch.load("models/FusionModel.pth", map_location="cpu"), strict=True)
```

---

## Dataset Provenance

- **Source Dataset:** EchoNext v1.1.1 (Structural Heart Disease from 12-lead ECG).
- **Target Variable:** `lvef_lte_45_flag` (Left Ventricular Ejection Fraction $\le$ 45%, identifying heart failure with reduced ejection fraction).
- **Leads:** Standard 12-lead ECG (I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6).
- **Sampling Frequency:** 250 Hz (2,500 samples per 10-second recording).
