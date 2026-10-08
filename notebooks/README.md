# CardioInsight Notebooks

This directory contains research and experimentation notebooks documenting the development, feature extraction, fusion training, and validation of CardioInsight.

---

## Notebooks

### [`final-fusion-model.ipynb`](./final-fusion-model.ipynb)
- **Objective:** Implements the complete adaptive reliability fusion pipeline from pre-extracted ResNet34 and BiGRU representations to final clinical risk calibration.
- **Workflow Modules:**
  1. **Feature Loading & Alignment:** Loads 512-d ResNet representations and 256-d BiGRU representations across official train (72,475), validation (4,626), and test (5,442) partitions.
  2. **Fusion Architecture:** Implements `AdaptiveReliabilityFusion` and `ECGFusionClassifier`.
  3. **Loss Function:** `HybridLoss` combining class-weighted Binary Cross-Entropy (handling 3.27:1 class imbalance) and Focal Loss ($\alpha=0.25, \gamma=2.0$) with modality entropy regularization.
  4. **Training Optimization:** Mixed-precision training (`GradScaler`), gradient clipping (`max_norm=1.0`), Cosine Annealing with warm restarts, Exponential Moving Average (EMA), and early stopping.
  5. **Validation & Calibration:** Temperature scaling for probability calibration and threshold optimization using Youden's J statistic.
  6. **Visualization:** Multi-panel ROC and PR curves, normalized confusion matrices, and modality weight kernel density estimation (KDE) distributions.
