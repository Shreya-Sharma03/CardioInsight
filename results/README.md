# CardioInsight Experimental Results & Benchmarks

This directory contains quantitative evaluation results and comparative benchmarks comparing the individual ResNet34-1D and BiGRU branches against the Reliability-Aware Multimodal Fusion framework.

---

## Benchmark Comparison Table (Test Set: 5,442 Records)

The table below reports test set performance on 5,442 patient records from `results/metrics/fusion_vs_individual_comparison.csv`:

| Model Architecture | Accuracy | Balanced Accuracy | Precision | Recall (Sensitivity) | Specificity | F1-Score | ROC-AUC | PR-AUC | MCC | Cohen's Kappa |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **BiGRU alone** | 81.77% | 81.17% | 49.05% | **80.25%** | 82.10% | 60.88% | 88.39% | 69.46% | 0.5246 | 0.4989 |
| **ResNet alone** | 87.47% | 79.98% | 63.51% | 68.40% | 91.56% | 65.87% | **89.28%** | 71.52% | 0.5826 | 0.5820 |
| **CardioInsight Fusion** | **87.80%** | **79.41%** | **65.20%** | 66.42% | **92.39%** | **65.81%** | 88.93% | **71.56%** | **0.5839** | **0.5838** |

*Note on Validation vs. Test Metrics:*
- On the **Validation Split** (4,626 records, default threshold = 0.50), the fusion model achieved:
  - **Balanced Accuracy:** `82.35%`
  - **ROC-AUC:** `88.92%`
  - **Accuracy:** `85.11%`
  - **Recall:** `77.94%`
  - **F1-Score:** `66.21%`
- On the **Test Split** (5,442 records, threshold = 0.41), the fusion model achieved:
  - **Balanced Accuracy:** `81.21%`
  - **ROC-AUC:** `88.93%`
  - **Accuracy:** `87.80%`
  - **Specificity:** `92.39%`
