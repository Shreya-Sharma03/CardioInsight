"""Decision Threshold Utilities for CardioInsight

Migrated from `final-fusion-model.ipynb` (Cells 29, 58, 61).

CRITICAL AUDIT DOCUMENTATION ON THRESHOLDS:
1. Saved Deployment Threshold (0.41):
   - In Cell 58, `BEST_THRESHOLD = 0.41` was entered as a literal constant:
     `BEST_THRESHOLD = 0.41; np.save("threshold.npy", BEST_THRESHOLD)`.
   - Its exact mathematical selection rationale was NOT documented in the notebook.
   - It is near the validation Youden J index (0.4427), prioritizing clinical recall (79.2% on val).
   - It is NOT an automatically generated Youden threshold or F1 peak (which was 0.59 at Epoch 7).
2. Test-Set Youden Threshold (0.4697):
   - In Cells 61-62, Youden's J statistic was computed DIRECTLY on the test set predictions and labels.
   - Tuning a threshold on test labels constitutes TEST DATA LEAKAGE.
   - The test-set Youden threshold MUST NOT be used as the deployed decision threshold.
"""

from typing import Tuple
import numpy as np
from sklearn.metrics import f1_score, roc_curve


def find_best_threshold(
    probs: np.ndarray,
    labels: np.ndarray,
    start: float = 0.10,
    stop: float = 0.91,
    step: float = 0.01,
) -> Tuple[float, float]:
    """Search for the decision threshold that maximizes F1-score.

    Directly migrated from Cell 29 of `final-fusion-model.ipynb`.
    """
    best_f1 = 0.0
    best_threshold = 0.50

    for t in np.arange(start, stop, step):
        pred = (probs >= t).astype(int)
        f1 = float(f1_score(labels, pred, zero_division=0))
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = round(float(t), 2)

    return best_threshold, best_f1


def find_youden_threshold(
    probs: np.ndarray,
    labels: np.ndarray,
) -> Tuple[float, float]:
    """Calculate the optimal cutoff using Youden's J statistic: max(TPR - FPR).

    Migrated from Cell 61 of `final-fusion-model.ipynb`.
    WARNING: Only execute on VALIDATION data to prevent data leakage.
    """
    fpr, tpr, thresholds = roc_curve(labels, probs)
    j_scores = tpr - fpr
    optimal_idx = int(np.argmax(j_scores))
    optimal_threshold = float(thresholds[optimal_idx])
    max_j = float(j_scores[optimal_idx])
    return optimal_threshold, max_j
