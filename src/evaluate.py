"""
Evaluation & Benchmarking Module for CardioInsight.

Computes comprehensive clinical and statistical classification metrics:
ROC-AUC, PR-AUC, Balanced Accuracy, Precision, Recall/Sensitivity, Specificity,
F1-Score, Cohen's Kappa, Matthews Correlation Coefficient, and Calibration metrics.
"""

from typing import Dict, Any, Union
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    matthews_corrcoef,
    cohen_kappa_score,
    brier_score_loss,
    log_loss,
)


def compute_clinical_metrics(
    y_true: Union[np.ndarray, list],
    y_probs: Union[np.ndarray, list],
    threshold: float = 0.50,
) -> Dict[str, float]:
    """
    Compute comprehensive clinical evaluation metrics for binary heart failure prediction.

    Parameters
    ----------
    y_true : array-like of shape (N,)
        Ground truth binary labels (0 = Low Risk / Normal, 1 = High Risk / LVEF <= 45%).
    y_probs : array-like of shape (N,)
        Predicted risk probabilities in [0, 1].
    threshold : float
        Decision boundary threshold (default 0.50).

    Returns
    -------
    metrics : dict
        Calculated statistical and clinical evaluation metrics.
    """
    y_true = np.asarray(y_true).astype(int)
    y_probs = np.asarray(y_probs).astype(float)
    y_pred = (y_probs >= threshold).astype(int)

    # Confusion matrix elements
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(specificity),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_probs)),
        "pr_auc": float(average_precision_score(y_true, y_probs)),
        "matthews_corrcoef": float(matthews_corrcoef(y_true, y_pred)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "brier_score": float(brier_score_loss(y_true, y_probs)),
        "log_loss": float(log_loss(y_true, y_probs)),
        "threshold": float(threshold),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
    }

    return metrics


def print_metrics_table(metrics: Dict[str, Any], title: str = "Evaluation Metrics") -> None:
    """Format and print metrics in a structured table."""
    print(f"\n{'='*20} {title} {'='*20}")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k:25s}: {v:.4f}")
        else:
            print(f"{k:25s}: {v}")
    print(f"{'='*(42 + len(title))}")


if __name__ == "__main__":
    # Smoke test on dummy ground truth and predictions
    np.random.seed(42)
    dummy_y = np.random.binomial(1, 0.25, size=1000)
    dummy_probs = np.clip(dummy_y * 0.6 + np.random.uniform(0.1, 0.4, size=1000), 0, 1)
    
    m = compute_clinical_metrics(dummy_y, dummy_probs, threshold=0.41)
    print_metrics_table(m, title="Smoke Test Metrics")
