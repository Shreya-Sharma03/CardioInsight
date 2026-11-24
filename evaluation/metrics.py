"""Diagnostic and Evaluation Metrics Module for CardioInsight

Migrated from `final-fusion-model.ipynb` (Cells 42, 52).
Computes all medical diagnostic metrics including Sensitivity, Specificity,
Balanced Accuracy, Precision, F1-Score, ROC-AUC, PR-AUC, MCC, Cohen's Kappa,
Log Loss, Brier Score, and Confusion Matrix.
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
    log_loss,
    brier_score_loss,
)


def compute_classification_metrics(
    labels: np.ndarray,
    probs: np.ndarray,
    threshold: float = 0.41,
) -> Dict[str, Any]:
    """Compute comprehensive performance metrics given probabilities and labels.

    Args:
        labels: 1D array of ground truth binary labels {0, 1}.
        probs: 1D array of predicted probabilities in [0, 1].
        threshold: Decision threshold for binary classification.

    Returns:
        Dictionary containing all diagnostic metrics and confusion matrix elements.
    """
    preds = (probs >= threshold).astype(int)
    cm = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()

    # Medical sensitivity (recall) and specificity
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    metrics = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(sensitivity),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "f1_score": float(f1_score(labels, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probs)),
        "pr_auc": float(average_precision_score(labels, probs)),
        "mcc": float(matthews_corrcoef(labels, preds)),
        "cohens_kappa": float(cohen_kappa_score(labels, preds)),
        "log_loss": float(log_loss(labels, probs)),
        "brier_score": float(brier_score_loss(labels, probs)),
        "confusion_matrix": cm,
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "total_samples": len(labels),
    }

    return metrics


def format_metrics_table(metrics: Dict[str, Any]) -> str:
    """Format metrics dictionary into a clean markdown table."""
    lines = [
        "| Metric | Value |",
        "|---|---|",
        f"| **Decision Threshold** | `{metrics['threshold']:.4f}` |",
        f"| **Accuracy** | `{metrics['accuracy'] * 100:.2f}%` ({metrics['accuracy']:.4f}) |",
        f"| **Balanced Accuracy** | `{metrics['balanced_accuracy'] * 100:.2f}%` ({metrics['balanced_accuracy']:.4f}) |",
        f"| **Sensitivity (Recall)** | `{metrics['sensitivity'] * 100:.2f}%` ({metrics['sensitivity']:.4f}) |",
        f"| **Specificity** | `{metrics['specificity'] * 100:.2f}%` ({metrics['specificity']:.4f}) |",
        f"| **Precision** | `{metrics['precision'] * 100:.2f}%` ({metrics['precision']:.4f}) |",
        f"| **F1-Score** | `{metrics['f1_score'] * 100:.2f}%` ({metrics['f1_score']:.4f}) |",
        f"| **ROC-AUC** | `{metrics['roc_auc'] * 100:.2f}%` ({metrics['roc_auc']:.4f}) |",
        f"| **PR-AUC** | `{metrics['pr_auc'] * 100:.2f}%` ({metrics['pr_auc']:.4f}) |",
        f"| **Matthews Corr Coef (MCC)** | `{metrics['mcc']:.4f}` |",
        f"| **Cohen's Kappa** | `{metrics['cohens_kappa']:.4f}` |",
        f"| **Log Loss** | `{metrics['log_loss']:.4f}` |",
        f"| **Brier Score** | `{metrics['brier_score']:.4f}` |",
    ]
    return "\n".join(lines)
