"""Evaluation Pipeline and Checkpoint Loader for CardioInsight

Migrated from `final-fusion-model.ipynb` (Cells 41, 42, 47-52, 61, 62).
Provides safe loading of fusion checkpoints and full inference evaluation.
"""

from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Union
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.fusion_model import ECGFusionClassifier
from .metrics import compute_classification_metrics
from config.config import PathConfig, DEFAULT_PATH_CONFIG


def load_fusion_checkpoint(
    checkpoint_path: Union[str, Path],
    model: Optional[ECGFusionClassifier] = None,
    device: Optional[torch.device] = None,
    strict: bool = True,
) -> ECGFusionClassifier:
    """Safely load a fusion checkpoint into ECGFusionClassifier.

    Handles both:
    1. Training checkpoints with metadata: {'epoch': ..., 'model': state_dict, 'score': ...}
    2. Pure state dictionaries: OrderedDict of tensors.
    """
    if device is None:
        device = torch.device("cpu")

    if model is None:
        model = ECGFusionClassifier()

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt

    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint mismatch! Missing: {missing}, Unexpected: {unexpected}"
        )

    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def evaluate_model(
    model: ECGFusionClassifier,
    res_features: np.ndarray,
    gru_features: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.41,
    batch_size: int = 1024,
    device: Optional[torch.device] = None,
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    """Perform forward inference over feature arrays and compute diagnostic metrics.

    Args:
        model: ECGFusionClassifier instance
        res_features: (N, 512) array of ResNet features
        gru_features: (N, 256) array of BiGRU features
        labels: (N,) ground-truth binary labels
        threshold: Decision cutoff for classification (default 0.41)
        batch_size: Batch size for batched inference
        device: Device to run evaluation on

    Returns:
        metrics: Dictionary of classification metrics
        probs: (N,) array of predicted probabilities
        alpha_res: (N,) array of ResNet reliability weights
        alpha_gru: (N,) array of BiGRU reliability weights
    """
    if device is None:
        device = torch.device("cpu")

    model.to(device)
    model.eval()

    all_probs = []
    all_alpha_res = []
    all_alpha_gru = []

    n_samples = len(labels)
    for i in range(0, n_samples, batch_size):
        end = min(i + batch_size, n_samples)
        res_batch = torch.from_numpy(res_features[i:end]).float().to(device)
        gru_batch = torch.from_numpy(gru_features[i:end]).float().to(device)

        logits, _, a_res, a_gru = model(res_batch, gru_batch)
        probs = torch.sigmoid(logits)

        all_probs.extend(probs.cpu().numpy().flatten())
        all_alpha_res.extend(a_res.cpu().numpy().flatten())
        all_alpha_gru.extend(a_gru.cpu().numpy().flatten())

    probs_arr = np.array(all_probs)
    alpha_res_arr = np.array(all_alpha_res)
    alpha_gru_arr = np.array(all_alpha_gru)

    metrics = compute_classification_metrics(labels, probs_arr, threshold=threshold)
    metrics["mean_alpha_res"] = float(np.mean(alpha_res_arr))
    metrics["mean_alpha_gru"] = float(np.mean(alpha_gru_arr))
    metrics["min_prob"] = float(np.min(probs_arr))
    metrics["max_prob"] = float(np.max(probs_arr))

    return metrics, probs_arr, alpha_res_arr, alpha_gru_arr


if __name__ == "__main__":
    from data.dataset import load_fusion_data

    paths = DEFAULT_PATH_CONFIG
    best_ckpt_path = paths.base_dir / paths.best_fusion_checkpoint

    print("Testing modular evaluator on test split...")
    model = load_fusion_checkpoint(best_ckpt_path)
    res_te, gru_te, y_te = load_fusion_data("test", paths)

    metrics, probs, a_res, a_gru = evaluate_model(model, res_te, gru_te, y_te, threshold=0.41)
    print(f"Test Accuracy        : {metrics['accuracy']:.4f}")
    print(f"Test Balanced Acc    : {metrics['balanced_accuracy']:.4f}")
    print(f"Test ROC-AUC         : {metrics['roc_auc']:.4f}")
    print(f"Mean Alpha Res       : {metrics['mean_alpha_res']:.4f}")
    print(f"Mean Alpha BiGRU     : {metrics['mean_alpha_gru']:.4f}")
