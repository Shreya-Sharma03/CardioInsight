"""CardioInsight Research Workflow Archive (Converted from exploratory notebook).

This module preserves the full research progression, experimental rationale,
exploratory calibration/t-SNE analyses, and historical provenance notes originally
authored in the exploratory Kaggle research notebook.

Rather than duplicating production code, this script reuses the modular project
packages (`config`, `data`, `models`, `training`, and `evaluation`).

Original Notebook Installation & Discovery Commands (Converted to CLI Notes):
    # Dependencies (use repository ML requirements from project root):
    #   pip install -r requirements-ml.txt
    # Original Kaggle installation cells installed:
    #   torch, matplotlib, seaborn, scikit-learn, scipy, pandas, tqdm
    # Note: torchvision was listed in the notebook's setup cell but is never imported.

Required Local Data & Checkpoints (from repository root):
    - Training features : resnet_features_train.npy, bigru_features_train.npy, resnet_labels_train (1).npy
    - Validation features: resnet_features_val.npy, bigru_features_val.npy, resnet_labels_val (1).npy
    - Holdout test features: resnet_features_test.npy, bigru_features_test.npy, resnet_labels_test (1).npy
    - Peak Epoch 7 fusion checkpoint: best_fusion_model.zip

Historical Provenance Notes Preserved from Exploratory Run:
    1. Training (`train.py` / `training/trainer.py`) uses precomputed 512-d ResNet34-1D
       and 256-d BiGRU feature vectors (not raw waveforms) with HybridLoss:
       0.70 * BCEWithLogitsLoss(pos_weight=3.2728) + 0.30 * FocalLoss(alpha=0.25, gamma=2.0).
    2. Early stopping halted training at Epoch 15 (`FusionModel.zip`). Cell 52 evaluated
       that Epoch 15 end-of-training state on validation (N=4,626) at threshold 0.50,
       observing ROC-AUC 88.92% and Balanced Accuracy 82.35%.
    3. The peak validation composite-score checkpoint was saved earlier at Epoch 7
       (`best_fusion_model.zip`, score 0.773426), achieving 89.12% validation ROC-AUC
       and 88.44% holdout test ROC-AUC (N=5,442) at deployed threshold 0.41.
    4. Cell 58 saved `BEST_THRESHOLD = 0.41` to `threshold.npy` as a literal constant.
    5. Cell 61 computed an exploratory Youden threshold (~0.4697) directly on the test
       predictions; final evaluation (`evaluation/visualize_results.py`) discards that
       test-derived threshold and uses the fixed 0.41 cutoff instead.

Usage:
    # Run standalone training via modular runner:
    python train.py --epochs 40 --batch-size 512 --device cpu

    # Generate holdout test evaluation & plots via modular runner:
    python evaluation/visualize_results.py --checkpoint best_fusion_model.zip --threshold 0.41

    # Run exploratory validation calibration / reliability / t-SNE analysis:
    python notebooks/final_fusion_model.py --split val --checkpoint best_fusion_model.zip --save-dir audit_results
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple

# Ensure repository root is importable when invoked as `python notebooks/final_fusion_model.py`
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import minimize
from sklearn.calibration import calibration_curve
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

from config.config import DEFAULT_EVAL_CONFIG, DEFAULT_PATH_CONFIG, DEFAULT_TRAIN_CONFIG
from data.dataset import FusionDataset, load_fusion_data
from evaluation.evaluator import evaluate_model, load_fusion_checkpoint
from models.fusion_model import ECGFusionClassifier
from training.trainer import FusionTrainer


# ---------------------------------------------------------------------------
# Exploratory Helpers Preserved from Original Research Workflow
# ---------------------------------------------------------------------------


def compute_class_pos_weight(labels: np.ndarray) -> float:
    """Compute BCEWithLogitsLoss positive class weight (negative / positive).

    Mirrors Module 14 (Cell 22): on the 72,475-sample training split
    (55,513 negative, 16,962 positive), negative / positive = 3.272786.
    """
    positive = int(np.sum(labels == 1))
    negative = int(np.sum(labels == 0))
    if positive == 0:
        raise ValueError("Cannot compute pos_weight with zero positive samples.")
    return float(negative / positive)


class TTAPredictor:
    """Exploratory Gaussian feature-perturbation Test-Time Augmentation wrapper (Cell 38)."""

    def __init__(self, model: ECGFusionClassifier, n_passes: int = 5, noise_std: float = 0.01):
        self.model = model
        self.n_passes = n_passes
        self.noise_std = noise_std

    @torch.no_grad()
    def predict(self, res_feat: torch.Tensor, gru_feat: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        preds = []
        for _ in range(self.n_passes):
            noise_r = torch.randn_like(res_feat) * self.noise_std
            noise_g = torch.randn_like(gru_feat) * self.noise_std
            logits, _, _, _ = self.model(res_feat + noise_r, gru_feat + noise_g)
            preds.append(torch.sigmoid(logits))
        return torch.stack(preds, dim=0).mean(dim=0)


class TemperatureScaling(nn.Module):
    """Post-hoc temperature scaling module from Cell 39 / Cell 43."""

    def __init__(self) -> None:
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature


@torch.no_grad()
def collect_fused_embeddings(
    model: ECGFusionClassifier,
    res_features: np.ndarray,
    gru_features: np.ndarray,
    batch_size: int = 1024,
    device: torch.device = torch.device("cpu"),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Collect raw logits, fused 512-d embeddings, and reliability weights (Cells 41-42)."""
    model.to(device)
    model.eval()

    all_logits = []
    all_fused = []
    all_alpha_res = []
    all_alpha_gru = []

    for start in range(0, len(res_features), batch_size):
        end = min(start + batch_size, len(res_features))
        res_b = torch.from_numpy(res_features[start:end]).float().to(device)
        gru_b = torch.from_numpy(gru_features[start:end]).float().to(device)
        logits_b, fused_b, a_res_b, a_gru_b = model(res_b, gru_b)

        all_logits.append(logits_b.cpu().numpy())
        all_fused.append(fused_b.cpu().numpy())
        all_alpha_res.append(a_res_b.cpu().numpy().flatten())
        all_alpha_gru.append(a_gru_b.cpu().numpy().flatten())

    return (
        np.concatenate(all_logits),
        np.concatenate(all_fused, axis=0),
        np.concatenate(all_alpha_res),
        np.concatenate(all_alpha_gru),
    )


def fit_temperature_scaling(logits: np.ndarray, labels: np.ndarray) -> float:
    """Optimize scalar temperature parameter on validation logits via NLL (Cell 43)."""

    def nll_loss(params: np.ndarray) -> float:
        temp = max(float(params[0]), 1e-4)
        probs = 1.0 / (1.0 + np.exp(-logits / temp))
        eps = 1e-8
        return float(-np.mean(labels * np.log(probs + eps) + (1.0 - labels) * np.log(1.0 - probs + eps)))

    result = minimize(nll_loss, x0=np.array([1.0]), method="Powell")
    return float(result.x[0])


def plot_calibration_curve(labels: np.ndarray, probs: np.ndarray, out_path: Path) -> None:
    """Plot 10-bin reliability calibration curve (Cell 51)."""
    true_prob, pred_prob = calibration_curve(labels, probs, n_bins=10)
    plt.figure(figsize=(6, 6))
    plt.plot(pred_prob, true_prob, marker="o", label="Fusion Model")
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Perfect Calibration")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Observed Frequency")
    plt.title("Validation Calibration Curve")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_reliability_scatter(alpha_res: np.ndarray, alpha_gru: np.ndarray, out_path: Path) -> None:
    """Plot per-sample ResNet vs BiGRU reliability weights along alpha_res + alpha_gru = 1.0 (Cell 50)."""
    plt.figure(figsize=(6.5, 6.5))
    plt.scatter(alpha_res, alpha_gru, alpha=0.35, s=8)
    plt.xlabel("ResNet34-1D Reliability (alpha_res)")
    plt.ylabel("BiGRU Reliability (alpha_gru)")
    plt.title("Per-Sample Adaptive Reliability Weight Distribution")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_latent_tsne(
    fused_embeddings: np.ndarray,
    labels: np.ndarray,
    out_path: Path,
    sample_size: int = 2000,
    seed: int = 42,
) -> None:
    """Run PCA (50 components) + t-SNE (2D) projection on fused 512-d embeddings (Cell 45)."""
    rng = np.random.default_rng(seed)
    if len(fused_embeddings) > sample_size:
        idx = rng.choice(len(fused_embeddings), size=sample_size, replace=False)
        feats_sub = fused_embeddings[idx]
        lbl_sub = labels[idx]
    else:
        feats_sub = fused_embeddings
        lbl_sub = labels

    pca = PCA(n_components=min(50, feats_sub.shape[1]), random_state=seed)
    feats_pca = pca.fit_transform(feats_sub)

    tsne = TSNE(
        n_components=2,
        perplexity=40,
        random_state=seed,
        init="pca",
        learning_rate="auto",
    )
    embedded = tsne.fit_transform(feats_pca)

    plt.figure(figsize=(8, 6.5))
    scatter = plt.scatter(embedded[:, 0], embedded[:, 1], c=lbl_sub, cmap="viridis", s=6, alpha=0.65)
    plt.colorbar(scatter, label="LVEF <= 45% Flag (0 / 1)")
    plt.title(f"Fused Embedding PCA+t-SNE Projection (n={len(feats_sub)})")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# ---------------------------------------------------------------------------
# End-to-End Orchestration Using Modular Packages
# ---------------------------------------------------------------------------


def run_exploratory_analysis(
    split: str = "val",
    checkpoint_path: str = DEFAULT_PATH_CONFIG.best_fusion_checkpoint,
    threshold: float = DEFAULT_EVAL_CONFIG.saved_threshold,
    save_dir: Path | None = None,
) -> Dict[str, float]:
    """Execute evaluation, temperature-scaling check, and optional exploratory plots."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths = DEFAULT_PATH_CONFIG

    resolved_ckpt = Path(checkpoint_path)
    if not resolved_ckpt.is_absolute():
        resolved_ckpt = PROJECT_ROOT / resolved_ckpt

    model = load_fusion_checkpoint(resolved_ckpt, device=device, strict=True)
    res_feats, gru_feats, labels = load_fusion_data(split, paths)

    metrics, probs, alpha_res, alpha_gru = evaluate_model(
        model,
        res_feats,
        gru_feats,
        labels,
        threshold=threshold,
        batch_size=DEFAULT_TRAIN_CONFIG.batch_size_val,
        device=device,
    )

    logits, fused_embeddings, _, _ = collect_fused_embeddings(
        model,
        res_feats,
        gru_feats,
        batch_size=DEFAULT_TRAIN_CONFIG.batch_size_val,
        device=device,
    )

    optimal_temp = fit_temperature_scaling(logits, labels)

    print(f"Split: {split} (N={len(labels):,}) | Checkpoint: {resolved_ckpt.name} | Threshold: {threshold:.2f}")
    print(f"  ROC-AUC           : {metrics['roc_auc'] * 100:.2f}%")
    print(f"  PR-AUC            : {metrics['pr_auc'] * 100:.2f}%")
    print(f"  Balanced Accuracy : {metrics['balanced_accuracy'] * 100:.2f}%")
    print(f"  Sensitivity       : {metrics['sensitivity'] * 100:.2f}%")
    print(f"  Specificity       : {metrics['specificity'] * 100:.2f}%")
    print(f"  Mean alpha_res    : {metrics['mean_alpha_res']:.4f}")
    print(f"  Mean alpha_gru    : {metrics['mean_alpha_gru']:.4f}")
    print(f"  Optimal Temp (NLL): {optimal_temp:.4f}")

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        plot_calibration_curve(labels, probs, save_dir / f"{split}_calibration_curve.png")
        plot_reliability_scatter(alpha_res, alpha_gru, save_dir / f"{split}_reliability_scatter.png")
        plot_latent_tsne(fused_embeddings, labels, save_dir / f"{split}_latent_tsne.png")
        print(f"Exploratory figures written to: {save_dir}")

    return {"roc_auc": metrics["roc_auc"], "optimal_temperature": optimal_temp}


def run_modular_training(save_path: str = "best_fusion_model_repro.pth") -> None:
    """Delegate fusion training to the modular FusionTrainer (replaces notebook Cells 14-37)."""
    paths = DEFAULT_PATH_CONFIG
    cfg = DEFAULT_TRAIN_CONFIG
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    res_tr, gru_tr, y_tr = load_fusion_data("train", paths)
    res_va, gru_va, y_va = load_fusion_data("val", paths)

    computed_pos_weight = compute_class_pos_weight(y_tr)
    print(f"Computed training pos_weight = {computed_pos_weight:.6f} (config: {cfg.pos_weight:.6f})")

    train_loader = DataLoader(FusionDataset(res_tr, gru_tr, y_tr), batch_size=cfg.batch_size_train, shuffle=True)
    val_loader = DataLoader(FusionDataset(res_va, gru_va, y_va), batch_size=cfg.batch_size_val, shuffle=False)

    model = ECGFusionClassifier()
    trainer = FusionTrainer(model=model, train_loader=train_loader, val_loader=val_loader, config=cfg, device=device)
    trainer.train(save_path=save_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CardioInsight Exploratory Research Workflow")
    parser.add_argument(
        "--mode",
        choices=["analyze", "train"],
        default="analyze",
        help="Run exploratory evaluation/calibration analysis or modular fusion training",
    )
    parser.add_argument("--split", choices=["val", "test"], default="val", help="Split for exploratory analysis")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=DEFAULT_PATH_CONFIG.best_fusion_checkpoint,
        help="Fusion checkpoint path",
    )
    parser.add_argument("--threshold", type=float, default=DEFAULT_EVAL_CONFIG.saved_threshold)
    parser.add_argument(
        "--save-dir",
        type=str,
        default="",
        help="Optional directory to export calibration, reliability scatter, and t-SNE plots",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="best_fusion_model_repro.pth",
        help="Output checkpoint path when --mode train is selected",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "train":
        run_modular_training(save_path=args.save_path)
    else:
        out_dir = Path(args.save_dir) if args.save_dir else None
        run_exploratory_analysis(
            split=args.split,
            checkpoint_path=args.checkpoint,
            threshold=args.threshold,
            save_dir=out_dir,
        )
