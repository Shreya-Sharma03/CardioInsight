"""Training and Validation Pipeline for CardioInsight

Extracted from `final-fusion-model.ipynb` (Cells 30, 31, 37).
Preserves the exact mixed-precision training, gradient clipping, CosineAnnealingWarmRestarts
scheduling, and composite validation score calculation:
Score = 0.40 * PR_AUC + 0.30 * ROC_AUC + 0.20 * F1 + 0.10 * Recall.

AUDIT / EXPERIMENT REALITY:
1. SWA: The code contains the scheduled SWA logic (epoch >= SWA_START).
   Because original training stopped at epoch 15 while SWA_START was 30,
   SWA averaging was NEVER activated during the actual training run.
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

from evaluation.threshold import find_best_threshold
from .losses import HybridLoss
from .callbacks import EarlyStopping, EMA, save_checkpoint
from config.config import TrainConfig, DEFAULT_TRAIN_CONFIG


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: GradScaler,
    device: torch.device,
    grad_clip_norm: float = 1.0,
) -> float:
    """Train the model for one epoch using mixed precision and gradient clipping.

    Directly migrated from Cell 30 of `final-fusion-model.ipynb`.
    """
    model.train()
    total_loss = 0.0

    for res_feat, gru_feat, labels in loader:
        res_feat = res_feat.to(device)
        gru_feat = gru_feat.to(device)
        labels = labels.float().to(device)

        optimizer.zero_grad()

        with autocast(enabled=device.type == "cuda"):
            logits, _, _, _ = model(res_feat, gru_feat)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * labels.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate the model on validation data and compute metrics.

    Directly migrated from Cell 31 of `final-fusion-model.ipynb`.
    """
    model.eval()
    total_loss = 0.0
    probs = []
    labels_all = []

    for res_feat, gru_feat, labels in loader:
        res_feat = res_feat.to(device)
        gru_feat = gru_feat.to(device)
        labels = labels.float().to(device)

        logits, _, _, _ = model(res_feat, gru_feat)
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        probs.extend(torch.sigmoid(logits).cpu().numpy())
        labels_all.extend(labels.cpu().numpy())

    probs = np.array(probs)
    labels_all = np.array(labels_all)

    # In Cell 31, best threshold is dynamically searched via F1 score
    threshold, best_f1 = find_best_threshold(probs, labels_all)
    preds = (probs >= threshold).astype(int)

    metrics = {
        "loss": total_loss / len(loader.dataset),
        "threshold": threshold,
        "accuracy": float(accuracy_score(labels_all, preds)),
        "precision": float(precision_score(labels_all, preds, zero_division=0)),
        "recall": float(recall_score(labels_all, preds, zero_division=0)),
        "f1": float(f1_score(labels_all, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels_all, probs)),
        "pr_auc": float(average_precision_score(labels_all, probs)),
    }

    return metrics


class FusionTrainer:
    """Encapsulates the complete training loop from Cell 37 of `final-fusion-model.ipynb`."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Optional[TrainConfig] = None,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config or DEFAULT_TRAIN_CONFIG
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        pos_weight_tensor = torch.tensor(
            [self.config.pos_weight], dtype=torch.float32, device=self.device
        )
        self.criterion = HybridLoss(
            pos_weight=pos_weight_tensor,
            alpha=self.config.focal_alpha,
            gamma=self.config.focal_gamma,
            focal_weight=self.config.focal_weight,
            bce_weight=self.config.bce_weight,
        )

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=self.config.t_0,
            T_mult=self.config.t_mult,
            eta_min=self.config.eta_min,
        )

        self.scaler = GradScaler(enabled=self.device.type == "cuda")
        self.early_stop = EarlyStopping(patience=self.config.early_stopping_patience)
        self.ema = EMA(self.model, decay=self.config.ema_decay)

    def train(self, save_path: str = "best_fusion_model.pth") -> Dict[str, list]:
        """Executes training loop with early stopping."""
        history = {
            "train_loss": [], "val_loss": [], "accuracy": [], "precision": [],
            "recall": [], "f1": [], "roc_auc": [], "pr_auc": [], "lr": []
        }
        best_score = -1.0

        for epoch in range(self.config.num_epochs):
            avg_train_loss = train_one_epoch(
                self.model, self.train_loader, self.optimizer, self.criterion,
                self.scaler, self.device, self.config.grad_clip_norm
            )
            metrics = validate_one_epoch(
                self.model, self.val_loader, self.criterion, self.device
            )

            self.scheduler.step()
            self.ema.update(self.model)

            history["train_loss"].append(avg_train_loss)
            history["val_loss"].append(metrics["loss"])
            history["accuracy"].append(metrics["accuracy"])
            history["precision"].append(metrics["precision"])
            history["recall"].append(metrics["recall"])
            history["f1"].append(metrics["f1"])
            history["roc_auc"].append(metrics["roc_auc"])
            history["pr_auc"].append(metrics["pr_auc"])
            history["lr"].append(self.optimizer.param_groups[0]["lr"])

            # Composite maximization score
            score = (
                0.40 * metrics["pr_auc"]
                + 0.30 * metrics["roc_auc"]
                + 0.20 * metrics["f1"]
                + 0.10 * metrics["recall"]
            )

            if score > best_score:
                best_score = score
                save_checkpoint(
                    {"epoch": epoch, "model": self.model.state_dict(), "score": score},
                    save_path,
                )

            if self.early_stop.step(score):
                break

        return history
