"""
Training pipeline for the fusion model.

Combines pre-extracted ResNet34-1D and BiGRU features using adaptive reliability
weighting, multi-head attention, and a hybrid BCE + focal loss function.
"""

import os
import argparse
import random
from typing import Tuple, Dict, Any
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    recall_score,
    accuracy_score,
    balanced_accuracy_score,
)

try:
    from .fusion_model import ECGFusionClassifier
except ImportError:
    from fusion_model import ECGFusionClassifier


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class FusionDataset(Dataset):
    """Dataset of paired ResNet features, BiGRU features, and binary labels."""

    def __init__(self, res_features: np.ndarray, gru_features: np.ndarray, labels: np.ndarray):
        self.res_features = torch.from_numpy(res_features).float()
        self.gru_features = torch.from_numpy(gru_features).float()
        self.labels = torch.from_numpy(labels).float()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.res_features[idx], self.gru_features[idx], self.labels[idx]


class HybridLoss(nn.Module):
    """
    Weighted combination of class-weighted Binary Cross-Entropy and Focal Loss.
    """

    def __init__(
        self,
        pos_weight: torch.Tensor,
        alpha: float = 0.25,
        gamma: float = 2.0,
        focal_weight: float = 0.30,
        bce_weight: float = 0.70,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.focal_weight = focal_weight
        self.bce_weight = bce_weight
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        bce = self.bce(logits, labels)
        prob = torch.sigmoid(logits)
        pt = torch.where(labels == 1, prob, 1.0 - prob)
        focal = -self.alpha * ((1.0 - pt) ** self.gamma) * torch.log(pt + 1e-8)
        focal = focal.mean()
        return self.bce_weight * bce + self.focal_weight * focal


def find_best_threshold(probs: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    """Scan decision thresholds to find the value that maximizes F1-score."""
    best_f1 = 0.0
    best_threshold = 0.50
    for t in np.arange(0.10, 0.91, 0.01):
        preds = (probs >= t).astype(int)
        score = f1_score(labels, preds, zero_division=0)
        if score > best_f1:
            best_f1 = float(score)
            best_threshold = float(t)
    return best_threshold, best_f1


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    entropy_weight: float = 0.01,
) -> float:
    """Train the model for a single epoch."""
    model.train()
    total_loss = 0.0

    for res, gru, labels in loader:
        res = res.to(device)
        gru = gru.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        if device.type == "cuda":
            with torch.cuda.amp.autocast():
                logits, _, alpha_res, alpha_gru = model(res, gru)
                loss = criterion(logits, labels)
                entropy = -(
                    alpha_res * torch.log(alpha_res + 1e-8)
                    + alpha_gru * torch.log(alpha_gru + 1e-8)
                ).mean()
                total_loss_batch = loss - entropy_weight * entropy
            scaler.scale(total_loss_batch).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits, _, alpha_res, alpha_gru = model(res, gru)
            loss = criterion(logits, labels)
            entropy = -(
                alpha_res * torch.log(alpha_res + 1e-8)
                + alpha_gru * torch.log(alpha_gru + 1e-8)
            ).mean()
            total_loss_batch = loss - entropy_weight * entropy
            total_loss_batch.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += total_loss_batch.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate_split(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run model inference over a dataloader split and return probabilities and labels."""
    model.eval()
    all_probs = []
    all_labels = []

    for res, gru, labels in loader:
        res = res.to(device)
        gru = gru.to(device)
        logits, _, _, _ = model(res, gru)
        probs = torch.sigmoid(logits)
        all_probs.extend(probs.cpu().numpy().flatten())
        all_labels.extend(labels.numpy().flatten())

    return np.array(all_probs), np.array(all_labels)


def train_fusion_pipeline(
    res_train: np.ndarray,
    gru_train: np.ndarray,
    y_train: np.ndarray,
    res_val: np.ndarray,
    gru_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 15,
    batch_size: int = 512,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    output_dir: str = "models",
    device_name: str = "cpu",
) -> Dict[str, Any]:
    """
    Execute complete fusion model training and validation.
    """
    device = torch.device(device_name)
    os.makedirs(output_dir, exist_ok=True)

    train_dataset = FusionDataset(res_train, gru_train, y_train)
    val_dataset = FusionDataset(res_val, gru_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size * 2, shuffle=False)

    model = ECGFusionClassifier().to(device)

    # Class balance adjustment
    neg_count = np.sum(y_train == 0)
    pos_count = np.sum(y_train == 1)
    pos_weight = torch.tensor([neg_count / max(pos_count, 1)], dtype=torch.float32).to(device)

    criterion = HybridLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=2, eta_min=1e-6
    )
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    best_score = -1.0
    best_threshold = 0.50
    best_checkpoint_path = os.path.join(output_dir, "best_fusion_model.pth")
    final_checkpoint_path = os.path.join(output_dir, "FusionModel.pth")

    for epoch in range(epochs):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            scaler=scaler,
            device=device,
        )
        scheduler.step()

        val_probs, val_labels = evaluate_split(model, val_loader, device)
        val_auc = roc_auc_score(val_labels, val_probs)
        val_pr_auc = average_precision_score(val_labels, val_probs)
        thresh, f1 = find_best_threshold(val_probs, val_labels)

        val_preds = (val_probs >= thresh).astype(int)
        rec = recall_score(val_labels, val_preds, zero_division=0)
        score = 0.40 * val_pr_auc + 0.30 * val_auc + 0.20 * f1 + 0.10 * rec

        print(
            f"Epoch {epoch + 1:2d}/{epochs:2d} | Train Loss: {train_loss:.4f} | "
            f"Val ROC-AUC: {val_auc:.4f} | PR-AUC: {val_pr_auc:.4f} | F1: {f1:.4f} | Thresh: {thresh:.2f}"
        )

        if score > best_score:
            best_score = score
            best_threshold = thresh
            torch.save(model.state_dict(), best_checkpoint_path)

    # Save final model state
    torch.save(model.state_dict(), final_checkpoint_path)
    np.save(os.path.join(output_dir, "threshold.npy"), float(best_threshold))

    return {
        "best_score": best_score,
        "best_threshold": best_threshold,
        "best_checkpoint": best_checkpoint_path,
        "final_checkpoint": final_checkpoint_path,
    }


def main():
    parser = argparse.ArgumentParser(description="Train CardioInsight Fusion Model")
    parser.add_argument("--data_dir", type=str, default=".", help="Directory containing feature and label .npy files")
    parser.add_argument("--output_dir", type=str, default="models", help="Directory to save checkpoints")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Compute device")
    args = parser.parse_args()

    set_seed(args.seed)

    # Resolve required feature and label files
    res_tr_path = os.path.join(args.data_dir, "resnet_features_train.npy")
    gru_tr_path = os.path.join(args.data_dir, "bigru_features_train.npy")
    y_tr_path = os.path.join(args.data_dir, "resnet_labels_train.npy")

    res_va_path = os.path.join(args.data_dir, "resnet_features_val.npy")
    gru_va_path = os.path.join(args.data_dir, "bigru_features_val.npy")
    y_va_path = os.path.join(args.data_dir, "resnet_labels_val.npy")

    required = [res_tr_path, gru_tr_path, y_tr_path, res_va_path, gru_va_path, y_va_path]
    missing = [p for p in required if not os.path.exists(p)]
    if missing:
        print(f"[Error] Required training data files not found: {missing}")
        print("Please supply pre-extracted features or point --data_dir to their directory.")
        return

    print(f"Loading data from {args.data_dir}...")
    res_train = np.load(res_tr_path)
    gru_train = np.load(gru_tr_path)
    y_train = np.load(y_tr_path)

    res_val = np.load(res_va_path)
    gru_val = np.load(gru_va_path)
    y_val = np.load(y_va_path)

    print(f"Training fusion model on {len(y_train)} samples, validating on {len(y_val)} samples...")
    results = train_fusion_pipeline(
        res_train=res_train,
        gru_train=gru_train,
        y_train=y_train,
        res_val=res_val,
        gru_val=gru_val,
        y_val=y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        output_dir=args.output_dir,
        device_name=args.device,
    )
    print("Training complete!")
    print(f"Saved checkpoints to {args.output_dir} (Optimal threshold: {results['best_threshold']:.2f})")


if __name__ == "__main__":
    main()
