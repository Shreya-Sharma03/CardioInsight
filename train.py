"""
CardioInsight — Fusion Model Training Script
Migrated from `final-fusion-model.ipynb` (Cells 2, 8, 14-37).

This executable script reproduces the exact multi-modal fusion training pipeline:
1. Loads precomputed 512-d ResNet34 and 256-d BiGRU latent feature representations.
2. Initializes the Adaptive Reliability Fusion Classifier (ECGFusionClassifier).
3. Configures AdamW optimizer with CosineAnnealingWarmRestarts and mixed-precision (AMP).
4. Uses the exact Hybrid Loss (Weighted BCE + Focal Loss with pos_weight=3.2728).
5. Runs training with gradient clipping, EarlyStopping (patience=8), and EMA shadow tracking.
6. Evaluates composite score (0.40*PR_AUC + 0.30*ROC_AUC + 0.20*F1 + 0.10*Recall) per epoch.
7. Saves the best checkpoint.

Usage:
    python train.py [--epochs 40] [--batch-size 512] [--device cuda/cpu]
"""

import sys
import argparse
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config.config import DEFAULT_PATH_CONFIG, DEFAULT_TRAIN_CONFIG, DEFAULT_MODEL_CONFIG, TrainConfig
from data.dataset import FusionDataset, load_fusion_data
from models.fusion_model import ECGFusionClassifier
from training.trainer import FusionTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train CardioInsight Adaptive Reliability Fusion Model")
    parser.add_argument("--epochs", type=int, default=DEFAULT_TRAIN_CONFIG.num_epochs, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_TRAIN_CONFIG.batch_size_train, help="Training batch size")
    parser.add_argument("--lr", type=float, default=DEFAULT_TRAIN_CONFIG.learning_rate, help="Learning rate (default: 3e-4)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    parser.add_argument("--save-path", type=str, default="best_fusion_model.pth", help="Output path for best checkpoint")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    print(f"============================================================")
    print(f" CardioInsight Fusion Model Training")
    print(f" Execution Device : {device}")
    print(f" Max Epochs       : {args.epochs}")
    print(f" Batch Size       : {args.batch_size}")
    print(f" Learning Rate    : {args.lr}")
    print(f" Output Checkpoint: {args.save_path}")
    print(f"============================================================\n")

    # 1. Load precomputed multimodal features
    paths = DEFAULT_PATH_CONFIG
    print("[1/4] Loading precomputed feature matrices and labels...")
    res_tr, gru_tr, y_tr = load_fusion_data("train", paths)
    res_va, gru_va, y_va = load_fusion_data("val", paths)
    print(f"  -> Train cohort: {len(y_tr):,} samples (ResNet: {res_tr.shape[1]}-d, BiGRU: {gru_tr.shape[1]}-d)")
    print(f"  -> Val cohort  : {len(y_va):,} samples")

    # 2. Formulate PyTorch DataLoaders
    print("\n[2/4] Initializing PyTorch DataLoaders...")
    train_dataset = FusionDataset(res_tr, gru_tr, y_tr)
    val_dataset = FusionDataset(res_va, gru_va, y_va)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=DEFAULT_TRAIN_CONFIG.batch_size_val,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    # 3. Instantiate Model
    print("\n[3/4] Initializing ECGFusionClassifier architecture...")
    model = ECGFusionClassifier(
        resnet_dim=DEFAULT_MODEL_CONFIG.resnet_dim,
        bigru_dim=DEFAULT_MODEL_CONFIG.bigru_dim,
        embed_dim=DEFAULT_MODEL_CONFIG.embed_dim,
        num_heads=DEFAULT_MODEL_CONFIG.num_heads,
        dropout=DEFAULT_MODEL_CONFIG.dropout,
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  -> Model instantiated: {total_params:,} parameters (100% trainable)")

    # 4. Train with FusionTrainer
    print("\n[4/4] Starting training loop with early stopping and EMA tracking...")
    custom_train_config = TrainConfig(
        num_epochs=args.epochs,
        batch_size_train=args.batch_size,
        learning_rate=args.lr,
    )

    trainer = FusionTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=custom_train_config,
        device=device,
    )

    history = trainer.train(save_path=args.save_path)
    print(f"\n============================================================")
    print(f" Training complete! Best checkpoint saved to: {args.save_path}")
    print(f" Best Validation Epochs Recorded: {len(history['val_loss'])}")
    print(f"============================================================")


if __name__ == "__main__":
    main()
