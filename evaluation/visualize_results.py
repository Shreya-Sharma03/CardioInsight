"""
CardioInsight — Research Paper Evaluation & Visualization Script
Migrated from `final-fusion-model.ipynb` (Cells 47-51, 61, 62).

This script performs full model evaluation on the holdout test set using the verified
best checkpoint and generates publication-grade visualizations:
1. Receiver Operating Characteristic (ROC) & Precision-Recall (PR) Curves
2. Normalized Confusion Matrix with optimal cutoff (0.41)
3. Modality Reliability Distributions (ResNet-34 vs. BiGRU Attention Weights)

Usage:
    python evaluation/visualize_results.py [--checkpoint best_fusion_model.zip] [--save-dir audit_results]
"""

import sys
import argparse
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)

from config.config import DEFAULT_PATH_CONFIG, DEFAULT_EVAL_CONFIG
from data.dataset import load_fusion_data
from evaluation.evaluator import load_fusion_checkpoint, evaluate_model


def set_publication_style():
    """Configure matplotlib defaults for clean publication aesthetics."""
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 13,
        'axes.titlesize': 13,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'figure.titlesize': 15,
        'figure.dpi': 300,
        'figure.facecolor': '#FFFFFF',
        'axes.facecolor': '#FFFFFF',
    })


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Research Paper Visualizations for CardioInsight")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(PROJECT_ROOT / DEFAULT_PATH_CONFIG.best_fusion_checkpoint),
        help="Path to trained fusion checkpoint (.zip or .pth)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_EVAL_CONFIG.saved_threshold,
        help="Classification decision threshold (default: 0.41)",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=str(PROJECT_ROOT / "audit_results"),
        help="Directory to save generated figures",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figures interactively via plt.show()",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_publication_style()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("============================================================")
    print(" CardioInsight Evaluation & Publication Visualizer")
    print(f" Checkpoint : {args.checkpoint}")
    print(f" Threshold  : {args.threshold}")
    print(f" Output Dir : {save_dir}")
    print("============================================================\n")

    # 1. Load trained checkpoint and holdout test data
    print("[1/4] Loading model and holdout test cohort...")
    model = load_fusion_checkpoint(args.checkpoint, strict=True)
    res_test, gru_test, y_test = load_fusion_data("test")
    print(f"  -> Holdout Test Samples: {len(y_test):,}")

    # 2. Run inference
    print("\n[2/4] Executing test set inference...")
    metrics, probs, alpha_res, alpha_gru = evaluate_model(
        model=model,
        res_features=res_test,
        gru_features=gru_test,
        labels=y_test,
        threshold=args.threshold,
    )

    print(f"  -> Test ROC-AUC         : {metrics['roc_auc']:.4f}")
    print(f"  -> Test PR-AUC (AP)     : {metrics['pr_auc']:.4f}")
    print(f"  -> Test Balanced Acc    : {metrics['balanced_accuracy']:.4f}")
    print(f"  -> Test Sensitivity     : {metrics['sensitivity']:.4f}")
    print(f"  -> Test Specificity     : {metrics['specificity']:.4f}")
    print(f"  -> Mean Alpha ResNet    : {metrics['mean_alpha_res']:.4f}")
    print(f"  -> Mean Alpha BiGRU     : {metrics['mean_alpha_gru']:.4f}")

    # 3. Generate ROC and PR Curves
    print("\n[3/4] Generating ROC and PR Curves...")
    fpr, tpr, _ = roc_curve(y_test, probs)
    precision, recall, _ = precision_recall_curve(y_test, probs)
    roc_auc = roc_auc_score(y_test, probs)
    pr_auc = average_precision_score(y_test, probs)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # ROC Plot
    axes[0].plot(fpr, tpr, color='#1f77b4', lw=2.5, label=f'Fusion Model (AUC = {roc_auc:.4f})')
    axes[0].plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1.5)
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel('False Positive Rate (1 - Specificity)')
    axes[0].set_ylabel('True Positive Rate (Sensitivity)')
    axes[0].set_title('Receiver Operating Characteristic (ROC)')
    axes[0].legend(loc='lower right')
    axes[0].grid(True, linestyle=':', alpha=0.6)

    # PR Plot
    axes[1].plot(recall, precision, color='#ff7f0e', lw=2.5, label=f'Fusion Model (AP = {pr_auc:.4f})')
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('Recall (Sensitivity)')
    axes[1].set_ylabel('Precision')
    axes[1].set_title('Precision-Recall (PR) Curve')
    axes[1].legend(loc='lower left')
    axes[1].grid(True, linestyle=':', alpha=0.6)

    fig.tight_layout()
    roc_pr_path = save_dir / "roc_pr_curves.png"
    fig.savefig(roc_pr_path, dpi=300)
    print(f"  -> Saved: {roc_pr_path}")

    # 4. Normalized Confusion Matrix & Reliability Distribution
    print("\n[4/4] Generating Confusion Matrix & Modality Attention Distributions...")
    preds_binary = (probs >= args.threshold).astype(int)
    cm = confusion_matrix(y_test, preds_binary, normalize='true')

    fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        cbar=True,
        ax=ax_cm,
        xticklabels=['Low Risk (0)', 'High Risk (1)'],
        yticklabels=['Low Risk (0)', 'High Risk (1)'],
    )
    ax_cm.set_xlabel('Predicted Label')
    ax_cm.set_ylabel('Actual Ground-Truth Label')
    ax_cm.set_title(f'Normalized Confusion Matrix (Threshold = {args.threshold})')
    fig_cm.tight_layout()
    cm_path = save_dir / "confusion_matrix.png"
    fig_cm.savefig(cm_path, dpi=300)
    print(f"  -> Saved: {cm_path}")

    # Modality Attention Distributions
    fig_dist, ax_dist = plt.subplots(figsize=(8.5, 4.2))
    sns.kdeplot(
        alpha_res.flatten(),
        label=r'ResNet-34 Reliability ($\alpha_{res}$)',
        color='#2ca02c',
        fill=True,
        alpha=0.3,
        lw=2,
        ax=ax_dist,
    )
    sns.kdeplot(
        alpha_gru.flatten(),
        label=r'BiGRU Reliability ($\alpha_{gru}$)',
        color='#d62728',
        fill=True,
        alpha=0.3,
        lw=2,
        ax=ax_dist,
    )
    ax_dist.set_xlabel('Adaptive Reliability Weight')
    ax_dist.set_ylabel('Density')
    ax_dist.set_title('Distribution of Modality Reliability Attention Weights')
    ax_dist.legend(loc='upper center')
    ax_dist.grid(True, linestyle=':', alpha=0.6)
    fig_dist.tight_layout()
    dist_path = save_dir / "reliability_distributions.png"
    fig_dist.savefig(dist_path, dpi=300)
    print(f"  -> Saved: {dist_path}")

    if args.show:
        plt.show()

    print(f"\n============================================================")
    print(f" All publication evaluation plots generated successfully!")
    print(f"============================================================")


if __name__ == "__main__":
    main()
