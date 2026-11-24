"""Loss Functions for CardioInsight Training

Extracted directly from `final-fusion-model.ipynb` (Cell 23).
Preserves the exact combination of positive-weighted Binary Cross-Entropy with Logits
and Alpha-Gamma Focal Loss.
"""

import torch
import torch.nn as nn


class HybridLoss(nn.Module):
    """Hybrid loss combining BCEWithLogitsLoss and Focal Loss.

    Loss = bce_weight * BCE + focal_weight * Focal

    Where:
        pt = prob if label == 1 else 1 - prob
        focal = -alpha * ((1 - pt) ** gamma) * log(pt + 1e-8)
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
