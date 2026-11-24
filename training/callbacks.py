"""Training Callbacks and Utilities for CardioInsight

Extracted from `final-fusion-model.ipynb` (Cells 26, 28, 32).
Preserves the exact EarlyStopping patience logic, EMA shadow updates, and checkpoint format.

AUDIT NOTES:
1. EMA: The EMA class was instantiated and updated per epoch during original training,
   but its shadow weights were never saved as a model checkpoint or used for final evaluation.
2. Checkpoint Saving: The checkpoint dictionary saved during peak epochs follows the exact
   schema: {"epoch": epoch, "model": model.state_dict(), "score": score}.
"""

from typing import Dict, Any, Optional
import torch
import torch.nn as nn


class EarlyStopping:
    """Early stopping callback based on a maximization score.

    Directly migrated from Cell 28 of `final-fusion-model.ipynb`.
    """

    def __init__(self, patience: int = 8):
        self.best: float = -1e9
        self.counter: int = 0
        self.patience: int = patience

    def step(self, score: float) -> bool:
        """Update counter and return True if early stopping should trigger."""
        if score > self.best:
            self.best = score
            self.counter = 0
            return False

        self.counter += 1
        return self.counter >= self.patience


class EMA:
    """Exponential Moving Average of model parameters.

    Directly migrated from Cell 26 of `final-fusion-model.ipynb`.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model: nn.Module) -> None:
        """Update shadow weights with decay."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = (
                    self.decay * self.shadow[name] + (1.0 - self.decay) * param.data
                )


def save_checkpoint(
    state: Dict[str, Any],
    filepath: str = "best_fusion_model.pth",
) -> None:
    """Save training checkpoint to disk.

    Directly migrated from Cell 32 of `final-fusion-model.ipynb`.
    """
    torch.save(state, filepath)
