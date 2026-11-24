"""CardioInsight Configuration Module

Preserves the exact hyperparameters, tensor dimensions, and paths used during the
original research experiment in `final-fusion-model.ipynb`.

LIMITATIONS & NOTES RECORDED FROM AUDIT:
1. Fusion training operates strictly on cached 512-dim ResNet and 256-dim BiGRU feature vectors.
2. The default threshold (0.41) was hardcoded as a literal constant in the original notebook.
   Its selection rationale is undocumented. It was not established as the F1-optimal validation threshold.
3. SWA was scheduled for epoch 30, but training stopped at epoch 15 via early stopping.
4. EMA shadow weights were updated per epoch but never evaluated or saved as a final checkpoint.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any


@dataclass
class ModelConfig:
    """Hyperparameters defining the multi-modal fusion network architecture."""

    resnet_dim: int = 512
    bigru_dim: int = 256
    embed_dim: int = 512
    num_heads: int = 8
    dropout: float = 0.30

    # Confidence estimator MLP dimensions: embed_dim -> 128 -> 1
    conf_hidden_dim: int = 128

    # Deep Classifier Head dimensions: 512 -> 512 -> 256 -> 64 -> 1
    classifier_stage1_dim: int = 512
    classifier_stage2_dim: int = 256
    classifier_stage3_dim: int = 64
    classifier_dropout1: float = 0.30
    classifier_dropout2: float = 0.20


@dataclass
class TrainConfig:
    """Training hyperparameters from Cells 22-29 and Cell 37."""

    seed: int = 42
    batch_size_train: int = 512
    batch_size_val: int = 1024
    batch_size_test: int = 1024
    num_epochs: int = 40

    # Optimizer (AdamW)
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4

    # Scheduler (CosineAnnealingWarmRestarts)
    t_0: int = 5
    t_mult: int = 2
    eta_min: float = 1e-6

    # Loss function (Hybrid BCE + Focal Loss)
    pos_weight: float = 3.2727861404418945  # Calculated as negative (55513) / positive (16962)
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    focal_weight: float = 0.30
    bce_weight: float = 0.70

    # Early stopping
    early_stopping_patience: int = 8

    # Gradient clipping & AMP
    grad_clip_norm: float = 1.0
    use_amp: bool = True

    # EMA
    ema_decay: float = 0.999

    # SWA schedule
    swa_start: int = 30
    swa_lr: float = 1e-4


@dataclass
class PathConfig:
    """Directory and file paths for the project."""

    base_dir: Path = Path(__file__).resolve().parent.parent

    # Model checkpoints
    best_fusion_checkpoint: str = "best_fusion_model.zip"  # Epoch 7 best model
    final_fusion_checkpoint: str = "FusionModel.zip"        # Epoch 15 end-of-training state dict
    config_checkpoint: str = "config.zip"
    resnet_checkpoint: str = "ResNet34_ECG.pth.zip"
    bigru_checkpoint: str = "checkpoint_backup/BiGRU_ECG_repacked.zip"

    # Raw ECG Waveforms (EchoNext Validation Cohort)
    val_waveforms_file: str = "data/raw/EchoNext_val_waveforms.npy"

    # Precomputed feature files
    resnet_features_train: str = "resnet_features_train.npy"
    resnet_features_val: str = "resnet_features_val.npy"
    resnet_features_test: str = "resnet_features_test.npy"

    bigru_features_train: str = "bigru_features_train.npy"
    bigru_features_val: str = "bigru_features_val.npy"
    bigru_features_test: str = "bigru_features_test.npy"

    # Label arrays (downloaded from Kaggle)
    labels_train: str = "resnet_labels_train (1).npy"
    labels_val: str = "resnet_labels_val (1).npy"
    labels_test: str = "resnet_labels_test (1).npy"

    # Saved decision threshold
    threshold_file: str = "threshold.npy"
    label_map_file: str = "labels.zip"


@dataclass
class EvaluationConfig:
    """Decision thresholds and evaluation parameters."""

    # Default saved threshold from Cell 58 (empirically hardcoded)
    saved_threshold: float = 0.41

    # Standard uncalibrated classification threshold
    default_threshold: float = 0.50

    # Note: Cell 61 Youden's J test-set threshold (0.4697) is documented as test-set leakage
    # and MUST NOT be used as the deployed decision threshold.


DEFAULT_MODEL_CONFIG = ModelConfig()
DEFAULT_TRAIN_CONFIG = TrainConfig()
DEFAULT_PATH_CONFIG = PathConfig()
DEFAULT_EVAL_CONFIG = EvaluationConfig()
