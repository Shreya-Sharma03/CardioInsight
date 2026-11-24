"""CardioInsight Dataset and DataLoader Module

Extracted from `final-fusion-model.ipynb` (Cells 14-16).
Manages loading of precomputed feature arrays and ground-truth binary labels.

DATA AUDIT & LIMITATIONS NOTICE:
1. This dataset operates on cached feature vectors (512-dim from ResNet, 256-dim from BiGRU).
2. ResNet representations and BiGRU representations were sourced from separate Kaggle repositories
   ('shreyasharma03/resnet34-1d' and 'vidhisidana/bigru-pth').
3. Array lengths match across all splits (Train: 72,475; Val: 4,626; Test: 5,442).
   However, sample-by-sample 1:1 cross-modal row alignment is NOT independently verified
   because sample ID manifests and upstream extraction scripts are absent.
4. Patient-level split isolation cannot be verified due to lack of patient identifiers.
"""

from pathlib import Path
from typing import Tuple, Dict, Optional
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from config.config import PathConfig, TrainConfig, DEFAULT_PATH_CONFIG, DEFAULT_TRAIN_CONFIG


class FusionDataset(Dataset):
    """PyTorch Dataset wrapping pre-extracted ResNet and BiGRU features and labels.

    Directly migrated from Cell 15 of `final-fusion-model.ipynb`.
    """

    def __init__(
        self,
        res_features: np.ndarray,
        gru_features: np.ndarray,
        labels: np.ndarray,
    ):
        """Args:
            res_features: Array of shape (N, 512)
            gru_features: Array of shape (N, 256)
            labels: Array of shape (N,)
        """
        assert len(res_features) == len(gru_features) == len(labels), (
            f"Array length mismatch: res={len(res_features)}, "
            f"gru={len(gru_features)}, labels={len(labels)}"
        )

        self.res_features = torch.from_numpy(res_features).float()
        self.gru_features = torch.from_numpy(gru_features).float()
        self.labels = torch.from_numpy(labels).float()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.res_features[idx],
            self.gru_features[idx],
            self.labels[idx],
        )


def load_fusion_data(
    split: str = "test",
    paths: Optional[PathConfig] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load precomputed features and labels for a specific split.

    Args:
        split: 'train', 'val', or 'test'
        paths: PathConfig instance (defaults to DEFAULT_PATH_CONFIG)

    Returns:
        Tuple of (res_features, gru_features, labels) as numpy arrays.
    """
    if paths is None:
        paths = DEFAULT_PATH_CONFIG

    base = paths.base_dir

    if split == "train":
        res_file = base / paths.resnet_features_train
        gru_file = base / paths.bigru_features_train
        lbl_file = base / paths.labels_train
    elif split == "val":
        res_file = base / paths.resnet_features_val
        gru_file = base / paths.bigru_features_val
        lbl_file = base / paths.labels_val
    elif split == "test":
        res_file = base / paths.resnet_features_test
        gru_file = base / paths.bigru_features_test
        lbl_file = base / paths.labels_test
    else:
        raise ValueError(f"Unknown split '{split}'. Expected 'train', 'val', or 'test'.")

    res = np.load(res_file)
    gru = np.load(gru_file)
    labels = np.load(lbl_file)

    return res, gru, labels


def get_fusion_dataloaders(
    paths: Optional[PathConfig] = None,
    train_cfg: Optional[TrainConfig] = None,
    num_workers: int = 0,  # 0 is safe cross-platform default (Windows/Linux)
) -> Dict[str, DataLoader]:
    """Create DataLoaders for train, val, and test splits.

    Migrated from Cell 16 of `final-fusion-model.ipynb`.
    """
    if paths is None:
        paths = DEFAULT_PATH_CONFIG
    if train_cfg is None:
        train_cfg = DEFAULT_TRAIN_CONFIG

    res_tr, gru_tr, y_tr = load_fusion_data("train", paths)
    res_va, gru_va, y_va = load_fusion_data("val", paths)
    res_te, gru_te, y_te = load_fusion_data("test", paths)

    train_ds = FusionDataset(res_tr, gru_tr, y_tr)
    val_ds = FusionDataset(res_va, gru_va, y_va)
    test_ds = FusionDataset(res_te, gru_te, y_te)

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg.batch_size_train,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg.batch_size_val,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=train_cfg.batch_size_test,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
    }
