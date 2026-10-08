"""
ResNet34-1D Architecture for 12-lead ECG Signal Analysis.

Extracts localized morphological patterns and waveform features (e.g., QRS complex,
ST segment, T wave dynamics) across 12 ECG channels.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def init_weights(module: nn.Module) -> None:
    """Kaiming normal initialization for conv, Xavier uniform for linear, constants for BN."""
    for m in module.modules():
        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class BasicBlock(nn.Module):
    """Basic 1D Residual Block with 2 Conv1d layers, BatchNorm1d, and shortcut connection."""
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class ResNet34_1D(nn.Module):
    """
    ResNet-34 1D Convolutional Neural Network for multi-lead ECG classification.

    Parameters
    ----------
    block : nn.Module
        Residual block class (default BasicBlock).
    in_channels : int
        Number of input leads/channels (default 12 for 12-lead ECG).

    Returns
    -------
    logit : torch.Tensor
        Classification logit of shape (B,).
    hidden_features : torch.Tensor
        Latent feature representation before the classification head of shape (B, 512).
    """

    def __init__(self, block=BasicBlock, in_channels: int = 12):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv1d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, 3)
        self.layer2 = self._make_layer(block, 128, 4, stride=2)
        self.layer3 = self._make_layer(block, 256, 6, stride=2)
        self.layer4 = self._make_layer(block, 512, 3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, 1)  # Single logit for binary risk classification
        self.feature_dim = 512       # Exposed feature dimension for fusion

        init_weights(self)

    def _make_layer(self, block, out_channels: int, blocks: int, stride: int = 1) -> nn.Sequential:
        layers = [block(self.in_channels, out_channels, stride)]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        """
        Forward pass.

        Input: (B, 12, T)
        Output:
            logit: (B,)
            hidden_features: (B, 512)
        """
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        hidden_features = torch.flatten(x, 1)  # (B, 512) pooled representation
        logit = self.fc(hidden_features).squeeze(-1)  # (B,) raw classification logit
        return logit, hidden_features


if __name__ == "__main__":
    model = ResNet34_1D()
    dummy = torch.randn(2, 12, 2500)
    logit, feat = model(dummy)
    print("ResNet34-1D initialized successfully.")
    print(f"Logit shape: {logit.shape}, Feature shape: {feat.shape}")
