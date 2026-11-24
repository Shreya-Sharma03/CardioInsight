"""Adaptive Reliability Fusion Architecture for Multi-Modal ECG Prediction

Extracted directly from `final-fusion-model.ipynb` (Cells 18-19).
Preserves the exact layer dimensions, parameter names, dropout rates,
multi-head attention configuration, residual MLP, and 4-stage classifier head.

Total model parameters: 2,514,819 (100% trainable).
Matches state_dict keys of `best_fusion_model.zip` and `FusionModel.zip` with zero missing/unexpected keys.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveReliabilityFusion(nn.Module):
    """Two-stream reliability-weighted multi-modal fusion module.

    1. Projects ResNet (512-dim) and BiGRU (256-dim) features into a shared 512-dim latent space.
    2. Uses non-linear confidence estimators to compute sample-specific reliability scalars (alpha_res, alpha_gru).
    3. Normalizes weights across modalities via Softmax: alpha_res + alpha_gru == 1.0.
    4. Formulates scaled tokens and applies 8-head self-attention.
    5. Mean-pools and refines through a residual feed-forward block with LayerNorm and Dropout.
    """

    def __init__(
        self,
        resnet_dim: int = 512,
        bigru_dim: int = 256,
        embed_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.30,
    ):
        super().__init__()
        self.res_proj = nn.Linear(resnet_dim, embed_dim)
        self.gru_proj = nn.Linear(bigru_dim, embed_dim)

        self.res_conf = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

        self.gru_conf = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm = nn.LayerNorm(embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, res: torch.Tensor, gru: torch.Tensor):
        # 1. Project to common embedding space
        res = self.res_proj(res)
        gru = self.gru_proj(gru)

        # 2. Compute modality confidence scores
        c_res = self.res_conf(res)
        c_gru = self.gru_conf(gru)

        # 3. Dynamic Softmax normalization across modalities
        weights = F.softmax(torch.cat([c_res, c_gru], dim=1), dim=1)
        alpha_res = weights[:, 0:1]
        alpha_gru = weights[:, 1:2]

        # 4. Scale embeddings by dynamic reliability
        res = res * alpha_res
        gru = gru * alpha_gru

        # 5. Formulate sequence of 2 tokens and apply cross-modal self-attention
        tokens = torch.stack([res, gru], dim=1)  # (Batch, 2, 512)
        fused, _ = self.attn(tokens, tokens, tokens)

        # 6. Mean pool tokens across modality dimension
        fused = fused.mean(dim=1)  # (Batch, 512)

        # 7. Residual refinement block + LayerNorm + Dropout
        fused = self.norm(fused + self.ffn(fused))
        fused = self.dropout(fused)

        return fused, alpha_res, alpha_gru


class ECGFusionClassifier(nn.Module):
    """Complete ECG Multi-Modal Fusion Classifier.

    Combines AdaptiveReliabilityFusion with a 4-stage regularized MLP classification head:
    512 -> 512 -> 256 -> 64 -> 1.
    """

    def __init__(
        self,
        resnet_dim: int = 512,
        bigru_dim: int = 256,
        embed_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.30,
    ):
        super().__init__()
        self.fusion = AdaptiveReliabilityFusion(
            resnet_dim=resnet_dim,
            bigru_dim=bigru_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.30),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, res: torch.Tensor, gru: torch.Tensor):
        """Forward pass.

        Args:
            res: Tensor of shape (Batch_Size, 512) containing ResNet features.
            gru: Tensor of shape (Batch_Size, 256) containing BiGRU features.

        Returns:
            logits: Tensor of shape (Batch_Size,) with uncalibrated logits.
            fused: Tensor of shape (Batch_Size, 512) with fused representations.
            alpha_res: Tensor of shape (Batch_Size, 1) with ResNet reliability weights.
            alpha_gru: Tensor of shape (Batch_Size, 1) with BiGRU reliability weights.
        """
        fused, alpha_res, alpha_gru = self.fusion(res, gru)
        logits = self.classifier(fused).squeeze(1)
        return logits, fused, alpha_res, alpha_gru


if __name__ == "__main__":
    model = ECGFusionClassifier()
    r = torch.randn(4, 512)
    g = torch.randn(4, 256)
    logits, fused, a_r, a_g = model(r, g)
    print("ECGFusionClassifier sanity test:")
    print("  logits shape   :", logits.shape)
    print("  fused shape    :", fused.shape)
    print("  alpha_res shape:", a_r.shape)
    print("  alpha_gru shape:", a_g.shape)
    print("  total params   :", sum(p.numel() for p in model.parameters()))
