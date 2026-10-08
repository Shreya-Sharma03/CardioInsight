"""
Adaptive Reliability-Aware Fusion Architecture.

Dynamically fuses representations learned by the ResNet34-1D morphological branch (512-d)
and the BiGRU temporal branch (256-d) using sample-specific confidence estimation,
dynamic softmax weighting, and multi-head cross-modal attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveReliabilityFusion(nn.Module):
    """
    Computes sample-level confidence scores for both branches, generates dynamic
    modality weights (alpha_res, alpha_gru summing to 1.0), and performs cross-modal
    multi-head attention fusion with residual refinement.

    Parameters
    ----------
    res_dim : int
        Feature dimension from ResNet34-1D (default 512).
    gru_dim : int
        Feature dimension from BiGRU (default 256).
    embed_dim : int
        Unified projection dimension for multimodal attention (default 512).
    heads : int
        Number of attention heads (default 8).
    dropout : float
        Dropout probability (default 0.30).
    """

    def __init__(
        self,
        res_dim: int = 512,
        gru_dim: int = 256,
        embed_dim: int = 512,
        heads: int = 8,
        dropout: float = 0.30,
    ):
        super().__init__()
        # Linear projection to unified embedding space
        self.res_proj = nn.Linear(res_dim, embed_dim)
        self.gru_proj = nn.Linear(gru_dim, embed_dim)

        # Modality reliability confidence estimators
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

        # Multi-head attention across modality tokens
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm = nn.LayerNorm(embed_dim)

        # Feed-Forward Network with residual refinement
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, res: torch.Tensor, gru: torch.Tensor):
        """
        Parameters
        ----------
        res : torch.Tensor of shape (B, 512)
            ResNet34 morphological features.
        gru : torch.Tensor of shape (B, 256)
            BiGRU temporal features.

        Returns
        -------
        fused : torch.Tensor of shape (B, 512)
            Unified multimodal representation.
        alpha_res : torch.Tensor of shape (B, 1)
            Dynamic sample-specific weight for ResNet branch.
        alpha_gru : torch.Tensor of shape (B, 1)
            Dynamic sample-specific weight for BiGRU branch.
        """
        # 1. Project into unified embedding space
        res_emb = self.res_proj(res)
        gru_emb = self.gru_proj(gru)

        # 2. Estimate sample-specific reliability scores
        res_score = self.res_conf(res_emb)  # (B, 1)
        gru_score = self.gru_conf(gru_emb)  # (B, 1)

        # 3. Softmax normalization enforces alpha_res + alpha_gru == 1.0
        weights = torch.softmax(torch.cat([res_score, gru_score], dim=1), dim=1)
        alpha_res = weights[:, 0:1]
        alpha_gru = weights[:, 1:2]

        # 4. Modality weighting
        res_weighted = res_emb * alpha_res
        gru_weighted = gru_emb * alpha_gru

        # 5. Multimodal token sequence (B, 2, embed_dim)
        tokens = torch.stack([res_weighted, gru_weighted], dim=1)

        # 6. Cross-modal multi-head attention
        attn_out, _ = self.attn(tokens, tokens, tokens)
        fused = attn_out.mean(dim=1)  # Pool across modality tokens -> (B, embed_dim)

        # 7. Residual connection + FFN + LayerNorm
        fused = self.norm(fused + self.ffn(fused))
        fused = self.dropout(fused)

        return fused, alpha_res, alpha_gru


class ECGFusionClassifier(nn.Module):
    """
    End-to-End ECG Fusion Classifier.

    Combines AdaptiveReliabilityFusion with a multi-layer deep classifier head.

    Returns
    -------
    logits : torch.Tensor of shape (B,)
        Raw binary classification logit.
    fused : torch.Tensor of shape (B, 512)
        Fused multimodal representation vector.
    alpha_res : torch.Tensor of shape (B, 1)
        ResNet reliability weight.
    alpha_gru : torch.Tensor of shape (B, 1)
        BiGRU reliability weight.
    """

    def __init__(
        self,
        res_dim: int = 512,
        gru_dim: int = 256,
        embed_dim: int = 512,
        heads: int = 8,
        dropout: float = 0.30,
    ):
        super().__init__()
        self.fusion = AdaptiveReliabilityFusion(
            res_dim=res_dim,
            gru_dim=gru_dim,
            embed_dim=embed_dim,
            heads=heads,
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
        fused, alpha_res, alpha_gru = self.fusion(res, gru)
        logits = self.classifier(fused).squeeze(-1)
        return logits, fused, alpha_res, alpha_gru


if __name__ == "__main__":
    model = ECGFusionClassifier()
    dummy_res = torch.randn(4, 512)
    dummy_gru = torch.randn(4, 256)
    logits, fused, a_res, a_gru = model(dummy_res, dummy_gru)
    print("ECGFusionClassifier initialized successfully.")
    print(f"Logits shape: {logits.shape}, Fused shape: {fused.shape}")
    print(f"Alpha Res: {a_res.flatten()[:2]}, Alpha GRU: {a_gru.flatten()[:2]}")
