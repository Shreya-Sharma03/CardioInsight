"""Stacked Bidirectional GRU with Temporal Attention for ECG Sequence Analysis

Migrated directly from `bigru_model.py` and notebook Cells 10-13.
Preserves the exact recurrent layer configuration, bidirectional processing,
residual projections, layer normalization, additive attention, and classification head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def init_weights(module: nn.Module):
    """Orthogonal init for recurrent weights, Xavier for linear, zero biases."""
    for name, param in module.named_parameters():
        if "weight_ih" in name:
            nn.init.xavier_uniform_(param.data)
        elif "weight_hh" in name:
            nn.init.orthogonal_(param.data)
        elif "bias" in name:
            param.data.fill_(0)
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            module.bias.data.fill_(0)


class ResidualBiGRULayer(nn.Module):
    """Single bidirectional GRU layer wrapped with LayerNorm + residual projection."""

    def __init__(self, input_size, hidden_size, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True, bidirectional=True)
        out_size = hidden_size * 2
        self.norm = nn.LayerNorm(out_size)
        self.dropout = nn.Dropout(dropout)
        self.residual_proj = (
            nn.Linear(input_size, out_size) if input_size != out_size else nn.Identity()
        )
        init_weights(self)

    def forward(self, x, lengths=None):
        if lengths is not None:
            packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            packed_out, h_n = self.gru(packed)
            out, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=x.size(1))
        else:
            out, h_n = self.gru(x)
        residual = self.residual_proj(x)
        out = self.norm(out + residual)
        out = self.dropout(out)
        return out, h_n


class StackedBiGRUEncoder(nn.Module):
    """Stacks several ResidualBiGRULayer blocks with input BatchNorm."""

    def __init__(self, input_channels=12, hidden_size=128, num_layers=3, dropout=0.2):
        super().__init__()
        self.input_bn = nn.BatchNorm1d(input_channels)
        layers = []
        in_size = input_channels
        for i in range(num_layers):
            layers.append(ResidualBiGRULayer(in_size, hidden_size, dropout=dropout))
            in_size = hidden_size * 2
        self.layers = nn.ModuleList(layers)
        self.output_dim = in_size

    def forward(self, x, lengths=None):
        # x: (B, 12, T) -> BatchNorm over channels -> (B, T, 12) for GRU
        x = self.input_bn(x)
        x = x.transpose(1, 2)  # (B, T, C)
        h_n_last = None
        for layer in self.layers:
            x, h_n_last = layer(x, lengths)
        return x, h_n_last  # x: (B, T, hidden*2)


class TemporalAttention(nn.Module):
    """Additive (Bahdanau-style) attention over the GRU's per-timestep outputs."""

    def __init__(self, hidden_dim, attn_dim=128):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, attn_dim)
        self.score = nn.Linear(attn_dim, 1, bias=False)
        init_weights(self)

    def forward(self, gru_out, mask=None):
        # gru_out: (B, T, H)
        energy = torch.tanh(self.proj(gru_out))          # (B, T, A)
        scores = self.score(energy).squeeze(-1)          # (B, T)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        weights = F.softmax(scores, dim=1)                # (B, T)
        context = torch.bmm(weights.unsqueeze(1), gru_out).squeeze(1)  # (B, H)
        return context, weights


class ResidualFCBlock(nn.Module):
    def __init__(self, dim, dropout=0.3):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        init_weights(self)

    def forward(self, x):
        residual = x
        x = F.gelu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return self.norm(x + residual)


class ClassificationHead(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden_dim)
        self.res_block = ResidualFCBlock(hidden_dim, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_dim, 1)
        init_weights(self)

    def forward(self, x):
        x = F.gelu(self.in_proj(x))
        x = self.res_block(x)
        x = self.dropout(x)
        logit = self.out(x)
        return logit.squeeze(-1)  # raw logit; sigmoid applied at inference/loss time


class BiGRUECGClassifier(nn.Module):
    """Complete BiGRU ECG Classifier with temporal attention and classification head."""

    def __init__(
        self,
        input_channels=12,
        hidden_size=128,
        num_gru_layers=3,
        attn_dim=128,
        fc_hidden=128,
        dropout=0.3,
    ):
        super().__init__()
        self.encoder = StackedBiGRUEncoder(input_channels, hidden_size, num_gru_layers, dropout)
        self.attention = TemporalAttention(self.encoder.output_dim, attn_dim)
        self.head = ClassificationHead(self.encoder.output_dim, fc_hidden, dropout)
        self.feature_dim = self.encoder.output_dim  # 256 dimensions

    def forward(self, x, lengths=None, mask=None, return_attention=False):
        gru_out, _ = self.encoder(x, lengths)                    # (B, T, H)
        hidden_features, attn_w = self.attention(gru_out, mask)  # (B, H) - pooled vector before head
        logit = self.head(hidden_features)
        if return_attention:
            return logit, hidden_features, attn_w
        return logit, hidden_features


if __name__ == "__main__":
    model = BiGRUECGClassifier()
    dummy = torch.randn(2, 12, 1000)
    logit, feats = model(dummy)
    print("BiGRUECGClassifier test:")
    print("  logit shape:", logit.shape)
    print("  feats shape:", feats.shape)
