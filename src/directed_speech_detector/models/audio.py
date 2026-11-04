"""Audio branch: strided conv stem + transformer encoder over log-mel frames."""

from __future__ import annotations

import torch
from torch import nn

from directed_speech_detector.models.layers import (
    AttentionPool,
    Normalizer,
    TransformerBlock,
    additive_mask,
    sinusoidal_positions,
)


class AudioEncoder(nn.Module):
    def __init__(
        self, n_mels: int, dim: int, n_heads: int, n_layers: int, ff_mult: int, dropout: float
    ) -> None:
        super().__init__()
        self.norm = Normalizer(n_mels)
        self.conv1 = nn.Conv1d(n_mels, dim, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv1d(dim, dim, kernel_size=5, stride=2, padding=2)
        self.act = nn.GELU()
        self.blocks = nn.ModuleList(TransformerBlock(dim, n_heads, ff_mult, dropout) for _ in range(n_layers))
        self.final_norm = nn.LayerNorm(dim)
        self.pool = AttentionPool(dim)

    def forward(self, feats: torch.Tensor, feat_mask: torch.Tensor) -> torch.Tensor:
        """feats (B, T, n_mels), feat_mask (B, T) -> embedding (B, dim)."""
        # Padded frames are zeroed before every convolution so that results do not depend on
        # how much padding a batch happens to contain.
        x = (self.norm(feats) * feat_mask.unsqueeze(-1)).transpose(1, 2)
        mask1 = feat_mask[:, ::2]
        x = self.act(self.conv1(x)) * mask1.unsqueeze(1)
        mask = mask1[:, ::2]
        x = (self.act(self.conv2(x)) * mask.unsqueeze(1)).transpose(1, 2)  # (B, ceil(T/4), D)
        x = x + sinusoidal_positions(x.shape[1], x.shape[2], x.device)
        bias = additive_mask(mask)
        for block in self.blocks:
            x = block(x, bias)
        return self.pool(self.final_norm(x), mask)
