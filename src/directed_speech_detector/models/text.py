"""Text branch: token embeddings + transformer encoder over the (ASR) transcript."""

from __future__ import annotations

import torch
from torch import nn

from directed_speech_detector.models.layers import (
    AttentionPool,
    TransformerBlock,
    additive_mask,
    sinusoidal_positions,
)


class TextEncoder(nn.Module):
    def __init__(
        self, vocab_size: int, dim: int, n_heads: int, n_layers: int, ff_mult: int, dropout: float
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.blocks = nn.ModuleList(TransformerBlock(dim, n_heads, ff_mult, dropout) for _ in range(n_layers))
        self.final_norm = nn.LayerNorm(dim)
        self.pool = AttentionPool(dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens (B, L) with 0 = padding -> embedding (B, dim)."""
        mask = (tokens != 0).to(torch.float32)
        x = self.embed(tokens) + sinusoidal_positions(
            tokens.shape[1], self.embed.embedding_dim, tokens.device
        )
        bias = additive_mask(mask)
        for block in self.blocks:
            x = block(x, bias)
        return self.pool(self.final_norm(x), mask)
