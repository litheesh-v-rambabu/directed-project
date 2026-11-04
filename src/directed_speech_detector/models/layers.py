"""Building blocks: pre-LN transformer encoder blocks, attention pooling, normalizers."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as functional
from torch import nn

MASK_FILL = -1e4


def additive_mask(mask: torch.Tensor) -> torch.Tensor:
    """(B, T) mask with 1 = valid -> additive attention bias (B, 1, 1, T)."""
    return ((1.0 - mask) * MASK_FILL).unsqueeze(1).unsqueeze(1)


def sinusoidal_positions(length: int, dim: int, device: torch.device) -> torch.Tensor:
    pos = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, dim, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / dim))
    pe = torch.zeros(length, dim, device=device)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class SelfAttention(nn.Module):
    """Multi-head self-attention built from plain ``nn.Linear`` layers.

    Using plain linears (rather than ``nn.MultiheadAttention``) keeps dynamic INT8 quantization
    and ONNX export straightforward.
    """

    def __init__(self, dim: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.n_heads, self.dropout = n_heads, dropout
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        qkv = self.qkv(x).view(b, t, 3, self.n_heads, d // self.n_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        out = functional.scaled_dot_product_attention(
            q, k, v, attn_mask=bias, dropout_p=self.dropout if self.training else 0.0
        )
        return self.proj(out.transpose(1, 2).reshape(b, t, d))


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, ff_mult: int, dropout: float) -> None:
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = SelfAttention(dim, n_heads, dropout)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_mult * dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(ff_mult * dim, dim)
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        x = x + self.drop(self.attn(self.norm1(x), bias))
        return x + self.drop(self.ff(self.norm2(x)))


class AttentionPool(nn.Module):
    """Learned masked attention pooling over time: (B, T, D) -> (B, D)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        logits = self.score(x).squeeze(-1) + (1.0 - mask) * MASK_FILL
        weights = torch.softmax(logits, dim=-1).unsqueeze(-1)
        return (weights * x).sum(dim=1)


class Normalizer(nn.Module):
    """Fixed per-dimension standardisation whose statistics are fit on the training split."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.mean: torch.Tensor
        self.std: torch.Tensor
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("std", torch.ones(dim))

    def set_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.mean.copy_(mean)
        self.std.copy_(std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std
