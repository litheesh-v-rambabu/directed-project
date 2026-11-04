"""Gated late fusion with modality dropout."""

from __future__ import annotations

import torch
from torch import nn


class GatedFusion(nn.Module):
    """Concatenate per-modality embeddings, apply a GLU block, and emit one logit.

    During training each modality embedding is independently replaced by a learned null vector with
    probability ``modality_dropout`` (at least one modality is always kept), which regularises the
    model and makes it robust to a missing modality.
    """

    def __init__(self, names: list[str], dim: int, dropout: float, modality_dropout: float) -> None:
        super().__init__()
        self.names, self.modality_dropout = names, modality_dropout
        self.nulls = nn.ParameterDict({n: nn.Parameter(torch.zeros(dim)) for n in names})
        self.glu = nn.Sequential(nn.Linear(len(names) * dim, 2 * dim), nn.GLU(dim=-1), nn.Dropout(dropout))
        self.head = nn.Linear(dim, 1)

    def _drop_mask(self, batch: int, device: torch.device) -> torch.Tensor:
        keep = (torch.rand(batch, len(self.names), device=device) >= self.modality_dropout).float()
        none_kept = keep.sum(dim=1) == 0
        forced = torch.randint(len(self.names), (batch,), device=device)
        keep[none_kept, forced[none_kept]] = 1.0
        return keep

    def forward(self, embeddings: dict[str, torch.Tensor]) -> torch.Tensor:
        parts = [embeddings[n] for n in self.names]
        if self.training and self.modality_dropout > 0 and len(parts) > 1:
            keep = self._drop_mask(parts[0].shape[0], parts[0].device)
            parts = [
                keep[:, i : i + 1] * p + (1 - keep[:, i : i + 1]) * self.nulls[n]
                for i, (n, p) in enumerate(zip(self.names, parts, strict=True))
            ]
        return self.head(self.glu(torch.cat(parts, dim=-1))).squeeze(-1)
