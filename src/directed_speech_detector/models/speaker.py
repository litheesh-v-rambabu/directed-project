"""Enrolled-speaker branch: compares the utterance's voice statistics with the enrollment profile."""

from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch import nn

from directed_speech_detector.features import speaker_profile
from directed_speech_detector.models.layers import Normalizer


class SpeakerBranch(nn.Module):
    """Shared MLP embeds the utterance and enrollment statistics; their interaction is projected.

    When no enrollment is available (``has_enroll == 0``) a learned null embedding is used instead,
    so the same model serves enrolled and un-enrolled users.
    """

    def __init__(self, n_mels: int, dim: int, dropout: float) -> None:
        super().__init__()
        self.norm = Normalizer(2 * n_mels)
        self.embed = nn.Sequential(
            nn.Linear(2 * n_mels, dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim, dim)
        )
        self.interaction = nn.Sequential(nn.Linear(2 * dim + 1, dim), nn.GELU())
        self.null = nn.Parameter(torch.zeros(dim))

    def forward(
        self,
        feats: torch.Tensor,
        feat_mask: torch.Tensor,
        enroll_profile: torch.Tensor,
        has_enroll: torch.Tensor,
    ) -> torch.Tensor:
        utt = self.embed(self.norm(speaker_profile(feats, feat_mask > 0)))
        enr = self.embed(self.norm(enroll_profile))
        cosine = functional.cosine_similarity(utt, enr, dim=-1).unsqueeze(-1)
        joint = self.interaction(torch.cat([utt * enr, (utt - enr).abs(), cosine], dim=-1))
        gate = has_enroll.unsqueeze(-1)
        return gate * joint + (1.0 - gate) * self.null
