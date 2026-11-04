"""Top-level directed-speech detector composed of optional audio / text / speaker branches."""

from __future__ import annotations

import torch
from torch import nn

from directed_speech_detector.config import ModelConfig
from directed_speech_detector.models.audio import AudioEncoder
from directed_speech_detector.models.fusion import GatedFusion
from directed_speech_detector.models.speaker import SpeakerBranch
from directed_speech_detector.models.text import TextEncoder

INPUT_NAMES = ("feats", "feat_mask", "tokens", "enroll_profile", "has_enroll")


class DirectedSpeechDetector(nn.Module):
    """Returns one logit per utterance; positive means *directed at the assistant*."""

    def __init__(self, cfg: ModelConfig, vocab_size: int, n_mels: int) -> None:
        super().__init__()
        self.modalities = list(cfg.modalities)
        d = cfg.d_model
        self.audio = (
            AudioEncoder(n_mels, d, cfg.n_heads, cfg.audio_layers, cfg.ff_mult, cfg.dropout)
            if "audio" in self.modalities
            else None
        )
        self.text = (
            TextEncoder(vocab_size, d, cfg.n_heads, cfg.text_layers, cfg.ff_mult, cfg.dropout)
            if "text" in self.modalities
            else None
        )
        self.speaker = SpeakerBranch(n_mels, d, cfg.dropout) if "speaker" in self.modalities else None
        self.fusion = GatedFusion(self.modalities, d, cfg.dropout, cfg.modality_dropout)

    def required_inputs(self) -> list[str]:
        """Names of the forward() inputs this configuration actually consumes."""
        needed = set()
        if "audio" in self.modalities:
            needed |= {"feats", "feat_mask"}
        if "text" in self.modalities:
            needed.add("tokens")
        if "speaker" in self.modalities:
            needed |= {"feats", "feat_mask", "enroll_profile", "has_enroll"}
        return [n for n in INPUT_NAMES if n in needed]

    def forward(
        self,
        feats: torch.Tensor,
        feat_mask: torch.Tensor,
        tokens: torch.Tensor,
        enroll_profile: torch.Tensor,
        has_enroll: torch.Tensor,
    ) -> torch.Tensor:
        emb: dict[str, torch.Tensor] = {}
        if self.audio is not None:
            emb["audio"] = self.audio(feats, feat_mask)
        if self.text is not None:
            emb["text"] = self.text(tokens)
        if self.speaker is not None:
            emb["speaker"] = self.speaker(feats, feat_mask, enroll_profile, has_enroll)
        return self.fusion(emb)

    def set_feature_stats(self, stats: dict[str, torch.Tensor]) -> None:
        if self.audio is not None:
            self.audio.norm.set_stats(stats["mel_mean"], stats["mel_std"])
        if self.speaker is not None:
            self.speaker.norm.set_stats(stats["profile_mean"], stats["profile_std"])

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
