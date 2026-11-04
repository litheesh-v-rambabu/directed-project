"""Inference wrapper shared by the API and tests (FP32 / dynamic-INT8 / ONNX backends)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from directed_speech_detector.config import ExperimentConfig
from directed_speech_detector.data.text import Vocabulary
from directed_speech_detector.evaluation.metrics import sigmoid
from directed_speech_detector.export import OnnxPredictor, quantize_int8
from directed_speech_detector.features import LogMelExtractor, speaker_profile
from directed_speech_detector.models.detector import INPUT_NAMES, DirectedSpeechDetector
from directed_speech_detector.training.checkpoint import load_checkpoint
from directed_speech_detector.utils import Timer


@dataclass(frozen=True)
class Prediction:
    probability: float
    is_directed: bool
    threshold: float
    enrollment_used: bool
    feature_ms: float
    model_ms: float


class Detector:
    """Turns (audio, transcript, optional enrollment audio) into a calibrated directed-speech decision."""

    def __init__(
        self,
        model: DirectedSpeechDetector,
        cfg: ExperimentConfig,
        vocab: Vocabulary,
        temperature: float = 1.0,
        threshold: float = 0.5,
        backend: str = "fp32",
        onnx_path: Path | None = None,
    ) -> None:
        self.cfg, self.vocab, self.temperature, self.threshold, self.backend = (
            cfg,
            vocab,
            temperature,
            threshold,
            backend,
        )
        self.modalities = list(cfg.model.modalities)
        self.n_parameters = model.num_parameters()
        self.extractor = LogMelExtractor(cfg.features, cfg.data.sample_rate)
        self._onnx = OnnxPredictor(onnx_path) if backend == "onnx" and onnx_path else None
        if backend == "onnx" and self._onnx is None:
            raise ValueError("backend 'onnx' requires an exported model (run `dsd export`)")
        self._model = quantize_int8(model) if backend == "int8" else model.eval()

    @classmethod
    def from_run_dir(cls, run_dir: str | Path, backend: str = "fp32") -> Detector:
        run = Path(run_dir)
        model, cfg, vocab = load_checkpoint(run / "model.pt")
        calib_path = run / "calibration.json"
        calib = json.loads(calib_path.read_text()) if calib_path.exists() else {}
        return cls(
            model,
            cfg,
            vocab,
            calib.get("temperature", 1.0),
            calib.get("threshold", 0.5),
            backend,
            run / "export" / "model.onnx",
        )

    def _features(self, audio: np.ndarray) -> torch.Tensor:
        return self.extractor(torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32)).unsqueeze(0))

    def predict(
        self, audio: np.ndarray, transcript: str, enrollment_audio: np.ndarray | None = None
    ) -> Prediction:
        with Timer() as feature_timer:
            feats = self._features(audio)
            mask = torch.ones(feats.shape[:2])
            use_enrollment = enrollment_audio is not None and "speaker" in self.modalities
            if use_enrollment:
                assert enrollment_audio is not None
                enr_feats = self._features(enrollment_audio)
                profile = speaker_profile(enr_feats, torch.ones(enr_feats.shape[:2], dtype=torch.bool))
            else:
                profile = torch.zeros(1, 2 * self.cfg.features.n_mels)
            tokens = torch.tensor([self.vocab.encode(transcript, self.cfg.model.max_text_len)])
        batch = {
            "feats": feats,
            "feat_mask": mask,
            "tokens": tokens,
            "enroll_profile": profile,
            "has_enroll": torch.tensor([1.0 if use_enrollment else 0.0]),
        }
        with Timer() as model_timer:
            if self._onnx is not None:
                logit = float(self._onnx(batch)[0])
            else:
                with torch.inference_mode():
                    logit = float(self._model(*(batch[n] for n in INPUT_NAMES))[0])
        prob = float(sigmoid(np.array(logit / self.temperature)))
        return Prediction(
            prob,
            prob > self.threshold,
            self.threshold,
            bool(use_enrollment),
            feature_timer.elapsed_ms,
            model_timer.elapsed_ms,
        )

    def info(self) -> dict[str, object]:
        return {
            "name": self.cfg.name,
            "modalities": self.modalities,
            "backend": self.backend,
            "n_parameters": self.n_parameters,
            "threshold": self.threshold,
            "temperature": self.temperature,
        }
