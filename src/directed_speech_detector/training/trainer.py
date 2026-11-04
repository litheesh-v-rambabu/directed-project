"""Training loop: AdamW + warmup/cosine schedule, early stopping on validation AUC."""

from __future__ import annotations

import copy
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader

from directed_speech_detector.config import ExperimentConfig, save_config
from directed_speech_detector.data.dataset import (
    DirectedSpeechDataset,
    SplitArrays,
    fit_feature_stats,
    get_split,
    make_loader,
)
from directed_speech_detector.data.text import Vocabulary
from directed_speech_detector.evaluation.metrics import auc_score
from directed_speech_detector.models.detector import DirectedSpeechDetector
from directed_speech_detector.training.checkpoint import build_model, save_checkpoint
from directed_speech_detector.utils import get_logger, resolve_device, seed_everything

LOGGER = get_logger(__name__)


@dataclass
class TrainResult:
    run_dir: Path
    best_epoch: int
    best_val_auc: float
    n_parameters: int
    train_seconds: float
    history: list[dict[str, Any]] = field(default_factory=list)


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def forward_batch(model: DirectedSpeechDetector, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    return model(
        batch["feats"], batch["feat_mask"], batch["tokens"], batch["enroll_profile"], batch["has_enroll"]
    )


@torch.no_grad()
def predict_logits(
    model: DirectedSpeechDetector,
    loader: DataLoader[dict[str, Any]],
    device: torch.device,
    force_no_enrollment: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (logits, labels) for every item in ``loader`` in dataset order."""
    model.eval()
    logits, labels = [], []
    for batch in loader:
        batch = move_batch(batch, device)
        if force_no_enrollment:
            batch["has_enroll"] = torch.zeros_like(batch["has_enroll"])
        logits.append(forward_batch(model, batch).cpu())
        labels.append(batch["label"].cpu())
    return torch.cat(logits).numpy(), torch.cat(labels).numpy()


def _lr_at(step: int, total: int, warmup: int, base_lr: float) -> float:
    if step < warmup:
        return base_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def prepare_data(cfg: ExperimentConfig) -> tuple[SplitArrays, SplitArrays, Vocabulary]:
    train, val = get_split(cfg, "train"), get_split(cfg, "val")
    return train, val, Vocabulary.build(train.texts)


def train_model(cfg: ExperimentConfig, run_dir: Path | None = None) -> TrainResult:
    """Train one configuration and write ``model.pt``, ``config.yaml`` and ``history.json``."""
    tc = cfg.train
    run_dir = run_dir or Path(cfg.output_dir) / cfg.name
    run_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(tc.seed)
    device = resolve_device(tc.device)

    train_split, val_split, vocab = prepare_data(cfg)
    train_ds = DirectedSpeechDataset(
        train_split, vocab, cfg.model.max_text_len, tc.word_dropout, tc.enrollment_dropout, tc.seed
    )
    val_ds = DirectedSpeechDataset(val_split, vocab, cfg.model.max_text_len)
    train_loader = make_loader(train_ds, tc.batch_size, shuffle=True, seed=tc.seed)
    val_loader = make_loader(val_ds, tc.batch_size * 2, shuffle=False)

    model = build_model(cfg, len(vocab))
    model.set_feature_stats(fit_feature_stats(train_split))
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=tc.lr, weight_decay=tc.weight_decay)
    total_steps = tc.epochs * len(train_loader)
    warmup = max(1, int(tc.warmup_frac * total_steps))

    best_auc, best_epoch, best_state = -1.0, 0, copy.deepcopy(model.state_dict())
    history: list[dict[str, Any]] = []
    step, started = 0, time.perf_counter()
    for epoch in range(1, tc.epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            for group in optimizer.param_groups:
                group["lr"] = _lr_at(step, total_steps, warmup, tc.lr)
            batch = move_batch(batch, device)
            loss = functional.binary_cross_entropy_with_logits(forward_batch(model, batch), batch["label"])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
            optimizer.step()
            losses.append(loss.item())
            step += 1
        val_logits, val_labels = predict_logits(model, val_loader, device)
        val_auc = auc_score(val_logits, val_labels)
        val_acc = float(((val_logits > 0) == (val_labels > 0.5)).mean())
        history.append(
            {"epoch": epoch, "train_loss": float(np.mean(losses)), "val_auc": val_auc, "val_acc": val_acc}
        )
        LOGGER.info(
            "[%s] epoch %d loss=%.4f val_auc=%.4f val_acc=%.4f",
            cfg.name,
            epoch,
            history[-1]["train_loss"],
            val_auc,
            val_acc,
        )
        if val_auc > best_auc:
            best_auc, best_epoch, best_state = val_auc, epoch, copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= tc.patience:
            LOGGER.info("[%s] early stopping at epoch %d", cfg.name, epoch)
            break

    model.load_state_dict(best_state)
    model.cpu()
    save_checkpoint(run_dir / "model.pt", model, cfg, vocab)
    save_config(cfg, run_dir / "config.yaml")
    elapsed = time.perf_counter() - started
    result = TrainResult(run_dir, best_epoch, best_auc, model.num_parameters(), elapsed, history)
    (run_dir / "history.json").write_text(
        json.dumps(
            {
                "best_epoch": best_epoch,
                "best_val_auc": best_auc,
                "n_parameters": result.n_parameters,
                "train_seconds": elapsed,
                "history": history,
            },
            indent=2,
        )
    )
    return result
