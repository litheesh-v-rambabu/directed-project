"""Model optimization: dynamic INT8 quantization and ONNX export with a parity check."""

from __future__ import annotations

import copy
import io
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from directed_speech_detector.data.dataset import DirectedSpeechDataset, collate_batch, get_split
from directed_speech_detector.models.detector import INPUT_NAMES, DirectedSpeechDetector
from directed_speech_detector.training.checkpoint import load_checkpoint

ONNX_OPSET = 17
PARITY_TOLERANCE = 1e-3


def quantize_int8(model: DirectedSpeechDetector) -> DirectedSpeechDetector:
    """Dynamic INT8 quantization of every ``nn.Linear`` (weights int8, activations quantized on the fly)."""
    quantized = torch.ao.quantization.quantize_dynamic(
        copy.deepcopy(model).eval(), {nn.Linear}, dtype=torch.qint8
    )
    return quantized


def state_dict_size_mb(model: nn.Module) -> float:
    """Serialized size of the model's state dict in MB."""
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.tell() / 1e6


class _OnnxWrapper(nn.Module):
    """Exposes only the inputs a configuration consumes (unused inputs would be pruned by ONNX)."""

    def __init__(self, model: DirectedSpeechDetector) -> None:
        super().__init__()
        self.model = model
        self.names = model.required_inputs()

    def forward(self, *args: torch.Tensor) -> torch.Tensor:
        given = dict(zip(self.names, args, strict=True))
        ref = args[0]
        batch = ref.shape[0]
        n_mels2 = 2 * (given["feats"].shape[-1] if "feats" in given else 1)
        dummy: dict[str, torch.Tensor] = {
            "feats": torch.zeros(batch, 4, 1),
            "feat_mask": torch.ones(batch, 4),
            "tokens": torch.ones(batch, 1, dtype=torch.long),
            "enroll_profile": torch.zeros(batch, n_mels2),
            "has_enroll": torch.zeros(batch),
        }
        dummy.update(given)
        return self.model(*(dummy[n] for n in INPUT_NAMES))


def example_batch(model: DirectedSpeechDetector, run_dir: Path, n: int = 16) -> dict[str, torch.Tensor]:
    _, cfg, vocab = load_checkpoint(run_dir / "model.pt")
    split = get_split(cfg, "val").subset(n)
    ds = DirectedSpeechDataset(split, vocab, cfg.model.max_text_len)
    return collate_batch([ds[i] for i in range(len(ds))])


def export_onnx(model: DirectedSpeechDetector, path: Path, batch: dict[str, torch.Tensor]) -> list[str]:
    """Export ``model`` to ONNX with dynamic batch/time axes. Returns the ONNX input names."""
    model.eval()
    wrapper = _OnnxWrapper(model).eval()
    names = wrapper.names
    axes = {
        "feats": {0: "batch", 1: "time"},
        "feat_mask": {0: "batch", 1: "time"},
        "tokens": {0: "batch", 1: "length"},
        "enroll_profile": {0: "batch"},
        "has_enroll": {0: "batch"},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        tuple(batch[n] for n in names),
        str(path),
        input_names=names,
        output_names=["logit"],
        dynamic_axes={**{n: axes[n] for n in names}, "logit": {0: "batch"}},
        opset_version=ONNX_OPSET,
        dynamo=False,
    )
    model.eval()  # torch.onnx.export restores the wrapper's original (train) mode on exit
    return names


class OnnxPredictor:
    """ONNX Runtime wrapper that accepts the same batch dict as the PyTorch model."""

    def __init__(self, path: Path, num_threads: int = 1) -> None:
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = num_threads
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
        self.input_names = [i.name for i in self.session.get_inputs()]

    def __call__(self, batch: dict[str, torch.Tensor]) -> np.ndarray:
        feed = {n: batch[n].numpy() for n in self.input_names}
        out: np.ndarray = self.session.run(None, feed)[0]
        return out


def parity_check(
    model: DirectedSpeechDetector, onnx_path: Path, batches: list[dict[str, torch.Tensor]]
) -> dict[str, Any]:
    """Max |PyTorch - ONNX Runtime| logit difference over several batch shapes."""
    model.eval()
    predictor = OnnxPredictor(onnx_path)
    worst = 0.0
    with torch.no_grad():
        for batch in batches:
            ref = model(*(batch[n] for n in INPUT_NAMES)).numpy()
            worst = max(worst, float(np.abs(ref - predictor(batch)).max()))
    return {
        "max_abs_logit_diff": worst,
        "tolerance": PARITY_TOLERANCE,
        "passed": worst < PARITY_TOLERANCE,
        "n_batches": len(batches),
        "batch_sizes": [int(b["feats"].shape[0]) for b in batches],
    }


def export_all(run_dir: Path) -> dict[str, Any]:
    """Create ``run_dir/export/model.onnx`` plus a parity report; also report FP32/INT8 sizes."""
    model, _, _ = load_checkpoint(run_dir / "model.pt")
    full = example_batch(model, run_dir, n=16)
    onnx_path = run_dir / "export" / "model.onnx"
    export_onnx(model, onnx_path, full)
    single = {k: v[:1] for k, v in full.items()}
    parity = parity_check(model, onnx_path, [full, single, {k: v[:5] for k, v in full.items()}])
    report = {
        "onnx_path": str(onnx_path),
        "opset": ONNX_OPSET,
        "parity": parity,
        "size_mb": {
            "fp32_state_dict": state_dict_size_mb(model),
            "int8_state_dict": state_dict_size_mb(quantize_int8(model)),
            "onnx": onnx_path.stat().st_size / 1e6,
        },
    }
    (run_dir / "export" / "export_report.json").write_text(json.dumps(report, indent=2))
    if not parity["passed"]:
        raise RuntimeError(f"ONNX parity check failed: {parity}")
    return report


Backend = Callable[[dict[str, torch.Tensor]], np.ndarray]
