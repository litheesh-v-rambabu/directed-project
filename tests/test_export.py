import numpy as np
import torch

from directed_speech_detector.export import (
    OnnxPredictor,
    export_onnx,
    parity_check,
    quantize_int8,
    state_dict_size_mb,
)
from directed_speech_detector.models.detector import INPUT_NAMES


def test_int8_quantization_changes_modules_and_shrinks(random_model, mini_batch) -> None:
    quantized = quantize_int8(random_model)
    assert state_dict_size_mb(quantized) < state_dict_size_mb(random_model)
    with torch.no_grad():
        ref = random_model(*(mini_batch[n] for n in INPUT_NAMES))
        out = quantized(*(mini_batch[n] for n in INPUT_NAMES))
    assert torch.isfinite(out).all() and np.corrcoef(ref.numpy(), out.numpy())[0, 1] > 0.9


def test_onnx_parity_across_dynamic_shapes(tmp_path, random_model, mini_batch) -> None:
    path = tmp_path / "m.onnx"
    names = export_onnx(random_model, path, mini_batch)
    assert set(names) == set(random_model.required_inputs())
    shorter = {k: (v[:3, :20] if k in ("feats", "feat_mask") else v[:3]) for k, v in mini_batch.items()}
    report = parity_check(random_model, path, [mini_batch, shorter])
    assert report["passed"], report
    assert OnnxPredictor(path)(shorter).shape == (3,)
