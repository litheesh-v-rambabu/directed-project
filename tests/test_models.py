import pytest
import torch

from directed_speech_detector.config import ModelConfig
from directed_speech_detector.models.detector import INPUT_NAMES, DirectedSpeechDetector
from directed_speech_detector.models.fusion import GatedFusion


def _forward(model, batch):
    return model(*(batch[n] for n in INPUT_NAMES))


@pytest.mark.parametrize("modalities", [["audio"], ["text"], ["audio", "text"], ["audio", "text", "speaker"]])
def test_forward_shape_for_each_ablation(modalities, mini_batch) -> None:
    cfg = ModelConfig(modalities=modalities, d_model=32, audio_layers=1, text_layers=1)
    model = DirectedSpeechDetector(cfg, vocab_size=300, n_mels=40).eval()
    assert _forward(model, mini_batch).shape == (8,)


def test_required_inputs_per_modality() -> None:
    def names(m):
        return DirectedSpeechDetector(ModelConfig(modalities=m, d_model=32), 50, 40).required_inputs()

    assert names(["text"]) == ["tokens"]
    assert names(["audio"]) == ["feats", "feat_mask"]
    assert set(names(["speaker"])) == {"feats", "feat_mask", "enroll_profile", "has_enroll"}


def test_padding_does_not_change_output(random_model, mini_batch) -> None:
    single = {k: v[:1] for k, v in mini_batch.items()}
    length = int(single["feat_mask"].sum())
    single = {**single, "feats": single["feats"][:, :length], "feat_mask": single["feat_mask"][:, :length]}
    padded = {
        **single,
        "feats": torch.nn.functional.pad(single["feats"], (0, 0, 0, 37), value=5.0),
        "feat_mask": torch.nn.functional.pad(single["feat_mask"], (0, 37)),
        "tokens": torch.nn.functional.pad(single["tokens"], (0, 5)),
    }
    with torch.no_grad():
        assert torch.allclose(_forward(random_model, single), _forward(random_model, padded), atol=1e-4)


def test_missing_enrollment_changes_speaker_output(random_model, mini_batch) -> None:
    with_enr = {**mini_batch, "has_enroll": torch.ones(8)}
    without = {**mini_batch, "has_enroll": torch.zeros(8)}
    with torch.no_grad():
        assert not torch.allclose(_forward(random_model, with_enr), _forward(random_model, without))


def test_gradients_flow_to_every_branch(random_model, mini_batch) -> None:
    random_model.train()
    _forward(random_model, mini_batch).sum().backward()
    for name in ("audio", "text", "speaker"):
        branch = getattr(random_model, name)
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in branch.parameters()), name


def test_modality_dropout_never_drops_everything() -> None:
    fusion = GatedFusion(["audio", "text", "speaker"], 8, 0.0, modality_dropout=0.99).train()
    keep = fusion._drop_mask(500, torch.device("cpu"))
    assert (keep.sum(dim=1) >= 1).all()
