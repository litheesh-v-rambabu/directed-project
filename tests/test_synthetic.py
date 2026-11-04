import numpy as np

from directed_speech_detector.data.synthetic import (
    corrupt_transcript,
    generate_households,
    sample_transcript,
)


def test_generation_is_deterministic() -> None:
    a = generate_households(3, 10, 5, 4.0)
    b = generate_households(3, 10, 5, 4.0)
    assert np.array_equal(a[0]["records"][0]["wave"], b[0]["records"][0]["wave"])
    assert a[1]["records"][2]["text"] == b[1]["records"][2]["text"]


def test_different_seeds_differ() -> None:
    a = generate_households(1, 5, 5, 4.0)[0]["records"][0]["wave"]
    b = generate_households(2, 5, 5, 4.0)[0]["records"][0]["wave"]
    assert a.shape != b.shape or not np.array_equal(a, b)


def test_worker_count_does_not_change_output() -> None:
    serial = generate_households(5, 40, 5, 3.0, num_workers=1)
    parallel = generate_households(5, 40, 5, 3.0, num_workers=2)
    assert [h["household"] for h in serial] == [h["household"] for h in parallel]
    assert np.array_equal(serial[3]["records"][1]["wave"], parallel[3]["records"][1]["wave"])


def test_records_are_valid() -> None:
    households = generate_households(9, 30, 10, 3.0)
    records = [r for h in households for r in h["records"]]
    assert len(records) == 30
    for r in records:
        assert r["wave"].dtype == np.float32 and np.isfinite(r["wave"]).all()
        assert 0.1 < r["wave"].size / 16_000 <= 3.0
        assert np.abs(r["wave"]).max() <= 1.0 and r["text"].strip()
    assert 0.2 < np.mean([r["label"] for r in records]) < 0.8


def test_asr_corruption_increases_with_low_snr() -> None:
    rng = np.random.default_rng(0)
    text = "set a timer for ten minutes please now"
    clean = [corrupt_transcript(text, 30.0, rng) == text for _ in range(300)]
    noisy = [corrupt_transcript(text, -2.0, rng) == text for _ in range(300)]
    assert sum(clean) > sum(noisy)


def test_directed_and_background_text_overlap() -> None:
    rng = np.random.default_rng(1)
    directed = {sample_transcript(True, rng) for _ in range(600)}
    background = {sample_transcript(False, rng) for _ in range(600)}
    assert directed & background, "text must be ambiguous for some utterances by design"
