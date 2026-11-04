"""Synthetic proxy data: speech-like audio plus ASR-style transcripts for directed/background speech.

This module deliberately does NOT produce real speech. It builds a controllable *proxy* in which
- audio is a formant-synthesised, syllable-timed voice with per-speaker pitch/vocal-tract
  parameters, room reverberation, and pink/babble noise;
- directed utterances are, on average, nearer-field (higher SNR/level, less reverb), slower and
  with a wider pitch range, but every acoustic distribution overlaps with background speech;
- transcripts come from slot-filled templates in which some command-like sentences are valid in
  *both* classes (e.g. "can you turn the kitchen lights off"), so text alone is ambiguous;
- the enrolled household user speaks most directed utterances but also many background ones.

All sampling is seeded per (split, household), so generation is reproducible and independent of
the number of worker processes. See docs/DATA.md for caveats.
"""

from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass
from typing import Any

import numpy as np

SAMPLE_RATE = 16_000

# (F1, F2, F3) in Hz for a reference adult voice; scaled per speaker by ``formant_scale``.
_VOWELS: dict[str, tuple[float, float, float]] = {
    "a": (730.0, 1090.0, 2440.0),
    "e": (530.0, 1840.0, 2480.0),
    "i": (270.0, 2290.0, 3010.0),
    "o": (570.0, 840.0, 2410.0),
    "u": (300.0, 870.0, 2240.0),
    "x": (660.0, 1720.0, 2410.0),
}
_FORMANT_WIDTH_HZ = (110.0, 150.0, 220.0)
_FORMANT_WEIGHT = (1.0, 0.7, 0.35)

_SLOTS: dict[str, list[str]] = {
    "num": ["five", "ten", "fifteen", "twenty", "thirty", "forty", "two", "three"],
    "day": ["today", "tomorrow", "tonight", "on friday", "this weekend", "on monday", "next week"],
    "genre": ["jazz", "rock", "classical", "pop", "country", "lofi", "blues", "podcast"],
    "person": ["mom", "dad", "john", "sarah", "the doctor", "my brother", "alex", "grandma"],
    "room": ["kitchen", "bedroom", "living room", "bathroom", "garage", "hallway"],
    "onoff": ["on", "off"],
    "updown": ["up", "down"],
    "chore": ["buy milk", "call the dentist", "water the plants", "take out the trash", "pay rent"],
    "time": ["noon", "six pm", "eight thirty", "seven am", "five o'clock", "midnight"],
    "place": ["the airport", "the office", "the gym", "downtown", "the pharmacy", "the station"],
    "city": ["london", "tokyo", "chicago", "paris", "mumbai", "sydney", "boston"],
    "food": ["eggs", "bread", "rice", "apples", "cheese", "pasta", "coffee", "tomatoes"],
    "sport": ["football", "baseball", "tennis", "hockey", "cricket"],
    "when": ["last night", "yesterday", "this morning", "on sunday"],
    "adj": ["great", "boring", "amazing", "terrible", "funny", "long", "nice", "weird"],
    "meal": ["breakfast", "lunch", "dinner"],
    "topic": ["the election", "the new phone", "the trip", "work", "the budget", "the exam"],
    "msg": ["running late", "see you soon", "happy birthday", "call me back", "on my way"],
}

# Directed, unambiguous requests.
_DIRECTED_CLEAR = [
    "set a timer for {num} minutes",
    "what's the weather {day}",
    "play some {genre} music",
    "call {person}",
    "send a message to {person} saying {msg}",
    "turn {onoff} the {room} lights",
    "remind me to {chore} at {time}",
    "how long does it take to get to {place}",
    "what time is it in {city}",
    "add {food} to my shopping list",
    "turn the volume {updown}",
    "skip this song",
    "set an alarm for {time}",
    "show me the route to {place}",
    "what's on my calendar {day}",
]
# Directed, but phrased like casual conversation or a polite human request.
_DIRECTED_AMBIGUOUS = [
    "can you turn the {room} lights {onoff}",
    "could you play {genre} music please",
    "what time is it",
    "do you know what the weather is like {day}",
    "tell me a joke",
    "how are you doing today",
    "what do you think about {topic}",
    "i was wondering how long it takes to get to {place}",
    "can you remind me to {chore}",
    "call {person} for me",
    "what did {person} say",
]
# Background, unambiguous conversation.
_BACKGROUND_CLEAR = [
    "did you see the {sport} game {when}",
    "i'll pick up {food} on the way home",
    "{person} said the meeting moved to {time}",
    "we should go to {place} {day}",
    "that movie was really {adj}",
    "i can't believe it's already {day}",
    "what did you have for {meal}",
    "honestly i'm {adj} about {topic}",
    "the new assistant at the office is {adj}",
    "i told {person} we would be there at {time}",
    "this soup needs more salt",
    "are you coming to {place} {day}",
]
# Background, but phrased like a command or question that an assistant could receive.
_BACKGROUND_COMMANDLIKE = [
    "can you turn the {room} lights {onoff}",
    "turn the volume {updown} it's too loud",
    "call {person} when you get home",
    "what time is it",
    "play {genre} music in the car",
    "remind me to {chore} {day}",
    "could you pass the {food}",
    "add {food} to the list",
    "what's the weather like {day}",
    "how long will it take to get to {place}",
    "set the table for {meal}",
    "tell me a joke",
    "what do you think about {topic}",
]
_WAKE_WORDS = ["hey assistant", "ok assistant", "assistant"]


@dataclass(frozen=True)
class Voice:
    f0: float
    formant_scale: float
    tilt: float
    breathiness: float


def sample_voice(rng: np.random.Generator) -> Voice:
    female = rng.random() < 0.5
    f0 = rng.normal(205.0, 22.0) if female else rng.normal(115.0, 16.0)
    scale = rng.normal(1.17, 0.04) if female else rng.normal(1.0, 0.04)
    return Voice(
        f0=float(np.clip(f0, 80.0, 280.0)),
        formant_scale=float(scale),
        tilt=float(rng.uniform(0.7, 1.4)),
        breathiness=float(rng.uniform(0.02, 0.15)),
    )


def _fill(template: str, rng: np.random.Generator) -> str:
    out = template
    for key, values in _SLOTS.items():
        token = "{" + key + "}"
        while token in out:
            out = out.replace(token, values[int(rng.integers(len(values)))], 1)
    return out


def sample_transcript(directed: bool, rng: np.random.Generator) -> str:
    """Sample a clean transcript for the given class."""
    if directed:
        ambiguous = rng.random() < 0.4
        pool = _DIRECTED_AMBIGUOUS if ambiguous else _DIRECTED_CLEAR
        wake_prob = 0.08 if ambiguous else 0.15
    else:
        commandlike = rng.random() < 0.45
        if commandlike:
            # Half of command-like background speech reuses the *exact* directed templates
            # (a person asking another person), making text alone genuinely ambiguous.
            shared = rng.random() < 0.5
            pool = _DIRECTED_CLEAR + _DIRECTED_AMBIGUOUS if shared else _BACKGROUND_COMMANDLIKE
        else:
            pool = _BACKGROUND_CLEAR
        wake_prob = 0.03
    text = _fill(pool[int(rng.integers(len(pool)))], rng)
    if rng.random() < wake_prob:
        text = f"{_WAKE_WORDS[int(rng.integers(len(_WAKE_WORDS)))]} {text}"
    return text


def _lexicon() -> list[str]:
    words: set[str] = set()
    for group in (_DIRECTED_CLEAR, _DIRECTED_AMBIGUOUS, _BACKGROUND_CLEAR, _BACKGROUND_COMMANDLIKE):
        for template in group:
            words.update(w for w in template.split() if not w.startswith("{"))
    for values in _SLOTS.values():
        for value in values:
            words.update(value.split())
    words.update(w for phrase in _WAKE_WORDS for w in phrase.split())
    return sorted(words)


_LEXICON = _lexicon()


def corrupt_transcript(text: str, snr_db: float, rng: np.random.Generator) -> str:
    """Simulate ASR errors (deletions/substitutions) whose rate grows as SNR drops."""
    error_rate = 0.02 + 0.18 * float(np.clip((16.0 - snr_db) / 16.0, 0.0, 1.0))
    out: list[str] = []
    for word in text.split():
        roll = rng.random()
        if roll < error_rate / 2:
            continue
        if roll < error_rate:
            out.append(_LEXICON[int(rng.integers(len(_LEXICON)))])
        else:
            out.append(word)
    return " ".join(out) if out else text.split()[0]


def _vowel_for_chunk(chunk: str, rng: np.random.Generator) -> str:
    for ch in chunk:
        if ch in "aeiou":
            return ch
        if ch == "y":
            return "i"
    return "x" if rng.random() < 0.7 else str(rng.choice(list("aeiou")))


@dataclass(frozen=True)
class Style:
    """Utterance-level speaking style (differs on average between classes)."""

    rate: float
    pitch_depth: float


def synth_speech(
    text: str, voice: Voice, style: Style, rng: np.random.Generator, sr: int = SAMPLE_RATE
) -> np.ndarray:
    """Formant-synthesise a syllable-timed utterance for ``text`` (float32 waveform)."""
    syllables: list[tuple[float, float, str, bool]] = []  # (start_s, dur_s, vowel, has_onset)
    cursor = float(rng.uniform(0.15, 0.4))
    for word in text.lower().split():
        n_syl = max(1, round(len(word) / 3.2))
        step = max(1, len(word) // n_syl)
        for s in range(n_syl):
            chunk = word[s * step : (s + 1) * step if s < n_syl - 1 else len(word)]
            dur = float(rng.uniform(0.11, 0.2)) / style.rate
            syllables.append((cursor, dur, _vowel_for_chunk(chunk, rng), rng.random() < 0.7))
            cursor += dur
        cursor += float(rng.uniform(0.04, 0.12)) / style.rate
    total = cursor + float(rng.uniform(0.15, 0.4))
    n = int(total * sr)
    wave = np.zeros(n, dtype=np.float32)

    centers = np.array([s[0] + s[1] / 2 for s in syllables])
    progress = centers / max(total, 1e-3)
    contour = (
        -0.1 * progress
        + 0.5 * np.sin(2 * np.pi * (progress * rng.uniform(0.8, 1.6) + rng.random()))
        + rng.normal(0.0, 0.25, size=centers.size)
    )
    f0_targets = voice.f0 * (1.0 + style.pitch_depth * contour)
    t_axis = np.arange(n) / sr
    f0_track = np.interp(t_axis, centers, f0_targets).astype(np.float64)

    for (start, dur, vowel, onset), f0_c in zip(syllables, f0_targets, strict=True):
        a, b = int(start * sr), min(n, int((start + dur) * sr))
        if b - a < 32:
            continue
        length = b - a
        env = np.hanning(length).astype(np.float32) ** 0.7
        seg_f0 = f0_track[a:b]
        phase = 2 * np.pi * np.cumsum(seg_f0) / sr
        n_harm = int(min(4200.0, sr / 2 - 100) / max(f0_c, 60.0))
        ks = np.arange(1, n_harm + 1)
        freqs = ks * f0_c
        envelope = np.zeros(n_harm)
        for form, width, weight in zip(_VOWELS[vowel], _FORMANT_WIDTH_HZ, _FORMANT_WEIGHT, strict=True):
            envelope += weight * np.exp(-0.5 * ((freqs - form * voice.formant_scale) / width) ** 2)
        amps = (envelope + 0.02) * ks ** (-voice.tilt)
        voiced = (np.sin(np.outer(ks, phase)) * amps[:, None]).sum(axis=0)
        voiced /= max(np.abs(voiced).max(), 1e-6)
        breath = rng.normal(0.0, voice.breathiness, size=length)
        wave[a:b] += (env * (voiced + breath)).astype(np.float32)
        if onset:
            burst_len = min(int(0.03 * sr), length)
            burst = np.diff(rng.normal(0.0, 1.0, size=burst_len + 1)) * 0.35
            wave[a : a + burst_len] += (burst * np.hanning(burst_len)).astype(np.float32)
    return wave


def _pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    spectrum = np.fft.rfft(rng.normal(size=n))
    freqs = np.maximum(np.arange(spectrum.size), 1)
    return np.fft.irfft(spectrum / np.sqrt(freqs), n=n).astype(np.float32)


def _reverberate(wave: np.ndarray, rt60: float, drr: float, rng: np.random.Generator) -> np.ndarray:
    length = int(min(rt60, 0.9) * SAMPLE_RATE)
    t = np.arange(length) / SAMPLE_RATE
    tail = rng.normal(size=length) * np.exp(-6.9 * t / rt60) * drr
    tail[0] = 1.0
    n_fft = 1 << int(np.ceil(np.log2(wave.size + length)))
    out = np.fft.irfft(np.fft.rfft(wave, n_fft) * np.fft.rfft(tail, n_fft), n_fft)[: wave.size]
    return out.astype(np.float32)


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)) + 1e-12))


def _babble(n: int, rng: np.random.Generator) -> np.ndarray:
    mix = np.zeros(n, dtype=np.float32)
    for _ in range(int(rng.integers(2, 4))):
        text = sample_transcript(bool(rng.random() < 0.5), rng)
        wave = synth_speech(text, sample_voice(rng), Style(1.0, 0.12), rng)
        reps = int(np.ceil(n / wave.size))
        mix += np.tile(wave, reps)[:n] / max(_rms(wave), 1e-6)
    return mix


@dataclass(frozen=True)
class AcousticCondition:
    snr_db: float
    rt60: float
    level_db: float
    noise_type: str  # "pink" | "babble"


def sample_condition(directed: bool, rng: np.random.Generator) -> AcousticCondition:
    snr = rng.normal(16.0 if directed else 10.0, 7.0)
    rt60 = float(np.exp(rng.normal(np.log(0.25 if directed else 0.45), 0.4)))
    level = rng.normal(-24.0 if directed else -30.0, 5.5)
    return AcousticCondition(
        snr_db=float(np.clip(snr, -2.0, 30.0)),
        rt60=float(np.clip(rt60, 0.1, 0.9)),
        level_db=float(level),
        noise_type="pink" if rng.random() < 0.5 else "babble",
    )


def render_utterance(
    text: str, voice: Voice, style: Style, cond: AcousticCondition, rng: np.random.Generator
) -> np.ndarray:
    """Synthesise ``text`` and apply reverberation, noise at ``cond.snr_db`` and level."""
    clean = synth_speech(text, voice, style, rng)
    wet = _reverberate(clean, cond.rt60, drr=float(rng.uniform(0.2, 0.5)), rng=rng)
    noise = _pink_noise(wet.size, rng) if cond.noise_type == "pink" else _babble(wet.size, rng)
    noise *= _rms(wet) / (_rms(noise) * 10 ** (cond.snr_db / 20.0))
    mix = wet + noise
    mix *= 10 ** (cond.level_db / 20.0) / _rms(mix)
    return np.clip(mix, -1.0, 1.0).astype(np.float32)


def sample_style(directed: bool, rng: np.random.Generator) -> Style:
    if directed:
        return Style(rate=float(np.clip(rng.normal(0.95, 0.12), 0.7, 1.3)),
                     pitch_depth=float(np.clip(rng.normal(0.18, 0.06), 0.05, 0.4)))
    return Style(rate=float(np.clip(rng.normal(1.1, 0.15), 0.7, 1.5)),
                 pitch_depth=float(np.clip(rng.normal(0.10, 0.05), 0.02, 0.3)))


def generate_household(task: tuple[int, int, int, float, int]) -> dict[str, Any]:
    """Generate one household's enrollment clip and utterances (picklable worker function)."""
    split_seed, household, n_utts, max_seconds, directed_prior_milli = task
    rng = np.random.default_rng([split_seed, household])
    enrolled = sample_voice(rng)
    family = [sample_voice(rng) for _ in range(3)]

    enroll_cond = AcousticCondition(snr_db=22.0, rt60=0.2, level_db=-24.0, noise_type="pink")
    enroll_text = " ".join(sample_transcript(True, rng) for _ in range(2))
    enrollment = render_utterance(enroll_text, enrolled, Style(1.0, 0.15), enroll_cond, rng)

    records: list[dict[str, Any]] = []
    max_len = int(max_seconds * SAMPLE_RATE)
    for _ in range(n_utts):
        directed = bool(rng.random() < directed_prior_milli / 1000.0)
        by_enrolled = bool(rng.random() < (0.88 if directed else 0.35))
        if by_enrolled:
            voice = enrolled
        elif rng.random() < 0.5:
            voice = family[int(rng.integers(len(family)))]
        else:
            voice = sample_voice(rng)
        cond = sample_condition(directed, rng)
        clean_text = sample_transcript(directed, rng)
        wave = render_utterance(clean_text, voice, sample_style(directed, rng), cond, rng)[:max_len]
        records.append(
            {
                "wave": wave,
                "text": corrupt_transcript(clean_text, cond.snr_db, rng),
                "label": int(directed),
                "is_enrolled_speaker": int(by_enrolled),
                "snr_db": cond.snr_db,
                "rt60": cond.rt60,
                "level_db": cond.level_db,
                "noise_type": cond.noise_type,
            }
        )
    return {"household": household, "enrollment": enrollment[:max_len], "records": records}


def generate_households(
    split_seed: int,
    n_utterances: int,
    utterances_per_household: int,
    max_seconds: float,
    num_workers: int = 1,
    directed_prior: float = 0.5,
) -> list[dict[str, Any]]:
    """Generate ``n_utterances`` across households; deterministic for a given ``split_seed``."""
    tasks = []
    remaining = n_utterances
    household = 0
    while remaining > 0:
        count = min(utterances_per_household, remaining)
        tasks.append((split_seed, household, count, max_seconds, int(directed_prior * 1000)))
        remaining -= count
        household += 1
    if num_workers <= 1 or len(tasks) < 4:
        return [generate_household(t) for t in tasks]
    with mp.get_context("fork" if "fork" in mp.get_all_start_methods() else "spawn").Pool(num_workers) as pool:
        return pool.map(generate_household, tasks, chunksize=2)
