"""Word-level vocabulary and tokenizer for (ASR-style) transcripts."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable

PAD, UNK = "<pad>", "<unk>"
_TOKEN_RE = re.compile(r"[a-z0-9']+")


def normalize_text(text: str) -> list[str]:
    """Lower-case and split into word tokens (punctuation dropped)."""
    return _TOKEN_RE.findall(text.lower())


class Vocabulary:
    """Maps words to integer ids; id 0 is padding, id 1 is unknown."""

    def __init__(self, words: Iterable[str]) -> None:
        self.itos: list[str] = [PAD, UNK, *words]
        self.stoi: dict[str, int] = {w: i for i, w in enumerate(self.itos)}

    @classmethod
    def build(cls, texts: Iterable[str], min_count: int = 1) -> Vocabulary:
        counts: Counter[str] = Counter()
        for text in texts:
            counts.update(normalize_text(text))
        words = sorted(w for w, c in counts.items() if c >= min_count)
        return cls(words)

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, text: str, max_len: int) -> list[int]:
        """Token ids truncated to ``max_len``; empty text becomes a single ``<unk>``."""
        ids = [self.stoi.get(w, 1) for w in normalize_text(text)][:max_len]
        return ids or [1]

    def to_list(self) -> list[str]:
        return self.itos[2:]
