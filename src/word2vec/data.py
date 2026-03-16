from __future__ import annotations

import collections
import pathlib
import re
from dataclasses import dataclass

import numpy as np


TOKEN_PATTERN = re.compile(r"[A-Za-z']+")


@dataclass(frozen=True)
class Vocabulary:
    word_to_id: dict[str, int]
    id_to_word: list[str]
    counts: np.ndarray

    @property
    def size(self) -> int:
        return len(self.id_to_word)


def read_text_file(path: str | pathlib.Path) -> str:
    return pathlib.Path(path).read_text(encoding="utf-8", errors="ignore")


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(text)]


def build_vocabulary(tokens: list[str], min_count: int = 5, max_vocab_size: int | None = None) -> Vocabulary:
    counter = collections.Counter(tokens)
    sorted_items = sorted(counter.items(), key=lambda item: (-item[1], item[0]))

    if max_vocab_size is not None:
        sorted_items = sorted_items[:max_vocab_size]

    filtered = [(word, count) for word, count in sorted_items if count >= min_count]

    word_to_id: dict[str, int] = {}
    id_to_word: list[str] = []
    counts: list[int] = []

    for idx, (word, count) in enumerate(filtered):
        word_to_id[word] = idx
        id_to_word.append(word)
        counts.append(count)

    return Vocabulary(word_to_id=word_to_id, id_to_word=id_to_word, counts=np.asarray(counts, dtype=np.int64))


def encode_tokens(tokens: list[str], vocabulary: Vocabulary, max_tokens: int | None = None) -> np.ndarray:
    encoded = [vocabulary.word_to_id[token] for token in tokens if token in vocabulary.word_to_id]
    if max_tokens is not None:
        encoded = encoded[:max_tokens]
    return np.asarray(encoded, dtype=np.int32)


def build_negative_sampling_distribution(counts: np.ndarray, power: float = 0.75) -> np.ndarray:
    weights = counts.astype(np.float64) ** power
    distribution = weights / weights.sum()
    return distribution


def generate_skipgram_pairs(
    token_ids: np.ndarray,
    window_size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    centers: list[int] = []
    contexts: list[int] = []

    n_tokens = int(token_ids.shape[0])
    for index in range(n_tokens):
        dynamic_window = int(rng.integers(1, window_size + 1))
        left = max(0, index - dynamic_window)
        right = min(n_tokens, index + dynamic_window + 1)

        center_id = int(token_ids[index])
        for ctx_index in range(left, right):
            if ctx_index == index:
                continue
            centers.append(center_id)
            contexts.append(int(token_ids[ctx_index]))

    return np.asarray(centers, dtype=np.int32), np.asarray(contexts, dtype=np.int32)
