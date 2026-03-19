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


def subsample_tokens(
    token_ids: np.ndarray,
    counts: np.ndarray,
    threshold: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Subsample frequent words using the word2vec probability formula.

    Each word w is discarded with probability:
        P(discard w) = 1 - sqrt(threshold / f(w))
    where f(w) = count(w) / total_tokens.

    Keeping probability: P(keep w) = min(1, sqrt(threshold / f(w))).

    This balances the representation of rare and frequent words, improving
    embedding quality by reducing the dominance of stop words.

    Args:
        token_ids: 1-D integer array of token indices.
        counts: Vocabulary-wide word count array (counts[i] = frequency of word i).
        threshold: Subsampling threshold t (original paper recommends 1e-5 to 1e-3).
        rng: NumPy random generator for reproducibility.

    Returns:
        Filtered 1-D integer array with frequent words probabilistically dropped.
    """
    total = float(counts.sum())
    # Compute keep probability for every word in the vocabulary
    freqs = counts[token_ids].astype(np.float64) / total
    keep_prob = np.minimum(1.0, np.sqrt(threshold / freqs))
    mask = rng.random(token_ids.shape[0]) < keep_prob
    return token_ids[mask]


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
