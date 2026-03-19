from __future__ import annotations

import numpy as np

from word2vec.data import (
    build_negative_sampling_distribution,
    build_vocabulary,
    encode_tokens,
    generate_skipgram_pairs,
    subsample_tokens,
    tokenize,
)


def test_tokenize_and_vocab_roundtrip() -> None:
    text = "Cats chase cats and dogs. Dogs run fast."
    tokens = tokenize(text)
    vocab = build_vocabulary(tokens, min_count=1)
    token_ids = encode_tokens(tokens, vocab)

    assert len(tokens) == token_ids.shape[0]
    assert "cats" in vocab.word_to_id
    assert vocab.size >= 4


def test_negative_sampling_distribution_sums_to_one() -> None:
    counts = np.array([10, 5, 1], dtype=np.int64)
    p = build_negative_sampling_distribution(counts)

    assert np.isclose(p.sum(), 1.0)
    assert np.all(p > 0.0)


def test_generate_skipgram_pairs_shapes() -> None:
    rng = np.random.default_rng(123)
    token_ids = np.array([0, 1, 2, 3, 4], dtype=np.int32)

    centers, contexts = generate_skipgram_pairs(token_ids, window_size=2, rng=rng)

    assert centers.shape == contexts.shape
    assert centers.ndim == 1
    assert centers.shape[0] > 0


def test_subsample_tokens_reduces_frequent_words() -> None:
    """Highly frequent words should be dropped more often than rare ones."""
    rng = np.random.default_rng(0)

    # Word 0 appears 10000 times, word 1 appears 10 times in a hypothetical corpus.
    counts = np.array([10000, 10], dtype=np.int64)
    # Create a sequence of 10000 tokens: 5000 frequent (word 0), 5000 rare (word 1).
    token_ids = np.array([0] * 5000 + [1] * 5000, dtype=np.int32)

    result = subsample_tokens(token_ids, counts=counts, threshold=1e-3, rng=rng)

    freq_kept = int(np.sum(result == 0))
    rare_kept = int(np.sum(result == 1))

    # The frequent word should be discarded much more than the rare word.
    assert freq_kept < rare_kept
    # Result must be a strict subset.
    assert result.shape[0] < token_ids.shape[0]
