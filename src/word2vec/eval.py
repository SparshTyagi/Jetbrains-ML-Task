from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np


def load_artifacts(artifacts_dir: str) -> tuple[np.ndarray, list[str], dict[str, int]]:
    base = pathlib.Path(artifacts_dir)
    model_data = np.load(base / "model.npz")
    vocab = json.loads((base / "vocab.json").read_text(encoding="utf-8"))
    word_to_id = {word: i for i, word in enumerate(vocab)}

    embeddings = model_data["input_embeddings"] + model_data["output_embeddings"]
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
    embeddings = embeddings / norms
    return embeddings, vocab, word_to_id


def nearest_neighbors(
    embeddings: np.ndarray,
    vocab: list[str],
    word_to_id: dict[str, int],
    query_word: str,
    top_k: int,
) -> list[tuple[str, float]]:
    if query_word not in word_to_id:
        raise ValueError(f"Word '{query_word}' not found in vocabulary")

    qid = word_to_id[query_word]
    qvec = embeddings[qid]
    similarities = embeddings @ qvec
    similarities[qid] = -np.inf

    top_ids = np.argpartition(similarities, -top_k)[-top_k:]
    top_ids = top_ids[np.argsort(similarities[top_ids])[::-1]]

    return [(vocab[i], float(similarities[i])) for i in top_ids]


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect nearest neighbors from trained embeddings")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts/default-run")
    parser.add_argument("--word", type=str, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    embeddings, vocab, word_to_id = load_artifacts(args.artifacts_dir)
    neighbors = nearest_neighbors(embeddings, vocab, word_to_id, args.word.lower(), args.top_k)

    print(f"Query: {args.word}")
    for rank, (word, score) in enumerate(neighbors, start=1):
        print(f"{rank:2d}. {word:20s} {score:.4f}")


if __name__ == "__main__":
    main()
