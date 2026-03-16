from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np

from .config import TrainingConfig
from .data import (
    build_negative_sampling_distribution,
    build_vocabulary,
    encode_tokens,
    generate_skipgram_pairs,
    read_text_file,
    tokenize,
)
from .model import SkipGramNegativeSampling


def parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser(description="Train skip-gram word2vec with negative sampling in pure NumPy")
    parser.add_argument("--text-path", type=str, default="data/text8.txt")
    parser.add_argument("--output-dir", type=str, default="artifacts/default-run")
    parser.add_argument("--embedding-dim", type=int, default=100)
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--negative-samples", type=int, default=5)
    parser.add_argument("--min-count", type=int, default=5)
    parser.add_argument("--max-vocab-size", type=int, default=50000)
    parser.add_argument("--max-tokens", type=int, default=300000)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=0.025)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=20000)
    args = parser.parse_args()

    return TrainingConfig(
        text_path=args.text_path,
        output_dir=args.output_dir,
        embedding_dim=args.embedding_dim,
        window_size=args.window_size,
        negative_samples=args.negative_samples,
        min_count=args.min_count,
        max_vocab_size=args.max_vocab_size,
        max_tokens=args.max_tokens,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        seed=args.seed,
        log_every=args.log_every,
    )


def save_artifacts(
    output_dir: pathlib.Path,
    model: SkipGramNegativeSampling,
    id_to_word: list[str],
    config: TrainingConfig,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_dir / "model.npz",
        input_embeddings=model.input_embeddings,
        output_embeddings=model.output_embeddings,
    )

    (output_dir / "vocab.json").write_text(json.dumps(id_to_word, ensure_ascii=True), encoding="utf-8")
    (output_dir / "config.json").write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")


def run_training(config: TrainingConfig) -> None:
    rng = np.random.default_rng(config.seed)

    print(f"Reading corpus from {config.text_path}")
    raw_text = read_text_file(config.text_path)
    tokens = tokenize(raw_text)
    print(f"Tokenized corpus size: {len(tokens):,}")

    vocabulary = build_vocabulary(
        tokens=tokens,
        min_count=config.min_count,
        max_vocab_size=config.max_vocab_size,
    )
    print(f"Vocabulary size: {vocabulary.size:,}")

    token_ids = encode_tokens(tokens=tokens, vocabulary=vocabulary, max_tokens=config.max_tokens)
    print(f"Encoded token count: {token_ids.shape[0]:,}")

    negative_distribution = build_negative_sampling_distribution(vocabulary.counts)

    model = SkipGramNegativeSampling(
        vocab_size=vocabulary.size,
        embedding_dim=config.embedding_dim,
        rng=rng,
    )

    start_time = time.time()
    global_step = 0

    for epoch in range(config.epochs):
        centers, contexts = generate_skipgram_pairs(
            token_ids=token_ids,
            window_size=config.window_size,
            rng=rng,
        )
        order = rng.permutation(centers.shape[0])
        centers = centers[order]
        contexts = contexts[order]

        epoch_loss = 0.0
        total_steps = max(1, config.epochs * centers.shape[0])

        for idx in range(centers.shape[0]):
            decay = max(0.0001, 1.0 - (global_step / total_steps))
            lr = config.learning_rate * decay

            negatives = rng.choice(
                vocabulary.size,
                size=config.negative_samples,
                replace=True,
                p=negative_distribution,
            ).astype(np.int32)

            loss = model.train_example(
                center_id=int(centers[idx]),
                positive_context_id=int(contexts[idx]),
                negative_context_ids=negatives,
                lr=lr,
            )

            epoch_loss += loss
            global_step += 1

            if global_step % config.log_every == 0:
                avg_loss = epoch_loss / (idx + 1)
                print(f"step={global_step:,} epoch={epoch + 1} avg_loss={avg_loss:.4f} lr={lr:.6f}")

        avg_epoch_loss = epoch_loss / max(1, centers.shape[0])
        print(f"epoch={epoch + 1} completed avg_loss={avg_epoch_loss:.4f}")

    elapsed = time.time() - start_time
    print(f"Training finished in {elapsed:.2f}s")

    save_artifacts(pathlib.Path(config.output_dir), model, vocabulary.id_to_word, config)
    print(f"Saved artifacts to {config.output_dir}")


def main() -> None:
    config = parse_args()
    run_training(config)


if __name__ == "__main__":
    main()
