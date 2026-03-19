from __future__ import annotations

from dataclasses import asdict, dataclass

from .constants import DEFAULT_RANDOM_SEED, DEFAULT_SUBSAMPLE_THRESHOLD


@dataclass
class TrainingConfig:
    text_path: str
    output_dir: str
    embedding_dim: int = 100
    window_size: int = 5
    negative_samples: int = 5
    min_count: int = 5
    max_vocab_size: int | None = 50000
    max_tokens: int | None = 300000
    subsample_threshold: float = DEFAULT_SUBSAMPLE_THRESHOLD
    epochs: int = 3
    learning_rate: float = 0.025
    seed: int = DEFAULT_RANDOM_SEED
    log_every: int = 20000

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
