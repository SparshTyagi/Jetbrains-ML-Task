from __future__ import annotations

import numpy as np

from word2vec.model import SkipGramNegativeSampling


def _loss_for_example(
    input_embeddings: np.ndarray,
    output_embeddings: np.ndarray,
    center_id: int,
    positive_context_id: int,
    negative_context_ids: np.ndarray,
) -> float:
    v_c = input_embeddings[center_id]
    u_o = output_embeddings[positive_context_id]
    u_neg = output_embeddings[negative_context_ids]

    pos_score = float(np.dot(v_c, u_o))
    neg_scores = u_neg @ v_c

    def sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))

    eps = 1e-12
    return float(-np.log(sigmoid(np.array([pos_score]))[0] + eps) - np.sum(np.log(sigmoid(-neg_scores) + eps)))


def test_single_step_gradient_matches_finite_difference() -> None:
    rng = np.random.default_rng(7)

    vocab_size = 6
    dim = 4
    center_id = 1
    positive_context_id = 2
    negative_context_ids = np.array([3, 4, 4], dtype=np.int32)
    lr = 1e-5

    model = SkipGramNegativeSampling(vocab_size=vocab_size, embedding_dim=dim, rng=rng)
    model.input_embeddings = rng.normal(0.0, 0.1, size=(vocab_size, dim))
    model.output_embeddings = rng.normal(0.0, 0.1, size=(vocab_size, dim))

    original_input = model.input_embeddings.copy()
    original_output = model.output_embeddings.copy()

    _ = model.train_example(center_id, positive_context_id, negative_context_ids, lr=lr)

    analytic_grad_v = (original_input[center_id] - model.input_embeddings[center_id]) / lr

    eps = 1e-6
    numeric_grad_v = np.zeros_like(analytic_grad_v)

    for i in range(dim):
        plus_input = original_input.copy()
        minus_input = original_input.copy()
        plus_input[center_id, i] += eps
        minus_input[center_id, i] -= eps

        loss_plus = _loss_for_example(
            plus_input,
            original_output,
            center_id,
            positive_context_id,
            negative_context_ids,
        )
        loss_minus = _loss_for_example(
            minus_input,
            original_output,
            center_id,
            positive_context_id,
            negative_context_ids,
        )
        numeric_grad_v[i] = (loss_plus - loss_minus) / (2 * eps)

    np.testing.assert_allclose(analytic_grad_v, numeric_grad_v, atol=1e-5, rtol=1e-4)
