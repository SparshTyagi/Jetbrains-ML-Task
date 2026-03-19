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


def _make_model(vocab_size: int = 6, dim: int = 4, seed: int = 7) -> SkipGramNegativeSampling:
    rng = np.random.default_rng(seed)
    model = SkipGramNegativeSampling(vocab_size=vocab_size, embedding_dim=dim, rng=rng)
    model.input_embeddings = rng.normal(0.0, 0.1, size=(vocab_size, dim))
    model.output_embeddings = rng.normal(0.0, 0.1, size=(vocab_size, dim))
    return model


CENTER = 1
POSITIVE = 2
NEGATIVES = np.array([3, 4, 4], dtype=np.int32)
EPS_FD = 1e-6
LR = 1e-5


def test_gradient_v_c_matches_finite_difference() -> None:
    """Analytic gradient for the center embedding v_c matches finite differences."""
    model = _make_model()
    original_input = model.input_embeddings.copy()
    original_output = model.output_embeddings.copy()

    model.train_example(CENTER, POSITIVE, NEGATIVES, lr=LR)
    analytic_grad = (original_input[CENTER] - model.input_embeddings[CENTER]) / LR

    dim = original_input.shape[1]
    numeric_grad = np.zeros(dim)
    for i in range(dim):
        plus = original_input.copy()
        minus = original_input.copy()
        plus[CENTER, i] += EPS_FD
        minus[CENTER, i] -= EPS_FD
        numeric_grad[i] = (
            _loss_for_example(plus, original_output, CENTER, POSITIVE, NEGATIVES)
            - _loss_for_example(minus, original_output, CENTER, POSITIVE, NEGATIVES)
        ) / (2 * EPS_FD)

    np.testing.assert_allclose(analytic_grad, numeric_grad, atol=1e-5, rtol=1e-4)


def test_gradient_u_o_matches_finite_difference() -> None:
    """Analytic gradient for the positive context embedding u_o matches finite differences."""
    model = _make_model()
    original_input = model.input_embeddings.copy()
    original_output = model.output_embeddings.copy()

    model.train_example(CENTER, POSITIVE, NEGATIVES, lr=LR)
    analytic_grad = (original_output[POSITIVE] - model.output_embeddings[POSITIVE]) / LR

    dim = original_output.shape[1]
    numeric_grad = np.zeros(dim)
    for i in range(dim):
        plus = original_output.copy()
        minus = original_output.copy()
        plus[POSITIVE, i] += EPS_FD
        minus[POSITIVE, i] -= EPS_FD
        numeric_grad[i] = (
            _loss_for_example(original_input, plus, CENTER, POSITIVE, NEGATIVES)
            - _loss_for_example(original_input, minus, CENTER, POSITIVE, NEGATIVES)
        ) / (2 * EPS_FD)

    np.testing.assert_allclose(analytic_grad, numeric_grad, atol=1e-5, rtol=1e-4)


def test_gradient_u_neg_matches_finite_difference() -> None:
    """Analytic gradient for a unique negative-sample embedding u_n matches finite differences.

    Word 4 appears twice in NEGATIVES; np.add.at accumulates both contributions,
    which must equal the sum of the two individual finite-difference estimates.
    """
    model = _make_model()
    original_input = model.input_embeddings.copy()
    original_output = model.output_embeddings.copy()

    model.train_example(CENTER, POSITIVE, NEGATIVES, lr=LR)

    dim = original_output.shape[1]
    for neg_word in np.unique(NEGATIVES):
        analytic_grad = (original_output[neg_word] - model.output_embeddings[neg_word]) / LR

        numeric_grad = np.zeros(dim)
        for i in range(dim):
            plus = original_output.copy()
            minus = original_output.copy()
            plus[neg_word, i] += EPS_FD
            minus[neg_word, i] -= EPS_FD
            numeric_grad[i] = (
                _loss_for_example(original_input, plus, CENTER, POSITIVE, NEGATIVES)
                - _loss_for_example(original_input, minus, CENTER, POSITIVE, NEGATIVES)
            ) / (2 * EPS_FD)

        np.testing.assert_allclose(analytic_grad, numeric_grad, atol=1e-5, rtol=1e-4)


# Keep the original combined test name so existing CI / review scripts still find it.
def test_single_step_gradient_matches_finite_difference() -> None:
    test_gradient_v_c_matches_finite_difference()
    test_gradient_u_o_matches_finite_difference()
    test_gradient_u_neg_matches_finite_difference()
