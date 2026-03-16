from __future__ import annotations

import numpy as np


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    x_arr = np.asarray(x)
    positive = x_arr >= 0
    negative = ~positive

    result = np.empty_like(x_arr, dtype=np.float64)
    result[positive] = 1.0 / (1.0 + np.exp(-x_arr[positive]))
    exp_x = np.exp(x_arr[negative])
    result[negative] = exp_x / (1.0 + exp_x)

    if np.isscalar(x):
        return float(result.item())
    return result


class SkipGramNegativeSampling:
    def __init__(self, vocab_size: int, embedding_dim: int, rng: np.random.Generator) -> None:
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        scale = 0.5 / embedding_dim
        self.input_embeddings = rng.normal(0.0, scale, size=(vocab_size, embedding_dim)).astype(np.float64)
        self.output_embeddings = np.zeros((vocab_size, embedding_dim), dtype=np.float64)

    def train_example(self, center_id: int, positive_context_id: int, negative_context_ids: np.ndarray, lr: float) -> float:
        v_c = self.input_embeddings[center_id].copy()
        u_o = self.output_embeddings[positive_context_id].copy()
        u_neg = self.output_embeddings[negative_context_ids].copy()

        pos_score = float(np.dot(v_c, u_o))
        neg_scores = u_neg @ v_c

        pos_sigmoid = float(sigmoid(pos_score))
        neg_sigmoid = sigmoid(neg_scores)

        eps = 1e-12
        loss = -np.log(pos_sigmoid + eps) - np.sum(np.log(sigmoid(-neg_scores) + eps))

        grad_pos = pos_sigmoid - 1.0
        grad_neg = neg_sigmoid

        grad_v_c = grad_pos * u_o + grad_neg @ u_neg
        grad_u_o = grad_pos * v_c
        grad_u_neg = np.outer(grad_neg, v_c)

        self.input_embeddings[center_id] -= lr * grad_v_c
        self.output_embeddings[positive_context_id] -= lr * grad_u_o
        np.add.at(self.output_embeddings, negative_context_ids, -lr * grad_u_neg)

        return float(loss)
