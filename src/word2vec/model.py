from __future__ import annotations

import numpy as np


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    """Numerically stable sigmoid: avoids overflow for large |x|.

    For x >= 0: sigma(x) = 1 / (1 + exp(-x))
    For x  < 0: sigma(x) = exp(x) / (1 + exp(x))

    Both forms are mathematically equivalent but each is well-conditioned on
    its respective domain, ensuring that exp() never receives a large positive
    argument.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    exp_neg_abs = np.exp(-np.abs(x_arr))
    out = np.where(
        x_arr >= 0,
        1.0 / (1.0 + exp_neg_abs),
        exp_neg_abs / (1.0 + exp_neg_abs),
    )
    return float(out.item()) if np.isscalar(x) or x_arr.ndim == 0 else out


class SkipGramNegativeSampling:
    def __init__(self, vocab_size: int, embedding_dim: int, rng: np.random.Generator) -> None:
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        scale = 0.5 / embedding_dim
        self.input_embeddings = rng.normal(0.0, scale, size=(vocab_size, embedding_dim)).astype(np.float64)
        self.output_embeddings = np.zeros((vocab_size, embedding_dim), dtype=np.float64)

    def train_example(self, center_id: int, positive_context_id: int, negative_context_ids: np.ndarray, lr: float) -> float:
        """Run one SGNS training step and update embeddings in place.

        Objective for one (center, positive, negatives) triple:
            L = -log σ(u_o · v_c) - Σ_i log σ(-u_nᵢ · v_c)

        Analytical gradients:
            ∂L/∂v_c   = (σ(u_o · v_c) - 1) u_o  +  Σ_i σ(u_nᵢ · v_c) u_nᵢ
            ∂L/∂u_o   = (σ(u_o · v_c) - 1) v_c
            ∂L/∂u_nᵢ = σ(u_nᵢ · v_c) v_c

        Args:
            center_id: Index of the center word in the input embedding matrix.
            positive_context_id: Index of the observed context word in the output matrix.
            negative_context_ids: 1-D array of negative-sample word indices.
            lr: Current learning rate.

        Returns:
            Scalar loss value for this example.
        """
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
