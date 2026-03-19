# JetBrains ML Task: Word2Vec in Pure NumPy

This repository contains a from-scratch implementation of the core word2vec optimization loop in pure NumPy, created for the JetBrains internship assignment.

The implemented variant is skip-gram with negative sampling (SGNS), including:

- Manual forward pass and objective computation.
- Manual analytical gradients and in-place parameter updates.
- Negative sampling with a unigram distribution raised to the 0.75 power.
- Subsampling of frequent words using the probability formula from Mikolov et al. 2013.
- Reproducible training runs and test coverage (including finite-difference gradient checks).

## Assignment Compliance

- Uses pure NumPy for model training logic.
- No PyTorch, TensorFlow, JAX, or autograd engine.
- Explicitly implements loss, gradients, and updates.

## Repository Structure

- `src/word2vec/data.py`: tokenization, vocabulary building, corpus encoding, subsampling, skip-gram pair generation.
- `src/word2vec/model.py`: SGNS model, numerically stable sigmoid, per-example training step.
- `src/word2vec/train.py`: end-to-end training pipeline and artifact saving.
- `src/word2vec/eval.py`: nearest-neighbor inspection via cosine similarity.
- `scripts/download_text8.py`: helper to download the text8 corpus.
- `tests/test_data.py`: data utility tests.
- `tests/test_gradients.py`: finite-difference gradient validation.

## Setup

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

## Dataset Options

The assignment allows any text dataset. This repo is configured for text8 by default.

Download text8:

```bash
python scripts/download_text8.py
```

You can also use any plain `.txt` corpus via `--text-path`.

## Train

```bash
# Option 1: direct module
set PYTHONPATH=src
python -m word2vec.train --text-path data/text8.txt --output-dir artifacts/text8-run

# Option 2: wrapper CLI
set PYTHONPATH=src
python -m word2vec.cli train --text-path data/text8.txt --output-dir artifacts/text8-run
```

Important arguments:

- `--embedding-dim` (default: `100`)
- `--window-size` (default: `5`)
- `--negative-samples` (default: `5`)
- `--epochs` (default: `3`)
- `--learning-rate` (default: `0.025`)
- `--min-count` (default: `5`)
- `--max-vocab-size` (default: `50000`)
- `--max-tokens` (default: `300000`, useful for faster experiments)
- `--subsample-threshold` (default: `1e-4`, set to `0` to disable)

## Evaluate Nearest Neighbors

```bash
set PYTHONPATH=src
python -m word2vec.eval --artifacts-dir artifacts/text8-run --word one --top-k 6
```

### Example Results

Training on a small 50k-token subset of `text8` for just 1 epoch yields the following nearest neighbors for the word **"one"**:

```text
Query: one
 1. the                  0.7482
 2. in                   0.7353
 3. of                   0.7029
 4. nine                 0.6693
 5. zero                 0.6521
 6. two                  0.6102
```

*Note: Even with a tiny dataset and single epoch, the skip-gram objective begins clustering numbers together ('nine', 'zero', 'two'). Full training on the complete corpus yields much sharper semantic relationships.*

## Objective and Gradients

For one center word $c$, positive context word $o$, and negatives $n_1..n_k$:

$$
\mathcal{L} = -\log\sigma(u_o^T v_c) - \sum_{i=1}^{k}\log\sigma(-u_{n_i}^T v_c)
$$

Where:

- $v_c$ is the center/input embedding.
- $u_w$ is the context/output embedding.

Gradients used in code:

- $\nabla_{v_c} = (\sigma(u_o^T v_c)-1)u_o + \sum_i \sigma(u_{n_i}^T v_c)u_{n_i}$
- $\nabla_{u_o} = (\sigma(u_o^T v_c)-1)v_c$
- $\nabla_{u_{n_i}} = \sigma(u_{n_i}^T v_c)v_c$

## Correctness and Quality Checks

- Unit tests for tokenization, vocabulary, pair generation, subsampling, and negative-sampling distribution.
- Finite-difference gradient checks for all three gradient directions: center embedding (v_c), positive context (u_o), and negative samples (u_nᵢ).
- Stable sigmoid implementation to avoid overflow issues.
- Deterministic random seeding for reproducibility.

Run tests:

```bash
python -m pytest -q
```

## Notes on Tradeoffs & Potential Optimizations

- **SGNS vs Hierarchical Softmax:** SGNS scales efficiently to large vocabularies by only updating a small subset of negative weights per step. It is generally superior for producing high-quality frequent-word representations in dense corpora compared to full softmax or hierarchical trees.
- **Batched Computations:** The current implementation processes one `(center, context, negatives)` tuple at a time. A production variant would batch these pairs into matrices, trading higher memory footprint for drastically faster vectorized NumPy execution.
- **Alias Sampling:** Negative samples are currently drawn via `np.random.choice` using probabilities. This requires $O(V)$ operations per call. Implementing Walker's Alias Method would reduce the sample-draw time to $O(1)$, significantly speeding up the inner training loop.
- **Evaluation Embeddings:** The `eval.py` script sums `input_embeddings` and `output_embeddings` to compute cosine similarity. While the original word2vec paper evaluates only on input embeddings, summing them is a common empirical practice (e.g., popularized by GloVe) that often improves neighbor quality on standard tasks.