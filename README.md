# Restricted Boltzmann Machine — Movie Recommendations

A PyTorch implementation of a Restricted Boltzmann Machine trained on MovieLens data to predict whether a user will like a movie.

## What It Does

The RBM learns latent features from user–movie rating data. Ratings are binarized into **liked** (≥3 stars → 1) and **not liked** (<3 stars → 0), with unrated movies masked out during training. After training, the model reconstructs ratings for unseen movies to generate recommendations.

## Architecture

The model uses the **Contrastive Divergence (CD-k)** algorithm with k=10 Gibbs sampling steps per weight update.

```
Visible Layer (1682 movie units)
        ↕  W (weight matrix)
Hidden Layer  (100 hidden units)
```

| Hyperparameter | Value |
|----------------|-------|
| Hidden units | 100 |
| Batch size | 100 |
| Epochs | 10 |
| CD steps (k) | 10 |
| Loss metric | Mean Absolute Error |

## Dataset

Included under `Boltzmann_Machines/`:

| Dataset | Users | Movies | Ratings |
|---------|-------|--------|---------|
| MovieLens 100K | 943 | 1,682 | 100,000 |
| MovieLens 1M | 6,040 | 3,952 | 1,000,209 |

Training and testing uses the ML-100K `u1.base` / `u1.test` split.

## 🛠 Tech Stack

| | Technology | Purpose |
|---|-----------|---------|
| 🐍 | Python 3 | Runtime |
| 🔥 | PyTorch | Tensor ops and model |
| 🔢 | NumPy | Data preprocessing |
| 🐼 | pandas | CSV loading |

## Installation

```bash
pip install torch numpy pandas
```

## Usage

```bash
cd Boltzmann_Machines
python rbm.py
```

Prints training loss per epoch and a final test loss. The script auto-detects CUDA and runs on GPU when available.

## Modernization Changelog

The original code was updated to follow current best practices:

- Removed unused ML-1M dataset loading (movies, users, ratings DataFrames were loaded but never used)
- Replaced deprecated `torch.FloatTensor()` with `torch.tensor(..., dtype=torch.float32)`
- Added `torch.no_grad()` context manager around test/inference loop
- Added `if __name__ == "__main__"` guard
- Added automatic GPU/CUDA device detection
- Replaced hardcoded file paths with `os.path` relative to script location
- Added error handling for missing data files
- Added `.clone()` to avoid in-place modification of training data during CD sampling
- Extracted configuration into named constants
- Added docstrings and type hints
- Modernized print statements to f-strings
- Renamed `train()` method to `update_weights()` for clarity

## ⚠️ Known Issues

- The ML-1M dataset is included in the repo but not used by the model — it's kept for reference.
- The RBM uses manual weight updates (not `nn.Module` / autograd) which is intentional for the CD algorithm but means no optimizer or learning-rate scheduler is used.

## Reference

The included `AItRBM-proof.pdf` contains a mathematical treatment of RBMs and the Contrastive Divergence algorithm.

## License

MIT — Kaustabh Ganguly, 2018
