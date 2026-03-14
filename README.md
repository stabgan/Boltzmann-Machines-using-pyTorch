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

## Datasets

Included under `Boltzmann_Machines/`:

| Dataset | Users | Movies | Ratings |
|---------|-------|--------|---------|
| MovieLens 100K | 943 | 1,682 | 100,000 |
| MovieLens 1M | 6,040 | 3,952 | 1,000,209 |

Training/testing uses the ML-100K `u1.base` / `u1.test` split.

## Dependencies

```bash
pip install torch numpy pandas
```

## How to Run

```bash
cd Boltzmann_Machines
python rbm.py
```

Prints training loss per epoch and a final test loss.

## Tech Stack

| | Technology |
|---|-----------|
| 🐍 | Python 3 |
| 🔥 | PyTorch |
| 🔢 | NumPy |
| 🐼 | pandas |

## Known Issues

- **No GPU support** — all tensors run on CPU; no `.to(device)` calls.
- **ML-1M loaded but unused** — the 1M dataset is read at startup but not used in training or evaluation.
- **Hardcoded paths** — dataset paths are relative; the script must be run from inside `Boltzmann_Machines/`.
- **Repo naming** — titled "Boltzmann Machines" but implements a single-layer RBM, not a Deep Boltzmann Machine.

## Reference

The included `AItRBM-proof.pdf` contains a mathematical treatment of RBMs and the Contrastive Divergence algorithm.

## License

MIT — Kaustabh Ganguly, 2018
