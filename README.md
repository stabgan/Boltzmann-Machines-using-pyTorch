# Restricted Boltzmann Machine for Movie Recommendations

A PyTorch implementation of a Restricted Boltzmann Machine (RBM) trained on the MovieLens dataset to predict whether a user will like a movie.

## What This Does

The model learns latent features from user–movie rating data using Contrastive Divergence (CD-k). Ratings are binarized into "liked" (≥3 stars → 1) and "not liked" (<3 stars → 0), and unrated movies are masked out during training. After training, the RBM reconstructs ratings for unseen movies to generate recommendations.

## Architecture

```
Visible Layer (nv = 1682 movies)
        ↕  W (weight matrix)
Hidden Layer (nh = 100 units)
```

- **Model:** Single-layer RBM (not a Deep Boltzmann Machine despite the repo name)
- **Training:** CD-10 (10 Gibbs sampling steps per weight update)
- **Loss:** Mean Absolute Error between original and reconstructed visible units
- **Hyperparameters:** 100 hidden units, batch size 100, 10 epochs

## Datasets

Both are included in the repo under `Boltzmann_Machines/`:

| Dataset | Users | Movies | Ratings |
|---------|-------|--------|---------|
| [MovieLens 100K](https://grouplens.org/datasets/movielens/100k/) | 943 | 1,682 | 100,000 |
| [MovieLens 1M](https://grouplens.org/datasets/movielens/1m/) | 6,040 | 3,952 | 1,000,209 |

Training and testing use the ML-100K `u1.base`/`u1.test` split. The ML-1M data is loaded but not used in the training loop.

## Dependencies

- Python 3.x
- PyTorch
- NumPy
- pandas

```bash
pip install torch numpy pandas
```

## Usage

```bash
cd Boltzmann_Machines
python rbm.py
```

The script prints training loss per epoch and a final test loss.

## Known Issues and Deprecations

| Issue | Details |
|-------|---------|
| `torch.autograd.Variable` import | Imported but never used. `Variable` has been deprecated since PyTorch 0.4 — tensors now track gradients natively. |
| Unused imports | `torch.nn`, `torch.nn.parallel`, `torch.optim`, `torch.utils.data` are imported but never referenced. |
| ML-1M loaded but unused | `movies.dat`, `users.dat`, and `ratings.dat` from ML-1M are read into DataFrames at startup but play no role in training or evaluation. |
| `pd.read_csv` with `sep='::'` | Using a multi-character separator triggers a `ParserWarning` in modern pandas and forces the Python parsing engine. |
| No GPU support | All tensors are on CPU. There are no `.cuda()` or `.to(device)` calls. |
| Naming | The repo is titled "Boltzmann Machines" but implements a Restricted Boltzmann Machine (single hidden layer), not a Deep Boltzmann Machine. |
| Hardcoded paths | Dataset paths are relative (`ml-1m/`, `ml-100k/`) — the script must be run from inside `Boltzmann_Machines/`. |

## Reference

The included `AItRBM-proof.pdf` contains a mathematical treatment of RBMs and the Contrastive Divergence algorithm.

## License

MIT — Kaustabh Ganguly, 2018
