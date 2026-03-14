"""
Restricted Boltzmann Machine for Movie Recommendations

Uses Contrastive Divergence (CD-k) on the MovieLens 100K dataset
to learn latent features from user-movie rating data and predict
whether a user will like a movie.
"""

import os
import numpy as np
import pandas as pd
import torch

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NB_HIDDEN = 100
BATCH_SIZE = 100
NB_EPOCHS = 10
CD_K = 10  # Gibbs sampling steps per weight update


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def load_data():
    """Load and prepare the MovieLens 100K training/test splits."""
    train_path = os.path.join(SCRIPT_DIR, "ml-100k", "u1.base")
    test_path = os.path.join(SCRIPT_DIR, "ml-100k", "u1.test")

    if not os.path.isfile(train_path):
        raise FileNotFoundError(
            f"Training data not found at {train_path}. "
            "Make sure the ml-100k dataset is in the Boltzmann_Machines directory."
        )
    if not os.path.isfile(test_path):
        raise FileNotFoundError(
            f"Test data not found at {test_path}. "
            "Make sure the ml-100k dataset is in the Boltzmann_Machines directory."
        )

    training_set = pd.read_csv(train_path, delimiter="\t", header=None).values.astype(int)
    test_set = pd.read_csv(test_path, delimiter="\t", header=None).values.astype(int)
    return training_set, test_set


def build_user_movie_matrix(data, nb_users, nb_movies):
    """Convert sparse (user, movie, rating) rows into a dense user×movie matrix."""
    matrix = []
    for user_id in range(1, nb_users + 1):
        user_mask = data[:, 0] == user_id
        movie_ids = data[:, 1][user_mask]
        user_ratings = np.zeros(nb_movies)
        user_ratings[movie_ids - 1] = data[:, 2][user_mask]
        matrix.append(user_ratings.tolist())
    return matrix


def binarize_ratings(tensor):
    """Convert numeric ratings to binary: liked (>=3) → 1, not liked (<3) → 0, unrated → -1."""
    tensor[tensor == 0] = -1
    tensor[tensor == 1] = 0
    tensor[tensor == 2] = 0
    tensor[tensor >= 3] = 1
    return tensor


# ---------------------------------------------------------------------------
# RBM Model
# ---------------------------------------------------------------------------
class RBM:
    """Restricted Boltzmann Machine trained with Contrastive Divergence."""

    def __init__(self, nv, nh, device=DEVICE):
        self.device = device
        self.W = torch.randn(nh, nv, device=device)
        self.a = torch.randn(1, nh, device=device)   # hidden bias
        self.b = torch.randn(1, nv, device=device)    # visible bias

    def sample_h(self, x):
        """Sample hidden units given visible units."""
        wx = torch.mm(x, self.W.t())
        activation = wx + self.a.expand_as(wx)
        p_h_given_v = torch.sigmoid(activation)
        return p_h_given_v, torch.bernoulli(p_h_given_v)

    def sample_v(self, y):
        """Sample visible units given hidden units."""
        wy = torch.mm(y, self.W)
        activation = wy + self.b.expand_as(wy)
        p_v_given_h = torch.sigmoid(activation)
        return p_v_given_h, torch.bernoulli(p_v_given_h)

    def update_weights(self, v0, vk, ph0, phk):
        """Single CD weight update step."""
        self.W += torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)
        self.b += torch.sum(v0 - vk, dim=0)
        self.a += torch.sum(ph0 - phk, dim=0)


# ---------------------------------------------------------------------------
# Training & Evaluation
# ---------------------------------------------------------------------------
def train_rbm(rbm, training_set, nb_users):
    """Train the RBM using Contrastive Divergence."""
    print(f"Training RBM on {DEVICE} for {NB_EPOCHS} epochs ...\n")
    for epoch in range(1, NB_EPOCHS + 1):
        train_loss = 0.0
        counter = 0.0
        for start in range(0, nb_users - BATCH_SIZE, BATCH_SIZE):
            vk = training_set[start : start + BATCH_SIZE].clone()
            v0 = training_set[start : start + BATCH_SIZE]
            ph0, _ = rbm.sample_h(v0)
            for _ in range(CD_K):
                _, hk = rbm.sample_h(vk)
                _, vk = rbm.sample_v(hk)
                vk[v0 < 0] = v0[v0 < 0]  # keep unrated movies masked
            phk, _ = rbm.sample_h(vk)
            rbm.update_weights(v0, vk, ph0, phk)
            train_loss += torch.mean(torch.abs(v0[v0 >= 0] - vk[v0 >= 0])).item()
            counter += 1.0
        print(f"  Epoch {epoch:>2d}/{NB_EPOCHS}  —  loss: {train_loss / counter:.4f}")
    print()


def test_rbm(rbm, training_set, test_set, nb_users):
    """Evaluate the RBM on the held-out test set."""
    test_loss = 0.0
    counter = 0.0
    with torch.no_grad():
        for user_id in range(nb_users):
            v = training_set[user_id : user_id + 1]
            vt = test_set[user_id : user_id + 1]
            if len(vt[vt >= 0]) > 0:
                _, h = rbm.sample_h(v)
                _, v = rbm.sample_v(h)
                test_loss += torch.mean(torch.abs(vt[vt >= 0] - v[vt >= 0])).item()
                counter += 1.0
    print(f"Test loss: {test_loss / counter:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # Load data
    raw_train, raw_test = load_data()

    nb_users = int(max(raw_train[:, 0].max(), raw_test[:, 0].max()))
    nb_movies = int(max(raw_train[:, 1].max(), raw_test[:, 1].max()))

    # Build dense matrices and convert to tensors
    training_set = torch.tensor(
        build_user_movie_matrix(raw_train, nb_users, nb_movies),
        dtype=torch.float32,
        device=DEVICE,
    )
    test_set = torch.tensor(
        build_user_movie_matrix(raw_test, nb_users, nb_movies),
        dtype=torch.float32,
        device=DEVICE,
    )

    # Binarize ratings
    training_set = binarize_ratings(training_set)
    test_set = binarize_ratings(test_set)

    # Create and train the RBM
    nv = training_set.shape[1]
    rbm = RBM(nv, NB_HIDDEN, device=DEVICE)
    train_rbm(rbm, training_set, nb_users)
    test_rbm(rbm, training_set, test_set, nb_users)


if __name__ == "__main__":
    main()
