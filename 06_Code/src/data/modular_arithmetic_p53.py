"""Modular addition dataset for the isolated Salah & Yevick Dropout setup.

Paper: Salah & Yevick, arXiv:2507.11645 / Neural Processing Letters.
Modulus p = 53, train fraction 0.5. Inputs are the pair (i, j), not a
Transformer token sequence. This loader is separate from
modular_arithmetic.py so the Nanda-Unified p=113 substrate stays untouched.
"""

import random

from torch import tensor
from torch.utils.data import DataLoader, Dataset

MODULUS = 53
TRAIN_FRACTION = 0.5


class ModularArithmeticP53Dataset(Dataset):
    """Pairs (i, j) with label (i + j) mod 53."""

    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        a, b = self.pairs[idx]
        # Two integer inputs; the MLP embeds each separately.
        return tensor([a, b]), (a + b) % MODULUS


def get_dataloaders_p53(seed, train_fraction=TRAIN_FRACTION, modulus=MODULUS):
    """Salah & Yevick modular-addition split.

    All p^2 pairs are shuffled with `seed`. Train takes the first
    int(train_fraction * p^2) pairs; test takes the rest. Each loader is
    one full batch (full-batch training), matching the paper's setup and
    this project's usual grokking practice.
    """
    pairs = [(i, j) for i in range(modulus) for j in range(modulus)]
    random.seed(seed)
    random.shuffle(pairs)
    n_train = int(train_fraction * len(pairs))
    train_pairs = pairs[:n_train]
    test_pairs = pairs[n_train:]

    train_dataset = ModularArithmeticP53Dataset(train_pairs)
    test_dataset = ModularArithmeticP53Dataset(test_pairs)
    train_loader = DataLoader(
        train_dataset, batch_size=len(train_dataset), shuffle=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=len(test_dataset), shuffle=False
    )
    return train_loader, test_loader
