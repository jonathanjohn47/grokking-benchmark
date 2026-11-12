from pathlib import Path

import yaml
from torch import tensor
from torch.utils.data import Dataset, DataLoader
import random


def train_fraction_from_yaml():
    """Read task.train_fraction from the benchmark yaml. Not a hardcoded 0.3."""
    config_path = Path(__file__).resolve().parents[2] / "configs" / "nanda_unified.yaml"
    with open(config_path) as handle:
        cfg = yaml.safe_load(handle)
    return float(cfg["task"]["train_fraction"])


def get_dataloaders(number, batch_size, seed):
    """Nanda gen_train_test.

    pairs = [(i, j, p) for i in range(p) for j in range(p)]
    random.seed(seed); random.shuffle(pairs)
    train = pairs[:int(train_fraction * len(pairs))]; test = the rest.

    train_fraction is read from nanda_unified.yaml. Each loader is one
    full batch (the whole split), which is Nanda's full-batch step.
    The batch_size argument is kept so existing call sites still pass it;
    the loader uses the full split length, which equals
    int(train_fraction * p * p) for the training split.
    """
    del batch_size  # full batch is the split itself; see the docstring
    train_fraction = train_fraction_from_yaml()
    pairs = [(i, j, number) for i in range(number) for j in range(number)]
    random.seed(seed)
    random.shuffle(pairs)
    div = int(train_fraction * len(pairs))
    train_pairs = pairs[:div]
    test_pairs = pairs[div:]

    train_dataset = ModularArithmeticDataset(number, train_pairs)
    test_dataset = ModularArithmeticDataset(number, test_pairs)
    train_dataloader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
    return train_dataloader, test_dataloader


class ModularArithmeticDataset(Dataset):
    def __init__(self, number, pairs):
        self.number = number
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        a, b, _equals = self.pairs[idx]
        sequence = [a, b, self.number]
        label = (a + b) % self.number
        return (tensor(sequence), label)


if __name__ == "__main__":
    print(tensor([5, 3, 8, 113]))
