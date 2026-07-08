# Python Project Compilation

Project: `/Users/jonathanjohn/Documents/grokking-benchmark/src`

Total Python Files: **3**

---

## data/modular_arithmetic.py

```python
from torch import long, tensor
from torch.utils.data import Dataset, DataLoader, random_split

def generate_pairs(number):
    pairs = []
    for i in range(number):
        for j in range(number):
            pairs.append((i, j, (i + j) % number))
    return pairs


def get_dataloaders(number, batch_size=32):
    modular_arithmetic_dataset = ModularArithmeticDataset(number)
    train_size = int(0.3 * len(modular_arithmetic_dataset))
    test_size = len(modular_arithmetic_dataset) - train_size

    train_dataset, test_dataset = random_split(modular_arithmetic_dataset, [train_size, test_size])
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    return train_dataloader, test_dataloader


class ModularArithmeticDataset(Dataset):
    def __init__(self, number):
        self.pairs = generate_pairs(number)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return (self.get_tensor(self.pairs[idx]), self.pairs[idx][2])
    
    def get_tensor(self, item):
        sequence = [item[0], item[1], 97]
        return tensor(sequence)
    
    
if __name__ == "__main__":
    print(tensor([5, 3, 8, 97]))
```

---

## models/transformer.py

```python
import pandas as pd

from torch import arange, nn
import torch
torch.set_printoptions(profile="full")


class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(3, d_model)
        
    def forward(self, x):
        token_vectors = self.token_embedding(x)
        position_vectors = self.position_embedding(arange(x.size(1)))
        return token_vectors + position_vectors
        
        
        
        
if __name__ == "__main__":
    model = Transformer(vocab_size=97, d_model=128)
```

---

## train.py

```python

```

---

