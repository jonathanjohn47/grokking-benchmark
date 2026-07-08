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
        
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        
        self.mlp = nn.Linear(d_model, d_model)

    def forward(self, x):
        token_vectors = self.token_embedding(x)
        position_vectors = self.position_embedding(arange(x.size(1)))
        combined_vector = token_vectors + position_vectors
        query_vector = self.query(combined_vector)
        key_vector = self.key(combined_vector)
        value_vector = self.value(combined_vector)
        
        scores = torch.matmul(query_vector, key_vector.transpose(-2, -1)) / torch.sqrt(torch.tensor(query_vector.size(-1), dtype=torch.float32))
        attention_weights = torch.softmax(scores, dim=-1)
        attended_values = torch.matmul(attention_weights, value_vector)
        mlp_output = self.mlp(attended_values)
        
        print("MLP Output:", mlp_output.shape)
        
        return query_vector, key_vector, value_vector
        
        
if __name__ == "__main__":
    model = Transformer(vocab_size=98, d_model=128)
    
    x = torch.eye(2, 3, dtype=torch.long)
    print(x)
    model.forward(x)
```

---

## train.py

```python

```

---

