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
torch.set_printoptions(threshold=20, edgeitems=3)


class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(3, d_model)
        
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        
        self.mlp = nn.Linear(d_model, d_model)
        
        self.output_head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        token_vectors = self.token_embedding(x)
        print("token_vectors:", token_vectors.shape, "\n", token_vectors)
        position_vectors = self.position_embedding(arange(x.size(3)))
        print("position_vectors:", position_vectors.shape, "\n", position_vectors)
        combined_vector = token_vectors + position_vectors
        print("combined_vector:", combined_vector.shape, "\n", combined_vector)
        query_vector = self.query(combined_vector)
        print("query_vector:", query_vector.shape, "\n", query_vector)
        key_vector = self.key(combined_vector)
        print("key_vector:", key_vector.shape, "\n", key_vector)
        value_vector = self.value(combined_vector)
        print("value_vector:", value_vector.shape, "\n", value_vector)

        scores = torch.matmul(query_vector, key_vector.transpose(-2, -1)) / torch.sqrt(torch.tensor(query_vector.size(-1), dtype=torch.float32))
        print("scores:", scores.shape, "\n", scores)
        attention_weights = torch.softmax(scores, dim=-1)
        print("attention_weights:", attention_weights.shape, "\n", attention_weights)
        attended_values = torch.matmul(attention_weights, value_vector)
        print("attended_values:", attended_values.shape, "\n", attended_values)
        mlp_output = self.mlp(attended_values)
        print("mlp_output:", mlp_output.shape, "\n", mlp_output)

        logits = self.output_head(mlp_output)
        print("logits:", logits.shape, "\n", logits)
        
        return logits
        
        
if __name__ == "__main__":
    model = Transformer(vocab_size=98, d_model=128)
    
    x = torch.eye(2, 3, dtype=torch.long)
    model.forward(x)
```

---

## train.py

```python

```

---

