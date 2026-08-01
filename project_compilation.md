# Python Project Compilation

Project: `C:\Users\jonat\StudioProjects\grokking-benchmark\src`

Total Python Files: **4**

---

## data\modular_arithmetic.py

```python
from torch import long, tensor
from torch.utils.data import Dataset, DataLoader, random_split

def generate_pairs(number):
    pairs = []
    for i in range(number):
        for j in range(number):
            pairs.append((i, j, (i + j) % number))
    return pairs

def get_dataloaders(number, batch_size):
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

## models\transformer.py

```python
from torch import arange, nn
import torch


class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(3, d_model)

        self.query = nn.Linear(d_model, d_model, bias=False)
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)

        self.mlp_in = nn.Linear(d_model, 4 * d_model, bias=False)
        self.mlp_activation = nn.ReLU()
        self.mlp_out = nn.Linear(4 * d_model, d_model, bias=False)


        self.output_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x):
        token_vectors = self.token_embedding(x)
        position_vectors = self.position_embedding(arange(x.size(1), device=x.device))
        combined_vector = token_vectors + position_vectors
        query_vector = self.query(combined_vector)
        key_vector = self.key(combined_vector)
        value_vector = self.value(combined_vector)

        scores = torch.matmul(query_vector, key_vector.transpose(-2, -1)) / torch.sqrt(torch.tensor(query_vector.size(-1), dtype=torch.float32))
        attention_weights = torch.softmax(scores, dim=-1)
        attended_values = combined_vector + torch.matmul(attention_weights, value_vector)
        mlp_output = attended_values + self.mlp_out(self.mlp_activation(self.mlp_in(attended_values)))


        logits = self.output_head(mlp_output)

        return logits
        
        
if __name__ == "__main__":
    model = Transformer(vocab_size=98, d_model=128)
    
    x = torch.eye(2, 3, dtype=torch.long)
    model.forward(x)
```

---

## predictors\l2_norm.py

```python
"""
L2 Norm Predictor

This module implements the L2 norm predictor.
"""
import numpy as np
# Add your implementation here
def compute_l2_norm(model):
    squared_sum = 0.0
    for param in model.parameters():
        squared_sum += (param ** 2).sum().item()
    l2_norm = squared_sum ** 0.5
    return l2_norm


def compute_l2_norm_rate_of_decline(l2_norm_history):
    if len(l2_norm_history) < 2:
        return None  

    return np.diff(l2_norm_history) * -1


def detect_l2_norm_signal_epoch(rate_of_decline):
    if rate_of_decline is None or len(rate_of_decline) == 0:
        return None  

    threshold = np.mean(rate_of_decline) + 2 * np.std(rate_of_decline)
    signal_epochs = np.where(rate_of_decline > threshold)[0]
    return signal_epochs


def detect_l2_norm_drop(rate_of_decline, multiplier=2):
    if rate_of_decline is None or len(rate_of_decline) == 0:
        return None  

    running_average = 0.0
    for i in range(len(rate_of_decline)):
        if i == 0:
            continue
        elif i > 0:
            running_average = (running_average * (i - 1) + rate_of_decline[i - 1]) / i
        if rate_of_decline[i] > multiplier * running_average:
            return i  
    return None
```

---

## train.py

```python
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.optim import AdamW

from data.modular_arithmetic import get_dataloaders
from models.transformer import Transformer
from predictors.l2_norm import compute_l2_norm

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Device:", device)

data_loader = get_dataloaders(97, batch_size=int(0.3 * 97 * 97))
model = Transformer(vocab_size=98, d_model=128).to(device)
optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1.0)

print("Optimizer:", optimizer)
cross_entropy_loss = nn.CrossEntropyLoss()

num_epochs = 20000
train_acc_history = []
test_acc_history = []
loss_history = []
l2_norm_history = []

for epoch in range(num_epochs):
    total_correct = 0
    total_samples = 0
    for x, y in data_loader[0]:
        x, y = x.to(device), y.to(device)
        logit = model.forward(x)
        equal_sign_logit = logit[:, 2, :]

        loss = cross_entropy_loss(equal_sign_logit, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        predicted = equal_sign_logit.argmax(dim=1)
        total_correct += (predicted == y).sum().item()
        total_samples += len(y)
        
    test_total_correct = 0
    test_total_samples = 0
        
    for x_test, y_test in data_loader[1]:
        x_test, y_test = x_test.to(device), y_test.to(device)
        logit_test = model.forward(x_test)
        equal_sign_logit_test = logit_test[:, 2, :]
        predicted_test = equal_sign_logit_test.argmax(dim=1)
        test_total_correct += (predicted_test == y_test).sum().item()
        test_total_samples += len(y_test)

    train_acc_history.append(total_correct / total_samples)
    test_acc_history.append(test_total_correct / test_total_samples)
    loss_history.append(loss.item())
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch + 1}: Loss = {loss_history[-1]}, Train Acc = {train_acc_history[-1]}, Test Acc = {test_acc_history[-1]}")

    
    l2_norm = compute_l2_norm(model)
    l2_norm_history.append(l2_norm)

plt.figure(figsize=(8, 5))
plt.plot(range(1, num_epochs + 1), train_acc_history, label="Train Accuracy")
plt.plot(range(1, num_epochs + 1), test_acc_history, label="Test Accuracy")
plt.xscale("log")
plt.xlabel("Epoch (log scale)")
plt.ylabel("Accuracy")
plt.title("Grokking Curve")
plt.legend()
plt.savefig("grokking_curve.png")
plt.figure(figsize=(8, 5))
plt.plot(range(1, num_epochs + 1), l2_norm_history, label="L2 Norm")
plt.xscale("log")
plt.xlabel("Epoch (log scale)")
plt.ylabel("L2 Norm")
plt.title("Weight Norm Over Training")
plt.legend()
plt.savefig("l2_norm_curve.png")

plt.show()
```

---

