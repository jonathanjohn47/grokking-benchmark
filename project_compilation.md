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

        scores = torch.matmul(query_vector, key_vector.transpose(-2, -1)) / (query_vector.size(-1) ** 0.5)
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

## plot_results.py

```python
import numpy as np
import matplotlib.pyplot as plt
import os

# Look for files in project root (parent of src/)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)

# Load saved training data
train_acc = np.load("train_acc_history.npy")
test_acc = np.load("test_acc_history.npy")
loss = np.load("loss_history.npy")
l2_norm = np.load("l2_norm_history.npy")

num_epochs = len(train_acc)

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Grokking curve (Accuracy)
axes[0, 0].plot(range(1, num_epochs + 1), train_acc, label="Train Accuracy", linewidth=2)
axes[0, 0].plot(range(1, num_epochs + 1), test_acc, label="Test Accuracy", linewidth=2)
axes[0, 0].set_xscale("log")
axes[0, 0].set_xlabel("Epoch (log scale)")
axes[0, 0].set_ylabel("Accuracy")
axes[0, 0].set_title("Grokking Curve: Accuracy")
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Loss
axes[0, 1].plot(range(1, num_epochs + 1), loss, color="red", linewidth=2)
axes[0, 1].set_xscale("log")
axes[0, 1].set_xlabel("Epoch (log scale)")
axes[0, 1].set_ylabel("Loss")
axes[0, 1].set_title("Training Loss")
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: L2 Norm
axes[1, 0].plot(range(1, num_epochs + 1), l2_norm, color="green", linewidth=2)
axes[1, 0].set_xscale("log")
axes[1, 0].set_xlabel("Epoch (log scale)")
axes[1, 0].set_ylabel("L2 Norm")
axes[1, 0].set_title("Model Weight L2 Norm")
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Train-Test Accuracy Gap
gap = np.array(train_acc) - np.array(test_acc)
axes[1, 1].plot(range(1, num_epochs + 1), gap, color="purple", linewidth=2)
axes[1, 1].set_xscale("log")
axes[1, 1].set_xlabel("Epoch (log scale)")
axes[1, 1].set_ylabel("Train Acc - Test Acc")
axes[1, 1].set_title("Generalization Gap")
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("grokking_analysis.png", dpi=150, bbox_inches="tight")
print("✓ Plot saved to grokking_analysis.png")
print(f"  - Epochs trained: {num_epochs}")
print(f"  - Final train accuracy: {train_acc[-1]:.4f}")
print(f"  - Final test accuracy: {test_acc[-1]:.4f}")
print(f"  - Final loss: {loss[-1]:.6f}")
print(f"  - Final L2 norm: {l2_norm[-1]:.4f}")
```

---

## train.py

```python
import torch
from torch import nn
from torch.optim import AdamW
import numpy as np
import os

from data.modular_arithmetic import get_dataloaders
from models.transformer import Transformer

# Save results to project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)

def compute_l2_norm(model):
    all_params = torch.cat([p.flatten() for p in model.parameters()])
    l2_norm = torch.norm(all_params).item()
    return l2_norm

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
    l2_norm_history.append(compute_l2_norm(model))
    if (epoch + 1) % 100 == 0:
        compute_l2 = l2_norm_history[-1]
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Train Accuracy: {train_acc_history[-1]:.4f}, Test Accuracy: {test_acc_history[-1]:.4f}, L2 Norm: {compute_l2:.4f}")

np.save("train_acc_history.npy", train_acc_history)
np.save("test_acc_history.npy", test_acc_history)
np.save("loss_history.npy", loss_history)
np.save("l2_norm_history.npy", l2_norm_history)
print("Training data saved:")
print("  - train_acc_history.npy")
print("  - test_acc_history.npy")
print("  - loss_history.npy")
print("  - l2_norm_history.npy")
```

---

