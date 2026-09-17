import torch
from torch import nn
from torch.optim import AdamW
import numpy as np
import os

from data.modular_arithmetic import get_dataloaders
from models.model_shadow_with_layernorm import Transformer

# Save results to project root (same pattern as train.py)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)

# Fixed seed, so this run can later be repeated with the same starting
# weights and the same train/test split. NOTE: train.py itself does not
# currently set any seed. If you want a true apples-to-apples comparison
# against the original model, add this same line to train.py yourself
# before that run too (temporarily) - this script does not modify train.py.
torch.manual_seed(1337)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Device:", device)

data_loader = get_dataloaders(97, batch_size=int(0.3 * 97 * 97))
model = Transformer(vocab_size=98, d_model=128).to(device)
optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1.0)

print("Optimizer:", optimizer)
cross_entropy_loss = nn.CrossEntropyLoss()

num_epochs = 10000
train_acc_history = []
test_acc_history = []
loss_history = []

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

    if (epoch + 1) % 100 == 0 or epoch == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, "
              f"Train Accuracy: {train_acc_history[-1]:.4f}, "
              f"Test Accuracy: {test_acc_history[-1]:.4f}")

# Saved under a "shadow_ln_" prefix on purpose, so these files never
# overwrite train.py's own train_acc_history.npy / test_acc_history.npy /
# loss_history.npy, which the L2 Norm and Dropout predictors depend on.
np.save("shadow_ln_train_acc_history.npy", train_acc_history)
np.save("shadow_ln_test_acc_history.npy", test_acc_history)
np.save("shadow_ln_loss_history.npy", loss_history)

final_test_acc = test_acc_history[-1]
test_acc_array = np.array(test_acc_history)
grok_epoch = int(np.argmax(test_acc_array > 0.9)) if (test_acc_array > 0.9).any() else None

print("\n" + "=" * 60)
print("SHADOW MODEL (LayerNorm) RESULT")
print("=" * 60)
print(f"Final Test Accuracy: {final_test_acc:.4f}")
print(f"Grok epoch (test acc > 90%): {grok_epoch}")
print("Compare this Final Test Accuracy against the original model's known")
print("plateau value: 0.995446 (99.5446%) - see context.md, Aug 7 session.")
