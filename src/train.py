import torch
from torch import nn
from torch.optim import AdamW
import numpy as np
import os
from predictors.l2_norm import compute_l2_norm_rate_of_decline, detect_l2_norm_drop, compute_l2_norm
from data.modular_arithmetic import get_dataloaders
from models.transformer import Transformer

# Save results to project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)



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



# Compute rate of decline
rate_of_decline = compute_l2_norm_rate_of_decline(l2_norm_history)

# Detect when the rule fires
detection_epoch = detect_l2_norm_drop(rate_of_decline, multiplier=2)

# Find when test accuracy actually jumps (crosses 90%)
grok_epoch = np.argmax(np.array(test_acc_history) > 0.9)

# Lead time
if detection_epoch is not None:
    lead_time = grok_epoch - detection_epoch
    print(f"Detection epoch: {detection_epoch}")
    print(f"Grok epoch (test acc > 90%): {grok_epoch}")
    print(f"Lead time: {lead_time} epochs")
else:
    print("No detection")

# Debug: print rate of decline and threshold
rate_of_decline_array = np.array(rate_of_decline)
print(f"\nRate of decline stats:")
print(f"  Min: {rate_of_decline_array.min():.6f}")
print(f"  Max: {rate_of_decline_array.max():.6f}")
print(f"  Mean: {rate_of_decline_array.mean():.6f}")
print(f"  Std: {rate_of_decline_array.std():.6f}")

# Check a few spike indices before filtering
cum_sum = np.cumsum(rate_of_decline_array)
prior_sum = np.concatenate([[0], cum_sum[:-1]])
divisor = np.maximum(np.arange(len(rate_of_decline_array)), 1)
prior_average = prior_sum / divisor
threshold = 2 * prior_average

spike_indices_before_filter = np.where(rate_of_decline_array > threshold)[0]
print(f"\nSpike indices before filtering: {spike_indices_before_filter[:20]}")  # First 20
print(f"Total spikes before filtering: {len(spike_indices_before_filter)}")
