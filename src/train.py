import matplotlib.pyplot as plt
from torch import nn
from torch.optim import AdamW

from data.modular_arithmetic import get_dataloaders
from models.transformer import Transformer

data_loader = get_dataloaders(97, batch_size=int(0.3 * 97 * 97))
model = Transformer(vocab_size=98, d_model=128)
optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1.0)

print("Optimizer:", optimizer)
cross_entropy_loss = nn.CrossEntropyLoss()

num_epochs = 20000
train_acc_history = []
test_acc_history = []
loss_history = []

for epoch in range(num_epochs):
    total_correct = 0
    total_samples = 0
    for x, y in data_loader[0]:
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

plt.figure(figsize=(8, 5))
plt.plot(range(1, num_epochs + 1), train_acc_history, label="Train Accuracy")
plt.plot(range(1, num_epochs + 1), test_acc_history, label="Test Accuracy")
plt.xscale("log")
plt.xlabel("Epoch (log scale)")
plt.ylabel("Accuracy")
plt.title("Grokking Curve")
plt.legend()
plt.savefig("grokking_curve.png")
plt.show()
