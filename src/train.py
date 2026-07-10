from torch import nn
from torch.optim import AdamW

from data.modular_arithmetic import get_dataloaders
from models.transformer import Transformer

data_loader = get_dataloaders(97, batch_size=int(0.3 * 97 * 97))
model = Transformer(vocab_size=98, d_model=128)
optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1.0)

print("Optimizer:", optimizer)
cross_entropy_loss = nn.CrossEntropyLoss()

for epoch in range(5000):
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

    print(f"Epoch {epoch + 1}: Loss = {loss.item()}, Accuracy = {total_correct / total_samples}")
    print(f"Test Accuracy = {test_total_correct / test_total_samples}")
