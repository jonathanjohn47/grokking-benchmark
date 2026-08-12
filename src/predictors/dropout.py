import torch


def compute_accuracy(model, data_loader):
    device = next(model.parameters()).device
    total_correct = 0
    total_samples = 0
    for x, y in data_loader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            logit = model.forward(x)
        equal_sign_logit = logit[:, 2, :]
        predicted = equal_sign_logit.argmax(dim=1)
        total_correct += (predicted == y).sum().item()
        total_samples += len(y)
    return total_correct / total_samples if total_samples > 0 else 0.0


def compute_dropout_gap(model, data_loader, dropout_rate):
    model.train()
    model.dropout1.p = dropout_rate
    model.dropout2.p = dropout_rate
    train_accuracy = compute_accuracy(model, data_loader)

    model.eval()
    model.dropout1.p = 0.0
    model.dropout2.p = 0.0
    eval_accuracy = compute_accuracy(model, data_loader)

    return train_accuracy - eval_accuracy, train_accuracy, eval_accuracy