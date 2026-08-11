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
    """
    Compute the accuracy gap between training and evaluation modes of the model.
    This is done by comparing the accuracy with dropout enabled (training mode)
    and dropout disabled (evaluation mode).
    """
    # Set model to training mode to enable dropout
    model.train()
    train_accuracy = compute_accuracy(model, data_loader)

    # Set model to evaluation mode to disable dropout
    model.eval()
    eval_accuracy = compute_accuracy(model, data_loader)

    # Calculate the gap
    dropout_gap = train_accuracy - eval_accuracy

    return dropout_gap, train_accuracy, eval_accuracy