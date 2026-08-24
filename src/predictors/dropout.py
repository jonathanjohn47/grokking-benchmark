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

def compute_dropout_gap_multi_rate(model, data_loader, dropout_rates):
    # Step 1: START (inputs: model, data_loader, dropout_rates)

    # Step 2: Setup Evaluation Mode
    model.eval()
    model.dropout1.p = 0.0
    model.dropout2.p = 0.0

    # Step 3: Compute Clean Accuracy (execute ONLY ONCE)
    clean_accuracy = compute_accuracy(model, data_loader)

    # Step 4: Initialize Results Storage
    results = {}

    # Step 5: FOR each rate in dropout_rates
    for dropout_rate in dropout_rates:
        # --- Switch to train mode with dropout active ---
        model.train()
        model.dropout1.p = dropout_rate
        model.dropout2.p = dropout_rate

        # Compute train accuracy (with dropout)
        train_accuracy = compute_accuracy(model, data_loader)

        # Compute dropout gap (train_accuracy - clean_accuracy)
        dropout_gap = train_accuracy - clean_accuracy

        # Store results
        results[dropout_rate] = {
            "train_accuracy": train_accuracy,
            "eval_accuracy": clean_accuracy,      # == clean_accuracy (dropout=0, eval mode)
            "dropout_gap": dropout_gap,
            "clean_accuracy": clean_accuracy
        }

    # Step 6: Loop Decision — "More rates?" (handled by for-loop)

    # Step 7: Final Cleanup — restore model to clean eval state
    model.eval()
    model.dropout1.p = 0.0
    model.dropout2.p = 0.0

    # Step 8: RETURN results — END
    return results