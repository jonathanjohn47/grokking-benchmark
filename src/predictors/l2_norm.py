import numpy as np
import torch

def compute_l2_norm(model):
    all_params = torch.cat([p.flatten() for p in model.parameters()])
    l2_norm = torch.norm(all_params).item()
    return l2_norm

def compute_l2_norm_rate_of_decline(l2_norm_history):
    return np.diff(l2_norm_history) *-1


def detect_l2_norm_drop(rate_of_decline, threshold=0.05, skip_epochs=200):
    for i, rate in enumerate(rate_of_decline):
        if i < skip_epochs:  # Skip early epochs
            continue
        if rate > threshold:
            return i
    return None





