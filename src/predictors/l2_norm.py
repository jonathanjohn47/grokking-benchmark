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