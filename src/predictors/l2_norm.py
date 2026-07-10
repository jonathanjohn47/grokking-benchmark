"""
L2 Norm Predictor

This module implements the L2 norm predictor.
"""

# Add your implementation here
def compute_l2_norm(model):
    squared_sum = 0.0
    for param in model.parameters():
        squared_sum += (param ** 2).sum().item()
    l2_norm = squared_sum ** 0.5
    return l2_norm