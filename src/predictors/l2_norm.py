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


def detect_l2_norm_signal_epoch(rate_of_decline):
    if rate_of_decline is None or len(rate_of_decline) == 0:
        return None  

    threshold = np.mean(rate_of_decline) + 2 * np.std(rate_of_decline)
    signal_epochs = np.where(rate_of_decline > threshold)[0]
    return signal_epochs


def detect_l2_norm_drop(rate_of_decline, multiplier=2):
    if rate_of_decline is None or len(rate_of_decline) == 0:
        return None  

    running_average = 0.0
    for i in range(len(rate_of_decline)):
        if i == 0:
            continue
        else:
            running_average = (running_average * (i - 1) + rate_of_decline[i - 1]) / i
            if rate_of_decline[i] > multiplier * running_average:
                return i  
    return None