import numpy as np
import torch

def compute_l2_norm(model):
    all_params = torch.cat([p.flatten() for p in model.parameters()])
    l2_norm = torch.norm(all_params).item()
    return l2_norm

def compute_l2_norm_rate_of_decline(l2_norm_history):
    return np.diff(l2_norm_history) *-1


def detect_l2_norm_drop(rate_of_decline, multiplier=2):
    cum_sum = np.cumsum(rate_of_decline)
    prior_sum = np.concatenate([[0], cum_sum[:-1]])
    
    # Divide by the count of prior elements, avoiding 0/0
    divisor = np.maximum(np.arange(len(rate_of_decline)), 1)
    prior_average = prior_sum / divisor
    
    threshold = multiplier * prior_average
    spike_indices = np.where(rate_of_decline > threshold)[0]
    
    # Exclude epoch 0 (no prior data)
    spike_indices = spike_indices[spike_indices > 0]
    
    return spike_indices[0] if len(spike_indices) > 0 else None



