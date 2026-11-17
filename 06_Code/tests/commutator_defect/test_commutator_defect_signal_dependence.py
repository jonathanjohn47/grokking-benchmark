import os
import sys
import numpy as np
import pytest
import torch
import torch.nn as nn

CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from src.predictors.commutator_defect import compute_commutator_defect

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3, 1, bias=False)
        # Initialize weights so gradients are non-zero
        self.fc.weight.data = torch.ones(1, 3)

def test_zero_defect():
    """Test Category 1: Null Signal (Identical Gradients).
    If batch A and batch B are identical, their gradients are identical, 
    meaning they perfectly commute. The defect must be exactly zero.
    """
    model = DummyModel()
    # A single example repeated 10 times
    x = torch.tensor([[1.0, 2.0, 3.0]]).repeat(10, 1)
    y = torch.tensor([[1.0]]).repeat(10, 1)
    
    # Pass identical inputs for train and batch sampling
    res = compute_commutator_defect(
        model, x, y, eta=1e-3, k_pairs=1, batch_size=10, seed=42
    )
    
    assert np.isclose(res["median"], 0.0, atol=1e-7), f"Expected 0.0, got {res['median']}"

def test_high_defect():
    """Test Category 2: Injected Feature (Non-Commuting Gradients).
    If batch A and batch B point in completely different directions 
    (highly non-linear loss landscape), the defect should be positive.
    """
    model = DummyModel()
    # Extremely varied inputs to create conflicting gradients
    x = torch.randn(100, 3)
    y = torch.randn(100, 1)
    
    res = compute_commutator_defect(
        model, x, y, eta=1e-3, k_pairs=2, batch_size=10, seed=42
    )
    
    assert res["median"] > 1e-5, f"Expected > 0, got {res['median']}"
