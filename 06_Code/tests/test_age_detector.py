"""Unit tests for AGE detector causality."""
import os
import sys
import pytest
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from predictors.age import compute_age_metrics_for_checkpoint

class MockModel:
    def __init__(self):
        self.output_head = torch.nn.Linear(128, 113)
        self._hook_data = None
        self._hook_fn = None
    
    def register_forward_hook(self, fn):
        self._hook_fn = fn
        class MockHandle:
            def remove(self):
                pass
        return MockHandle()
    
    def eval(self):
        pass
    
    def __call__(self, x):
        # Simulate feature extraction
        h = torch.randn(x.shape[0], 128)
        if self._hook_fn:
            self._hook_fn(None, (h,), None)
        return self.output_head(h)

class MockLoader:
    def __init__(self, n_batches=10, batch_size=383):
        self.n_batches = n_batches
        self.batch_size = batch_size
        self._iter = 0
    
    def __iter__(self):
        self._iter = 0
        return self
    
    def __next__(self):
        if self._iter >= self.n_batches:
            raise StopIteration
        self._iter += 1
        x = torch.randn(self.batch_size, 3)
        y = torch.randint(0, 113, (self.batch_size,))
        x[:, 0] = y.float()
        return x, y

def test_featureless_null():
    """Test 1: Random noise should not trigger false positives."""
    model = MockModel()
    loader = MockLoader()
    
    metrics = compute_age_metrics_for_checkpoint(model, loader, "cpu")
    
    # NC1 should be high and stable for random features
    assert metrics["nc1"] > 0.5, "NC1 should be high for random features"
    assert metrics["N"] == 3830, "Should have correct number of samples"

def test_injected_feature():
    """Test 2: Synthetic data with perfect class separation should show low NC1."""
    class PerfectModel(MockModel):
        def __call__(self, x):
            # Create perfectly separated classes
            y = x[:, 0].long()
            h = torch.zeros(x.shape[0], 128)
            for i in range(x.shape[0]):
                h[i, y[i] % 128] = 10.0  # Perfect separation
            
            if self._hook_fn:
                self._hook_fn(None, (h,), None)
            return self.output_head(h)
    
    model = PerfectModel()
    loader = MockLoader()
    
    metrics = compute_age_metrics_for_checkpoint(model, loader, "cpu")
    
    # NC1 should be very low for perfectly separated classes
    assert metrics["nc1"] < 0.01, "NC1 should be very low for perfect separation"
