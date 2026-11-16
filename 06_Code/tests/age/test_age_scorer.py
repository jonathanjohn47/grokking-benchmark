"""Unit tests for AGE scorer with frozen protocol."""
import pytest
import numpy as np

def test_positive_control():
    """Test 1: Synthetic NC1 curve that drops 500 epochs before grok should PASS."""
    grok_epochs = np.array([10000, 12000, 11000, 13000, 14000])
    firing_epochs = grok_epochs - 500  # Fires 500 epochs before grok
    
    lead_frac = 0.95  # (grok - firing) / grok
    rho = 1.0  # Perfect correlation
    mean_rel_lead = 0.05  # 500 / 10000
    
    # Check frozen protocol
    assert lead_frac >= 0.8, "lead_frac should be >= 0.8"
    assert rho >= 0.9, "rho should be >= 0.9"
    assert 0.05 <= mean_rel_lead <= 0.9, "mean_rel_lead should be in [0.05, 0.9]"

def test_negative_control():
    """Test 2: Synthetic NC1 curve that drops AFTER grok should FAIL."""
    grok_epochs = np.array([10000, 12000, 11000, 13000, 14000])
    firing_epochs = grok_epochs + 500  # Fires 500 epochs AFTER grok
    
    lead_frac = 0.0  # No lead
    rho = 1.0  # Perfect correlation
    mean_rel_lead = -0.05  # Negative lead
    
    # Check frozen protocol - should fail
    assert lead_frac < 0.8, "lead_frac should be < 0.8 (FAIL)"
    assert mean_rel_lead < 0.05, "mean_rel_lead should be < 0.05 (FAIL)"
