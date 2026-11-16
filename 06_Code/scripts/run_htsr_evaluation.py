#!/usr/bin/env python3
"""Runner to compute HTSR Alpha across all checkpoints for all 5 seeds."""
import json
import os
import sys
import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(REPO_ROOT, "06_Code", "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from models.transformer_four_head import TransformerFourHead
from predictors.htsr_alpha import compute_htsr_metrics_for_checkpoint

RESULTS_DIR = os.path.join(REPO_ROOT, "08_Experiments", "results", "results_per_nanda_power")
MODULUS = 113
D_MODEL = 128
NUM_HEADS = 4
CHECKPOINT_EPOCHS = list(range(0, 40001, 100))

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    for seed in range(5):
        seed_dir = os.path.join(RESULTS_DIR, f"seed_{seed}")
        ckpt_dir = os.path.join(seed_dir, "checkpoints")
        
        with open(os.path.join(seed_dir, "summary.json")) as f:
            summary = json.load(f)
            grok_epoch = summary["grok_epoch"]

        print(f"[Seed {seed}] Computing HTSR Alpha over {len(CHECKPOINT_EPOCHS)} checkpoints...")
        
        model = TransformerFourHead(vocab_size=MODULUS + 1, d_model=D_MODEL, num_heads=NUM_HEADS).to(device)
        model.eval()
        
        alpha_history = []
        
        for epoch in CHECKPOINT_EPOCHS:
            ckpt_path = os.path.join(ckpt_dir, f"model_epoch_{epoch}.pt")
            state = torch.load(ckpt_path, map_location=device, weights_only=True)
            model.load_state_dict(state)
            
            metrics = compute_htsr_metrics_for_checkpoint(model)
            alpha_history.append(metrics["alpha"])
            
            if epoch % 5000 == 0:
                print(f"  epoch {epoch}: alpha = {metrics['alpha']:.4f}")

        alpha_arr = np.array(alpha_history)
        alpha_min_idx = int(np.nanargmin(alpha_arr))
        alpha_min_epoch = CHECKPOINT_EPOCHS[alpha_min_idx]

        out_dir = os.path.join(seed_dir, "htsr")
        os.makedirs(out_dir, exist_ok=True)
        
        block = {
            "htsr_checkpoints": CHECKPOINT_EPOCHS,
            "alpha_history": [float(a) for a in alpha_arr],
            "grok_epoch": grok_epoch,
            "alpha_min_epoch": alpha_min_epoch,
            "alpha_min_value": float(alpha_arr[alpha_min_idx]),
            "alpha_min_to_grok_ratio": float(alpha_min_epoch) / grok_epoch if grok_epoch else None
        }
        with open(os.path.join(out_dir, "htsr_signal.json"), "w") as f:
            json.dump(block, f, indent=2)
            
        print(f"[Seed {seed}] Done. Alpha Min: {alpha_min_epoch}, Grok: {grok_epoch}")

if __name__ == "__main__":
    main()
