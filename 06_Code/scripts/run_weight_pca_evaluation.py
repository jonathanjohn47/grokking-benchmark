#!/usr/bin/env python3
"""Runner to compute Weight-PCA metrics across all checkpoints for all 5 seeds."""
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
from predictors.weight_pca import compute_weight_pca_metrics_for_checkpoint

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

        print(f"[Seed {seed}] Computing Weight-PCA over {len(CHECKPOINT_EPOCHS)} checkpoints...")
        
        model = TransformerFourHead(vocab_size=MODULUS + 1, d_model=D_MODEL, num_heads=NUM_HEADS).to(device)
        model.eval()
        
        norm_eff_rank_history = []
        
        for epoch in CHECKPOINT_EPOCHS:
            ckpt_path = os.path.join(ckpt_dir, f"model_epoch_{epoch}.pt")
            state = torch.load(ckpt_path, map_location=device, weights_only=True)
            model.load_state_dict(state)
            
            metrics = compute_weight_pca_metrics_for_checkpoint(model)
            norm_eff_rank_history.append(metrics["norm_eff_rank"])
            
            if epoch % 5000 == 0:
                print(f"  epoch {epoch}: norm_eff_rank = {metrics['norm_eff_rank']:.4f}")

        rank_arr = np.array(norm_eff_rank_history)
        min_idx = int(np.argmin(rank_arr))
        min_epoch = CHECKPOINT_EPOCHS[min_idx]

        out_dir = os.path.join(seed_dir, "weight_pca")
        os.makedirs(out_dir, exist_ok=True)
        
        block = {
            "weight_pca_checkpoints": CHECKPOINT_EPOCHS,
            "norm_eff_rank_history": [float(r) for r in rank_arr],
            "grok_epoch": grok_epoch,
            "norm_eff_rank_min_epoch": min_epoch,
            "norm_eff_rank_min_value": float(rank_arr[min_idx]),
            "norm_eff_rank_min_to_grok_ratio": float(min_epoch) / grok_epoch if grok_epoch else None
        }
        with open(os.path.join(out_dir, "weight_pca_signal.json"), "w") as f:
            json.dump(block, f, indent=2)
            
        print(f"[Seed {seed}] Done. Min Epoch: {min_epoch}, Grok: {grok_epoch}")

if __name__ == "__main__":
    main()
