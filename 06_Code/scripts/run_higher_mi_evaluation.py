#!/usr/bin/env python3
"""Runner to compute Higher-MI (O-information) across all checkpoints for all 5 seeds."""
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
from predictors.higher_mi import compute_higher_mi_logit_triplet_for_checkpoint, o_information_three_way
from data.modular_arithmetic import get_dataloaders

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

        print(f"[Seed {seed}] Computing Higher-MI over {len(CHECKPOINT_EPOCHS)} checkpoints...")
        
        # Rebuild the exact test split used in training
        _, test_loader = get_dataloaders(number=MODULUS, batch_size=int(0.3 * MODULUS * MODULUS), seed=seed)
        
        model = TransformerFourHead(vocab_size=MODULUS + 1, d_model=D_MODEL, num_heads=NUM_HEADS).to(device)
        model.eval()
        
        omega_history = []
        
        for epoch in CHECKPOINT_EPOCHS:
            ckpt_path = os.path.join(ckpt_dir, f"model_epoch_{epoch}.pt")
            state = torch.load(ckpt_path, map_location=device, weights_only=True)
            model.load_state_dict(state)
            
            triplet = compute_higher_mi_logit_triplet_for_checkpoint(model, test_loader, device)
            omega_mean, _ = o_information_three_way(triplet["correct"], triplet["runner_up"], triplet["rest_mean"])
            omega_history.append(omega_mean)
            
            if epoch % 5000 == 0:
                print(f"  epoch {epoch}: O-info = {omega_mean:.4f}")

        omega_arr = np.array(omega_history)
        max_idx = int(np.argmax(omega_arr))
        max_epoch = CHECKPOINT_EPOCHS[max_idx]

        out_dir = os.path.join(seed_dir, "higher_mi")
        os.makedirs(out_dir, exist_ok=True)
        
        block = {
            "higher_mi_checkpoints": CHECKPOINT_EPOCHS,
            "population_omega_history": [float(o) for o in omega_arr],
            "grok_epoch": grok_epoch,
            "population_omega_max_epoch": max_epoch,
            "population_omega_max_value": float(omega_arr[max_idx]),
            "population_omega_max_to_grok_ratio": float(max_epoch) / grok_epoch if grok_epoch else None
        }
        with open(os.path.join(out_dir, "higher_mi_signal.json"), "w") as f:
            json.dump(block, f, indent=2)
            
        print(f"[Seed {seed}] Done. Max Epoch: {max_epoch}, Grok: {grok_epoch}")

if __name__ == "__main__":
    main()
