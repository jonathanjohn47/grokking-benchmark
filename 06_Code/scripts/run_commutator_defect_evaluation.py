#!/usr/bin/env python3
"""Runner to compute Commutator Defect across all checkpoints for all 5 seeds."""
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
from predictors.commutator_defect import compute_commutator_defect, find_commutator_defect_event_epoch
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

        print(f"[Seed {seed}] Computing Commutator Defect over {len(CHECKPOINT_EPOCHS)} checkpoints...")
        
        # Rebuild the exact training split
        train_loader, _ = get_dataloaders(number=MODULUS, batch_size=int(0.3 * MODULUS * MODULUS), seed=seed)
        train_inputs, train_labels = next(iter(train_loader))
        
        model = TransformerFourHead(vocab_size=MODULUS + 1, d_model=D_MODEL, num_heads=NUM_HEADS).to(device)
        model.eval()
        
        median_history = []
        
        for epoch in CHECKPOINT_EPOCHS:
            ckpt_path = os.path.join(ckpt_dir, f"model_epoch_{epoch}.pt")
            state = torch.load(ckpt_path, map_location=device, weights_only=True)
            model.load_state_dict(state)
            
            res = compute_commutator_defect(model, train_inputs, train_labels, eta=1e-3, k_pairs=5, batch_size=512, seed=20261103)
            median_history.append(res["median"])
            
            if epoch % 5000 == 0:
                print(f"  epoch {epoch}: median D = {res['median']:.6f}")

        # Find the event epoch (the onset of the drop)
        event_epoch = find_commutator_defect_event_epoch(CHECKPOINT_EPOCHS, median_history)

        out_dir = os.path.join(seed_dir, "commutator_defect")
        os.makedirs(out_dir, exist_ok=True)
        
        block = {
            "commutator_defect_checkpoints": CHECKPOINT_EPOCHS,
            "median_history": [float(m) for m in median_history],
            "grok_epoch": grok_epoch,
            "event_epoch": event_epoch,
            "event_to_grok_ratio": float(event_epoch) / grok_epoch if (event_epoch is not None and grok_epoch) else None
        }
        with open(os.path.join(out_dir, "commutator_defect_signal.json"), "w") as f:
            json.dump(block, f, indent=2)
            
        print(f"[Seed {seed}] Done. Event Epoch: {event_epoch}, Grok: {grok_epoch}")

if __name__ == "__main__":
    main()
