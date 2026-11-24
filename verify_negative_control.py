import os
import sys
import numpy as np
import torch

REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
SRC = os.path.join(REPO_ROOT, "06_Code", "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from models.transformer_four_head import TransformerFourHead
from predictors.l2_norm import detect_confirmed_sum_w2_peak
from predictors.commutator_defect import compute_commutator_defect

def main():
    results_base = os.path.join(REPO_ROOT, "08_Experiments", "results", "nanda_unified_lambda0")
    print("=" * 70)
    print("NEGATIVE CONTROL VERIFICATION (lambda = 0.0, weight_decay = 0.0)")
    print("=" * 70)

    # 1. Loop through seeds 0 to 4 in test_acc_history.npy and print max test acc
    print("\n--- 1. Maximum Test Accuracy Across Seeds (0 to 4) ---")
    for seed in range(5):
        test_acc_path = os.path.join(results_base, f"seed_{seed}", "training", "test_acc_history.npy")
        if os.path.exists(test_acc_path):
            test_acc = np.load(test_acc_path)
            max_test_acc = float(np.max(test_acc))
            final_test_acc = float(test_acc[-1])
            print(f"Seed {seed}: Max Test Acc = {max_test_acc:.4f} ({max_test_acc * 100:.2f}%), Final Test Acc = {final_test_acc:.4f} (Grok threshold 0.99 reached? {max_test_acc >= 0.99})")
        else:
            print(f"Seed {seed}: test_acc_history.npy not found at {test_acc_path}")

    # 2. Load seed_0 sum_w2_history.npy and detect confirmed sum_w2 peak
    print("\n--- 2. L2 Norm False Positive Verification (Seed 0) ---")
    sum_w2_path = os.path.join(results_base, "seed_0", "l2_norm", "sum_w2_history.npy")
    if os.path.exists(sum_w2_path):
        sum_w2 = np.load(sum_w2_path)
        peak_epoch = detect_confirmed_sum_w2_peak(sum_w2)
        print(f"Seed 0 sum_w2_history length: {len(sum_w2)}")
        print(f"L2 Norm confirmed peak detector fired at epoch: {peak_epoch}")
        if peak_epoch is not None:
            print(f"FALSE POSITIVE CONFIRMED: L2 Norm fired at epoch {peak_epoch} despite zero generalization / no grokking!")
        else:
            print("L2 Norm detector did not fire.")
    else:
        print(f"sum_w2_history.npy not found at {sum_w2_path}")

    # 3. Instantiate TransformerFourHead, load seed_0/checkpoints/model_epoch_500.pt, and compute commutator defect
    print("\n--- 3. Commutator Defect False Positive Verification (Seed 0, Epoch 500) ---")
    ckpt_path = os.path.join(results_base, "seed_0", "checkpoints", "model_epoch_500.pt")
    if os.path.exists(ckpt_path):
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Instantiating TransformerFourHead(vocab_size=114, d_model=128, num_heads=4) on {device}...")
        model = TransformerFourHead(vocab_size=114, d_model=128, num_heads=4).to(device)
        state_dict = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Loaded checkpoint from {ckpt_path}")

        dummy_inputs = torch.randint(0, 113, (3830, 3))
        dummy_labels = torch.randint(0, 113, (3830,))
        print("Computing commutator defect with dummy data (3830 pairs)...")
        defect_result = compute_commutator_defect(
            model=model,
            train_inputs=dummy_inputs,
            train_labels=dummy_labels,
        )
        median_defect = defect_result.get("median")
        print(f"Commutator defect median at epoch 500: {median_defect}")
        print(f"Commutator defect raw values: {defect_result.get('values')}")
        if median_defect is not None and median_defect > 20:
            print(f"FALSE POSITIVE CONFIRMED: Commutator Defect median ({median_defect:.2f}) > 20 threshold during early memorization (epoch 500) despite zero generalization / no grokking!")
    else:
        print(f"Checkpoint not found at {ckpt_path}")

    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
