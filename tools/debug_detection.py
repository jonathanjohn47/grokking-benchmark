import numpy as np

# Load the data
l2_norm_history = np.load("l2_norm_history.npy")
test_acc_history = np.load("test_acc_history.npy")

# Compute rate of decline
rate_of_decline = np.diff(l2_norm_history) * -1

# Find grok
grok_epoch = np.where(np.array(test_acc_history) > 0.9)[0][0]
print(f"Grok epoch: {grok_epoch}\n")

# Test different thresholds with skip_epochs=200
print("Testing thresholds (skip_epochs=200):")
print("Threshold | Detection Epoch | Lead Time")
print("-" * 45)

for threshold in [0.008, 0.010, 0.012, 0.014, 0.016, 0.020]:
    # Find first epoch after skip where rate > threshold
    detection_epoch = None
    for i in range(200, len(rate_of_decline)):
        if rate_of_decline[i] > threshold:
            detection_epoch = i
            break

    if detection_epoch is not None:
        lead_time = grok_epoch - detection_epoch
        print(f"{threshold:.3f}    | {detection_epoch:14d} | {lead_time:9d}")
    else:
        print(f"{threshold:.3f}    | {'None':14s} | {'N/A':>9s}")

# Show rates around skip boundary and grok
print(f"\nRates around skip boundary (epoch 195-210):")
for i in range(195, min(210, len(rate_of_decline))):
    print(f"  rate[{i:4d}] = {rate_of_decline[i]:.6f}")

print(f"\nRates around grok (epoch {grok_epoch-5} to {grok_epoch+5}):")
for i in range(max(0, grok_epoch-5), min(grok_epoch+5, len(rate_of_decline))):
    print(f"  rate[{i:4d}] = {rate_of_decline[i]:.6f}")
