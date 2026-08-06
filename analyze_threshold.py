import numpy as np

# Load the data
l2_norm_history = np.load("l2_norm_history.npy")
test_acc_history = np.load("test_acc_history.npy")

# Compute rate of decline (how much L2 norm drops each epoch)
rate_of_decline = np.diff(l2_norm_history) * -1

# Find when test accuracy crosses 90% (grok event)
grok_epoch = np.where(np.array(test_acc_history) > 0.9)[0][0]
print(f"Grok epoch (test acc > 90%): {grok_epoch}")
print(f"L2 norm at grok: {l2_norm_history[grok_epoch]:.4f}")

# Analyze rate of decline
print(f"\nRate of decline statistics:")
print(f"  Min: {rate_of_decline.min():.6f}")
print(f"  Max: {rate_of_decline.max():.6f}")
print(f"  Mean: {rate_of_decline.mean():.6f}")
print(f"  Median: {np.median(rate_of_decline):.6f}")
print(f"  Std: {rate_of_decline.std():.6f}")

# Look at rate of decline around the grok region (500 epochs before and after)
grok_window_start = max(0, grok_epoch - 500)
grok_window_end = min(len(rate_of_decline), grok_epoch + 500)
rates_around_grok = rate_of_decline[grok_window_start:grok_window_end]

print(f"\nRate of decline in the grok window (epoch {grok_window_start}-{grok_window_end}):")
print(f"  Mean: {rates_around_grok.mean():.6f}")
print(f"  Max: {rates_around_grok.max():.6f}")
print(f"  Percentile 75: {np.percentile(rates_around_grok, 75):.6f}")
print(f"  Percentile 90: {np.percentile(rates_around_grok, 90):.6f}")

# Look at rates BEFORE the grok (epoch 500-grok_epoch)
if grok_epoch > 500:
    pre_grok_start = 500
    rates_before_grok = rate_of_decline[pre_grok_start:grok_epoch]
    print(f"\nRate of decline BEFORE grok (epoch {pre_grok_start}-{grok_epoch}):")
    print(f"  Mean: {rates_before_grok.mean():.6f}")
    print(f"  Max: {rates_before_grok.max():.6f}")
    print(f"  Percentile 75: {np.percentile(rates_before_grok, 75):.6f}")
    print(f"  Percentile 90: {np.percentile(rates_before_grok, 90):.6f}")

# Recommended thresholds
print(f"\n=== THRESHOLD RECOMMENDATIONS ===")
pre_grok_mean = rates_before_grok.mean()
pre_grok_p90 = np.percentile(rates_before_grok, 90)
grok_window_mean = rates_around_grok.mean()
grok_window_max = rates_around_grok.max()

print(f"Pre-grok mean rate: {pre_grok_mean:.6f}")
print(f"Pre-grok 90th percentile: {pre_grok_p90:.6f}")
print(f"Grok window mean rate: {grok_window_mean:.6f}")
print(f"Grok window max rate: {grok_window_max:.6f}")

print(f"\nSuggested thresholds (aim for detection ~500-1000 epochs before grok):")
print(f"  Conservative (low false positives): {pre_grok_p90:.6f}")
print(f"  Balanced: {grok_window_mean:.6f}")
print(f"  Aggressive (catch early): {pre_grok_mean:.6f}")
