import numpy as np
import torch
from scipy.ndimage import uniform_filter1d

def compute_l2_norm(model):
    all_params = torch.cat([p.flatten() for p in model.parameters()])
    l2_norm = torch.norm(all_params).item()
    return l2_norm


def compute_sum_of_squared_weights(model):
    """
    Sum of squared weights over ALL parameters (sum w^2). This is the exact
    quantity Nanda et al. plot in Figure 7 ("Total Sum of Squared Weights").
    Note compute_l2_norm(model) == sqrt(compute_sum_of_squared_weights(model));
    both are logged so the Nanda-Unified weight-norm curve can be compared
    to the paper directly without re-squaring.
    """
    total = 0.0
    for p in model.parameters():
        total += torch.sum(p.detach() ** 2).item()
    return total


# Which top-level module names roll up into which reported group. Used by
# compute_per_module_sum_of_squared_weights(). Groups follow
# configs/nanda_unified.yaml -> logging.weight_norm_per_module_sum_w2.
PER_MODULE_GROUPS = {
    "token_embedding": ["token_embedding"],
    "position_embedding": ["position_embedding"],
    "attention_qkv": ["query", "key", "value"],
    "mlp": ["mlp_in", "mlp_out"],
    "output_head": ["output_head"],
}


def compute_per_module_sum_of_squared_weights(model):
    """
    Sum w^2 split by module group (token_embedding, position_embedding,
    attention_qkv, mlp, output_head). Lets the weight-space predictors see
    WHERE the weight norm moves — e.g. confirm the early transient (if any)
    is the embedding table and a pre-grok bump is in attention/MLP.

    Returns a dict {group_name: sum_w2}. Any parameter whose top-level name
    is not in PER_MODULE_GROUPS is ignored (there are none in the current
    model, but this keeps the function safe against future modules).
    """
    prefix_to_group = {}
    for group, prefixes in PER_MODULE_GROUPS.items():
        for prefix in prefixes:
            prefix_to_group[prefix] = group

    out = {group: 0.0 for group in PER_MODULE_GROUPS}
    for name, p in model.named_parameters():
        top_level = name.split(".")[0]
        group = prefix_to_group.get(top_level)
        if group is not None:
            out[group] += torch.sum(p.detach() ** 2).item()
    return out

def compute_l2_norm_rate_of_decline(l2_norm_history):
    return np.diff(l2_norm_history) *-1


def detect_l2_norm_drop(rate_of_decline, threshold=0.05, skip_epochs=200):
    for i, rate in enumerate(rate_of_decline):
        if i < skip_epochs:  # Skip early epochs
            continue
        if rate > threshold:
            return i
    return None


def compute_acceleration(rate_of_decline):
    """
    Compute the second derivative of L2 norm (acceleration).
    acceleration[i] = rate[i] - rate[i-1]
    Positive = decay rate is increasing (speeding up)
    Negative = decay rate is decreasing (slowing down)
    Sign change = inflection point (change in decay character)
    """
    acceleration = np.diff(rate_of_decline)
    return acceleration


def detect_inflection(acceleration, skip_epochs=200):
    """
    Detect inflection point where acceleration changes sign.
    This is where the decay pattern changes character — exactly what happens
    before/during the grok transition.
    Returns the epoch index where the sign flip occurs, or None.
    """
    if len(acceleration) <= skip_epochs:
        return None

    # Check from skip_epochs onwards for sign changes
    for i in range(skip_epochs, len(acceleration)):
        # acceleration[i-1] * acceleration[i] < 0 means opposite signs (sign flip)
        if i > 0 and acceleration[i-1] * acceleration[i] < 0:
            return i

    return None


def apply_moving_average(data, window_size=50):
    """
    Apply moving average smoothing to reduce noise.
    window_size: number of epochs to average over (default 50).
    Returns smoothed array (same length as input, padded at edges).
    """
    return uniform_filter1d(np.array(data), size=window_size, mode='nearest')


def resample_to_log_uniform_grid(data, num_points=None):
    """
    Resample data onto a grid uniformly spaced in log10(epoch) — i.e. the
    SAME spacing the data has when viewed on a log-x plot.
    Without this, a fixed window_size (in raw epoch index) covers a huge
    visual chunk of the plot near epoch 1 and a tiny sliver near epoch N,
    which is why smoothing looked over-smoothed early and scribbly late.
    Returns (epoch_grid, resampled_data) where epoch_grid is uniform in
    log10-space (non-uniform in linear epoch terms).
    """
    data = np.array(data)
    num_epochs = len(data)
    epochs = np.arange(1, num_epochs + 1)
    log_epochs = np.log10(epochs)

    if num_points is None:
        num_points = num_epochs

    log_grid = np.linspace(log_epochs[0], log_epochs[-1], num_points)
    resampled = np.interp(log_grid, log_epochs, data)
    epoch_grid = 10 ** log_grid

    return epoch_grid, resampled


def compute_fast_slow_moving_averages(l2_norm_history, fast_window=50, slow_window=200, num_points=None):
    """
    Compute two moving averages on a LOG-EPOCH-UNIFORM grid, so the window
    covers equal VISUAL width everywhere on a log-x plot (fixes both the
    over-smoothing near epoch 1 and the scribbling near epoch N).

    Steps:
      1. Log-transform L2 norm values (handles the exponential decay in y).
      2. Resample onto a grid uniform in log10(epoch) (handles the log-x axis).
      3. Apply the moving average on that resampled grid.
      4. Convert back to linear L2 norm space.

    fast_window / slow_window are now measured in grid points, which (since
    the grid is log-uniform) corresponds to a constant *proportional* width
    in epochs everywhere along the curve.

    Returns (epoch_grid, fast_ma, slow_ma) — always plot fast_ma/slow_ma
    against epoch_grid, not against range(1, num_epochs+1).
    """
    l2_norm_history = np.array(l2_norm_history)
    log_l2_norm = np.log(l2_norm_history)

    epoch_grid, log_l2_norm_resampled = resample_to_log_uniform_grid(log_l2_norm, num_points=num_points)

    fast_ma_log = apply_moving_average(log_l2_norm_resampled, window_size=fast_window)
    slow_ma_log = apply_moving_average(log_l2_norm_resampled, window_size=slow_window)

    fast_ma = np.exp(fast_ma_log)
    slow_ma = np.exp(slow_ma_log)

    return epoch_grid, fast_ma, slow_ma


def detect_ma_crossover(epoch_grid, fast_ma, slow_ma, skip_epochs=100):
    """
    Detect where fast MA crosses above slow MA, searching along the
    log-uniform grid. skip_epochs is a real epoch value (not a grid index) —
    crossovers before this epoch are ignored (initialization noise).
    Returns the real epoch (float) of the crossover, or None.
    """
    if len(fast_ma) != len(slow_ma):
        return None

    for i in range(len(fast_ma) - 1):
        if epoch_grid[i] < skip_epochs:
            continue
        # Previous: fast <= slow, Current: fast > slow (crossover from below)
        if fast_ma[i] <= slow_ma[i] and fast_ma[i + 1] > slow_ma[i + 1]:
            return epoch_grid[i + 1]

    return None


def compute_ma_of_slow_ma(slow_ma, fast_window=20):
    """
    Apply a second, faster-window moving average on top of the already-smoothed
    slow_ma. Since slow_ma has already filtered out raw L2 norm noise, this
    reveals slow_ma's OWN turning points cleanly — the difference between the
    two tracks how sharply slow_ma is currently bending.
    Returns (fast_ma_of_slow_ma, diff) where diff = fast_ma_of_slow_ma - slow_ma.
    """
    fast_ma_of_slow_ma = apply_moving_average(slow_ma, window_size=fast_window)
    diff = fast_ma_of_slow_ma - slow_ma
    return fast_ma_of_slow_ma, diff


def compute_noise_floor(diff, epoch_grid, quiet_epoch_cutoff=90):
    """
    Estimate the normal noise level of the MA-of-MA difference from the quiet
    early-training region (before any real dynamics begin), so later spikes
    can be judged against what "flat" actually looks like for this run.
    """
    quiet_mask = epoch_grid < quiet_epoch_cutoff
    return np.std(diff[quiet_mask])


def detect_ma_of_ma_trigger(epoch_grid, diff, noise_floor, threshold_multiplier=10, skip_epochs=100):
    """
    Fire on the first epoch (after skip_epochs) where diff climbs past
    threshold_multiplier x noise_floor — the rising edge of a real signal.
    Causal: only needs the noise floor (known early) and points seen so far,
    unlike picking the curve's peak, which requires seeing the whole future.
    Returns the real epoch (float) of the trigger, or None.
    """
    threshold = threshold_multiplier * noise_floor
    for i in range(len(diff)):
        if epoch_grid[i] < skip_epochs:
            continue
        if diff[i] > threshold:
            return epoch_grid[i]
    return None


def detect_ma_of_ma_zero_crossing(epoch_grid, diff, skip_epochs=100):
    """
    Fire on the first epoch (after skip_epochs) where diff crosses from
    positive to negative (the first "green -> red" zero crossing).
    Confirmed on two separate training runs to land in a consistent,
    unambiguous spot even though the crossing's magnitude is weak.
    Returns the real epoch (float) of the crossing, or None.
    """
    for i in range(len(diff) - 1):
        if epoch_grid[i] < skip_epochs:
            continue
        if diff[i] >= 0 and diff[i + 1] < 0:
            return epoch_grid[i + 1]
    return None

