# Figures for the 40,000-epoch retrain

These plots are drawn only from `aggregate.json` and the `seed_0` … `seed_4` json and npy files in `08_Experiments/results/results_per_nanda_power`. No checkpoint (`.pt`) file is used. The script is `06_Code/scripts/plot_results_per_nanda_power.py`.

The horizontal axis on the accuracy, loss, and L2 plots is the stored epoch index: array position `0` is the measurement before the first optimizer step, and the grok numbers in `aggregate.json` are that same index. The axis is logarithmic, so index `0` is left outside the frame. Vertical lines on the accuracy plot are the grok epochs at test-accuracy cuts 0.99 (primary), 0.95, and 0.9, plus the dropout-variance peak epoch saved in each seed's summary. The L2 plot marks the same three grok cuts, the global peak of the sum of squared weights, and the confirmed-peak firing epoch.

The L2 curve is the per-epoch total sum of squared weights, `l2_norm/sum_w2_history.npy`. The peak epoch and the confirmed-peak epoch are computed from that array. The parameter L2 norm is not drawn.

The dropout-variance curve uses `dropout_variance_checkpoints.npy`. The saved checkpoint at epoch 0 is omitted from the line because a log axis cannot place it. The variance peak is the epoch stored in `dropout_variance_signal.json`. On this retrain that peak is after the grok epoch.

A curve of dropout gap against epoch was not drawn. `dropout_gap_epochs.npy` contains only the final epoch, so there is no gap history to plot.

## Overview

Test accuracy for seeds 0–4. Each dot is that seed's stored 0.99 grok index.

![Test accuracy, all seeds](overview_test_accuracy.png)

Grok epoch for each seed at cuts 0.99, 0.95, and 0.9, taken from `aggregate.json`.

![Grok epochs by seed and cut](overview_grok_epochs.png)

## Seed 0

Train and test accuracy, with grok cuts and the L2 and dropout-variance detection epochs.

![Seed 0 accuracy](seed_0_train_test_accuracy.png)

Train and test cross-entropy loss (log vertical axis), with the three grok cuts.

![Seed 0 loss](seed_0_train_test_loss.png)

Total sum of squared weights, with grok cuts 0.99, 0.95, and 0.9, the global peak epoch, and the confirmed-peak firing epoch.

![Seed 0 L2 norm](seed_0_l2_norm.png)

Variance of test accuracy across 100 stochastic forward passes at dropout rate 0.5, on the saved checkpoint grid.

![Seed 0 dropout variance](seed_0_dropout_variance.png)

## Seed 1

Train and test accuracy, with grok cuts and the L2 and dropout-variance detection epochs.

![Seed 1 accuracy](seed_1_train_test_accuracy.png)

Train and test cross-entropy loss (log vertical axis), with the three grok cuts.

![Seed 1 loss](seed_1_train_test_loss.png)

Total sum of squared weights, with grok cuts 0.99, 0.95, and 0.9, the global peak epoch, and the confirmed-peak firing epoch.

![Seed 1 L2 norm](seed_1_l2_norm.png)

Variance of test accuracy across 100 stochastic forward passes at dropout rate 0.5, on the saved checkpoint grid.

![Seed 1 dropout variance](seed_1_dropout_variance.png)

## Seed 2

Train and test accuracy, with grok cuts and the L2 and dropout-variance detection epochs.

![Seed 2 accuracy](seed_2_train_test_accuracy.png)

Train and test cross-entropy loss (log vertical axis), with the three grok cuts.

![Seed 2 loss](seed_2_train_test_loss.png)

Total sum of squared weights, with grok cuts 0.99, 0.95, and 0.9, the global peak epoch, and the confirmed-peak firing epoch.

![Seed 2 L2 norm](seed_2_l2_norm.png)

Variance of test accuracy across 100 stochastic forward passes at dropout rate 0.5, on the saved checkpoint grid.

![Seed 2 dropout variance](seed_2_dropout_variance.png)

## Seed 3

Train and test accuracy, with grok cuts and the L2 and dropout-variance detection epochs.

![Seed 3 accuracy](seed_3_train_test_accuracy.png)

Train and test cross-entropy loss (log vertical axis), with the three grok cuts.

![Seed 3 loss](seed_3_train_test_loss.png)

Total sum of squared weights, with grok cuts 0.99, 0.95, and 0.9, the global peak epoch, and the confirmed-peak firing epoch.

![Seed 3 L2 norm](seed_3_l2_norm.png)

Variance of test accuracy across 100 stochastic forward passes at dropout rate 0.5, on the saved checkpoint grid.

![Seed 3 dropout variance](seed_3_dropout_variance.png)

## Seed 4

Train and test accuracy, with grok cuts and the L2 and dropout-variance detection epochs.

![Seed 4 accuracy](seed_4_train_test_accuracy.png)

Train and test cross-entropy loss (log vertical axis), with the three grok cuts.

![Seed 4 loss](seed_4_train_test_loss.png)

Total sum of squared weights, with grok cuts 0.99, 0.95, and 0.9, the global peak epoch, and the confirmed-peak firing epoch.

![Seed 4 L2 norm](seed_4_l2_norm.png)

Variance of test accuracy across 100 stochastic forward passes at dropout rate 0.5, on the saved checkpoint grid.

![Seed 4 dropout variance](seed_4_dropout_variance.png)
