import torch
from torch import nn
from torch.optim import AdamW
import numpy as np
import os
from predictors.l2_norm import (
    compute_l2_norm,
    compute_fast_slow_moving_averages,
    detect_ma_crossover,
    compute_ma_of_slow_ma,
    compute_noise_floor,
    detect_ma_of_ma_zero_crossing,
)
from data.modular_arithmetic import get_dataloaders
from models.transformer import Transformer
from predictors.dropout import compute_dropout_gap

# Save results to project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(project_root)



device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Device:", device)

data_loader = get_dataloaders(97, batch_size=int(0.3 * 97 * 97))
model = Transformer(vocab_size=98, d_model=128).to(device)
optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1.0)

print("Optimizer:", optimizer)
cross_entropy_loss = nn.CrossEntropyLoss()

num_epochs = 10000
train_acc_history = []
test_acc_history = []
loss_history = []
l2_norm_history = []
dropout_gap_epochs = []
dropout_gap_history = []
dropout_train_acc_history = []
dropout_eval_acc_history = []

for epoch in range(num_epochs):
    total_correct = 0
    total_samples = 0
    for x, y in data_loader[0]:
        x, y = x.to(device), y.to(device)
        logit = model.forward(x)
        equal_sign_logit = logit[:, 2, :]

        loss = cross_entropy_loss(equal_sign_logit, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        predicted = equal_sign_logit.argmax(dim=1)
        total_correct += (predicted == y).sum().item()
        total_samples += len(y)

    test_total_correct = 0
    test_total_samples = 0

    for x_test, y_test in data_loader[1]:
        x_test, y_test = x_test.to(device), y_test.to(device)
        logit_test = model.forward(x_test)
        equal_sign_logit_test = logit_test[:, 2, :]
        predicted_test = equal_sign_logit_test.argmax(dim=1)
        test_total_correct += (predicted_test == y_test).sum().item()
        test_total_samples += len(y_test)

    train_acc_history.append(total_correct / total_samples)
    test_acc_history.append(test_total_correct / test_total_samples)
    loss_history.append(loss.item())
    l2_norm_history.append(compute_l2_norm(model))

    # Dropout Gap Predictor: now computed at every epoch (previously throttled
    # to every 100th epoch, because it needs two full passes over the test set
    # — dropout ON, then dropout OFF — which is costly). Jonathan has since
    # asked for full per-epoch resolution here as well, so this block now
    # runs unconditionally. This will make each epoch noticeably slower than
    # before, since two extra passes over the test set now happen every time.
    dropout_gap, dropout_train_acc, dropout_eval_acc = compute_dropout_gap(
        model, data_loader[1], dropout_rate=0.9
    )
    dropout_gap_epochs.append(epoch + 1)
    dropout_gap_history.append(dropout_gap)
    dropout_train_acc_history.append(dropout_train_acc)
    dropout_eval_acc_history.append(dropout_eval_acc)

    # compute_dropout_gap() leaves the model in eval() mode with p reset to
    # 0.0. Training must resume in train() mode, so we restore it here before
    # the next epoch's training pass begins.
    model.train()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Train Accuracy: {train_acc_history[-1]:.4f}, Test Accuracy: {test_acc_history[-1]:.4f}, L2 Norm: {l2_norm_history[-1]:.4f}, Dropout Gap: {dropout_gap:.4f}")

np.save("train_acc_history.npy", train_acc_history)
np.save("test_acc_history.npy", test_acc_history)
np.save("loss_history.npy", loss_history)
np.save("l2_norm_history.npy", l2_norm_history)
np.save("dropout_gap_epochs.npy", dropout_gap_epochs)
np.save("dropout_gap_history.npy", dropout_gap_history)
np.save("dropout_train_acc_history.npy", dropout_train_acc_history)
np.save("dropout_eval_acc_history.npy", dropout_eval_acc_history)
print("Training data saved:")
print("  - train_acc_history.npy")
print("  - test_acc_history.npy")
print("  - loss_history.npy")
print("  - l2_norm_history.npy")
print("  - dropout_gap_epochs.npy")
print("  - dropout_gap_history.npy")
print("  - dropout_train_acc_history.npy")
print("  - dropout_eval_acc_history.npy")



# L2 Norm Predictor: Moving Average Crossover Strategy
print("\n" + "="*60)
print("L2 NORM PREDICTOR: Moving Average Crossover")
print("="*60)

# Compute fast and slow moving averages on a log-epoch-uniform grid
# (matches the log-x plot's visual spacing, so the window covers equal
# visual width everywhere instead of scribbling at high epochs)
# Fast MA (window=50 grid points): responsive to recent changes
# Slow MA (window=200 grid points): captures overall trend
epoch_grid, fast_ma, slow_ma = compute_fast_slow_moving_averages(l2_norm_history, fast_window=50, slow_window=200)

# Detect crossover point: where fast MA crosses above slow MA
detection_epoch = detect_ma_crossover(epoch_grid, fast_ma, slow_ma, skip_epochs=100)

# Find when test accuracy actually jumps (crosses 90%)
grok_epoch = np.argmax(np.array(test_acc_history) > 0.9)

# Save data for plotting
np.save("epoch_grid.npy", epoch_grid)
np.save("fast_ma.npy", fast_ma)
np.save("slow_ma.npy", slow_ma)

# Report results
print(f"\nDetection Results:")
if detection_epoch is not None:
    lead_time = grok_epoch - detection_epoch
    print(f"  MA Crossover epoch: {detection_epoch:.1f}")
    print(f"  Grok epoch (test acc > 90%): {grok_epoch}")
    print(f"  Lead time: {lead_time:.1f} epochs")
else:
    print("  No MA crossover detected")
    print(f"  Grok epoch (test acc > 90%): {grok_epoch}")

# Debug stats
print(f"\nL2 Norm Stats:")
print(f"  Min: {np.min(l2_norm_history):.4f}")
print(f"  Max: {np.max(l2_norm_history):.4f}")
print(f"  Final: {l2_norm_history[-1]:.4f}")


# L2 Norm Predictor: MA-of-MA Zero-Crossing Trigger Strategy
print("\n" + "="*60)
print("L2 NORM PREDICTOR: MA of Slow MA — Zero-Crossing Trigger")
print("="*60)

# Apply a second, faster-window MA on top of slow_ma itself. slow_ma has
# already filtered out raw L2 norm noise, so this reveals slow_ma's own
# turning points cleanly instead of chasing noise in the raw signal.
fast_ma_of_slow_ma, ma_of_ma_diff = compute_ma_of_slow_ma(slow_ma, fast_window=20)

# Noise floor: how big this difference is during the quiet early-training
# region, before any real dynamics begin — printed for context only, no
# longer used to decide the trigger (see detect_ma_of_ma_zero_crossing).
noise_floor = compute_noise_floor(ma_of_ma_diff, epoch_grid, quiet_epoch_cutoff=90)

# Trigger: first epoch (after skip_epochs) where the difference crosses from
# positive to negative. Confirmed on two runs to land in a consistent,
# unambiguous spot even though the crossing itself is weak in magnitude.
trigger_epoch = detect_ma_of_ma_zero_crossing(epoch_grid, ma_of_ma_diff, skip_epochs=100)

np.save("fast_ma_of_slow_ma.npy", fast_ma_of_slow_ma)
np.save("ma_of_ma_diff.npy", ma_of_ma_diff)

print(f"\nNoise floor: {noise_floor:.6f}")
if trigger_epoch is not None:
    trigger_lead_time = grok_epoch - trigger_epoch
    print(f"  Trigger epoch (first zero-crossing): {trigger_epoch:.1f}")
    print(f"  Grok epoch (test acc > 90%): {grok_epoch}")
    print(f"  Lead time: {trigger_lead_time:.1f} epochs")
else:
    print("  No trigger detected")
    print(f"  Grok epoch (test acc > 90%): {grok_epoch}")



# Dropout Predictor: gap tracked at every epoch during training (see loop
# above). We simply report the last tracked value here — no need to run
# compute_dropout_gap() a second time on the already-trained model.
print("\n" + "="*60)
print("DROPOUT PREDICTOR: Gap tracked every epoch")
print("="*60)
print(f"\nFinal Dropout Gap Check (rate=0.9, epoch {dropout_gap_epochs[-1]}):")
print(f"  Accuracy with dropout (train mode, p=0.9): {dropout_train_acc_history[-1]:.4f}")
print(f"  Accuracy without dropout (eval mode, p=0.0): {dropout_eval_acc_history[-1]:.4f}")
print(f"  Gap: {dropout_gap_history[-1]:.4f}")


# ======================================================================
# PDF REPORT GENERATION
# ======================================================================
# matplotlib is imported only here, at the very end, after training is
# fully finished. 'Agg' backend must be selected before importing pyplot,
# otherwise matplotlib tries to load a DLL that Windows blocks on this
# machine — same workaround already confirmed working in
# src/plot_results.py.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

print("\n" + "="*60)
print("PDF REPORT GENERATION")
print("="*60)

epochs_axis = range(1, num_epochs + 1)

with PdfPages("training_report.pdf") as pdf:

    # Page 1: Grokking curve — Train vs. Test accuracy
    fig1, ax1 = plt.subplots(figsize=(12, 7))
    ax1.plot(epochs_axis, train_acc_history, label="Train Accuracy", color="steelblue", linewidth=2)
    ax1.plot(epochs_axis, test_acc_history, label="Test Accuracy", color="seagreen", linewidth=2)
    ax1.set_xscale("log")
    ax1.set_xlabel("Epoch (log scale)")
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Grokking Curve: Train vs. Test Accuracy")
    ax1.legend(loc="center right")
    ax1.grid(True, alpha=0.3)
    pdf.savefig(fig1)
    plt.close(fig1)

    # Page 2: Loss curve
    fig2, ax2 = plt.subplots(figsize=(12, 7))
    ax2.plot(epochs_axis, loss_history, color="darkorange", linewidth=2)
    ax2.set_xscale("log")
    ax2.set_xlabel("Epoch (log scale)")
    ax2.set_ylabel("Loss")
    ax2.set_title("Training Loss")
    ax2.grid(True, alpha=0.3)
    pdf.savefig(fig2)
    plt.close(fig2)

    # Page 3: L2 norm curve, with the MA crossover detection epoch marked
    # (detection_epoch was computed earlier in this same script, in the
    # L2 Norm Predictor section).
    fig3, ax3 = plt.subplots(figsize=(12, 7))
    ax3.plot(epochs_axis, l2_norm_history, color="purple", linewidth=2, label="L2 Norm")
    if detection_epoch is not None:
        ax3.axvline(x=detection_epoch, color="red", linestyle="--", linewidth=2,
                    label=f"MA Crossover (epoch {detection_epoch:.0f})")
    ax3.set_xscale("log")
    ax3.set_xlabel("Epoch (log scale)")
    ax3.set_ylabel("L2 Norm")
    ax3.set_title("L2 Norm of Model Weights")
    ax3.legend(loc="upper right")
    ax3.grid(True, alpha=0.3)
    pdf.savefig(fig3)
    plt.close(fig3)

    # Page 4: Dropout Gap — now recorded at every epoch, same resolution as
    # the other three plots, so it uses the same log-x style for consistency.
    fig4, ax4 = plt.subplots(figsize=(12, 7))
    ax4.plot(dropout_gap_epochs, dropout_gap_history, color="crimson", linewidth=2)
    ax4.set_xscale("log")
    ax4.set_xlabel("Epoch (log scale)")
    ax4.set_ylabel("Dropout Gap")
    ax4.set_title("Dropout Gap")
    ax4.grid(True, alpha=0.3)
    pdf.savefig(fig4)
    plt.close(fig4)

    # ------------------------------------------------------------------
    # Numeric data pages: besides the four graphs above, Jonathan asked
    # for the actual per-epoch numbers to be present in the PDF as well,
    # not only the plots. Since num_epochs can run into the thousands,
    # the numbers are laid out as a table and split across multiple
    # pages (rows_per_page rows each), so every page stays readable
    # instead of one unreadable giant table.
    # ------------------------------------------------------------------
    rows_per_page = 45
    table_columns = ["Epoch", "Loss", "Train Acc", "Test Acc", "L2 Norm", "Dropout Gap"]

    table_rows = []
    for i in range(num_epochs):
        table_rows.append([
            str(i + 1),
            f"{loss_history[i]:.4f}",
            f"{train_acc_history[i]:.4f}",
            f"{test_acc_history[i]:.4f}",
            f"{l2_norm_history[i]:.4f}",
            f"{dropout_gap_history[i]:.4f}",
        ])

    num_table_pages = (len(table_rows) + rows_per_page - 1) // rows_per_page

    for page_idx in range(num_table_pages):
        start = page_idx * rows_per_page
        end = start + rows_per_page
        page_rows = table_rows[start:end]

        fig_table, ax_table = plt.subplots(figsize=(8.5, 11))
        ax_table.axis("off")
        ax_table.set_title(
            f"Per-Epoch Results (page {page_idx + 1} of {num_table_pages})",
            fontsize=12, pad=20
        )
        table = ax_table.table(
            cellText=page_rows,
            colLabels=table_columns,
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.3)
        pdf.savefig(fig_table)
        plt.close(fig_table)

print(f"[OK] PDF report saved to training_report.pdf ({4 + num_table_pages} pages: 4 graphs + {num_table_pages} table pages)")