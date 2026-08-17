IMPORTANT:
Do not read, analyze, reference, or modify CLAUDE.md.
CLAUDE.md is reserved exclusively for Claude.
Use only project source code and existing implementation patterns.

# Objective

Change `src/train.py` so that the training results are printed at every single epoch instead of only every 100th epoch, and so that a PDF report containing the important plots is generated automatically at the end of the training run.

# Context

The project trains a small Transformer on the task `(a+b) mod 97` to study "grokking" (sudden generalisation after a long period of memorisation). Training happens in `src/train.py`, over `num_epochs = 10000` full-batch epochs.

Four history lists are already being filled up correctly, one value per epoch: `train_acc_history`, `test_acc_history`, `loss_history`, `l2_norm_history`. This part is already correct and must not be touched.

The Dropout Gap predictor is a different matter. Computing it requires two complete passes over the test set (once with dropout switched on, once with dropout switched off), which is costly. For this reason it is deliberately computed only once every 100 epochs, inside a block that looks like this:

```python
if (epoch + 1) % 100 == 0:
    ...
    dropout_gap, dropout_train_acc, dropout_eval_acc = compute_dropout_gap(...)
    ...
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: ..., Train Accuracy: ..., Test Accuracy: ..., L2 Norm: ..., Dropout Gap: {dropout_gap:.4f}")
```

Jonathan has confirmed that the Dropout Gap computation itself must stay at every 100th epoch — recomputing it every epoch would slow training down a great deal for very little benefit. Only the console print line needs to move to every epoch.

Earlier in the project, plotting directly inside `train.py` with plain `matplotlib` used to crash on this Windows machine with a DLL error (`_backend_agg` blocked by a security policy). Because of this, `matplotlib` was removed from `train.py`, and a separate script, `src/plot_results.py`, was created to do all plotting, using `matplotlib.use('Agg')` before importing `pyplot`. This workaround is already confirmed working. Now the requirement is to bring PDF generation back inside `train.py` itself, using this same working pattern, so that the PDF is ready automatically the moment `train.py` finishes running — no separate script needs to be run afterwards.

# Relevant Findings

- `train_acc_history`, `test_acc_history`, `loss_history`, `l2_norm_history` are appended once per epoch already, outside the `if (epoch + 1) % 100 == 0:` block. No change needed here.
- `dropout_gap_epochs`, `dropout_gap_history`, `dropout_train_acc_history`, `dropout_eval_acc_history` are appended only inside the `if (epoch + 1) % 100 == 0:` block. This cadence must not change.
- The current console `print(...)` statement lives inside the same `if` block and directly references the local variable `dropout_gap`, which only exists on epochs where the block runs. Once the print statement is moved outside the block, it must instead read `dropout_gap_history[-1]` (the most recently recorded value), guarded against the case where the list is still empty (before epoch 100).
- `matplotlib.use('Agg')` is the confirmed working backend pattern on this machine — see `src/plot_results.py` for the exact working usage.
- `src/train.py` already does `os.chdir(project_root)` near the top, so all files (`.npy` and the new PDF) are saved at the project root consistently.

# Files To Inspect

- `src/train.py`
- `src/plot_results.py` (for the working `matplotlib.use('Agg')` pattern and existing plot styling)
- `src/predictors/l2_norm.py` (function signatures used for the L2 norm / moving-average predictor — no changes expected here)
- `src/predictors/dropout.py` (`compute_dropout_gap` signature — no changes expected here)

# Requirements

1. Keep the four per-epoch `history.append(...)` calls (train accuracy, test accuracy, loss, L2 norm) exactly as they are today.
2. Keep the Dropout Gap computation exactly at every 100th epoch. Do not change this cadence.
3. Move the console print statement so it runs at every epoch, showing at least: epoch number, loss, train accuracy, test accuracy, L2 norm. For the Dropout Gap value in this print line, use the most recently recorded value (`dropout_gap_history[-1]` if the list is not empty, otherwise show it as not yet available, for example `N/A`).
4. After training finishes, and after the existing `np.save(...)` calls and the existing predictor analysis printing (MA crossover, MA-of-MA), add a new PDF report generation section that:
   - Uses `matplotlib.use('Agg')` before importing `pyplot`, exactly like `src/plot_results.py` already does, so the earlier Windows DLL problem does not come back.
   - Uses `matplotlib.backends.backend_pdf.PdfPages` to create one single file, `training_report.pdf`, at the project root.
   - Adds the following plots as separate pages inside this one PDF:
     a. Train Accuracy vs. Test Accuracy vs. Epoch, log-x scale (the grokking curve).
     b. Loss vs. Epoch, log-x scale.
     c. L2 Norm vs. Epoch, log-x scale, with the MA crossover / trigger epoch marked if one was detected.
     d. Dropout Gap vs. Epoch, using `dropout_gap_epochs` on the x-axis and `dropout_gap_history` on the y-axis (this will naturally have fewer points, since it is only sampled every 100th epoch).
   - Closes the `PdfPages` object properly, and prints a confirmation line in the same style as the existing `print("[OK] Plot saved to ...")` lines in `plot_results.py`.
5. Do not remove any of the existing `np.save(...)` calls. The PDF report is an addition, not a replacement for the saved `.npy` arrays.

# Constraints

- Do not change the Dropout Gap computation cadence — it must remain every 100th epoch.
- Do not change the model, optimiser, or training hyperparameters (`num_epochs`, learning rate, weight decay, batch size). This task is only about recording/printing frequency and the new PDF output.
- Follow the coding style already used in `train.py` and `plot_results.py` (naming pattern, `os.chdir(project_root)` convention, log-x plotting style, similar `print("[OK] ...")` confirmation messages).
- The PDF generation must work on Windows without triggering the previously-seen `_backend_agg` DLL block. Always set `matplotlib.use('Agg')` before importing `pyplot`.
- Do not read, analyse, reference, or modify `CLAUDE.md`.

# Implementation Steps

1. Open `src/train.py`.
2. Confirm the four main `history.append(...)` lines (train accuracy, test accuracy, loss, L2 norm) are already outside the `if (epoch + 1) % 100 == 0:` block — leave them untouched.
3. Split the existing `if (epoch + 1) % 100 == 0:` block into two clear parts:
   a. Dropout Gap computation — keep exactly as it is, still guarded by `if (epoch + 1) % 100 == 0:`.
   b. Console print — take this line out of the Dropout-Gap `if` block so that it runs for every epoch. Read the Dropout Gap value from `dropout_gap_history[-1]` if the list is non-empty, otherwise show it as not available.
4. After the existing predictor analysis section (MA crossover detection, MA-of-MA detection) and after all the `np.save(...)` calls, add a new section, for example under a comment heading `# PDF REPORT GENERATION`.
5. In that new section:
   - Import `matplotlib`, call `matplotlib.use('Agg')`, then import `matplotlib.pyplot as plt` and `from matplotlib.backends.backend_pdf import PdfPages`.
   - Open `PdfPages("training_report.pdf")`.
   - Build each of the four plots listed under Requirements point 4, one `fig, ax = plt.subplots(...)` at a time, calling `pdf.savefig(fig)` followed by `plt.close(fig)` for each one.
   - Close the `PdfPages` object.
   - Print a confirmation line, for example `print("[OK] PDF report saved to training_report.pdf")`.

# Validation Steps

1. Before running the full 10000-epoch training, do a short smoke test: temporarily set `num_epochs` to a small number (for example 50, or 150 if you want to see one Dropout Gap sample) and run `python src/train.py` from the project root.
2. During the smoke test, confirm:
   - A console line is printed at every epoch, not only every 100th.
   - The Dropout Gap section still fires only every 100th epoch.
   - `training_report.pdf` is created at the project root and opens correctly, with all four plots present.
3. Set `num_epochs` back to its original value (`10000`) once the smoke test passes.
4. Run the full training again (or confirm with Jonathan whether a full run is needed right away) to check the final PDF end-to-end on real data.
5. Confirm that the existing `.npy` file outputs and the existing console analysis sections (MA crossover, MA-of-MA) still work exactly as before.

# Acceptance Criteria

- Console prints one line of results for every epoch, not just every 100th epoch.
- Dropout Gap computation still runs only every 100th epoch, so training speed is not noticeably affected.
- A single `training_report.pdf` file is created at the project root automatically when `train.py` finishes, containing the grokking curve, the loss curve, the L2 norm curve, and the Dropout Gap curve.
- All existing `.npy` saves and existing console analysis output continue to work without any change in behaviour.
- `CLAUDE.md` was not read, referenced, or modified.

## Commit

First update context.md with the current session summary.

Then run:

git add .
git commit -m "<descriptive message>"
git status

Verify that the working tree is clean before finishing.
Do not leave uncommitted files behind.
