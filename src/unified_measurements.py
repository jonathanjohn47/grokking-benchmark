"""
Unified measurement collection for L2 Norm and Dropout predictors.
Ensures consistent, complete measurements across single-head and four-head models.
"""

import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


class PredictorMeasurements:
    """Collects and saves all measurements for both predictors."""

    def __init__(self, output_dir, model_type="single_head"):
        self.output_dir = output_dir
        self.model_type = model_type
        self.create_subdirs()

    def create_subdirs(self):
        """Create subdirectories for each predictor."""
        self.l2_norm_dir = os.path.join(self.output_dir, "l2_norm")
        self.dropout_dir = os.path.join(self.output_dir, "dropout")
        self.spectral_dir = os.path.join(self.output_dir, "spectral")
        self.reports_dir = os.path.join(self.output_dir, "reports")
        self.training_dir = os.path.join(self.output_dir, "training")

        for dir_path in [self.l2_norm_dir, self.dropout_dir, self.spectral_dir,
                          self.reports_dir, self.training_dir]:
            os.makedirs(dir_path, exist_ok=True)

    def save_training_data(self, train_acc, test_acc, loss):
        """Save core training metrics."""
        np.save(os.path.join(self.training_dir, "train_acc_history.npy"), train_acc)
        np.save(os.path.join(self.training_dir, "test_acc_history.npy"), test_acc)
        np.save(os.path.join(self.training_dir, "loss_history.npy"), loss)

    # ========== L2 NORM MEASUREMENTS ==========

    def save_l2_norm_data(self, l2_norm_history, epoch_grid, fast_ma, slow_ma,
                         fast_ma_of_slow_ma, ma_of_ma_diff, detection_epoch=None,
                         sum_w2_history=None, per_module_sum_w2_history=None):
        """Save all L2 Norm measurements.

        sum_w2_history: optional per-epoch list of sum-of-squared-weights
            (the Nanda Figure 7 quantity; == l2_norm_history ** 2).
        per_module_sum_w2_history: optional per-epoch list of dicts
            {group_name: sum_w2} from
            predictors.l2_norm.compute_per_module_sum_of_squared_weights.
            Saved as per_module_sum_w2.npy shaped [n_groups, n_epochs] with
            the row order recorded in per_module_sum_w2_names.npy.
        """

        # Raw L2 norm
        np.save(os.path.join(self.l2_norm_dir, "l2_norm_history.npy"), l2_norm_history)

        # sum w^2 (Nanda Fig 7 quantity) — total and per module
        if sum_w2_history is not None:
            np.save(os.path.join(self.l2_norm_dir, "sum_w2_history.npy"),
                    np.asarray(sum_w2_history, dtype=float))
        if per_module_sum_w2_history:
            group_names = list(per_module_sum_w2_history[0].keys())
            per_module_matrix = np.array(
                [[row[g] for row in per_module_sum_w2_history] for g in group_names],
                dtype=float,
            )
            np.save(os.path.join(self.l2_norm_dir, "per_module_sum_w2.npy"), per_module_matrix)
            np.save(os.path.join(self.l2_norm_dir, "per_module_sum_w2_names.npy"),
                    np.array(group_names))

        # Smoothed L2 norm (simple low-pass filter)
        l2_norm_smoothed = self._smooth_signal(l2_norm_history, window=50)
        np.save(os.path.join(self.l2_norm_dir, "l2_norm_smoothed.npy"), l2_norm_smoothed)

        # Moving averages
        np.save(os.path.join(self.l2_norm_dir, "epoch_grid.npy"), epoch_grid)
        np.save(os.path.join(self.l2_norm_dir, "fast_ma.npy"), fast_ma)
        np.save(os.path.join(self.l2_norm_dir, "slow_ma.npy"), slow_ma)
        np.save(os.path.join(self.l2_norm_dir, "fast_ma_of_slow_ma.npy"), fast_ma_of_slow_ma)
        np.save(os.path.join(self.l2_norm_dir, "ma_of_ma_diff.npy"), ma_of_ma_diff)

        # Acceleration derivatives
        accel_raw = np.diff(l2_norm_history)
        accel_smoothed = self._smooth_signal(accel_raw, window=50)
        accel_double_smoothed = self._smooth_signal(accel_smoothed, window=50)

        np.save(os.path.join(self.l2_norm_dir, "acceleration_raw.npy"), accel_raw)
        np.save(os.path.join(self.l2_norm_dir, "acceleration_smoothed.npy"), accel_smoothed)
        np.save(os.path.join(self.l2_norm_dir, "acceleration_double_smoothed.npy"), accel_double_smoothed)

        if detection_epoch is not None:
            np.save(os.path.join(self.l2_norm_dir, "detection_epoch.npy"), np.array([detection_epoch]))

    # ========== DROPOUT MEASUREMENTS ==========

    def save_dropout_data(self, dropout_gap_epochs, dropout_gap_history_by_rate,
                         dropout_train_acc_by_rate, dropout_eval_acc_by_rate, dropout_rates):
        """Save all Dropout measurements — full multi-rate sweep only.

        There is no single-rate (p=0.9) record any more. Every array is
        indexed [rate_index, epoch_index], with rate_index following the
        order of dropout_rates.npy.
        """

        np.save(os.path.join(self.dropout_dir, "dropout_gap_epochs.npy"), dropout_gap_epochs)
        np.save(os.path.join(self.dropout_dir, "dropout_rates.npy"), np.array(dropout_rates))

        gap_by_rate = np.array([dropout_gap_history_by_rate[r] for r in dropout_rates])
        train_acc_by_rate = np.array([dropout_train_acc_by_rate[r] for r in dropout_rates])
        eval_acc_by_rate = np.array([dropout_eval_acc_by_rate[r] for r in dropout_rates])

        np.save(os.path.join(self.dropout_dir, "dropout_gap_by_rate.npy"), gap_by_rate)
        np.save(os.path.join(self.dropout_dir, "dropout_train_acc_by_rate.npy"), train_acc_by_rate)
        np.save(os.path.join(self.dropout_dir, "dropout_eval_acc_by_rate.npy"), eval_acc_by_rate)

    # ========== SPECTRAL MEASUREMENTS ==========

    def save_spectral_data(self, spectral_checkpoints, spectral_history):
        """Save all Spectral (Predictor 3) measurements — Canatar et al.
        2021 task-model-alignment version (NOT the old per-weight-matrix
        stable_rank version, which has been removed).

        spectral_checkpoints: list/array of epoch indices, one per
            checkpoint that was evaluated (same role as
            dropout_variance_checkpoints.npy).
        spectral_history: dict of per-checkpoint lists, all the same
            length as spectral_checkpoints:
                k_90               : list[int]
                k_95               : list[int]
                alignment_score    : list[float]
                entropy            : list[float]
                eigenvalues_history: list[list[float]]  (top-50 eta_k per checkpoint)
                cumulative_power_history: list[list[float]]  (C(1..100) per checkpoint)
            built by calling
            predictors.spectral.compute_spectral_metrics_for_checkpoint at
            each checkpoint and collecting each returned field.

        Saves (all under self.spectral_dir):
            spectral_checkpoints.npy      [n_checkpoints]
            spectral_k90.npy             [n_checkpoints]
            spectral_k95.npy             [n_checkpoints]
            spectral_alignment.npy       [n_checkpoints]
            spectral_entropy.npy         [n_checkpoints]
            spectral_eigenvalues.npy     [n_checkpoints, 50]
            spectral_cumulative_power.npy [n_checkpoints, 100]
        """
        np.save(os.path.join(self.spectral_dir, "spectral_checkpoints.npy"),
                np.array(spectral_checkpoints, dtype=int))
        np.save(os.path.join(self.spectral_dir, "spectral_k90.npy"),
                np.array(spectral_history["k_90"], dtype=float))
        np.save(os.path.join(self.spectral_dir, "spectral_k95.npy"),
                np.array(spectral_history["k_95"], dtype=float))
        np.save(os.path.join(self.spectral_dir, "spectral_alignment.npy"),
                np.array(spectral_history["alignment_score"], dtype=float))
        np.save(os.path.join(self.spectral_dir, "spectral_entropy.npy"),
                np.array(spectral_history["entropy"], dtype=float))

        eig = self._stack_ragged(spectral_history["eigenvalues_history"], width=50)
        np.save(os.path.join(self.spectral_dir, "spectral_eigenvalues.npy"), eig)

        cum = self._stack_ragged(spectral_history["cumulative_power_history"], width=100)
        np.save(os.path.join(self.spectral_dir, "spectral_cumulative_power.npy"), cum)

    @staticmethod
    def _stack_ragged(list_of_lists, width):
        """[n_checkpoints] lists of <= `width` floats -> float array
        [n_checkpoints, width], right-padded with the last value (NaN if a
        row is empty). For the Nanda-Unified run every row is already
        exactly `width` long (N = 3830 >> width), so this is just a guard."""
        n = len(list_of_lists)
        out = np.full((n, width), np.nan, dtype=float)
        for i, row in enumerate(list_of_lists):
            row = np.asarray(row, dtype=float)
            m = min(len(row), width)
            out[i, :m] = row[:m]
            if 0 < m < width:
                out[i, m:] = row[m - 1]
        return out

    # ========== VISUALIZATION GENERATION ==========

    def generate_l2_norm_visualizations(self, l2_norm_history, epoch_grid, fast_ma, slow_ma,
                                       fast_ma_of_slow_ma, ma_of_ma_diff, test_acc_history,
                                       grok_epoch, detection_epoch=None, trigger_epoch=None):
        """Generate standalone L2 Norm visualization graphs."""

        num_epochs = len(l2_norm_history)
        epochs_axis = range(1, num_epochs + 1)

        # 1. L2 Norm curve
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.plot(epochs_axis, l2_norm_history, color="purple", linewidth=2)
        if detection_epoch is not None:
            ax.axvline(x=detection_epoch, color="red", linestyle="--", linewidth=2,
                       label=f"MA Crossover (epoch {detection_epoch:.0f})")
        ax.axvline(x=grok_epoch, color="green", linestyle=":", linewidth=2,
                   label=f"Grok epoch (epoch {grok_epoch})")
        ax.set_xscale("log")
        ax.set_xlabel("Epoch (log scale)")
        ax.set_ylabel("L2 Norm")
        ax.set_title("L2 Norm of Model Weights")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.l2_norm_dir, "l2_norm_curve.png"), dpi=150, bbox_inches='tight')
        plt.close(fig)

        # 2. MA crossover detection
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.plot(epoch_grid, fast_ma, label="Fast MA (window=50)", linewidth=2, color="blue")
        ax.plot(epoch_grid, slow_ma, label="Slow MA (window=200)", linewidth=2, color="orange")
        if detection_epoch is not None:
            ax.axvline(x=detection_epoch, color="red", linestyle="--", linewidth=2,
                       label=f"Crossover (epoch {detection_epoch:.0f})")
        ax.axvline(x=grok_epoch, color="green", linestyle=":", linewidth=2,
                   label=f"Grok epoch (epoch {grok_epoch})")
        ax.set_xscale("log")
        ax.set_xlabel("Epoch (log scale)")
        ax.set_ylabel("L2 Norm")
        ax.set_title("Moving Average Crossover Detection")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.l2_norm_dir, "ma_of_slow_ma_crossover.png"), dpi=150, bbox_inches='tight')
        plt.close(fig)

        # 3. MA-of-MA differential
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.plot(epoch_grid, ma_of_ma_diff, color="darkred", linewidth=2, label="MA-of-MA Differential")
        ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5, alpha=0.5)
        if trigger_epoch is not None:
            ax.axvline(x=trigger_epoch, color="red", linestyle="--", linewidth=2,
                       label=f"Zero-crossing (epoch {trigger_epoch:.0f})")
        ax.axvline(x=grok_epoch, color="green", linestyle=":", linewidth=2,
                   label=f"Grok epoch (epoch {grok_epoch})")
        ax.set_xscale("log")
        ax.set_xlabel("Epoch (log scale)")
        ax.set_ylabel("Differential")
        ax.set_title("MA-of-MA Differential: Zero-Crossing Trigger")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.l2_norm_dir, "ma_of_ma_diff.png"), dpi=150, bbox_inches='tight')
        plt.close(fig)

        # 4. MA-of-MA differential vs grokking (linear scale)
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.plot(epoch_grid, ma_of_ma_diff, color="darkred", linewidth=2, label="MA-of-MA Differential")
        ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5, alpha=0.5)
        if trigger_epoch is not None:
            ax.axvline(x=trigger_epoch, color="red", linestyle="--", linewidth=2,
                       label=f"Zero-crossing (epoch {trigger_epoch:.0f})")
        ax.axvline(x=grok_epoch, color="green", linestyle=":", linewidth=2,
                   label=f"Grok epoch (epoch {grok_epoch})")
        ax.set_xscale("linear")
        ax.set_xlabel("Epoch (linear scale)")
        ax.set_ylabel("Differential")
        ax.set_title("MA-of-MA Differential: Linear Scale")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.l2_norm_dir, "ma_of_ma_diff_linear.png"), dpi=150, bbox_inches='tight')
        plt.close(fig)

    def generate_dropout_visualizations(self, dropout_gap_epochs, dropout_gap_history_by_rate,
                                       dropout_rates, test_acc_history, grok_epoch):
        """Generate standalone Dropout visualization graphs."""

        # Multi-rate sweep: all 5 rates on one plot
        fig, ax = plt.subplots(figsize=(12, 7))
        for rate in dropout_rates:
            ax.plot(
                dropout_gap_epochs,
                dropout_gap_history_by_rate[rate],
                linewidth=2,
                label=f"rate={rate}"
            )
        ax.axvline(x=grok_epoch, color="green", linestyle=":", linewidth=2,
                   label=f"Grok epoch ({grok_epoch})")
        ax.set_xscale("log")
        ax.set_xlabel("Epoch (log scale)")
        ax.set_ylabel("Dropout Gap")
        ax.set_title("Dropout Gap: Multi-Rate Sweep (p=0.1, 0.3, 0.5, 0.7, 0.9)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.dropout_dir, "dropout_gap_curve.png"), dpi=150, bbox_inches='tight')
        plt.close(fig)

    def generate_combined_report(self, train_acc, test_acc, loss, l2_norm_history,
                                dropout_gap_epochs, dropout_gap_history_by_rate, dropout_rates,
                                epoch_grid, fast_ma, slow_ma, grok_epoch, detection_epoch=None):
        """Generate combined PDF report with all measurements."""

        num_epochs = len(train_acc)
        epochs_axis = range(1, num_epochs + 1)

        with PdfPages(os.path.join(self.reports_dir, "training_report.pdf")) as pdf:

            # Page 1: Grokking curve
            fig, ax = plt.subplots(figsize=(12, 7))
            ax.plot(epochs_axis, train_acc, label="Train Accuracy", color="steelblue", linewidth=2)
            ax.plot(epochs_axis, test_acc, label="Test Accuracy", color="seagreen", linewidth=2)
            ax.axvline(x=grok_epoch, color="green", linestyle=":", linewidth=2,
                       label=f"Grok epoch ({grok_epoch})")
            ax.set_xscale("log")
            ax.set_xlabel("Epoch (log scale)")
            ax.set_ylabel("Accuracy")
            ax.set_title("Grokking Curve: Train vs. Test Accuracy")
            ax.legend()
            ax.grid(True, alpha=0.3)
            pdf.savefig(fig)
            plt.close(fig)

            # Page 2: Loss curve
            fig, ax = plt.subplots(figsize=(12, 7))
            ax.plot(epochs_axis, loss, color="darkorange", linewidth=2)
            ax.set_xscale("log")
            ax.set_xlabel("Epoch (log scale)")
            ax.set_ylabel("Loss")
            ax.set_title("Training Loss")
            ax.grid(True, alpha=0.3)
            pdf.savefig(fig)
            plt.close(fig)

            # Page 3: L2 Norm with MA crossover
            fig, ax = plt.subplots(figsize=(12, 7))
            ax.plot(epochs_axis, l2_norm_history, color="purple", linewidth=2, label="L2 Norm")
            ax.plot(epoch_grid, fast_ma, label="Fast MA", linewidth=2, color="blue", alpha=0.7)
            ax.plot(epoch_grid, slow_ma, label="Slow MA", linewidth=2, color="orange", alpha=0.7)
            if detection_epoch is not None:
                ax.axvline(x=detection_epoch, color="red", linestyle="--", linewidth=2,
                           label=f"Crossover ({detection_epoch:.0f})")
            ax.axvline(x=grok_epoch, color="green", linestyle=":", linewidth=2,
                       label=f"Grok ({grok_epoch})")
            ax.set_xscale("log")
            ax.set_xlabel("Epoch (log scale)")
            ax.set_ylabel("L2 Norm")
            ax.set_title("L2 Norm & Moving Averages")
            ax.legend()
            ax.grid(True, alpha=0.3)
            pdf.savefig(fig)
            plt.close(fig)

            # Page 4: Dropout Gap (multi-rate)
            fig, ax = plt.subplots(figsize=(12, 7))
            for rate in dropout_rates:
                ax.plot(
                    dropout_gap_epochs,
                    dropout_gap_history_by_rate[rate],
                    linewidth=2,
                    label=f"rate={rate}"
                )
            ax.axvline(x=grok_epoch, color="green", linestyle=":", linewidth=2,
                       label=f"Grok ({grok_epoch})")
            ax.set_xscale("log")
            ax.set_xlabel("Epoch (log scale)")
            ax.set_ylabel("Dropout Gap")
            ax.set_title("Dropout Gap: Multi-Rate Sweep")
            ax.legend()
            ax.grid(True, alpha=0.3)
            pdf.savefig(fig)
            plt.close(fig)

        return os.path.join(self.reports_dir, "training_report.pdf")

    # ========== UTILITIES ==========

    @staticmethod
    def _smooth_signal(signal, window=50):
        """Apply simple moving average smoothing."""
        if window > len(signal):
            window = len(signal)
        return np.convolve(signal, np.ones(window)/window, mode='same')
