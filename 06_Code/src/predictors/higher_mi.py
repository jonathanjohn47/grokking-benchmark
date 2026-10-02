"""
Higher-MI predictor (Predictor 7 of 8) -- O-information (population-based,
ADAPTED) / Transfer Entropy (exploratory only)
(v4 -- 2026-10-03 (later still): placed into 06_Code/src/predictors/ and
wired into run_nanda_benchmark.py's CHECKPOINT_PREDICTOR_FUNCS (the
checkpoint -> per-example logit-triplet extraction wrapper is now
compute_higher_mi_logit_triplet_for_checkpoint below). NOT yet wired into
analyze_nanda_unified.py's build_events() / PRIMARY_SIGNALS -- deliberately
deferred, see "STILL OPEN" near the end of this docstring.)

SIGNALS THIS MODULE PRODUCES (summary -- full reasoning for each is in
the sections below):
  1. PRIMARY     -- population O-information, global-maximum event rule.
  2. EXPLORATORY -- population O-information, zero-crossing event rule.
  3. EXPLORATORY CONTROL -- temporal/windowed O-information (W=100,
     stride=10) on per-checkpoint MEAN logits, both event rules too.
     Purpose: empirically test the population-vs-temporal conceptual
     risk flagged in "MAJOR ADAPTATION" below, by direct comparison --
     NOT a second candidate for PRIMARY.
  4. EXPLORATORY -- Transfer Entropy table (tau = 1..10, both
     directions), unchanged since 2026-10-02.

SOURCE PAPER
------------
  Pomarico, D.; Cilli, R.; Monaco, A.; Bellantuono, L.; La Rocca, M.;
  Maggipinto, T.; Magnifico, G.; Ontivero Ortega, M.; Pantaleo, E.;
  Tangaro, S.; Stramaglia, S.; Bellotti, R.; Amoroso, N.
  "Transfer entropy and O-information to detect grokking in tensor
  network multi-class classification problems." arXiv:2507.23346 (2025).

KSG ESTIMATOR (paper's Appendix A -- Kraskov-Stogbauer-Grassberger /
Frenzel-Pompe nearest-neighbour method, Chebyshev/max norm throughout,
locked to the paper, NOT adapted):

    I(X;Y)   = psi(k) + psi(T) - < psi(Nx+1)  + psi(Ny+1)  >        (Eq A7)
    I(X;Y|Z) = psi(k) + < psi(Nz+1) - psi(Nxz+1) - psi(Nyz+1) >     (Eq A9)

  psi = digamma. eps_k(i) = Chebyshev distance from point i to its k-th
  nearest OTHER point in the FULL joint space. Nx/Ny/Nz/Nxz/Nyz(i) = how
  many OTHER points fall STRICTLY within eps_k(i) in that marginal
  subspace (paper, Section II.D: "points with distance strictly less
  than eps_k(t)" -- the strict '<' matters, see IMPLEMENTATION NOTE).

    Omega(F0,F1,F2) = I(F0;F1) - I(F0;F1|F2)                        (Eq 13)
  averaged over all 3! = 6 orderings (paper's Fig. 7 caption convention:
  "mean ... of the six permutations"), since the KSG *estimator* is not
  perfectly permutation-invariant at finite sample size even though
  O-information is, in theory.
  Positive = redundant. Negative = synergistic.

  k = 4 is the only k value the source paper states explicitly, and
  only for Transfer Entropy (Section II.D) -- reused here for
  O-information too, for consistency. ASSUMPTION, confirmed with
  Jonathan 2026-10-02 (same status as Weight-PCA's threshold-90
  assumption: flagged, not independently re-derived).

IMPLEMENTATION NOTE (not in the paper -- an estimator-correctness issue
found and fixed while building this module, verified numerically):
  A naive nearest-neighbour implementation (scipy cKDTree ball-point
  query with radius = eps) counts points with distance <= eps
  (inclusive). The paper requires STRICT '<'. These are NOT equivalent
  here: because eps is itself the Chebyshev (max-norm) distance to the
  k-th joint neighbour, whichever coordinate was the *binding* one for
  that neighbour pair sits at EXACTLY distance eps in that coordinate's
  own marginal -- this is common (observed on ~50% of points in a
  cross-check run), not a rare floating-point coincidence. Using
  inclusive counting changed Omega by roughly 2.6x in one test case.
  Fix: query with r = np.nextafter(eps, 0) (the largest representable
  float strictly less than eps), which makes scipy's inclusive '<='
  query exactly equivalent to the paper's strict '<'. Verified to match
  a brute-force O(N^2) strict-inequality reference implementation
  EXACTLY (bit-for-bit) on N=500 synthetic data; the naive inclusive
  version did not match (248/500 points differed). This is why
  `_strict_count` below exists rather than a one-line ball-point call.

MAJOR ADAPTATION -- population-based O-information (CONFIRMED with
Jonathan, 2026-10-03, REPLACES the "windowed-over-training-sweeps"
design from the 2026-10-02 draft)
----------------------------------------------------------------------
Source paper: at each training sweep t, (F0_W(t), F1_W(t), F2_W(t)) is
ONE joint observation (the 3 class-output scores at that sweep), and
O-information is estimated by treating TRAINING SWEEPS as the samples
(T = total sweeps in Eq. 13-15). The quantity being measured is
therefore about dynamics: "do the three output processes co-evolve
redundantly or synergistically AS TRAINING PROGRESSES".

This project's version: at a FIXED checkpoint t, (F0, F1, F2) is
computed PER TEST EXAMPLE (correct-class logit, runner-up logit,
rest-mean logit over the other 111 classes), and O-information is
estimated by treating the ~8938 TEST EXAMPLES as the samples. The
quantity measured is therefore different in kind: it is about
POPULATION STRUCTURE at one frozen point in training -- "across
different inputs, how related are these three per-example quantities"
-- not about training dynamics.

This was done to fix a real problem in the 10-02 windowed-time design
(an undocumented window size, and only ~100 samples per window, making
the KSG estimate noisy) with a much larger, well-defined sample
(~8938 >> 100, no arbitrary window parameter). That part is a clear
improvement. BUT it is worth being explicit, for the thesis writeup/
defence, that this changes WHAT is being measured, not just HOW: the
source paper's own physical story for why O-information should track
grokking ("random, uncoordinated dynamics early -> structured,
coordinated dynamics after learning" -- i.e. a story about training
dynamics) does not automatically carry over to a population reading.
A concrete risk worth checking once real numbers are in: correct_logit,
runner_up_logit and rest_mean_logit all come from the SAME forward pass
on the SAME input, so across the test population they likely share a
common "example difficulty" factor (easy examples: high correct, low
runner-up, very low rest; hard/confused examples: correct and
runner-up close together) REGARDLESS of whether the model has grokked.
That could make the population Omega(t) persistently positive
(redundant) throughout training from this shared-cause structure alone,
with only a magnitude change rather than the sign change / transition
the source paper reports for its own (temporal) O-information. This
isn't a reason not to try it -- it's a reason to look at the raw
Omega(t) curve once computed, before trusting the event epoch it
implies. Recommend naming this "Population O-information" in any
write-up to keep it clearly distinct from Pomarico's own (temporal)
O-information.

EXPLORATORY CONTROL -- temporal/windowed O-information (CONFIRMED with
Jonathan, 2026-10-03 (later), added specifically to test the
population-vs-temporal risk raised above empirically rather than
resolve it by more a priori reasoning)
----------------------------------------------------------------------
Jonathan's own words: "Proceeding anyway with raw curve inspection
first, and adding temporal sliding-window (W=100 on mean logits) as
exploratory control to directly test this concern. ... Will run both
and report raw Omega(t) before trusting event epochs."

This restores the ORIGINAL 2026-10-02 windowed-over-training-sweeps
design (W=100 checkpoints, stride=10, Jonathan's confirmed defaults
from that earlier exchange) -- but it is no longer a competing
proposal for PRIMARY; the population signal above already took that
role. Its only job now is to be a second, independently-computed
signal that measures the thing the source paper itself measures
(co-evolution ACROSS TRAINING TIME, same sense as Eq. 13-15's own
"samples = training sweeps"), so the two can be plotted side by side:
  - If they tell the SAME story (same rough shape / transition
    timing), the population signal's shared-example-difficulty
    confound (see MAJOR ADAPTATION above) is probably not dominating,
    and population O-info is a reasonable, less-noisy stand-in.
  - If they tell DIFFERENT stories (e.g. population Omega stays
    positive/flat throughout while temporal Omega shows a genuine
    sign transition), that is direct empirical evidence the two
    quantities are NOT interchangeable, and the thesis writeup should
    say so plainly rather than quietly using the less faithful one.
This is exactly the kind of "verify another way" check the KSG
estimator itself already went through (see IMPLEMENTATION NOTE above)
-- same discipline, applied to the predictor's scientific validity
rather than its numerical correctness.

Operates on per-checkpoint MEAN logits (correct_mean, runner_up_mean,
rest_mean_mean -- one scalar per checkpoint), NOT the population's
per-example (C, N) arrays -- this is the key structural difference from
the PRIMARY signal, not just a parameter change. Reuses the same
o_information_three_way() core; only what counts as "the samples fed to
KSG" differs (a window of checkpoints here, vs. the test population
there).

EVENT RULE -- ASSUMPTION, recommend PRIMARY = global maximum, not
zero-crossing (open for Jonathan to confirm or override)
----------------------------------------------------------------------
Jonathan proposed both find_global_maximum_epoch and
find_zero_crossing_epoch, primary/exploratory, and asked for both.
Both are implemented below. Recommendation: use the GLOBAL MAXIMUM as
PRIMARY, not the zero-crossing, for two reasons: (1) it matches this
project's existing PRIMARY-rule convention across every other
predictor (AGE's nc1_min, HTSR's alpha_min, Weight-PCA's
effrank_min -- all global extrema, frozen 2026-09-17); (2) a
zero-crossing rule can return None (the signal may never cross, or
may cross back and forth on noise before any real transition, making
"the first crossing" an unstable choice) -- `evaluate_frozen_protocol()`
in analyze_nanda_unified.py needs one epoch per seed to compute
lead_frac/rho/mean_rel_lead; a None breaks that for whichever seed hits
it. The global maximum is always defined. Zero-crossing stays available
as EXPLORATORY, same status as the steepest-drop / threshold-crossing
rules elsewhere in this project.

Both find_global_maximum_epoch and find_zero_crossing_epoch are
GENERIC over any (epochs, signal) pair, so they are reused as-is on the
temporal control signal's (window_center_epochs, omega_mean) output --
no separate control-specific event-rule functions were needed.

Transfer Entropy -- EXPLORATORY ONLY (confirmed with Jonathan,
2026-10-02, unchanged in this revision): computed once per seed over
the FULL (non-windowed) trajectory of per-checkpoint MEAN logits (not
population-based -- this part is unchanged from the paper's own
Eq. 10-12), for TE_{correct->runner_up} and TE_{runner_up->correct}
at delays tau = 1..10, matching the paper's Table I shape. No event
epoch, not fed into the frozen-protocol verdict. "Averaged over seeds"
(Jonathan's phrasing) is a downstream aggregation at the
analyze_nanda_unified.py level, not inside this module.

CHECKPOINT -> LOGIT-TRIPLET EXTRACTION (added 2026-10-03 (later still))
----------------------------------------------------------------------
compute_higher_mi_logit_triplet_for_checkpoint(model, test_loader, device)
below does the extraction this module's own functions above all assume
is already done: it evaluates a frozen checkpoint on the full test set
and returns the per-example (correct, runner_up, rest_mean) arrays. See
its own docstring for the 114->113->3 class-reduction detail (slicing
off the never-trained "=" class before the 113->3 reduction already
described above) and the exact true-class / runner-up / rest-of-field
split used.

This function assumes `model` is already in eval mode with the right
checkpoint's state_dict loaded (the caller's job, same convention as
every other compute_*_metrics_for_checkpoint in this project — see
predictors/weight_pca.py, predictors/htsr_alpha.py).

STILL OPEN (deliberately deferred, per Jonathan's instruction)
----------------------------------------------------------------------
This module (plus its checkpoint predictor wrapper in
run_nanda_benchmark.py) is NOT YET wired into analyze_nanda_unified.py's
build_events() or its PRIMARY_SIGNALS list — so higher_mi's verdict is
not yet persisted to predictor_events.json via the generalized
frozen-protocol mechanism built for HTSR Alpha / Weight-PCA. That is the
next step once real 5-seed results are in and the raw Omega(t) curves
(population vs. temporal control) have been inspected, per "Proceeding
anyway with raw curve inspection first ... before trusting event
epochs."
"""

from itertools import permutations

import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy.special import digamma

K_DEFAULT = 4  # ASSUMPTION (see module docstring): paper states k=4 only
               # for Transfer Entropy; reused for O-information too.


# --------------------------------------------------------------------------
# KSG core (paper's Appendix A -- Eq. A4-A9), KD-tree implementation.
# A dense O(N^2) distance-matrix implementation was tried first and
# measured impractical at the population scale this predictor needs
# (N ~ 8938 test points per checkpoint): ~19s per checkpoint, ~10.7
# hours for all 5 seeds x 401 checkpoints. This KD-tree version was
# benchmarked at ~0.7s per checkpoint, ~22 minutes total for 5 seeds x
# 401 checkpoints -- verified to give BIT-FOR-BIT IDENTICAL results to
# the dense reference implementation on N=500 synthetic data (both the
# naive inclusive-count tree version AND the dense version were used as
# cross-checks against each other; see IMPLEMENTATION NOTE above for
# why the naive tree version initially did NOT match).
# --------------------------------------------------------------------------

def _col(a):
    return np.asarray(a, dtype=float).reshape(-1, 1)


def _kth_nn_distances(joint_points, k):
    """Chebyshev distance from every point to its k-th nearest OTHER
    point in `joint_points` (T, d). This is eps_k(i) in the paper."""
    tree = cKDTree(joint_points)
    d, _ = tree.query(joint_points, k=k + 1, p=np.inf, workers=-1)
    return d[:, -1]


def _strict_count(marginal_points, eps):
    """Count OTHER points STRICTLY within eps[i] (Chebyshev norm) in
    `marginal_points`, for every point i. See IMPLEMENTATION NOTE above
    for why r=np.nextafter(eps, 0) is used instead of r=eps directly."""
    tree = cKDTree(marginal_points)
    r_strict = np.nextafter(eps, 0)
    counts = tree.query_ball_point(marginal_points, r=r_strict, p=np.inf,
                                    workers=-1, return_length=True)
    return np.asarray(counts) - 1  # exclude the point itself


def mutual_information_ksg(x, y, k=K_DEFAULT):
    """I(X;Y) via KSG, Eq. A7. x, y: 1-D arrays, same length T."""
    x, y = _col(x), _col(y)
    T = len(x)
    eps = _kth_nn_distances(np.hstack([x, y]), k)
    n_x = _strict_count(x, eps)
    n_y = _strict_count(y, eps)
    return float(digamma(k) + digamma(T) - np.mean(digamma(n_x + 1) + digamma(n_y + 1)))


def conditional_mutual_information_ksg(x, y, z, k=K_DEFAULT):
    """I(X;Y|Z) via KSG, Eq. A9. x, y, z: 1-D arrays, same length T."""
    x, y, z = _col(x), _col(y), _col(z)
    eps = _kth_nn_distances(np.hstack([x, y, z]), k)
    n_z = _strict_count(z, eps)
    n_xz = _strict_count(np.hstack([x, z]), eps)
    n_yz = _strict_count(np.hstack([y, z]), eps)
    return float(digamma(k) + np.mean(digamma(n_z + 1) - digamma(n_xz + 1) - digamma(n_yz + 1)))


def o_information_three_way(f0, f1, f2, k=K_DEFAULT):
    """Omega(F0,F1,F2) = I(F0;F1) - I(F0;F1|F2), Eq. 13, averaged over
    all 3! = 6 orderings (paper's Fig. 7 caption convention). Returns
    (mean, std) over the six estimates -- std is a finite-sample
    diagnostic, not part of the signal itself."""
    vals = []
    for a, b, c in permutations([np.asarray(f0), np.asarray(f1), np.asarray(f2)]):
        vals.append(mutual_information_ksg(a, b, k)
                     - conditional_mutual_information_ksg(a, b, c, k))
    return float(np.mean(vals)), float(np.std(vals))


# --------------------------------------------------------------------------
# Population O-information signal (MAJOR ADAPTATION -- see module
# docstring). One Omega per checkpoint, computed across the test
# population at that checkpoint -- no sliding window over training time.
# --------------------------------------------------------------------------

def compute_population_o_information_signal(checkpoints, correct_logits,
                                              runner_up_logits, rest_mean_logits,
                                              k=K_DEFAULT):
    """checkpoints: (C,) array of checkpoint epochs.
    correct_logits, runner_up_logits, rest_mean_logits: (C, N) arrays --
    per-checkpoint, per-test-example values (N ~ 8938 in this project).
    Each row c is fed into o_information_three_way() as the N samples
    for that checkpoint.

    Returns (signal_epochs, omega_mean, omega_std): all shape (C,), one
    entry per checkpoint (NOT one per window -- that was the 10-02
    design, replaced)."""
    checkpoints = np.asarray(checkpoints)
    correct_logits = np.asarray(correct_logits, dtype=float)
    runner_up_logits = np.asarray(runner_up_logits, dtype=float)
    rest_mean_logits = np.asarray(rest_mean_logits, dtype=float)
    C = len(checkpoints)
    assert correct_logits.shape == (C, correct_logits.shape[1])
    assert runner_up_logits.shape == correct_logits.shape
    assert rest_mean_logits.shape == correct_logits.shape

    means = np.empty(C)
    stds = np.empty(C)
    for c in range(C):
        m, sd = o_information_three_way(
            correct_logits[c], runner_up_logits[c], rest_mean_logits[c], k=k)
        means[c] = m
        stds[c] = sd
    return checkpoints.copy(), means, stds


def find_global_maximum_epoch(checkpoints, signal):
    """RECOMMENDED PRIMARY event rule (see module docstring "EVENT
    RULE"). Epoch of the global MAXIMUM of the population O-information
    signal (peak redundancy). Mirrors the global-extremum convention of
    every other PRIMARY rule in this project (AGE/HTSR/Weight-PCA all
    use an extremum epoch). Always defined (no None case).

    Returns (event_epoch, event_value, event_index)."""
    arr = np.asarray(signal, dtype=float)
    idx = int(np.nanargmax(arr))
    return int(checkpoints[idx]), float(arr[idx]), idx


def find_zero_crossing_epoch(checkpoints, signal):
    """EXPLORATORY (per recommendation above -- Jonathan to confirm or
    override). Epoch of the first negative-to-positive zero crossing
    (synergy -> redundancy), matching the source paper's own
    description of its Fig. 7. Returns None if the signal never
    crosses from negative to positive -- this None-risk is exactly why
    it is not recommended as PRIMARY (see module docstring)."""
    arr = np.asarray(signal, dtype=float)
    sign = np.sign(arr)
    for i in range(1, len(sign)):
        if sign[i - 1] < 0 and sign[i] >= 0:
            return int(checkpoints[i])
    return None


# --------------------------------------------------------------------------
# Temporal O-information signal (EXPLORATORY CONTROL -- see module
# docstring). Restores the 2026-10-02 windowed-over-training-sweeps
# design, now explicitly a control against the population signal above,
# not a PRIMARY candidate. Operates on per-checkpoint MEAN logits.
# --------------------------------------------------------------------------

def compute_temporal_o_information_signal(checkpoints, correct_mean,
                                            runner_up_mean, rest_mean_mean,
                                            window=100, stride=10, k=K_DEFAULT):
    """EXPLORATORY CONTROL signal (see module docstring "EXPLORATORY
    CONTROL"). checkpoints: (C,) array of checkpoint epochs.
    correct_mean, runner_up_mean, rest_mean_mean: (C,) arrays -- ONE
    SCALAR PER CHECKPOINT (the per-checkpoint mean logit over the test
    batch), NOT the population signal's (C, N) per-example arrays --
    this is the key structural difference, not just a parameter change.

    Slides a window of `window` consecutive checkpoints (stride
    `stride`) along training time and feeds the `window` values inside
    each window to o_information_three_way() as the KSG samples -- this
    is the paper's own sense of "samples = training sweeps" (Eq.
    13-15), just applied to checkpoints, and only over a local window so
    the signal stays time-resolved (matching Fig. 7's time evolution).
    window=100, stride=10 are Jonathan's confirmed defaults
    (2026-10-02): for this project's 401 checkpoints that gives ~31
    window-centers, comparable in spirit to Fig. 7's resolution.

    Returns (window_center_epochs, omega_mean, omega_std): each shape
    (num_windows,) -- NOT shape (C,), since each value summarizes a
    WINDOW of checkpoints, not a single one. window_center_epochs holds
    the checkpoint epoch at the midpoint of each window, for plotting
    against the same x-axis as the population signal."""
    checkpoints = np.asarray(checkpoints)
    correct_mean = np.asarray(correct_mean, dtype=float)
    runner_up_mean = np.asarray(runner_up_mean, dtype=float)
    rest_mean_mean = np.asarray(rest_mean_mean, dtype=float)
    C = len(checkpoints)
    assert correct_mean.shape == (C,)
    assert runner_up_mean.shape == (C,)
    assert rest_mean_mean.shape == (C,)

    starts = list(range(0, C - window + 1, stride))
    if not starts:
        raise ValueError(
            f"window={window} is longer than the available {C} "
            f"checkpoints -- cannot form even one window.")

    centers = np.empty(len(starts))
    means = np.empty(len(starts))
    stds = np.empty(len(starts))
    for i, start in enumerate(starts):
        end = start + window
        m, sd = o_information_three_way(
            correct_mean[start:end], runner_up_mean[start:end],
            rest_mean_mean[start:end], k=k)
        centers[i] = checkpoints[start + window // 2]
        means[i] = m
        stds[i] = sd
    return centers, means, stds


# --------------------------------------------------------------------------
# Transfer entropy -- EXPLORATORY ONLY, full (non-windowed) trajectory
# of per-checkpoint MEAN logits (unchanged from the 10-02 draft).
# --------------------------------------------------------------------------

def transfer_entropy_ksg(source, target, tau=1, k=K_DEFAULT):
    """TE_{source->target} via KSG, Eq. 11-12 collapsed to the identity
    the source paper itself derives: TE = I(target(t+tau); source(t) |
    target(t)). EXPLORATORY ONLY -- not windowed, no event epoch, not
    scored. source, target: 1-D arrays, one entry per checkpoint
    (per-checkpoint MEAN logit, not population-based)."""
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    target_future = target[tau:]
    target_past = target[:-tau]
    source_past = source[:-tau]
    return conditional_mutual_information_ksg(target_future, source_past, target_past, k=k)


def transfer_entropy_table(correct_mean, runner_up_mean, taus=range(1, 11), k=K_DEFAULT):
    """Reproduces the source paper's Table I shape: TE in both
    directions between the correct-class and runner-up-class mean-logit
    processes, for each delay tau. correct_mean, runner_up_mean: 1-D
    arrays, one entry per checkpoint (per-checkpoint mean over the test
    batch). Returns {tau: {"correct_to_runner_up": .., "runner_up_to_correct": ..}}."""
    out = {}
    for tau in taus:
        out[tau] = {
            "correct_to_runner_up": transfer_entropy_ksg(correct_mean, runner_up_mean, tau, k),
            "runner_up_to_correct": transfer_entropy_ksg(runner_up_mean, correct_mean, tau, k),
        }
    return out


# --------------------------------------------------------------------------
# Checkpoint -> per-example logit-triplet extraction (added 2026-10-03
# (later still) -- see module docstring, "CHECKPOINT -> LOGIT-TRIPLET
# EXTRACTION"). This is the piece run_nanda_benchmark.py's
# _checkpoint_predictor_higher_mi calls once per checkpoint, same role as
# predictors/weight_pca.py's compute_weight_pca_metrics_for_checkpoint.
# --------------------------------------------------------------------------

def compute_higher_mi_logit_triplet_for_checkpoint(model, test_loader, device):
    """Evaluate a frozen checkpoint on the FULL test_loader (model is
    already in eval() mode with the checkpoint's state_dict loaded by the
    caller -- same convention as every other compute_*_metrics_for_checkpoint
    in this project) and extract, per test example, the three per-example
    logit quantities this predictor's O-information needs -- the project's
    113->3 class reduction (see module docstring, MAJOR ADAPTATION):

        F0 = correct_logit    -- logit of the true class
        F1 = runner_up_logit  -- highest logit among the OTHER valid classes
        F2 = rest_mean_logit  -- mean logit of the remaining valid classes
                                 (excludes BOTH the true class and the
                                 runner-up, so F1 and F2 never share a
                                 value and F2 is not pulled toward F1)

    IMPLEMENTATION NOTE -- slicing off the never-trained "=" class: the
    model's output_head has vocab_size = p + 1 logits (one extra class
    for the "=" token id, which is NEVER a target label -- see
    data/modular_arithmetic.py's ModularArithmeticDataset). "the OTHER
    classes" here means the other p - 1 VALID answer classes [0, p-1],
    not all p non-true logits out of p + 1 -- logits[:, :p] is sliced off
    FIRST, so the never-trained "=" class cannot contaminate runner_up /
    rest_mean with a logit the model was never taught to produce.
    ASSUMPTION (same flagging discipline as the paper's own k=4 and the
    113->3 reduction above) -- flag for Jonathan to confirm if this
    matters for the write-up. rest_mean is computed from
    total - correct - runner_up, dividing by (p - 2) -- exact, no
    masking-with -inf-then-mean bug (mean() over a -inf-masked tensor
    would itself be -inf, not a mean over the remaining entries).

    Returns a dict of three (N,) float64 numpy arrays -- "correct",
    "runner_up", "rest_mean" -- N = the full test-set size (this
    project's p=113 config: ~8939 examples), accumulated across however
    many batches test_loader actually yields (NOT assumed to be a single
    batch -- get_dataloaders() reuses the training batch_size for the
    test DataLoader too, so the ~8939-example test set is itself split
    into a few batches)."""
    p = model.output_head.out_features - 1
    if p <= 2:
        raise ValueError(
            f"compute_higher_mi_logit_triplet_for_checkpoint needs at least "
            f"3 valid classes (p > 2) to form correct/runner_up/rest_mean; "
            f"got p={p}.")
    model.eval()
    correct_chunks, runner_up_chunks, rest_mean_chunks = [], [], []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = model.forward(x)[:, 2, :][:, :p]  # drop the never-trained "=" class
            n = logits.shape[0]
            idx = torch.arange(n, device=logits.device)
            correct = logits[idx, y]
            total = logits.sum(dim=1)
            masked = logits.clone()
            masked[idx, y] = float("-inf")             # exclude the true class from the max
            runner_up, _ = masked.max(dim=1)
            rest_mean = (total - correct - runner_up) / (p - 2)
            correct_chunks.append(correct.detach().cpu().numpy())
            runner_up_chunks.append(runner_up.detach().cpu().numpy())
            rest_mean_chunks.append(rest_mean.detach().cpu().numpy())
    return {
        "correct": np.concatenate(correct_chunks).astype(float),
        "runner_up": np.concatenate(runner_up_chunks).astype(float),
        "rest_mean": np.concatenate(rest_mean_chunks).astype(float),
    }


# --------------------------------------------------------------------------
# Self-test -- synthetic sanity checks, re-run after switching to the
# KD-tree implementation to confirm the fix didn't change the textbook
# answers (Jonathan's "verify another way" practice).
# --------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    T = 3000

    x_indep = rng.normal(size=T)
    y_indep = rng.normal(size=T)
    mi_indep = mutual_information_ksg(x_indep, y_indep)

    x_corr = rng.normal(size=T)
    y_corr = x_corr + 0.1 * rng.normal(size=T)
    mi_corr = mutual_information_ksg(x_corr, y_corr)

    print("MI(independent)   ~ 0   ->", round(mi_indep, 4))
    print("MI(near-copy)     >> 0  ->", round(mi_corr, 4))
    assert abs(mi_indep) < 0.1
    assert mi_corr > 1.0

    f0_r = rng.normal(size=T)
    f1_r = f0_r + 0.1 * rng.normal(size=T)
    f2_r = f0_r + 0.1 * rng.normal(size=T)
    omega_redundant, _ = o_information_three_way(f0_r, f1_r, f2_r)

    f0_s = rng.normal(size=T)
    f1_s = rng.normal(size=T)
    f2_s = f0_s * f1_s + 0.05 * rng.normal(size=T)
    omega_synergistic, _ = o_information_three_way(f0_s, f1_s, f2_s)

    print("Omega(redundant)   > 0  ->", round(omega_redundant, 4))
    print("Omega(synergistic) < 0  ->", round(omega_synergistic, 4))
    assert omega_redundant > 0
    assert omega_synergistic < 0

    # Population-signal + event-rule smoke test on synthetic data shaped
    # like the real use case (C checkpoints x N test examples).
    C, N = 50, 2000
    correct = np.tile(np.linspace(-2, 3, C), (N, 1)).T + 0.3 * rng.normal(size=(C, N))
    runner_up = np.tile(np.linspace(-1, 1, C), (N, 1)).T + 0.3 * rng.normal(size=(C, N))
    rest_mean = -0.5 + 0.1 * rng.normal(size=(C, N))
    checkpoints = np.arange(C) * 100

    sig_epochs, omega_mean, omega_std = compute_population_o_information_signal(
        checkpoints, correct, runner_up, rest_mean)
    gmax = find_global_maximum_epoch(sig_epochs, omega_mean)
    zc = find_zero_crossing_epoch(sig_epochs, omega_mean)
    print("population signal computed, C =", C, " omega range:",
          round(omega_mean.min(), 3), "to", round(omega_mean.max(), 3))
    print("global max epoch:", gmax[0], " zero-crossing epoch:", zc)

    # Temporal control signal (EXPLORATORY CONTROL) smoke test. NOTE:
    # this needs its OWN synthetic trajectory, not correct.mean(axis=1)
    # from the population test above -- averaging that population data
    # (N=2000) over its axis collapses almost all the per-checkpoint
    # noise, leaving three near-perfectly-smooth, near-deterministic
    # curves; inside any one window they are then (numerically) almost
    # collinear / near-constant, which degenerates the KSG estimate to
    # an uninformative 0.0 everywhere -- a property of that averaged
    # synthetic input, not a bug in the estimator (discovered by
    # actually inspecting the first version of this smoke test's
    # output, same "check don't assume" habit as the IMPLEMENTATION
    # NOTE bug above). A fresh, directly-constructed MEAN-logit-style
    # trajectory (trend + per-checkpoint noise, redundant construction
    # as in the omega_redundant check above) avoids that degeneracy.
    C_ctrl = 200
    checkpoints_ctrl = np.arange(C_ctrl) * 25
    trend = np.linspace(-1, 1, C_ctrl)
    correct_mean_traj = trend + 0.05 * rng.normal(size=C_ctrl)
    runner_up_mean_traj = trend + 0.05 * rng.normal(size=C_ctrl)
    rest_mean_traj = trend + 0.05 * rng.normal(size=C_ctrl)

    ctr_epochs, ctr_omega_mean, ctr_omega_std = compute_temporal_o_information_signal(
        checkpoints_ctrl, correct_mean_traj, runner_up_mean_traj, rest_mean_traj,
        window=30, stride=5)
    ctr_gmax = find_global_maximum_epoch(ctr_epochs, ctr_omega_mean)
    ctr_zc = find_zero_crossing_epoch(ctr_epochs, ctr_omega_mean)
    print("temporal control signal computed, num_windows =", len(ctr_epochs),
          " omega range:", round(ctr_omega_mean.min(), 3), "to",
          round(ctr_omega_mean.max(), 3))
    print("control global max epoch:", ctr_gmax[0],
          " control zero-crossing epoch:", ctr_zc)
    # All three series share one trend -> this construction is
    # redundant by design (same reasoning as omega_redundant above), so
    # every window's Omega should come out positive.
    assert (ctr_omega_mean > 0).all()

    # Sanity: the windowed function must raise cleanly (not silently
    # return garbage) when the window is longer than the trajectory --
    # this would otherwise be a quiet way to misconfigure W=100 against
    # a benchmark run shorter than 100 checkpoints.
    try:
        compute_temporal_o_information_signal(
            checkpoints_ctrl, correct_mean_traj, runner_up_mean_traj, rest_mean_traj,
            window=C_ctrl + 1, stride=5)
        raise AssertionError("expected ValueError for window > C, got none")
    except ValueError:
        pass

    # compute_higher_mi_logit_triplet_for_checkpoint smoke test -- a tiny
    # fake model (just enough attributes/methods for the function to run:
    # .output_head.out_features and .eval()/.forward()) with HAND-PICKED
    # logits, so correct/runner_up/rest_mean can be checked against a
    # value worked out by hand, not just "didn't crash".
    import types

    class _FakeModel:
        def __init__(self, logits_at_pos2):
            # logits_at_pos2: (N, out_features) -- returned at seq position 2
            self._logits = logits_at_pos2
            self.output_head = types.SimpleNamespace(
                out_features=logits_at_pos2.shape[1])

        def eval(self):
            pass

        def forward(self, x):
            # x is unused (a batch of row-indices here); return shape
            # (N, 3, out_features) so [:, 2, :] recovers this batch's rows.
            idx = x.numpy() if hasattr(x, "numpy") else np.asarray(x)
            rows = self._logits[idx]
            out = np.zeros((len(idx), 3, self._logits.shape[1]), dtype=float)
            out[:, 2, :] = rows
            return torch.as_tensor(out, dtype=torch.float32)

    # out_features = 6 -> p = 5 valid classes [0..4] + 1 never-trained "="
    # class at index 5 (must be sliced off, see IMPLEMENTATION NOTE above).
    # Row 0: y=1 -> correct=5.0; others (excluding true)=[1,2,0,3] ->
    #   runner_up=3.0 (index 4); rest=[1,2,0] -> mean=1.0.
    # Row 1: y=3 -> correct=9.0; others (excluding true)=[4,1,2,-7] ->
    #   runner_up=4.0 (index 0); rest=[1,2,-7] -> mean=-4/3=-1.3333.
    fake_logits = np.array([
        [1.0, 5.0, 2.0, 0.0, 3.0, 99.0],   # row 0 -- index 5 must be ignored
        [4.0, 1.0, 2.0, 9.0, -7.0, -99.0],  # row 1
    ])
    fake_y = torch.tensor([1, 3], dtype=torch.long)
    fake_x = torch.tensor([0, 1], dtype=torch.long)  # row-index lookup, see forward() above
    fake_test_loader = [(fake_x, fake_y)]
    fake_model = _FakeModel(fake_logits)

    triplet = compute_higher_mi_logit_triplet_for_checkpoint(
        fake_model, fake_test_loader, device=torch.device("cpu"))
    print("logit-triplet smoke test:", triplet)
    assert np.allclose(triplet["correct"], [5.0, 9.0])
    assert np.allclose(triplet["runner_up"], [3.0, 4.0])
    assert np.allclose(triplet["rest_mean"], [1.0, -4.0 / 3.0])
    # The never-trained "=" class (99.0 / -99.0, both far larger in
    # magnitude than anything else in their row) must NOT have leaked
    # into runner_up or rest_mean -- if it had, runner_up would be 99/-99
    # or rest_mean would be wildly off; the asserts above already rule
    # this out, but say so explicitly since it's the one bug this test
    # exists to catch.

    print("\nAll sanity checks passed.")
