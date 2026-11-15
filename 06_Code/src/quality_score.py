"""Predictor scoring: Superiority Index and legacy Quality Index S.

Superiority Index (control redesign, November 2026)
-------------------------------------------------
The validity gate decides first. A signal that fails any of

  Lead fraction  >= 0.8
  Correlation    >= 0.9
  Noise margin   in [0.05, 0.9]

receives Superiority Index = 0.

A passing signal is scored by the arithmetic mean of three normalised
margins above the gate floor toward the theoretical perfect values:

  m_lead  = (lead  - 0.8) / (1.0 - 0.8)
  m_corr  = (rho   - 0.9) / (1.0 - 0.9)
  m_noise = (noise - 0.05) / (0.5 - 0.05)   # 0.5 is the ideal relative lead
  SI      = (m_lead + m_corr + m_noise) / 3

Setting all three raw values to fraction t of the way from the gate floor
to their perfect values yields SI = t exactly (e.g. t = 0.10 -> SI = 0.10).

Legacy Predictor Quality Index S (PR #35)
-----------------------------------------
Still used by analyze_nanda_unified.py for the L2 row. Unchanged formulas:

  S_rho, S_lead, S_sigma, S_raw, gated S as before.
"""

import numpy as np

# Validity-gate floors and theoretical perfect values for Superiority Index.
LEAD_THRESH = 0.8
LEAD_PERFECT = 1.0
CORR_THRESH = 0.9
CORR_PERFECT = 1.0
NOISE_LO = 0.05
NOISE_HI = 0.9
NOISE_PERFECT = 0.5  # ideal mean relative lead (Control A / oracle)


def passes(result, rho_thresh):
    """Check if a result satisfies the frozen protocol PASS gate."""
    if "passed" in result and result["passed"] is not None:
        return bool(result["passed"])
    rho = result.get("rho")
    if rho is None or (isinstance(rho, float) and np.isnan(rho)):
        return False
    n_firing = result.get("n_firing")
    if n_firing is not None and n_firing < 4:
        return False
    lead_frac = result.get("lead_frac", 0.0)
    mean_rel_lead = result.get("mean_rel_lead", 0.0)
    return (
        lead_frac >= LEAD_THRESH
        and rho >= rho_thresh
        and NOISE_LO <= mean_rel_lead <= NOISE_HI
    )


def raw_values_for_superiority_target(t):
    """Reverse-engineer Lead, Correlation, Noise for target Superiority Index t.

    Places each raw value exactly fraction t of the way from its gate floor
    to its theoretical perfect value. For a passing target, t in [0, 1]
    yields SI = t when scored by superiority_index().
    """
    t = float(t)
    lead = LEAD_THRESH + t * (LEAD_PERFECT - LEAD_THRESH)
    corr = CORR_THRESH + t * (CORR_PERFECT - CORR_THRESH)
    noise = NOISE_LO + t * (NOISE_PERFECT - NOISE_LO)
    return {
        "lead_frac": lead,
        "rho": corr,
        "noise_margin": noise,
    }


def superiority_index(lead_frac, rho, noise_margin, corr_thresh=None):
    """Superiority Index from the three raw gate metrics.

    Returns 0.0 if the validity gate fails. Otherwise returns the mean of
    the three normalised margins toward the perfect values.
    """
    if corr_thresh is None:
        corr_thresh = CORR_THRESH

    lead_frac = float(lead_frac)
    rho = float(rho)
    noise_margin = float(noise_margin)

    gate_ok = (
        lead_frac >= LEAD_THRESH
        and rho >= corr_thresh
        and NOISE_LO <= noise_margin <= NOISE_HI
    )
    if not gate_ok:
        return 0.0

    m_lead = (lead_frac - LEAD_THRESH) / (LEAD_PERFECT - LEAD_THRESH)
    m_corr = (rho - corr_thresh) / (CORR_PERFECT - corr_thresh)
    m_noise = (noise_margin - NOISE_LO) / (NOISE_PERFECT - NOISE_LO)

    # Clamp so values past the perfect point cannot inflate the score.
    m_lead = float(np.clip(m_lead, 0.0, 1.0))
    m_corr = float(np.clip(m_corr, 0.0, 1.0))
    m_noise = float(np.clip(m_noise, 0.0, 1.0))

    return (m_lead + m_corr + m_noise) / 3.0


def superiority_index_from_result(result, corr_thresh=None):
    """Superiority Index from an evaluate()/metric result dict."""
    if corr_thresh is None:
        corr_thresh = CORR_THRESH
    lead_frac = result.get("lead_frac", 0.0)
    rho = result.get("rho", 0.0)
    noise = result.get("noise_margin", result.get("mean_rel_lead", 0.0))
    if rho is None or (isinstance(rho, float) and np.isnan(rho)):
        return 0.0
    return superiority_index(lead_frac, rho, noise, corr_thresh=corr_thresh)


def quality_score(result, rho_thresh, passed=None):
    """Compute the Predictor Quality Index S and its components.

    Parameters
    ----------
    result : dict
        Evaluation result containing at minimum:
        - "lead_frac": float
        - "rho": float (or NaN)
        - "r_i": list or array of relative leads for firing seeds, or None
        May optionally contain "mean_rel_lead", "n_firing", "passed".
    rho_thresh : float
        Spearman correlation threshold (e.g. 0.8999999999999998).
    passed : bool, optional
        Whether the row passed the gate. If None, passes(result, rho_thresh) is used.

    Returns
    -------
    dict with keys: S_rho, S_lead, sigma_r, S_sigma, S_raw, S, pass
    """
    if passed is None:
        passed = passes(result, rho_thresh)

    # Fail-fast branch: fewer than 4 firing seeds (r_i is None or len < 4)
    r_i = result.get("r_i")
    if r_i is None or len(r_i) < 4:
        return {
            "S_rho": 0.0,
            "S_lead": 0.0,
            "sigma_r": 0.0,
            "S_sigma": 0.0,
            "S_raw": 0.0,
            "S": 0.0,
            "pass": bool(passed),
        }

    r_i = np.asarray(r_i, dtype=float)
    rho = result.get("rho")
    lead_frac = result.get("lead_frac", 0.0)

    # S_rho: NaN -> 0
    if rho is None or (isinstance(rho, float) and np.isnan(rho)):
        S_rho = 0.0
    else:
        S_rho = float(max(0.0, float(rho)))

    # r_bar and S_lead
    r_bar = float(np.mean(r_i))
    S_lead = float(max(0.0, 1.0 - abs(r_bar - 0.5) / 0.5))

    # sigma_r and S_sigma
    sigma_r = float(np.std(r_i, ddof=0))
    if r_bar <= 0.0:
        S_sigma = 0.0
    else:
        S_sigma = float(max(0.0, 1.0 - sigma_r / r_bar))

    # S_raw and gated S
    S_raw = float(lead_frac * S_rho * S_lead * S_sigma)
    S = S_raw if passed else 0.0

    return {
        "S_rho": S_rho,
        "S_lead": S_lead,
        "sigma_r": sigma_r,
        "S_sigma": S_sigma,
        "S_raw": S_raw,
        "S": S,
        "pass": bool(passed),
    }
