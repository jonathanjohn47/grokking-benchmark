"""Predictor Quality Index S (PR #35, November 2026).

The gate (passes()) decides validity: a candidate row that fails PASS receives
S = 0 and is not ranked. For rows that pass, S in [0, 1] ranks predictive
quality along four independent axes:

  S_rho   = max(0, rho)                      -- rank-correlation quality; NaN -> 0
  r_i     = (G_i - P_i) / G_i                -- per-seed relative lead (firing seeds only)
  r_bar   = mean(r_i)
  S_lead  = max(0, 1 - |r_bar - 0.5| / 0.5)  -- rewards lead fraction centred on 0.5;
                                                own choice, no paper fixes this target
  sigma_r = np.std(r_i, ddof=0)              -- population std, same convention as the
                                                rest of the project
  S_sigma = max(0, 1 - sigma_r / r_bar)      if r_bar > 0, else 0
                                             -- rewards low cross-seed variability;
                                                own choice, no paper fixes this formula;
                                                guard: r_bar <= 0 sets S_sigma = 0 to
                                                avoid division by zero / sign flip
  S_raw   = lead_frac * S_rho * S_lead * S_sigma
  S       = S_raw  if passes(result, rho_thresh)  else 0.0

The gate fires first: a row that fails PASS gets S = 0 regardless of S_raw.
Fewer than 4 firing seeds -> every component and S = 0.
"""

import numpy as np


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
        lead_frac >= 0.8
        and rho >= rho_thresh
        and 0.05 <= mean_rel_lead <= 0.9
    )


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
