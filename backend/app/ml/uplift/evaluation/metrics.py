"""Causal / uplift evaluation on a held-out split.

Ordinary classification accuracy is **not** used to judge uplift: a model can
have excellent ROC-AUC for ``P(Y|X,a)`` and still rank customers by *incremental*
value badly. The metrics here judge the ranking-by-uplift and the resulting
policy directly.

Inputs (one held-out frame):
  * ``arm`` / ``treatment`` -- which arm the row was randomly assigned to
  * ``recovered``           -- observed outcome
  * ``propensity``          -- logged joint assignment propensity (uniform here)
  * ``tau_hat``             -- model's predicted *best-action* uplift, max_a (mu_a - mu_0)
  * ``best_action``         -- argmax_a (mu_a - mu_0)
  * ``uplift_<action>``     -- predicted uplift for each action

Key honesty caveat baked into every treated-vs-control number: treated rows
carry a **uniformly random** action, not the model's recommended one. So the
Qini / uplift@k curves estimate the value of *targeting who to treat* while
still treating them with a random action -- a conservative lower bound on the
value of also choosing the action. The policy-value estimate does condition on
action match and is the number that reflects "choose who *and* what".
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_trapz = np.trapezoid  # numpy >= 2.0

from app.ml.uplift import config as _cfg
from app.ml.uplift.config import (
    RANDOM_STATE,
    TREATMENT_ACTIONS,
    UPLIFT_AT_K_FRACTIONS,
)


def _responders(y: np.ndarray) -> float:
    return float(np.sum(y))


def qini_curve(
    tau_hat: np.ndarray,
    treated: np.ndarray,
    y: np.ndarray,
    n_points: int = 20,
) -> dict:
    """Qini curve for the treat-vs-not decision.

    Rows sorted descending by ``tau_hat``. At population fraction ``phi`` the
    Qini value is the estimated number of *incremental* responders from treating
    that top slice::

        Q(phi) = Y_t(phi) - Y_c(phi) * N_t(phi) / N_c(phi)

    The random-targeting reference line is ``phi * Q(1.0)``. The Qini
    coefficient is the normalized area between the two.
    """
    order = np.argsort(-tau_hat, kind="stable")
    t = treated[order].astype(bool)
    yy = y[order].astype(int)
    n = len(order)

    nt_cum = np.cumsum(t)
    nc_cum = np.cumsum(~t)
    yt_cum = np.cumsum(np.where(t, yy, 0))
    yc_cum = np.cumsum(np.where(~t, yy, 0))

    with np.errstate(divide="ignore", invalid="ignore"):
        q = yt_cum - np.where(nc_cum > 0, yc_cum * nt_cum / nc_cum, 0.0)

    total_q = float(q[-1])
    fractions = np.linspace(1.0 / n_points, 1.0, n_points)
    idx = np.clip((fractions * n).astype(int) - 1, 0, n - 1)
    curve = [
        {"population_fraction": round(float(f), 4), "qini": round(float(q[i]), 4)}
        for f, i in zip(fractions, idx)
    ]

    # area between Qini curve and the random line, trapezoid over phi in [0,1]
    phi = np.arange(1, n + 1) / n
    random_line = phi * total_q
    area = float(_trapz(q - random_line, phi))
    # normalize by the area of the "perfect" wedge ~ |total_q| / 2
    denom = abs(total_q) / 2.0 if total_q != 0 else 1.0
    qini_coefficient = area / denom

    return {
        "curve": curve,
        "total_incremental_responders": round(total_q, 3),
        "qini_coefficient": round(qini_coefficient, 4),
    }


def _bootstrap_qini_ci(
    tau_hat: np.ndarray,
    treated: np.ndarray,
    y: np.ndarray,
    *,
    n_samples: int | None = None,
) -> dict:
    n_samples = n_samples or _cfg.QINI_BOOTSTRAP_SAMPLES
    rng = np.random.default_rng(RANDOM_STATE)
    n = len(y)
    coeffs = []
    for _ in range(n_samples):
        s = rng.integers(0, n, n)
        if treated[s].sum() < 5 or (~treated[s]).sum() < 5:
            continue
        coeffs.append(qini_curve(tau_hat[s], treated[s], y[s])["qini_coefficient"])
    if len(coeffs) < 20:
        return {"n_effective_samples": len(coeffs), "ci_95": None}
    lo, hi = np.percentile(coeffs, [2.5, 97.5])
    return {
        "n_effective_samples": len(coeffs),
        "ci_95": [round(float(lo), 4), round(float(hi), 4)],
        "std": round(float(np.std(coeffs)), 4),
    }


def uplift_at_k(
    tau_hat: np.ndarray,
    treated: np.ndarray,
    y: np.ndarray,
    fractions=UPLIFT_AT_K_FRACTIONS,
) -> list[dict]:
    order = np.argsort(-tau_hat, kind="stable")
    t = treated[order].astype(bool)
    yy = y[order].astype(int)
    n = len(order)
    rows = []
    for f in fractions:
        k = max(1, int(round(f * n)))
        head_t, head_c = t[:k], ~t[:k]
        rate_t = float(yy[:k][head_t].mean()) if head_t.any() else None
        rate_c = float(yy[:k][head_c].mean()) if head_c.any() else None
        rows.append(
            {
                "k_fraction": f,
                "n": k,
                "n_treated": int(head_t.sum()),
                "n_control": int(head_c.sum()),
                "treated_recovery_rate": _round(rate_t),
                "control_recovery_rate": _round(rate_c),
                "observed_uplift": (
                    _round(rate_t - rate_c) if rate_t is not None and rate_c is not None else None
                ),
                "mean_predicted_uplift": round(float(tau_hat[order][:k].mean()), 5),
            }
        )
    return rows


def observed_lift_by_action(frame: pd.DataFrame) -> dict:
    """Per-action observed recovery rate vs the control rate -- the raw,
    model-free incrementality the randomization buys us."""
    y = frame["recovered"].astype(int).to_numpy()
    arm = frame["arm"].astype(str).to_numpy()
    ctrl_mask = arm == "none"
    ctrl_rate = float(y[ctrl_mask].mean()) if ctrl_mask.any() else None
    ctrl_n = int(ctrl_mask.sum())
    out: dict[str, dict] = {
        "control": {"n": ctrl_n, "recovery_rate": _round(ctrl_rate)},
    }
    for a in TREATMENT_ACTIONS:
        m = arm == a
        n = int(m.sum())
        rate = float(y[m].mean()) if n else None
        se = (
            float(np.sqrt(rate * (1 - rate) / n + (ctrl_rate * (1 - ctrl_rate) / ctrl_n)))
            if rate is not None and ctrl_rate is not None and n and ctrl_n
            else None
        )
        lift = rate - ctrl_rate if rate is not None and ctrl_rate is not None else None
        out[a] = {
            "n": n,
            "recovery_rate": _round(rate),
            "observed_uplift_vs_control": _round(lift),
            "uplift_std_error": _round(se),
            "uplift_z": _round(lift / se) if lift is not None and se else None,
        }
    return out


def policy_value(frame: pd.DataFrame) -> dict:
    """Self-normalized inverse-propensity estimate of the expected recovery rate
    under several action policies, evaluated on the held-out *treated* rows
    (where an action and its propensity were logged).

        V(pi) = sum_i [ 1{a_i = pi(x_i)} / p_i * y_i ]  /  sum_i [ 1{a_i = pi(x_i)} / p_i ]

    Policies compared:
      * ``uplift_policy``   -- pi(x) = argmax_a uplift_a(x)   (this model)
      * ``best_marginal``   -- pi(x) = the single action with the highest overall
                               observed recovery rate (a strong non-personalized baseline)
      * ``random_action``   -- pi(x) ~ uniform (== the mean treated recovery rate)
      * ``treat_none``      -- the observed control recovery rate (no IPW needed)
    """
    treated = frame[frame["treatment"].astype(bool)].reset_index(drop=True)
    ctrl = frame[~frame["treatment"].astype(bool)]
    y = treated["recovered"].astype(int).to_numpy()
    a = treated["arm"].astype(str).to_numpy()
    p = treated["propensity"].to_numpy(dtype=float)
    w = 1.0 / np.clip(p, 1e-6, None)

    def snipw(mask: np.ndarray) -> tuple[float | None, int]:
        if not mask.any():
            return None, 0
        num = np.sum(w[mask] * y[mask])
        den = np.sum(w[mask])
        return (float(num / den) if den else None), int(mask.sum())

    rec = treated["best_action"].astype(str).to_numpy()
    v_uplift, n_uplift = snipw(a == rec)

    marginal_rates = {
        act: float(y[a == act].mean()) if (a == act).any() else -1.0
        for act in TREATMENT_ACTIONS
    }
    best_marginal_action = max(marginal_rates, key=marginal_rates.get)
    v_marginal, n_marginal = snipw(a == best_marginal_action)

    v_random = float(y.mean()) if len(y) else None
    v_none = float(ctrl["recovered"].astype(int).mean()) if len(ctrl) else None

    return {
        "estimator": "self_normalized_ipw_on_heldout_treated",
        "treat_none_control_rate": _round(v_none),
        "random_action_value": _round(v_random),
        "best_marginal": {
            "action": best_marginal_action,
            "value": _round(v_marginal),
            "n_matched": n_marginal,
        },
        "uplift_policy": {
            "value": _round(v_uplift),
            "n_matched": n_uplift,
            "gain_vs_random_action": (
                _round(v_uplift - v_random)
                if v_uplift is not None and v_random is not None
                else None
            ),
            "gain_vs_treat_none": (
                _round(v_uplift - v_none)
                if v_uplift is not None and v_none is not None
                else None
            ),
        },
        "recommended_action_distribution": (
            pd.Series(rec).value_counts().to_dict()
        ),
    }


def uplift_calibration(frame: pd.DataFrame, n_bins: int = 5) -> list[dict]:
    """Bin held-out rows by predicted best-action uplift; in each bin compare the
    mean predicted uplift to the observed (treated - control) recovery-rate
    difference. Needs both arms present per bin -- coarse bins on purpose."""
    tau = frame["tau_hat"].to_numpy(dtype=float)
    treated = frame["treatment"].astype(bool).to_numpy()
    y = frame["recovered"].astype(int).to_numpy()
    try:
        edges = np.quantile(tau, np.linspace(0, 1, n_bins + 1))
        edges = np.unique(edges)
    except Exception:  # noqa: BLE001
        return []
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (tau >= lo) & (tau <= hi)
        mt, mc = m & treated, m & ~treated
        if mt.sum() < 10 or mc.sum() < 10:
            continue
        rt, rc = float(y[mt].mean()), float(y[mc].mean())
        rows.append(
            {
                "bin": f"[{lo:.4f},{hi:.4f}]",
                "n": int(m.sum()),
                "mean_predicted_uplift": round(float(tau[m].mean()), 5),
                "observed_uplift": _round(rt - rc),
                "n_treated": int(mt.sum()),
                "n_control": int(mc.sum()),
            }
        )
    return rows


def evaluate_uplift(frame: pd.DataFrame) -> dict:
    """``frame`` must already carry ``tau_hat`` / ``best_action`` /
    ``uplift_<action>`` columns (added by the training driver from the fitted
    estimator's predictions on the held-out split)."""
    tau = frame["tau_hat"].to_numpy(dtype=float)
    treated = frame["treatment"].astype(bool).to_numpy()
    y = frame["recovered"].astype(int).to_numpy()

    n_t, n_c = int(treated.sum()), int((~treated).sum())
    pos_t, pos_c = int(y[treated].sum()), int(y[~treated].sum())

    qini = qini_curve(tau, treated, y)
    qini_ci = _bootstrap_qini_ci(tau, treated, y)

    reliable = pos_c >= 10 and n_c >= 100
    return {
        "n_heldout": int(len(frame)),
        "n_treated": n_t,
        "n_control": n_c,
        "positives_treated": pos_t,
        "positives_control": pos_c,
        "qini": {**qini, "bootstrap_ci": qini_ci},
        "uplift_at_k": uplift_at_k(tau, treated, y),
        "observed_lift_by_action": observed_lift_by_action(frame),
        "policy_value": policy_value(frame),
        "uplift_calibration": uplift_calibration(frame),
        "statistical_reliability": {
            "reliable": bool(reliable),
            "note": (
                "PASS: enough held-out control outcomes for a directional read."
                if reliable
                else f"WEAK: only {pos_c} control positives in the held-out split "
                "-- Qini and uplift-calibration numbers are high-variance; treat "
                "the bootstrap CI, not the point estimate, as the result."
            ),
        },
    }


def _round(x, nd: int = 6):
    return round(float(x), nd) if x is not None and np.isfinite(x) else None
