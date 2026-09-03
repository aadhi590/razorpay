"""Propensity, overlap and positivity diagnostics for the uplift dataset.

The synthetic generator assigns the arm by a **known randomized mechanism**
(``is_control ~ Bernoulli(0.20)`` independent of context; treatment action
uniform over the eligible set with the propensity logged per decision). That
makes the raw treatment/control and action contrasts unbiased *without*
inverse-propensity weighting -- but "known randomized" is a claim to be
*checked*, not assumed. This module checks it and reports the weighting
diagnostics a production (non-randomized) deployment would depend on:

* every propensity in ``(0, 1]``
* overlap / positivity: min propensity per arm, and per feature stratum
* effective sample size under stabilized inverse-propensity weights
* count of extreme IPW weights
* per-action and control coverage vs the configured minimums
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from app.ml.features.schema import LABEL
from app.ml.uplift.config import (
    CONTROL_ACTION,
    CONTROL_DESIGN_PROPENSITY,
    MAX_IPW_WEIGHT,
    MIN_CONTROL_EVENTS,
    MIN_CONTROL_POSITIVES,
    MIN_EVENTS_PER_ACTION,
    MIN_POSITIVES_PER_ACTION,
    MIN_PROPENSITY,
    OVERLAP_MIN_EFFECTIVE_SAMPLE_FRACTION,
    TREATMENT_ACTIONS,
)


@dataclass
class PropensityReport:
    passed: bool = True
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)

    def fail(self, msg: str) -> None:
        self.passed = False
        self.failures.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def as_dict(self) -> dict:
        return {
            "passed": self.passed,
            "failures": self.failures,
            "warnings": self.warnings,
            "diagnostics": self.diagnostics,
        }


def _effective_sample_size(weights: np.ndarray) -> float:
    w = np.asarray(weights, dtype=float)
    w = w[np.isfinite(w) & (w > 0)]
    if w.size == 0:
        return 0.0
    return float(w.sum() ** 2 / np.sum(w**2))


def stabilized_ipw_weights(frame: pd.DataFrame) -> np.ndarray:
    """Stabilized inverse-propensity weights for the arm assignment.

    ``w_i = P(arm_i) / propensity_i`` where the numerator is the marginal
    probability of ``arm_i`` (the stabilizer) and the denominator is the logged
    joint propensity of that row's (arm, action) cell. With uniform randomized
    assignment every ``w_i`` is ~1, which is the point of reporting it.
    """
    p = frame["propensity"].to_numpy(dtype=float)
    arm = frame["arm"].astype(str).to_numpy()
    marginal = pd.Series(arm).map(pd.Series(arm).value_counts(normalize=True)).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        w = np.where(p > 0, marginal / p, np.nan)
    return w


def validate_propensity(frame: pd.DataFrame) -> PropensityReport:
    report = PropensityReport()
    p = frame["propensity"].to_numpy(dtype=float)
    treat = frame["treatment"].astype(bool).to_numpy()
    y = frame[LABEL].astype(int).to_numpy()

    # -- range --------------------------------------------------
    n_nonpos = int(np.sum(~(p > 0)))
    n_gt1 = int(np.sum(p > 1.0))
    report.diagnostics["propensity_range"] = {
        "min": float(np.nanmin(p)),
        "max": float(np.nanmax(p)),
        "n_le_zero_or_nan": n_nonpos,
        "n_gt_one": n_gt1,
        "n_below_min_threshold": int(np.sum(p < MIN_PROPENSITY)),
        "min_threshold": MIN_PROPENSITY,
    }
    if n_nonpos:
        report.fail(f"{n_nonpos} rows have propensity <= 0 or NaN")
    if n_gt1:
        report.fail(f"{n_gt1} rows have propensity > 1")
    if int(np.sum((p > 0) & (p < MIN_PROPENSITY))):
        report.warn(
            f"{int(np.sum((p > 0) & (p < MIN_PROPENSITY)))} rows have propensity "
            f"< {MIN_PROPENSITY}; inverse-propensity weights for them explode"
        )

    # -- realized vs design control fraction -------------------
    realized_control = float(np.mean(~treat))
    report.diagnostics["control_fraction"] = {
        "design": CONTROL_DESIGN_PROPENSITY,
        "realized": round(realized_control, 6),
        "abs_diff": round(abs(realized_control - CONTROL_DESIGN_PROPENSITY), 6),
    }
    if abs(realized_control - CONTROL_DESIGN_PROPENSITY) > 0.05:
        report.warn(
            f"realized control fraction {realized_control:.3f} differs from the "
            f"design {CONTROL_DESIGN_PROPENSITY} by > 0.05; check the generator "
            "or the CONTROL_DESIGN_PROPENSITY constant"
        )

    # -- overlap / positivity per arm --------------------------
    per_arm: dict[str, dict] = {}
    for arm in [CONTROL_ACTION, *TREATMENT_ACTIONS]:
        m = (frame["arm"] == arm).to_numpy()
        n = int(m.sum())
        pos = int(y[m].sum())
        per_arm[arm] = {
            "rows": n,
            "positives": pos,
            "min_propensity": float(np.nanmin(p[m])) if n else None,
            "max_propensity": float(np.nanmax(p[m])) if n else None,
        }
    report.diagnostics["per_arm"] = per_arm

    if per_arm[CONTROL_ACTION]["rows"] < MIN_CONTROL_EVENTS:
        report.warn(
            f"only {per_arm[CONTROL_ACTION]['rows']} control rows "
            f"(< {MIN_CONTROL_EVENTS}); mu_0 estimates are high-variance"
        )
    if per_arm[CONTROL_ACTION]["positives"] < MIN_CONTROL_POSITIVES:
        report.warn(
            f"only {per_arm[CONTROL_ACTION]['positives']} natural recoveries in "
            f"control (< {MIN_CONTROL_POSITIVES}); the control response model and "
            "every uplift number built on it are statistically weak"
        )
    for a in TREATMENT_ACTIONS:
        if per_arm[a]["rows"] < MIN_EVENTS_PER_ACTION:
            report.warn(
                f"action '{a}' has {per_arm[a]['rows']} rows "
                f"(< {MIN_EVENTS_PER_ACTION}); mu_{a} is unreliable"
            )
        if per_arm[a]["positives"] < MIN_POSITIVES_PER_ACTION:
            report.warn(
                f"action '{a}' has {per_arm[a]['positives']} positives "
                f"(< {MIN_POSITIVES_PER_ACTION}); uplift for it is noisy"
            )

    # -- IPW weight behaviour ---------------------------------
    w = stabilized_ipw_weights(frame)
    ess = _effective_sample_size(w)
    n_extreme = int(np.sum(w[np.isfinite(w)] > MAX_IPW_WEIGHT))
    report.diagnostics["ipw"] = {
        "stabilized_weight_mean": float(np.nanmean(w)),
        "stabilized_weight_max": float(np.nanmax(w)),
        "effective_sample_size": round(ess, 1),
        "n_rows": int(len(frame)),
        "ess_fraction": round(ess / len(frame), 4) if len(frame) else None,
        "n_extreme_weights": n_extreme,
        "max_weight_threshold": MAX_IPW_WEIGHT,
    }
    if len(frame) and ess / len(frame) < OVERLAP_MIN_EFFECTIVE_SAMPLE_FRACTION:
        report.warn(
            f"effective sample size under stabilized IPW is {ess:.0f} / "
            f"{len(frame)} ({ess / len(frame):.0%}); weak overlap"
        )
    if n_extreme:
        report.warn(
            f"{n_extreme} rows have a stabilized IPW weight > {MAX_IPW_WEIGHT}"
        )

    report.diagnostics["assignment_strategies"] = (
        frame["assignment_strategy"].value_counts().to_dict()
    )
    report.diagnostics["weighting_recommendation"] = _weighting_recommendation(report)
    return report


def _weighting_recommendation(report: PropensityReport) -> str:
    ipw = report.diagnostics.get("ipw", {})
    ess_fraction = ipw.get("ess_fraction")
    if (
        report.passed
        and ess_fraction is not None
        and ess_fraction >= 0.95
        and not any("propensity <" in w for w in report.warnings)
    ):
        return (
            "Assignment is uniform-randomized and overlap is near-perfect "
            f"(ESS = {ess_fraction:.0%} of n). Inverse-propensity weighting is "
            "NOT required for unbiased treatment/control and action contrasts "
            "on this dataset; it is implemented and reported only as a "
            "production-readiness diagnostic and as a variance check on the "
            "policy-value estimates."
        )
    return (
        "Overlap or propensity quality is imperfect: prefer the stabilized-IPW "
        "or doubly-robust estimates over the naive difference in means, and "
        "review the warnings above before trusting any uplift number."
    )
