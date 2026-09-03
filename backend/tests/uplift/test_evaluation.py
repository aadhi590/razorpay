"""Phase 6: causal / uplift evaluation metrics."""
from __future__ import annotations

import numpy as np
import pytest

from app.ml.uplift.evaluation.metrics import (
    evaluate_uplift,
    observed_lift_by_action,
    policy_value,
    qini_curve,
    uplift_at_k,
)

pytestmark = pytest.mark.needs_data


def _attach(est, frame):
    from app.ml.features.schema import ALL_FEATURES
    from app.ml.uplift.config import TREATMENT_ACTIONS

    out = frame.reset_index(drop=True).copy()
    base = est.predict_baseline(out[ALL_FEATURES])
    U = np.column_stack(
        [est.predict_action(out[ALL_FEATURES], a) - base for a in TREATMENT_ACTIONS]
    )
    out["baseline_probability"] = base
    out["tau_hat"] = U.max(axis=1)
    out["best_action"] = [TREATMENT_ACTIONS[i] for i in U.argmax(axis=1)]
    for i, a in enumerate(TREATMENT_ACTIONS):
        out[f"uplift_{a}"] = U[:, i]
    return out


def test_qini_curve_basic_properties():
    rng = np.random.default_rng(0)
    n = 2000
    tau = rng.normal(size=n)
    treated = rng.random(n) < 0.5
    # outcome truly increases with tau for treated
    y = ((rng.random(n) < (0.1 + 0.2 * (tau > 0) * treated))).astype(int)
    q = qini_curve(tau, treated, y)
    assert len(q["curve"]) == 20
    assert q["curve"][-1]["population_fraction"] == 1.0
    # a real signal -> positive coefficient
    assert q["qini_coefficient"] > 0


def test_observed_lift_by_action_beats_control(uplift_dataset):
    lift = observed_lift_by_action(uplift_dataset.frame)
    assert lift["control"]["n"] > 0
    for a in ("retry", "sms_nudge", "whatsapp_nudge", "method_switch_prompt"):
        assert lift[a]["observed_uplift_vs_control"] > 0


def test_policy_value_uplift_beats_random(fitted_t_learner, uplift_split):
    pred = _attach(fitted_t_learner, uplift_split["test"])
    pv = policy_value(pred)
    assert pv["treat_none_control_rate"] is not None
    # the uplift policy should not do worse than random action assignment
    assert pv["uplift_policy"]["value"] >= pv["random_action_value"] - 0.03


def test_uplift_at_k_monotone_ish(fitted_t_learner, uplift_split):
    pred = _attach(fitted_t_learner, uplift_split["test"])
    rows = uplift_at_k(
        pred["tau_hat"].to_numpy(),
        pred["treatment"].astype(bool).to_numpy(),
        pred["recovered"].astype(int).to_numpy(),
    )
    assert [r["k_fraction"] for r in rows] == [0.1, 0.2, 0.3, 0.5]
    assert all(r["mean_predicted_uplift"] is not None for r in rows)


def test_evaluate_uplift_reports_reliability(fitted_t_learner, uplift_split):
    pred = _attach(fitted_t_learner, uplift_split["test"])
    ev = evaluate_uplift(pred)
    assert "qini" in ev and "policy_value" in ev and "uplift_at_k" in ev
    assert "statistical_reliability" in ev
    assert isinstance(ev["statistical_reliability"]["reliable"], bool)
    assert ev["n_control"] > 0
