"""Phase 2 / 3: control model, S-/T-learner training, uplift calc, robustness."""
from __future__ import annotations

import numpy as np
import pytest

from app.ml.uplift.config import CONTROL_ACTION, TREATMENT_ACTIONS
from app.ml.uplift.estimators.learners import SLearner, TLearner
from app.ml.uplift.features import ALL_FEATURES

pytestmark = pytest.mark.needs_data


def test_t_learner_trains_one_model_per_arm(uplift_split):
    est = TLearner(base="logreg").fit(uplift_split["train"])
    assert set(est._models) == {CONTROL_ACTION, *TREATMENT_ACTIONS}
    assert est.metadata.per_arm_training_rows[CONTROL_ACTION] > 0
    assert est.metadata.calibration_method == "sigmoid"


def test_control_model_probabilities_in_range(fitted_t_learner, uplift_split):
    mu0 = fitted_t_learner.predict_baseline(uplift_split["test"][ALL_FEATURES])
    assert mu0.shape[0] == len(uplift_split["test"])
    assert np.all((mu0 >= 0) & (mu0 <= 1))


def test_s_learner_trains_and_scores(uplift_split):
    est = SLearner(base="logreg").fit(uplift_split["train"])
    X = uplift_split["test"][ALL_FEATURES]
    mu0 = est.predict_baseline(X)
    for a in TREATMENT_ACTIONS:
        mua = est.predict_action(X, a)
        assert mua.shape == mu0.shape
        assert np.all((mua >= 0) & (mua <= 1))


def test_uplift_is_difference_of_probabilities(fitted_t_learner, uplift_split):
    X = uplift_split["test"][ALL_FEATURES].head(50)
    for a in TREATMENT_ACTIONS:
        u = fitted_t_learner.predict_uplift(X, a)
        expected = fitted_t_learner.predict_action(X, a) - fitted_t_learner.predict_baseline(X)
        assert np.allclose(u, expected)


def test_predict_row_shapes(fitted_t_learner, uplift_split):
    row = uplift_split["test"][ALL_FEATURES].head(1)
    pred = fitted_t_learner.predict_row(row)
    assert 0 <= pred.baseline_probability <= 1
    assert set(pred.treatment_probability) == set(TREATMENT_ACTIONS)
    assert set(pred.uplift) == set(TREATMENT_ACTIONS)


def test_directional_uplift_matches_generator(fitted_t_learner, uplift_split):
    """Averaged over the held-out set, every action's mean predicted uplift
    should be positive and method_switch >= retry (the generator's ordering)."""
    X = uplift_split["test"][ALL_FEATURES]
    means = {a: float(fitted_t_learner.predict_uplift(X, a).mean()) for a in TREATMENT_ACTIONS}
    assert means["method_switch_prompt"] > means["retry"]
    assert all(v > 0 for v in means.values())


def test_unseen_category_does_not_raise(fitted_t_learner, uplift_split):
    X = uplift_split["test"][ALL_FEATURES].head(5).copy()
    X["failure_reason"] = "a_reason_never_seen_at_fit_time"
    out = fitted_t_learner.predict_baseline(X)
    assert np.all(np.isfinite(out))


def test_missing_numeric_is_imputed(fitted_t_learner, uplift_split):
    X = uplift_split["test"][ALL_FEATURES].head(5).copy()
    X["cust_prior_success_ratio"] = np.nan
    out = fitted_t_learner.predict_baseline(X)
    assert np.all(np.isfinite(out))
