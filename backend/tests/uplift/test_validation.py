"""Phase 4 / 5: propensity, overlap, split integrity, leakage failure modes."""
from __future__ import annotations

import pytest

from app.ml.training.splitting import grouped_chronological_split
from app.ml.training.validation import DataQualityError
from app.ml.uplift.config import SPLIT_FRACTIONS
from app.ml.uplift.validation import (
    stabilized_ipw_weights,
    validate_propensity,
    validate_uplift_frame,
)

pytestmark = pytest.mark.needs_data


def test_split_separates_events(uplift_dataset):
    split = grouped_chronological_split(uplift_dataset.frame, fractions=SPLIT_FRACTIONS)
    r = split.report
    assert r["event_overlap"] == {"train_val": 0, "train_test": 0, "val_test": 0}
    tr, va, te = split.frames(uplift_dataset.frame)
    assert set(tr["recovery_event_id"]) & set(te["recovery_event_id"]) == set()
    assert len(tr) + len(va) + len(te) == len(uplift_dataset.frame)
    # every split must contain some control and some treatment
    for part in (tr, va, te):
        assert part["treatment"].astype(bool).any()
        assert (~part["treatment"].astype(bool)).any()


def test_propensity_validator_passes_and_reports_overlap(uplift_dataset):
    rep = validate_propensity(uplift_dataset.frame)
    assert rep.passed, rep.failures
    d = rep.diagnostics
    assert d["propensity_range"]["n_le_zero_or_nan"] == 0
    assert d["propensity_range"]["n_gt_one"] == 0
    # uniform randomized assignment -> near-perfect overlap
    assert d["ipw"]["ess_fraction"] > 0.9
    assert abs(d["control_fraction"]["realized"] - 0.20) < 0.05


def test_stabilized_weights_are_near_one(uplift_dataset):
    w = stabilized_ipw_weights(uplift_dataset.frame)
    assert 0.8 < float(w.mean()) < 1.25
    assert float(w.max()) < 5.0


def test_propensity_validator_flags_bad_propensity(uplift_dataset):
    bad = uplift_dataset.frame.copy()
    bad.loc[bad.index[:5], "propensity"] = 0.0
    rep = validate_propensity(bad)
    assert not rep.passed
    assert any("propensity <= 0" in f for f in rep.failures)


def test_integrity_validator_passes(uplift_dataset):
    split = grouped_chronological_split(uplift_dataset.frame, fractions=SPLIT_FRACTIONS)
    rep = validate_uplift_frame(
        uplift_dataset.frame, split_event_overlap=split.report["event_overlap"]
    )
    assert rep.passed
    assert rep.checks["duplicate_events"] == 0
    assert rep.checks["uplift_feature_sql_forbidden_tokens"] == []


def test_integrity_validator_fails_on_split_contamination(uplift_dataset):
    with pytest.raises(DataQualityError):
        validate_uplift_frame(
            uplift_dataset.frame, split_event_overlap={"train_test": 4}
        )


def test_integrity_validator_fails_on_duplicate_event(uplift_dataset):
    import pandas as pd

    dup = pd.concat(
        [uplift_dataset.frame, uplift_dataset.frame.iloc[[0]]], ignore_index=True
    )
    with pytest.raises(DataQualityError):
        validate_uplift_frame(dup)


def test_integrity_validator_fails_on_temporal_violation(uplift_dataset):
    bad = uplift_dataset.frame.copy()
    bad.loc[bad.index[0], "hours_since_failure"] = -3.0
    with pytest.raises(DataQualityError):
        validate_uplift_frame(bad)
