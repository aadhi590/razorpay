from __future__ import annotations

import numpy as np
import pytest

from app.ml.datasets.coverage import action_coverage
from app.ml.training.splitting import grouped_chronological_split
from app.ml.training.validation import DataQualityError, validate_training_frame

pytestmark = pytest.mark.needs_data


@pytest.fixture
def dataset(training_dataset):
    """Session-built (see tests/ml/conftest.py). Consumers here only read it or
    ``.copy()`` before mutating, so sharing one instance is safe."""
    return training_dataset


def test_dataset_one_row_per_observed_intervention(dataset, conn):
    from sqlalchemy import text

    n_iv = conn.execute(text("SELECT count(*) FROM interventions WHERE executed_at IS NOT NULL")).scalar()
    assert dataset.stats["n_rows"] == n_iv
    assert dataset.stats["n_positive"] + dataset.stats["n_negative"] == dataset.stats["n_rows"]


def test_action_coverage_reports_all_actions(dataset):
    cov = action_coverage(dataset.frame)
    from app.ml.config import ACTIONS

    assert set(cov.per_action) == set(ACTIONS)
    for a, m in cov.per_action.items():
        assert m["examples"] == m["recovered"] + m["not_recovered"]


def test_validation_passes_on_real_frame(dataset):
    report = validate_training_frame(dataset.frame)
    assert report.passed
    assert report.checks["temporal_violations"] == 0
    assert report.checks["feature_sql_forbidden_tokens_training"] == []


def test_validation_fails_on_temporal_violation(dataset):
    bad = dataset.frame.copy()
    bad.loc[bad.index[0], "hours_since_failure"] = -5.0
    with pytest.raises(DataQualityError):
        validate_training_frame(bad)


def test_validation_fails_on_non_binary_label(dataset):
    bad = dataset.frame.copy()
    bad.loc[bad.index[0], "recovered"] = 2
    with pytest.raises(DataQualityError):
        validate_training_frame(bad)


def test_validation_fails_on_impossible_value(dataset):
    bad = dataset.frame.copy()
    bad.loc[bad.index[0], "cust_prior_success_ratio"] = 1.5
    with pytest.raises(DataQualityError):
        validate_training_frame(bad)


def test_validation_fails_on_split_contamination(dataset):
    with pytest.raises(DataQualityError):
        validate_training_frame(
            dataset.frame, split_contamination={"train_test": 3}
        )


def test_split_is_grouped_and_chronological(dataset):
    frame = dataset.frame
    split = grouped_chronological_split(frame)
    r = split.report
    assert r["event_overlap"] == {"train_val": 0, "train_test": 0, "val_test": 0}

    tr, va, te = split.frames(frame)
    # every row of an event stays together
    assert set(tr["recovery_event_id"]) & set(te["recovery_event_id"]) == set()
    # chronological: train ends before test starts (allow tiny boundary overlap)
    assert tr["as_of"].max() <= te["as_of"].min() + np.timedelta64(2, "D")
    assert len(tr) + len(va) + len(te) == len(frame)
