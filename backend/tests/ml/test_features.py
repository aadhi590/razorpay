from __future__ import annotations

import pytest
from sqlalchemy import text

from app.ml.features.point_in_time import PointInTimeFeatureExtractor
from app.ml.features.schema import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    FORBIDDEN_LEAKAGE_SOURCES,
    LABEL,
    NUMERIC_FEATURES,
)

pytestmark = pytest.mark.needs_data


def test_feature_sql_has_no_leakage_sources():
    for mode in ("training", "inference"):
        sql = PointInTimeFeatureExtractor.feature_sql(mode).lower()
        for token in FORBIDDEN_LEAKAGE_SOURCES:
            assert token.lower() not in sql, (mode, token)


def test_training_frame_shape_and_label(training_frame):
    df = training_frame
    assert len(df) > 100
    for col in ALL_FEATURES + [LABEL]:
        assert col in df.columns
    assert set(df[LABEL].unique()) <= {0, 1}
    # every row is a treatment observation
    assert (df["is_control"] == False).all()  # noqa: E712


def test_no_temporal_violations(training_frame):
    df = training_frame
    assert (df["hours_since_failure"] >= 0).all()
    assert (df["attempt_number"] >= 1).all()
    assert (df["subscription_tenure_days"] >= 0).all()


def test_non_nullable_features_have_no_nulls(training_frame):
    df = training_frame
    for col in NUMERIC_FEATURES:
        if col == "cust_days_since_last_failure":
            continue
        assert df[col].isna().sum() == 0, col
    for col in CATEGORICAL_FEATURES:
        assert df[col].isna().sum() == 0, col


def test_point_in_time_customer_history_recomputed_independently(training_frame, conn):
    """Independently recompute cust_prior_failed_payments for a sample of rows
    from raw tables and confirm it matches (proves the < T filter is real)."""
    df = training_frame
    sample = df.sample(min(25, len(df)), random_state=0)
    for _, row in sample.iterrows():
        got = conn.execute(
            text(
                """
                SELECT count(*) FROM payments p
                JOIN subscriptions s ON s.id = p.subscription_id
                WHERE s.customer_id = :cid
                  AND p.id <> :pid
                  AND p.failed_at IS NOT NULL
                  AND p.failed_at < :t
                """
            ),
            {"cid": int(row["customer_id"]), "pid": int(row["payment_id"]), "t": row["as_of"]},
        ).scalar()
        assert int(row["cust_prior_failed_payments"]) == int(got)


def test_no_feature_uses_future_intervention(training_frame, conn):
    """attempt_number for a row must equal the count of that event's
    interventions strictly before its own executed_at, + 1."""
    df = training_frame
    sample = df.sample(min(25, len(df)), random_state=1)
    for _, row in sample.iterrows():
        prior = conn.execute(
            text(
                """
                SELECT count(*) FROM interventions i2
                WHERE i2.recovery_event_id = :re
                  AND i2.id <> :dp
                  AND i2.executed_at < :t
                """
            ),
            {"re": int(row["recovery_event_id"]), "dp": int(row["decision_point_id"]), "t": row["as_of"]},
        ).scalar()
        assert int(row["attempt_number"]) == int(prior) + 1


def test_inference_frame_has_one_row_per_action(require_training_data, conn):
    from app.ml.config import ACTIONS

    reid = conn.execute(text("SELECT recovery_event_id FROM interventions LIMIT 1")).scalar()
    fdf = PointInTimeFeatureExtractor().features_for_event(conn, int(reid))
    assert sorted(fdf["action"].tolist()) == sorted(ACTIONS)
    assert LABEL not in fdf.columns
    # context columns identical across the action rows
    assert fdf["amount_paise"].nunique() == 1


def test_inference_unknown_event_raises(conn):
    with pytest.raises(ValueError):
        PointInTimeFeatureExtractor().features_for_event(conn, 10**9)
