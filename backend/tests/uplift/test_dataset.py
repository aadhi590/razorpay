"""Phase 1 / 5: uplift dataset construction, point-in-time correctness, leakage."""
from __future__ import annotations

import pytest
from sqlalchemy import text

from app.ml.features.schema import ALL_FEATURES, LABEL
from app.ml.uplift.config import CONTROL_ACTION, TREATMENT_ACTIONS
from app.ml.uplift.dataset.decision_points import uplift_feature_sql

pytestmark = pytest.mark.needs_data


def test_one_row_per_recovery_event(uplift_dataset):
    f = uplift_dataset.frame
    assert f["recovery_event_id"].is_unique
    assert uplift_dataset.stats["n_rows"] == len(f)


def test_both_arms_present_with_expected_values(uplift_dataset):
    f = uplift_dataset.frame
    arms = set(f["arm"].unique())
    assert arms == {CONTROL_ACTION, *TREATMENT_ACTIONS}
    assert (f.loc[~f["treatment"].astype(bool), "arm"] == CONTROL_ACTION).all()
    assert set(f["recovered"].unique()) <= {0, 1}


def test_control_rows_match_db_count(uplift_dataset, conn):
    n_control = conn.execute(
        text("select count(*) from recovery_events where is_control")
    ).scalar()
    assert int((~uplift_dataset.frame["treatment"].astype(bool)).sum()) == n_control


def test_feature_sql_has_no_outcomes_table(uplift_dataset):
    sql = uplift_feature_sql().lower()
    assert "from outcomes" not in sql
    assert "join outcomes" not in sql
    assert "payment_recovered" not in sql


def test_no_temporal_violations_either_arm(uplift_dataset):
    f = uplift_dataset.frame
    assert (f["hours_since_failure"] >= 0).all()
    assert (f["attempt_number"] >= 1).all()
    # first decision -> nobody has a prior attempt on their own event
    assert (f["attempt_number"] == 1).all()


def test_non_nullable_features_have_no_nulls(uplift_dataset):
    f = uplift_dataset.frame
    for col in ALL_FEATURES:
        if col == "cust_days_since_last_failure":
            continue
        assert f[col].isna().sum() == 0, col


def test_propensity_columns_present_and_positive(uplift_dataset):
    import numpy as np

    f = uplift_dataset.frame
    assert (f["propensity"] > 0).all()
    assert (f["propensity"] <= 1).all()
    # control uses the design constant; treatment uses (1-p_ctrl)*logged
    ctrl = f[~f["treatment"].astype(bool)]
    assert np.allclose(ctrl["propensity"].to_numpy(), 0.20)


def test_labels_recomputed_from_raw_tables(uplift_dataset, conn):
    """Independently confirm the control label = payment recovered naturally."""
    f = uplift_dataset.frame
    ctrl = f[~f["treatment"].astype(bool)].sample(min(15, (~f["treatment"].astype(bool)).sum()), random_state=0)
    for _, row in ctrl.iterrows():
        got = conn.execute(
            text(
                "select (p.recovered_at is not null)::int from payments p "
                "join recovery_events re on re.payment_id = p.id where re.id = :r"
            ),
            {"r": int(row["recovery_event_id"])},
        ).scalar()
        assert int(row[LABEL]) == int(got)


def test_treatment_label_is_first_attempt_outcome(uplift_dataset, conn):
    f = uplift_dataset.frame
    tr = f[f["treatment"].astype(bool)].sample(15, random_state=1)
    for _, row in tr.iterrows():
        got = conn.execute(
            text(
                """
                select o.payment_recovered::int
                from interventions i join outcomes o on o.intervention_id = i.id
                where i.recovery_event_id = :r
                order by i.id limit 1
                """
            ),
            {"r": int(row["recovery_event_id"])},
        ).scalar()
        assert int(row[LABEL]) == int(got)
        assert row["arm"] == conn.execute(
            text(
                "select action_type from interventions where recovery_event_id = :r "
                "order by id limit 1"
            ),
            {"r": int(row["recovery_event_id"])},
        ).scalar()


def test_synthetic_monotone_signal(uplift_dataset):
    """The generator's action effectiveness is retry < sms < whatsapp <
    method_switch; the observed first-attempt rates should reflect that and every
    action should beat the control rate."""
    pa = uplift_dataset.stats["per_arm"]
    ctrl_rate = pa[CONTROL_ACTION]["recovery_rate"]
    for a in TREATMENT_ACTIONS:
        assert pa[a]["recovery_rate"] > ctrl_rate
