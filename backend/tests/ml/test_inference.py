from __future__ import annotations

import pandas as pd
import pytest

from app.ml.config import ACTIONS
from app.ml.features.schema import NUMERIC_FEATURES


def _feature_row() -> dict:
    row = {f: 1.0 for f in NUMERIC_FEATURES}
    row["failure_reason"] = "insufficient_funds"
    row["experiment_intervention_type"] = "none"
    return row


def test_predict_recovery_returns_probability(trained_model):
    p = trained_model.predict_recovery(_feature_row(), "retry")
    assert isinstance(p, float)
    assert 0.0 <= p <= 1.0


def test_predict_all_actions_returns_dict(trained_model):
    out = trained_model.predict_all_actions(_feature_row())
    assert set(out) == set(ACTIONS)
    assert all(0.0 <= v <= 1.0 for v in out.values())


def test_unknown_action_raises(trained_model):
    with pytest.raises(ValueError):
        trained_model.predict_recovery(_feature_row(), "carrier_pigeon")


def test_malformed_input_missing_feature_raises(trained_model):
    incomplete = {"failure_reason": "insufficient_funds", "action": "retry"}
    with pytest.raises(ValueError):
        trained_model.predict_from_frame(pd.DataFrame([incomplete]))


def test_predict_for_event_via_db(trained_model, conn, require_training_data):
    from sqlalchemy import text

    reid = conn.execute(text("SELECT recovery_event_id FROM interventions LIMIT 1")).scalar()
    scores = trained_model.predict_for_event(conn, int(reid))
    assert set(scores) == set(ACTIONS)
    for s in scores.values():
        assert 0.0 <= s.probability <= 1.0
        assert s.expected_value_paise == pytest.approx(
            s.probability * _amount(conn, int(reid)) - s.cost_paise, rel=1e-6
        )


def _amount(conn, reid: int) -> float:
    from sqlalchemy import text

    return float(
        conn.execute(
            text(
                "SELECT p.amount FROM payments p "
                "JOIN recovery_events re ON re.payment_id = p.id WHERE re.id = :r"
            ),
            {"r": reid},
        ).scalar()
    )


def test_explain_returns_supported_factors(trained_model):
    ex = trained_model.explain(_feature_row(), "method_switch_prompt")
    assert "top_factors" in ex and isinstance(ex["top_factors"], list)
    assert "model_global_importance_top" in ex
