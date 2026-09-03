from __future__ import annotations

import numpy as np
import pandas as pd

from app.ml.features.schema import ALL_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES
from app.ml.preprocessing.pipeline import build_preprocessor


def _row(**overrides) -> dict:
    base = {f: 1.0 for f in NUMERIC_FEATURES}
    base.update({f: "x" for f in CATEGORICAL_FEATURES})
    base["failure_reason"] = "insufficient_funds"
    base["action"] = "retry"
    base["experiment_intervention_type"] = "none"
    base.update(overrides)
    return base


def test_fit_transform_shape_stable():
    df = pd.DataFrame([_row(action=a) for a in ("retry", "sms_nudge", "whatsapp_nudge")])
    pre = build_preprocessor()
    Xt = pre.fit_transform(df[ALL_FEATURES])
    assert Xt.shape[0] == 3
    # transform of a new row keeps the same width
    new = pre.transform(pd.DataFrame([_row(action="method_switch_prompt")])[ALL_FEATURES])
    assert new.shape[1] == Xt.shape[1]


def test_unseen_category_does_not_error():
    train = pd.DataFrame([_row(failure_reason="insufficient_funds"),
                          _row(failure_reason="card_expired")])
    pre = build_preprocessor().fit(train[ALL_FEATURES])
    out = pre.transform(pd.DataFrame([_row(failure_reason="brand_new_reason")])[ALL_FEATURES])
    assert out.shape[0] == 1
    assert np.isfinite(out).all()


def test_missing_numeric_is_imputed():
    train = pd.DataFrame([_row(cust_days_since_last_failure=v) for v in (1.0, 2.0, 3.0)])
    pre = build_preprocessor().fit(train[ALL_FEATURES])
    row = _row()
    row["cust_days_since_last_failure"] = np.nan
    out = pre.transform(pd.DataFrame([row])[ALL_FEATURES])
    assert np.isfinite(out).all()  # NaN imputed, missing-indicator added
