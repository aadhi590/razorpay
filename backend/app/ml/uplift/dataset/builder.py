"""Build the point-in-time uplift dataset.

One row per eligible recovery decision (see ``decision_points.py``):

    recovery_event_id, customer_id, payment_id, as_of (decision time),
    <shared point-in-time context features>, arm ('none' | action),
    treatment (bool), experiment_id, variant,
    raw_action_propensity (logged P(action | treated, eligible)),
    propensity (joint P(this arm/action cell)),
    recovered (observed outcome), assignment_strategy

No future information: features come from the shared body (every subquery
filters ``< as_of``); the label and propensity are joined on by key afterwards,
never used as predictors.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sqlalchemy import Connection, text

from app.ml.features.point_in_time import PointInTimeFeatureExtractor
from app.ml.features.schema import ALL_FEATURES, LABEL, METADATA_COLUMNS
from app.ml.uplift.config import (
    CONTROL_ACTION,
    CONTROL_DESIGN_PROPENSITY,
    TREATMENT_ACTIONS,
    UPLIFT_FEATURE_VERSION,
)
from app.ml.uplift.dataset.decision_points import (
    UPLIFT_CONTROL_LABEL_SQL,
    UPLIFT_TREATMENT_LABEL_SQL,
    UPLIFT_TREATMENT_PROPENSITY_SQL,
    uplift_feature_sql,
)

UPLIFT_METADATA_COLUMNS: list[str] = [
    *METADATA_COLUMNS,
    "arm",
    "treatment",
    "raw_action_propensity",
    "propensity",
    "exploration",
    "assignment_strategy",
]


@dataclass
class UpliftDataset:
    frame: pd.DataFrame
    feature_version: str
    dataset_version: str
    created_at: str
    stats: dict = field(default_factory=dict)

    @property
    def X(self) -> pd.DataFrame:  # noqa: N802
        return self.frame[ALL_FEATURES]

    @property
    def y(self) -> pd.Series:
        return self.frame[LABEL].astype(int)

    @property
    def treatment(self) -> pd.Series:
        return self.frame["treatment"].astype(bool)

    @property
    def groups(self) -> pd.Series:
        return self.frame["recovery_event_id"]

    @property
    def control(self) -> pd.DataFrame:
        return self.frame[~self.frame["treatment"].astype(bool)].reset_index(drop=True)

    def action_rows(self, action: str) -> pd.DataFrame:
        return self.frame[self.frame["arm"] == action].reset_index(drop=True)


def _dataset_version(frame: pd.DataFrame) -> str:
    key = "|".join(
        [
            str(len(frame)),
            str(int(frame["recovery_event_id"].nunique())),
            str(frame["as_of"].min()),
            str(frame["as_of"].max()),
            str(int(frame[LABEL].sum())),
            str(int(frame["treatment"].sum())),
        ]
    )
    return "uds_" + hashlib.sha256(key.encode()).hexdigest()[:12]


def build_uplift_dataset(conn: Connection) -> UpliftDataset:
    raw = pd.read_sql(text(uplift_feature_sql()), conn)
    if raw.empty:
        raise ValueError(
            "no uplift decision points found -- generate data with "
            "`python -m app.scripts.generate_data --reset --randomized-assignment`"
        )

    control_labels = pd.read_sql(text(UPLIFT_CONTROL_LABEL_SQL), conn)
    treat_labels = pd.read_sql(text(UPLIFT_TREATMENT_LABEL_SQL), conn)
    treat_prop = pd.read_sql(text(UPLIFT_TREATMENT_PROPENSITY_SQL), conn)

    raw = raw.sort_values(["as_of", "recovery_event_id"]).reset_index(drop=True)
    is_control = raw["is_control"].astype(bool).to_numpy()

    # -- label -----------------------------------------------------
    control_map = dict(
        zip(control_labels["recovery_event_id"], control_labels["recovered"])
    )
    treat_map = dict(
        zip(treat_labels["decision_point_id"], treat_labels["recovered"])
    )
    recovered = np.where(
        is_control,
        raw["recovery_event_id"].map(control_map).to_numpy(),
        raw["decision_point_id"].map(treat_map).to_numpy(),
    )
    raw[LABEL] = pd.to_numeric(pd.Series(recovered), errors="coerce")

    # -- propensity ----------------------------------------------
    prop_map = dict(
        zip(treat_prop["decision_point_id"], treat_prop["raw_action_propensity"])
    )
    expl_map = dict(zip(treat_prop["decision_point_id"], treat_prop["exploration"]))
    strat_map = dict(
        zip(treat_prop["decision_point_id"], treat_prop["assignment_strategy"])
    )
    raw_action_propensity = np.where(
        is_control,
        np.nan,
        raw["decision_point_id"].map(prop_map).to_numpy(),
    )
    raw["raw_action_propensity"] = pd.to_numeric(
        pd.Series(raw_action_propensity), errors="coerce"
    )
    # Joint propensity of landing in this (arm, action) cell:
    #   control  -> P(is_control)                 = CONTROL_DESIGN_PROPENSITY
    #   action a -> P(treated) * P(a | eligible)  = (1 - p_ctrl) * logged
    raw["propensity"] = np.where(
        is_control,
        CONTROL_DESIGN_PROPENSITY,
        (1.0 - CONTROL_DESIGN_PROPENSITY) * raw["raw_action_propensity"],
    )
    raw["exploration"] = np.where(
        is_control, False, raw["decision_point_id"].map(expl_map).fillna(False)
    ).astype(bool)
    raw["assignment_strategy"] = np.where(
        is_control,
        "randomized_control",
        raw["decision_point_id"].map(strat_map).fillna("unknown"),
    )
    raw["arm"] = raw["action"].astype("string")
    raw["treatment"] = ~is_control

    # -- drop rows we cannot label / weight (never fabricate) -----
    n_before = len(raw)
    keep = raw[LABEL].notna()
    keep &= raw["treatment"].eq(False) | raw["propensity"].gt(0)
    dropped = raw[~keep]
    raw = raw[keep].reset_index(drop=True)

    # -- shared finalize (derived features, dtypes, column order) --
    finalized = PointInTimeFeatureExtractor._finalize(raw)
    extra = raw[
        [c for c in UPLIFT_METADATA_COLUMNS if c not in finalized.columns]
    ].reset_index(drop=True)
    frame = pd.concat([finalized.reset_index(drop=True), extra], axis=1)
    frame = frame.loc[:, ~frame.columns.duplicated()]

    stats = _describe(frame, n_dropped=n_before - len(frame), dropped=dropped)
    return UpliftDataset(
        frame=frame,
        feature_version=UPLIFT_FEATURE_VERSION,
        dataset_version=_dataset_version(frame),
        created_at=datetime.now(timezone.utc).isoformat(),
        stats=stats,
    )


def _describe(frame: pd.DataFrame, *, n_dropped: int, dropped: pd.DataFrame) -> dict:
    tr = frame["treatment"].astype(bool)
    y = frame[LABEL].astype(int)
    per_arm: dict[str, dict] = {}
    for arm in [CONTROL_ACTION, *TREATMENT_ACTIONS]:
        m = frame["arm"] == arm
        n = int(m.sum())
        per_arm[arm] = {
            "rows": n,
            "positives": int(y[m].sum()),
            "recovery_rate": round(float(y[m].mean()), 6) if n else None,
            "distinct_events": int(frame.loc[m, "recovery_event_id"].nunique()),
            "mean_propensity": (
                round(float(frame.loc[m, "propensity"].mean()), 6) if n else None
            ),
        }
    return {
        "n_rows": int(len(frame)),
        "n_recovery_events": int(frame["recovery_event_id"].nunique()),
        "n_customers": int(frame["customer_id"].nunique()),
        "n_control": int((~tr).sum()),
        "n_treatment": int(tr.sum()),
        "n_positive": int(y.sum()),
        "control_recovery_rate": round(float(y[~tr].mean()), 6) if (~tr).any() else None,
        "treatment_recovery_rate": round(float(y[tr].mean()), 6) if tr.any() else None,
        "naive_observed_lift": (
            round(float(y[tr].mean() - y[~tr].mean()), 6)
            if tr.any() and (~tr).any()
            else None
        ),
        "as_of_min": str(frame["as_of"].min()),
        "as_of_max": str(frame["as_of"].max()),
        "per_arm": per_arm,
        "n_dropped": int(n_dropped),
        "dropped_reason_counts": (
            {"unlabelled_or_no_propensity": int(len(dropped))} if n_dropped else {}
        ),
    }
