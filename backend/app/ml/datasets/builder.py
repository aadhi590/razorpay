"""Assemble the training dataset: one row per observed (recovery opportunity, action).

Only *observed* interventions become rows. We never fabricate a counterfactual
outcome for an action that was not actually tried on an event, and control
events (no interventions) are excluded from the response model entirely -- they
are kept aside for future uplift work.

A single recovery event can contribute several rows (escalation tried more than
one action, sometimes the same action twice); rows from one event never cross
the train/val/test boundary (see ``training.splitting``).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone

import pandas as pd
from sqlalchemy import Connection

from app.ml.config import FEATURE_VERSION
from app.ml.features.point_in_time import PointInTimeFeatureExtractor
from app.ml.features.schema import ALL_FEATURES, LABEL


@dataclass
class TrainingDataset:
    frame: pd.DataFrame
    feature_version: str
    dataset_version: str
    created_at: str
    stats: dict = field(default_factory=dict)

    @property
    def X(self) -> pd.DataFrame:  # noqa: N802 - conventional ML name
        return self.frame[ALL_FEATURES]

    @property
    def y(self) -> pd.Series:
        return self.frame[LABEL].astype(int)

    @property
    def groups(self) -> pd.Series:
        return self.frame["recovery_event_id"]


def _dataset_version(frame: pd.DataFrame) -> str:
    """Stable fingerprint of the observed data (not of the wall clock)."""
    key = "|".join(
        [
            str(len(frame)),
            str(int(frame["recovery_event_id"].nunique())),
            str(frame["as_of"].min()),
            str(frame["as_of"].max()),
            str(int(frame[LABEL].sum())),
        ]
    )
    return "ds_" + hashlib.sha256(key.encode()).hexdigest()[:12]


def build_training_dataset(conn: Connection) -> TrainingDataset:
    extractor = PointInTimeFeatureExtractor()
    frame = extractor.build_training_frame(conn)
    frame = frame.sort_values(["as_of", "decision_point_id"]).reset_index(drop=True)

    pos = int(frame[LABEL].sum())
    stats = {
        "n_rows": int(len(frame)),
        "n_recovery_events": int(frame["recovery_event_id"].nunique()),
        "n_customers": int(frame["customer_id"].nunique()),
        "n_positive": pos,
        "n_negative": int(len(frame) - pos),
        "positive_rate": round(pos / len(frame), 6) if len(frame) else 0.0,
        "as_of_min": str(frame["as_of"].min()),
        "as_of_max": str(frame["as_of"].max()),
        "action_counts": frame["action"].value_counts().to_dict(),
        "action_positive_counts": (
            frame.groupby("action")[LABEL].sum().astype(int).to_dict()
        ),
    }
    return TrainingDataset(
        frame=frame,
        feature_version=FEATURE_VERSION,
        dataset_version=_dataset_version(frame),
        created_at=datetime.now(timezone.utc).isoformat(),
        stats=stats,
    )
