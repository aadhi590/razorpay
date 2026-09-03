"""Shared point-in-time feature contract for the uplift layer.

The uplift models deliberately reuse the **exact same** feature definition as the
predictive recovery-response model (``app/ml/features/schema.py`` +
``point_in_time.py``). There is no second, incompatible feature set -- only the
set of *rows* differs (see ``app/ml/uplift/dataset``):

* predictive layer -> one row per observed intervention, ``as_of = executed_at``
* uplift layer     -> one row per eligible recovery *decision* (control events
                      included), ``as_of = recovery_event.created_at``

``BASELINE_FEATURES`` is ``ALL_FEATURES`` minus the ``action`` column: the
control response model and each per-action T-learner see a constant ``action``
within their training slice, so it carries no information there. The S-learner
uses the full ``ALL_FEATURES`` (``action`` is its treatment indicator).
"""
from __future__ import annotations

from app.ml.features.point_in_time import PointInTimeFeatureExtractor
from app.ml.features.schema import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    FEATURE_DESCRIPTIONS,
    FORBIDDEN_LEAKAGE_SOURCES,
    LABEL,
    METADATA_COLUMNS,
    NUMERIC_FEATURES,
)

TREATMENT_FEATURE = "action"

BASELINE_NUMERIC_FEATURES: list[str] = list(NUMERIC_FEATURES)
BASELINE_CATEGORICAL_FEATURES: list[str] = [
    c for c in CATEGORICAL_FEATURES if c != TREATMENT_FEATURE
]
BASELINE_FEATURES: list[str] = BASELINE_NUMERIC_FEATURES + BASELINE_CATEGORICAL_FEATURES

S_LEARNER_FEATURES: list[str] = list(ALL_FEATURES)

__all__ = [
    "PointInTimeFeatureExtractor",
    "ALL_FEATURES",
    "NUMERIC_FEATURES",
    "CATEGORICAL_FEATURES",
    "METADATA_COLUMNS",
    "FEATURE_DESCRIPTIONS",
    "FORBIDDEN_LEAKAGE_SOURCES",
    "LABEL",
    "TREATMENT_FEATURE",
    "BASELINE_FEATURES",
    "BASELINE_NUMERIC_FEATURES",
    "BASELINE_CATEGORICAL_FEATURES",
    "S_LEARNER_FEATURES",
]
