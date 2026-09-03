"""ML layer configuration: versions, paths, action list, thresholds.

The action list is imported from the existing rules layer
(``app.services.recovery_config.ACTION_TYPES``) so the two never diverge.
"""
from __future__ import annotations

from pathlib import Path

from app.services.recovery_config import ACTION_TYPES

# --- versioning -----------------------------------------------------------
MODEL_NAME = "recovery_response"
MODEL_VERSION = "ml_v1"
FEATURE_VERSION = "feat_v1"

# --- actions -------------------------------------------------------------
# Ordered list of the recovery actions the model scores.
ACTIONS: list[str] = list(ACTION_TYPES.keys())
ACTION_COST_PAISE: dict[str, int] = {
    a: int(ACTION_TYPES[a]["cost_paise"]) for a in ACTIONS
}

# --- artifact locations ------------------------------------------------
_BACKEND_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = _BACKEND_ROOT / "artifacts" / "ml"
ARTIFACT_FILENAME = f"{MODEL_NAME}_{MODEL_VERSION}.joblib"
LATEST_ARTIFACT_FILENAME = f"{MODEL_NAME}_latest.joblib"


def artifact_path() -> Path:
    return ARTIFACT_DIR / ARTIFACT_FILENAME


def latest_artifact_path() -> Path:
    return ARTIFACT_DIR / LATEST_ARTIFACT_FILENAME


# --- training / split configuration ---------------------------------
RANDOM_STATE = 42
SPLIT_FRACTIONS = (0.60, 0.20, 0.20)  # train, validation, test (by event, chronological)

# Calibration: isotonic needs many positives; with a small positive count the
# safe choice is Platt/sigmoid scaling.
ISOTONIC_MIN_POSITIVES = 1000
CALIBRATION_CV_FOLDS = 5

# --- data-quality thresholds --------------------------------------
MIN_EXAMPLES_PER_ACTION = 30          # WARN below this
MIN_POSITIVES_PER_ACTION_FOR_METRICS = 5  # per-action metrics only reported above this
MAX_UNEXPECTED_NULL_FRACTION = 0.20   # FAIL if a non-nullable feature exceeds this
TARGET_RATE_WARN_LOW = 0.02
TARGET_RATE_WARN_HIGH = 0.98

# Features that are ALLOWED to contain nulls (everything else must not).
NULLABLE_NUMERIC_FEATURES = {"cust_days_since_last_failure"}
