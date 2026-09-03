"""Uplift / causal-recovery layer configuration.

This layer sits *beside* the predictive recovery-response model
(``app/ml/``), never replacing it. Where the predictive model answers
``P(recovery | context, action)``, this layer estimates

    mu_0(X) = P(recovery | X, no intervention)         -- control response
    mu_a(X) = P(recovery | X, action = a)              -- action response
    uplift_a(X) = mu_a(X) - mu_0(X)                    -- incremental recovery

and turns that into an economic quantity (incremental expected revenue net of
intervention cost). See ``app/ml/uplift/README.md``.

Everything here is a *synthetic benchmark*: the data comes from
``app/scripts/generate_data.py --randomized-assignment``, whose outcome process
is a known closed form.
"""
from __future__ import annotations

from pathlib import Path

from app.ml.config import ACTION_COST_PAISE, ACTIONS, FEATURE_VERSION

# --- versioning --------------------------------------------------------
UPLIFT_MODEL_NAME = "uplift_recovery"
UPLIFT_MODEL_VERSION = "uplift_v1"
# The uplift layer reuses the predictive layer's point-in-time feature
# contract verbatim (see ``app/ml/uplift/features``); this pins which one.
UPLIFT_FEATURE_VERSION = FEATURE_VERSION

# --- actions ----------------------------------------------------------
# The "do nothing" pseudo-action. Control rows carry this value in the shared
# ``action`` categorical feature, so an S-learner can represent mu_0 as
# "predict with action = CONTROL_ACTION".
CONTROL_ACTION = "none"
TREATMENT_ACTIONS: list[str] = list(ACTIONS)
ALL_ARMS: list[str] = [CONTROL_ACTION, *TREATMENT_ACTIONS]
INTERVENTION_COST_PAISE: dict[str, int] = dict(ACTION_COST_PAISE)

# --- assignment design (synthetic generator) -------------------------
# ``generate_data.py`` draws ``is_control`` as Bernoulli(CONTROL_GROUP_FRACTION)
# independently of context, then (with --randomized-assignment) draws the
# treatment action uniformly over the eligible set and logs that propensity on
# each AgentEvent. The control fraction is NOT logged per event, so this
# documented design constant is used for the treatment-vs-control weighting and
# is cross-checked against the empirical fraction in the propensity diagnostics.
CONTROL_DESIGN_PROPENSITY = 0.20

# --- artifact locations ---------------------------------------------
_BACKEND_ROOT = Path(__file__).resolve().parents[3]
UPLIFT_ARTIFACT_DIR = _BACKEND_ROOT / "artifacts" / "ml"
UPLIFT_ARTIFACT_FILENAME = f"{UPLIFT_MODEL_NAME}_{UPLIFT_MODEL_VERSION}.joblib"
UPLIFT_LATEST_ARTIFACT_FILENAME = f"{UPLIFT_MODEL_NAME}_latest.joblib"


def uplift_artifact_path() -> Path:
    return UPLIFT_ARTIFACT_DIR / UPLIFT_ARTIFACT_FILENAME


def uplift_latest_artifact_path() -> Path:
    return UPLIFT_ARTIFACT_DIR / UPLIFT_LATEST_ARTIFACT_FILENAME


# --- training / split ----------------------------------------------
RANDOM_STATE = 42
SPLIT_FRACTIONS = (0.60, 0.20, 0.20)  # train / validation / test (event-grouped, chronological)

# Calibration: isotonic needs many positives. The control arm in particular is
# tiny, so sigmoid/Platt is the safe default everywhere.
ISOTONIC_MIN_POSITIVES = 1000
CALIBRATION_CV_FOLDS = 5

# --- data-sufficiency thresholds (WARN, never silently drop) --------
MIN_CONTROL_EVENTS = 300
MIN_CONTROL_POSITIVES = 30
MIN_EVENTS_PER_ACTION = 200
MIN_POSITIVES_PER_ACTION = 20

# --- propensity / overlap diagnostics ------------------------------
MIN_PROPENSITY = 1e-3          # below this, IPW weights explode
MAX_IPW_WEIGHT = 50.0          # a single row should not dominate a weighted mean
OVERLAP_MIN_EFFECTIVE_SAMPLE_FRACTION = 0.5  # ESS / n below this -> weak overlap

# --- uplift evaluation -------------------------------------------
UPLIFT_AT_K_FRACTIONS = (0.1, 0.2, 0.3, 0.5)
QINI_BOOTSTRAP_SAMPLES = 200
