"""Point-in-time-correct feature construction.

The SAME SQL body (:data:`_FEATURE_BODY_SQL`) is used for training and for
inference -- only the ``decision_points`` CTE that feeds it changes:

* training  -> one row per historical intervention, ``as_of = executed_at``
* inference -> one row per candidate action for a live recovery event,
               ``as_of = now()``

Every subquery filters on ``< as_of`` (or ``executed_at < as_of`` /
``failed_at < as_of`` / ``due_at < as_of``), so a feature can never see the
current intervention, its outcome, or anything that happened afterwards. The
body never references the ``outcomes`` table; the label is joined on
separately by :class:`~app.ml.datasets.builder` and checked by the
data-quality validator.

Customer payment history uses a reconstructed schedule
(``subscription.started_at + rank*30d``) because successful payments carry no
timestamp in the current schema -- a synthetic-data dependency documented in
``app/ml/README.md``.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
from sqlalchemy import Connection, text

from app.ml.config import ACTIONS, FEATURE_VERSION
from app.ml.features.schema import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    METADATA_COLUMNS,
    NUMERIC_FEATURES,
)

DECISION_TIME_COLUMN = "as_of"

# --- shared feature body (everything after the `decision_points` CTE) -----
_FEATURE_BODY_SQL = """
payment_schedule AS (
    SELECT
        p.id             AS payment_id,
        s.customer_id    AS customer_id,
        s.started_at
          + (row_number() OVER (PARTITION BY p.subscription_id ORDER BY p.id)
             * INTERVAL '30 days')          AS due_at,
        p.failed_at      AS failed_at,
        p.recovered_at   AS recovered_at
    FROM payments p
    JOIN subscriptions s ON s.id = p.subscription_id
)
SELECT
    dp.decision_point_id                                            AS decision_point_id,
    dp.recovery_event_id                                            AS recovery_event_id,
    cu.id                                                           AS customer_id,
    p.id                                                            AS payment_id,
    dp.as_of                                                        AS as_of,
    re.variant                                                      AS variant,
    re.is_control                                                   AS is_control,
    re.experiment_id                                                AS experiment_id,
    p.currency                                                      AS currency,

    -- numeric features -------------------------------------------------
    p.amount                                                        AS amount_paise,
    s.amount                                                        AS subscription_amount,
    re.priority                                                     AS priority,
    EXTRACT(EPOCH FROM (dp.as_of - p.failed_at))    / 3600.0        AS hours_since_failure,
    GREATEST(EXTRACT(EPOCH FROM (dp.as_of - s.started_at)) / 86400.0, 0.0)  AS subscription_tenure_days,
    1 + (
        SELECT count(*) FROM interventions i2
        WHERE i2.recovery_event_id = dp.recovery_event_id
          AND i2.id IS DISTINCT FROM dp.decision_point_id
          AND i2.executed_at < dp.as_of
    )                                                               AS attempt_number,
    (SELECT count(*) > 0 FROM interventions i2
        WHERE i2.recovery_event_id = dp.recovery_event_id
          AND i2.id IS DISTINCT FROM dp.decision_point_id
          AND i2.executed_at < dp.as_of
          AND i2.action_type = 'retry')::int                        AS already_tried_retry,
    (SELECT count(*) > 0 FROM interventions i2
        WHERE i2.recovery_event_id = dp.recovery_event_id
          AND i2.id IS DISTINCT FROM dp.decision_point_id
          AND i2.executed_at < dp.as_of
          AND i2.action_type = 'sms_nudge')::int                    AS already_tried_sms_nudge,
    (SELECT count(*) > 0 FROM interventions i2
        WHERE i2.recovery_event_id = dp.recovery_event_id
          AND i2.id IS DISTINCT FROM dp.decision_point_id
          AND i2.executed_at < dp.as_of
          AND i2.action_type = 'whatsapp_nudge')::int               AS already_tried_whatsapp_nudge,
    (SELECT count(*) > 0 FROM interventions i2
        WHERE i2.recovery_event_id = dp.recovery_event_id
          AND i2.id IS DISTINCT FROM dp.decision_point_id
          AND i2.executed_at < dp.as_of
          AND i2.action_type = 'method_switch_prompt')::int         AS already_tried_method_switch_prompt,

    (SELECT count(*) FROM payment_schedule ps
        WHERE ps.customer_id = cu.id AND ps.payment_id <> p.id
          AND ps.due_at < dp.as_of)                                 AS cust_prior_total_payments,
    (SELECT count(*) FROM payment_schedule ps
        WHERE ps.customer_id = cu.id AND ps.payment_id <> p.id
          AND ps.failed_at IS NOT NULL AND ps.failed_at < dp.as_of) AS cust_prior_failed_payments,
    (SELECT count(*) FROM payment_schedule ps
        WHERE ps.customer_id = cu.id AND ps.payment_id <> p.id
          AND ps.recovered_at IS NOT NULL AND ps.recovered_at < dp.as_of)
                                                                    AS cust_prior_recovered_payments,
    EXTRACT(EPOCH FROM (dp.as_of - (
        SELECT max(ps.failed_at) FROM payment_schedule ps
        WHERE ps.customer_id = cu.id AND ps.payment_id <> p.id
          AND ps.failed_at < dp.as_of
    ))) / 86400.0                                                   AS cust_days_since_last_failure,
    (SELECT count(*) FROM payment_schedule ps
        WHERE ps.customer_id = cu.id AND ps.payment_id <> p.id
          AND ps.failed_at >= dp.as_of - INTERVAL '90 days'
          AND ps.failed_at <  dp.as_of)                             AS cust_failures_last_90d,

    -- categorical features --------------------------------------------
    p.failure_reason                                                AS failure_reason,
    dp.action                                                       AS action,
    COALESCE(e.intervention_type, 'none')                           AS experiment_intervention_type

FROM decision_points dp
JOIN recovery_events re ON re.id = dp.recovery_event_id
JOIN payments p         ON p.id = re.payment_id
JOIN subscriptions s    ON s.id = p.subscription_id
JOIN customers cu       ON cu.id = s.customer_id
LEFT JOIN experiments e  ON e.id = re.experiment_id
"""

_TRAINING_DECISION_POINTS = """
WITH decision_points AS (
    SELECT
        i.id                AS decision_point_id,
        i.recovery_event_id AS recovery_event_id,
        i.action_type       AS action,
        i.executed_at       AS as_of
    FROM interventions i
    WHERE i.executed_at IS NOT NULL
),
"""

_LABEL_SQL = """
SELECT i.id AS decision_point_id, o.payment_recovered::int AS recovered
FROM interventions i
JOIN outcomes o ON o.intervention_id = i.id
WHERE i.executed_at IS NOT NULL
"""


def _inference_decision_points_cte() -> str:
    values = ", ".join(f"('{a}')" for a in ACTIONS)  # ACTIONS are trusted constants
    return f"""
WITH decision_points AS (
    SELECT
        CAST(NULL AS INTEGER)        AS decision_point_id,
        CAST(:re_id AS INTEGER)      AS recovery_event_id,
        v.action                     AS action,
        CAST(:as_of AS TIMESTAMPTZ)  AS as_of
    FROM (VALUES {values}) AS v(action)
),
"""


def training_feature_sql() -> str:
    return _TRAINING_DECISION_POINTS + _FEATURE_BODY_SQL


def inference_feature_sql() -> str:
    return _inference_decision_points_cte() + _FEATURE_BODY_SQL


def compose_feature_sql(decision_points_cte: str) -> str:
    """Attach a caller-supplied ``decision_points`` CTE to the shared feature
    body so a new consumer (e.g. the uplift dataset, which needs control events
    as decision points too) never forks the feature definition.

    ``decision_points_cte`` must be a ``WITH decision_points AS ( ... ),``
    fragment producing exactly the columns
    ``(decision_point_id, recovery_event_id, action, as_of)`` -- the same shape
    as :data:`_TRAINING_DECISION_POINTS`. The trailing comma is required: the
    shared body continues the ``WITH`` chain with ``payment_schedule AS (...)``.
    """
    stripped = decision_points_cte.strip()
    if not stripped.lower().startswith("with decision_points"):
        raise ValueError(
            "decision_points_cte must start with 'WITH decision_points AS ('"
        )
    if not stripped.endswith(","):
        raise ValueError(
            "decision_points_cte must end with ',' so the shared body can "
            "continue the WITH chain"
        )
    return decision_points_cte + _FEATURE_BODY_SQL


class PointInTimeFeatureExtractor:
    """Builds the model feature frame for training and for live inference."""

    feature_version = FEATURE_VERSION
    numeric_features = NUMERIC_FEATURES
    categorical_features = CATEGORICAL_FEATURES
    all_features = ALL_FEATURES
    metadata_columns = METADATA_COLUMNS

    # -- training -----------------------------------------------------
    def build_training_frame(self, conn: Connection) -> pd.DataFrame:
        feats = pd.read_sql(text(training_feature_sql()), conn)
        labels = pd.read_sql(text(_LABEL_SQL), conn)
        merged = feats.merge(labels, on="decision_point_id", how="inner")
        return self._finalize(merged)

    # -- inference --------------------------------------------------
    def features_for_event(
        self,
        conn: Connection,
        recovery_event_id: int,
        as_of: datetime | None = None,
    ) -> pd.DataFrame:
        as_of = as_of or datetime.now(timezone.utc)
        df = pd.read_sql(
            text(inference_feature_sql()),
            conn,
            params={"re_id": int(recovery_event_id), "as_of": as_of},
        )
        if df.empty:
            raise ValueError(
                f"recovery_event {recovery_event_id} not found / not joinable"
            )
        return self._finalize(df)

    # -- shared post-processing -----------------------------------
    @staticmethod
    def _finalize(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        total = df["cust_prior_total_payments"].astype("int64")
        failed = df["cust_prior_failed_payments"].astype("int64")
        successful = (total - failed).clip(lower=0)
        df["cust_prior_successful_payments"] = successful
        df["cust_prior_success_ratio"] = (successful / total.where(total > 0)).fillna(0.0)
        df["cust_prior_failure_ratio"] = (failed / total.where(total > 0)).fillna(0.0)

        for col in NUMERIC_FEATURES:
            # Decimals from EXTRACT() -> float; keep NaN for nullable features.
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

        for col in CATEGORICAL_FEATURES:
            df[col] = df[col].astype("string").fillna("__MISSING__")

        ordered = [c for c in METADATA_COLUMNS if c in df.columns]
        ordered += ALL_FEATURES
        if "recovered" in df.columns:
            ordered += ["recovered"]
        return df[ordered]

    # -- introspection (used by the leakage validator + tests) -------
    @staticmethod
    def feature_sql(mode: str = "training") -> str:
        if mode == "training":
            return training_feature_sql()
        if mode == "inference":
            return inference_feature_sql()
        raise ValueError(mode)
