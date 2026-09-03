"""``decision_points`` CTEs for the uplift dataset.

One row per **eligible recovery decision at the moment the recovery event was
opened** (``as_of = recovery_event.created_at``, which the generator sets equal
to ``payment.failed_at``). Using event-creation time -- not the first
intervention's ``executed_at`` -- keeps the decision time identical across the
control and treatment arms, so ``hours_since_failure`` and every other
point-in-time feature is comparable between arms and no post-decision
information (the execution latency, the outcome) can leak in.

Two arms, unioned:

* **control**   -- every ``is_control`` recovery event; ``action = 'none'``.
* **treatment** -- the *first* intervention (attempt 1) on every non-control
                   event -- the randomized initial action. "First" is by
                   ``interventions.id`` (insertion order = attempt order), NOT by
                   ``executed_at``: the generator's per-attempt execution latency
                   grows with the attempt number, so a later attempt can carry
                   an earlier ``executed_at``. Escalation attempts 2-3 are a
                   separate decision and are excluded from the v1 dataset.

The composed SQL feeds :func:`app.ml.features.point_in_time.compose_feature_sql`,
so the feature body is byte-for-byte the predictive layer's.
"""
from __future__ import annotations

from app.ml.features.point_in_time import compose_feature_sql
from app.ml.uplift.config import CONTROL_ACTION

# Attempt-1 intervention per treatment event (insertion order == attempt order).
_FIRST_INTERVENTION = """
    SELECT DISTINCT ON (i.recovery_event_id)
        i.id                AS intervention_id,
        i.recovery_event_id AS recovery_event_id,
        i.action_type       AS action_type,
        i.executed_at       AS executed_at
    FROM interventions i
    WHERE i.executed_at IS NOT NULL
    ORDER BY i.recovery_event_id, i.id
"""

_UPLIFT_DECISION_POINTS = f"""
WITH decision_points AS (
    SELECT
        CAST(NULL AS INTEGER)      AS decision_point_id,
        re.id                      AS recovery_event_id,
        CAST('{CONTROL_ACTION}' AS VARCHAR) AS action,
        re.created_at              AS as_of
    FROM recovery_events re
    WHERE re.is_control = true

    UNION ALL

    SELECT
        fi.intervention_id         AS decision_point_id,
        fi.recovery_event_id       AS recovery_event_id,
        fi.action_type             AS action,
        re.created_at              AS as_of
    FROM ({_FIRST_INTERVENTION}) fi
    JOIN recovery_events re ON re.id = fi.recovery_event_id
    WHERE re.is_control = false
),
"""

# Label: control -> did the payment recover naturally; treatment -> outcome of
# the first intervention. Never reads a post-decision *feature*, only the label.
UPLIFT_CONTROL_LABEL_SQL = """
SELECT
    re.id AS recovery_event_id,
    (p.recovered_at IS NOT NULL)::int AS recovered
FROM recovery_events re
JOIN payments p ON p.id = re.payment_id
WHERE re.is_control = true
"""

UPLIFT_TREATMENT_LABEL_SQL = f"""
SELECT
    fi.intervention_id AS decision_point_id,
    o.payment_recovered::int AS recovered
FROM ({_FIRST_INTERVENTION}) fi
JOIN outcomes o ON o.intervention_id = fi.intervention_id
"""

# Logged assignment propensity for the first (attempt = 1) intervention.
# ``input_context`` is a JSON column: json ``->>`` works without a jsonb cast.
UPLIFT_TREATMENT_PROPENSITY_SQL = f"""
SELECT
    fi.intervention_id AS decision_point_id,
    (ae.input_context -> 'assignment' ->> 'propensity')::float   AS raw_action_propensity,
    (ae.input_context -> 'assignment' ->> 'exploration')::boolean AS exploration,
    (ae.input_context -> 'assignment' ->> 'strategy')            AS assignment_strategy
FROM ({_FIRST_INTERVENTION}) fi
JOIN agent_events ae
  ON ae.recovery_event_id = fi.recovery_event_id
 AND ae.event_type = 'intervention_decision'
 AND (ae.input_context -> 'assignment' ->> 'chosen_action') = fi.action_type
 AND COALESCE(ae.input_context ->> 'attempt', '1') = '1'
"""


def uplift_feature_sql() -> str:
    return compose_feature_sql(_UPLIFT_DECISION_POINTS)
