"""Feature contract for the recovery-response model.

Every feature here represents information available **strictly before** the
intervention's decision time ``T`` (``intervention.executed_at`` in training,
``now()`` at inference). See ``point_in_time.py`` for how the temporal boundary
is enforced in SQL.

The ONLY column drawn from the ``outcomes`` table is the training label; no
feature is derived from an outcome (see ``FORBIDDEN_LEAKAGE_SOURCES``).
"""
from __future__ import annotations

LABEL = "recovered"

# --- numeric features -------------------------------------------------
NUMERIC_FEATURES: list[str] = [
    "amount_paise",                 # payment.amount
    "subscription_amount",          # subscription.amount
    "priority",                     # recovery_event.priority (ordinal 0/1/2)
    "hours_since_failure",          # T - payment.failed_at
    "subscription_tenure_days",     # T - subscription.started_at
    "attempt_number",               # 1 + prior interventions on this event before T
    "already_tried_retry",          # 0/1: retry already used on this event before T
    "already_tried_sms_nudge",
    "already_tried_whatsapp_nudge",
    "already_tried_method_switch_prompt",
    "cust_prior_total_payments",     # customer payments scheduled (due) before T
    "cust_prior_failed_payments",    # ... with failed_at < T
    "cust_prior_successful_payments",  # prior_total - prior_failed
    "cust_prior_recovered_payments",   # ... with recovered_at < T (a prior recovery)
    "cust_prior_success_ratio",        # successful / total  (0 when no history)
    "cust_prior_failure_ratio",        # failed / total      (0 when no history)
    "cust_days_since_last_failure",     # T - max(failed_at < T)  (NULL if none)
    "cust_failures_last_90d",           # failed_at in [T-90d, T)
]

# --- categorical features -------------------------------------------
CATEGORICAL_FEATURES: list[str] = [
    "failure_reason",                  # payment.failure_reason
    "action",                          # the action being scored  <-- treatment variable
    "experiment_intervention_type",    # experiment arm assigned at event creation ('none' if unassigned)
]

ALL_FEATURES: list[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# --- metadata columns (carried in the dataset, NOT fed to the model) ---
# Used for splitting, grouping, coverage, auditing and future uplift work.
METADATA_COLUMNS: list[str] = [
    "decision_point_id",   # intervention.id in training; NULL at inference
    "recovery_event_id",   # grouping key for the split
    "customer_id",
    "payment_id",
    "as_of",               # decision time T
    "variant",             # 'treatment' (all training rows) / 'control'
    "is_control",
    "experiment_id",
    "currency",
]

FEATURE_DESCRIPTIONS: dict[str, str] = {
    "amount_paise": "Failed payment amount in paise.",
    "subscription_amount": "Subscription's recurring amount in paise.",
    "priority": "Recovery-event priority (0 low / 1 mid / 2 high), derived from amount at event creation.",
    "hours_since_failure": "Hours between the payment failure and the decision time T.",
    "subscription_tenure_days": "Age of the subscription at T, in days.",
    "attempt_number": "Which recovery attempt this is (1-based): 1 + interventions on this event executed before T.",
    "already_tried_retry": "1 if a 'retry' intervention was executed on this event before T.",
    "already_tried_sms_nudge": "1 if an 'sms_nudge' intervention was executed on this event before T.",
    "already_tried_whatsapp_nudge": "1 if a 'whatsapp_nudge' intervention was executed on this event before T.",
    "already_tried_method_switch_prompt": "1 if a 'method_switch_prompt' intervention was executed on this event before T.",
    "cust_prior_total_payments": "Count of the customer's payments whose scheduled due date is before T.",
    "cust_prior_failed_payments": "Count of the customer's payments that failed before T.",
    "cust_prior_successful_payments": "prior_total_payments - prior_failed_payments.",
    "cust_prior_recovered_payments": "Count of the customer's payments recovered (recovered_at) before T.",
    "cust_prior_success_ratio": "prior_successful / prior_total (0.0 when there is no history).",
    "cust_prior_failure_ratio": "prior_failed / prior_total (0.0 when there is no history).",
    "cust_days_since_last_failure": "Days since the customer's most recent prior failure (NULL if none).",
    "cust_failures_last_90d": "Count of the customer's failures in the 90 days before T.",
    "failure_reason": "Gateway failure reason for the payment.",
    "action": "Recovery action being evaluated (the treatment variable).",
    "experiment_intervention_type": "intervention_type of the experiment the event was assigned to ('none' if unassigned).",
}

# Any feature-construction SQL that references these (other than to build the
# label) is a leakage bug. Enforced by the data-quality validator.
FORBIDDEN_LEAKAGE_SOURCES: list[str] = [
    "outcomes",
    "payment_recovered",
    "recovered_amount_paise",
    "recovery_time_seconds",
    "observed_at",
]
