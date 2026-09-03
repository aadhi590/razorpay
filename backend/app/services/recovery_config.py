"""Recovery decision constants.

These mirror the intervention model encoded in ``app/scripts/generate_data.py``
(``ACTION_TYPES``, ``FAILURE_REASONS``, ``DIMINISHING_RETURNS_DECAY``), which is
the agreed starting specification for the recovery decision engine. They are
duplicated here rather than imported from the offline data generator so the
runtime service layer carries no dependency on the data-generation tooling.
Keep the two in sync until the generator is refactored to share this module.
"""
from __future__ import annotations

# action_type -> economics used by the rules-based policy.
# effectiveness is the generator's per-action recovery multiplier.
ACTION_TYPES: dict[str, dict[str, float | int]] = {
    "retry": {"cost_paise": 50, "effectiveness": 0.35},
    "sms_nudge": {"cost_paise": 20, "effectiveness": 0.45},
    "whatsapp_nudge": {"cost_paise": 80, "effectiveness": 0.55},
    "method_switch_prompt": {"cost_paise": 150, "effectiveness": 0.65},
}

# failure_reason -> relative recoverability multiplier.
FAILURE_REASONS: dict[str, float] = {
    "insufficient_funds": 0.55,
    "card_expired": 0.20,
    "bank_timeout": 0.75,
    "issuer_decline": 0.35,
}

# Multiplier applied per additional intervention attempt (diminishing returns):
# decay ** (attempt_number - 1).
DIMINISHING_RETURNS_DECAY = 0.6

# Hard cap on intervention attempts per recovery event. The generator draws
# num_attempts from rng.choices([1, 2, 3], ...), so 3 is the effective ceiling.
MAX_INTERVENTION_ATTEMPTS = 3

# Recovery-probability clamp bounds (same range the generator clamps to).
MIN_RECOVERY_PROB = 0.02
MAX_RECOVERY_PROB = 0.9

# Used when a payment's failure_reason is missing or unrecognised.
DEFAULT_REASON_MULTIPLIER = 0.30

# Neutral prior for customer reliability when there is no payment history.
DEFAULT_RELIABILITY = 0.5

# The premium (most expensive) action is only offered to higher-value
# recoveries unless it is the only remaining untried option.
PREMIUM_ACTION = "method_switch_prompt"
PREMIUM_MIN_PRIORITY = 1
PREMIUM_MIN_AMOUNT_PAISE = 19900

# Recovery-event lifecycle values (mirrors app/scripts/generate_data.py).
STATUS_OPEN = "open"
STATUS_CLOSED = "closed"
STATUS_ABANDONED = "abandoned"
