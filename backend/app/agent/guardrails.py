"""Hard application-level guardrails.

These run in application code **unconditionally**, before any state-changing
tool executes, regardless of what Gemini reasoned or claimed. Gemini can never
talk its way past them: a violation is returned to the model as a structured
tool error so it can reconsider on its next genuine turn, and repeated
violations terminate the run safely (handled in the loop).

The checks mirror the ones the existing
:class:`~app.services.recovery_orchestrator.RecoveryOrchestratorService`
enforces inline, and share its constants
(:mod:`app.services.recovery_config`) so the two cannot drift. Eligibility /
premium-gating is delegated to the real
:class:`~app.services.recovery_policy.RulesBasedRecoveryPolicy`.
"""
from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.models.interventions import Intervention
from app.models.payment import Payment
from app.models.recovery_events import RecoveryEvent
from app.models.subscription import Subscription
from app.services.recovery_config import (
    ACTION_TYPES,
    MAX_INTERVENTION_ATTEMPTS,
    STATUS_OPEN,
)
from app.services.recovery_policy import PolicyContext, RulesBasedRecoveryPolicy


@dataclass(frozen=True)
class GuardrailResult:
    ok: bool
    code: str
    message: str

    @classmethod
    def allow(cls) -> "GuardrailResult":
        return cls(True, "ok", "allowed")

    @classmethod
    def deny(cls, code: str, message: str) -> "GuardrailResult":
        return cls(False, code, message)


_LOAD_OPTIONS = (
    selectinload(RecoveryEvent.interventions).selectinload(Intervention.outcome),
    selectinload(RecoveryEvent.payment)
    .selectinload(Payment.subscription)
    .selectinload(Subscription.customer),
)


def load_event(db: Session, recovery_event_id: int) -> RecoveryEvent | None:
    stmt = (
        select(RecoveryEvent)
        .where(RecoveryEvent.id == recovery_event_id)
        .options(*_LOAD_OPTIONS)
    )
    return db.scalars(stmt).one_or_none()


def prior_action_types(event: RecoveryEvent) -> list[str]:
    return [i.action_type for i in sorted(event.interventions, key=lambda i: i.id)]


def attempt_number(event: RecoveryEvent) -> int:
    return len(event.interventions) + 1


def payment_recovered(event: RecoveryEvent) -> bool:
    p = event.payment
    if p.status == "success" or p.recovered_at is not None:
        return True
    return any(
        i.outcome is not None and i.outcome.payment_recovered
        for i in event.interventions
    )


def build_policy_context(event: RecoveryEvent) -> PolicyContext:
    """The exact context the orchestrator would build for this event."""
    payment = event.payment
    customer = payment.subscription.customer
    return PolicyContext(
        failure_reason=payment.failure_reason,
        amount_paise=payment.amount,
        priority=event.priority,
        is_control=event.is_control,
        attempt_number=attempt_number(event),
        prior_action_types=prior_action_types(event),
        customer_successful_payments=customer.total_successful_payments,
        customer_failed_payments=customer.total_failed_payments,
        recovery_event_id=event.id,
    )


def eligible_action_types(event: RecoveryEvent) -> list[str]:
    """Untried, rules-eligible actions (includes premium gating), best-first."""
    if event.is_control:
        return []
    decision = RulesBasedRecoveryPolicy().decide(build_policy_context(event))
    return [c.action_type for c in decision.candidates]


# --- checks ----------------------------------------------------------

def check_event_actionable(event: RecoveryEvent | None) -> GuardrailResult:
    if event is None:
        return GuardrailResult.deny("event_not_found", "recovery event does not exist")
    if event.payment is None or event.payment.subscription is None:
        return GuardrailResult.deny(
            "missing_data", "recovery event has no payment/subscription/customer"
        )
    if event.is_control:
        return GuardrailResult.deny(
            "control_event", "control event: no intervention may be executed"
        )
    if payment_recovered(event):
        return GuardrailResult.deny(
            "already_recovered", "payment is already recovered; nothing to do"
        )
    if event.status != STATUS_OPEN:
        return GuardrailResult.deny(
            "not_open", f"recovery event is not open (status={event.status})"
        )
    if len(event.interventions) >= MAX_INTERVENTION_ATTEMPTS:
        return GuardrailResult.deny(
            "max_attempts_reached",
            f"max intervention attempts reached "
            f"({len(event.interventions)}/{MAX_INTERVENTION_ATTEMPTS})",
        )
    return GuardrailResult.allow()


def check_execute_action(
    event: RecoveryEvent | None, action_type: str
) -> GuardrailResult:
    base = check_event_actionable(event)
    if not base.ok:
        return base
    assert event is not None  # narrowed by check_event_actionable

    if action_type not in ACTION_TYPES:
        return GuardrailResult.deny(
            "unsupported_action",
            f"unknown action {action_type!r}; supported: {sorted(ACTION_TYPES)}",
        )
    if action_type in prior_action_types(event):
        return GuardrailResult.deny(
            "action_already_attempted",
            f"action {action_type!r} has already been attempted on this event",
        )
    eligible = eligible_action_types(event)
    if action_type not in eligible:
        return GuardrailResult.deny(
            "action_not_eligible",
            f"action {action_type!r} is not in the eligible set {eligible} "
            f"(rules policy: premium gating / already tried)",
        )
    return GuardrailResult.allow()
