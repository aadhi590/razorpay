"""Read-only context tools.

Each returns a *bounded, compact* view -- never raw database rows. They populate
the corresponding slot on :class:`~app.agent.state.RecoveryAgentState` so the
run trace shows exactly what Gemini asked for and when.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from app.agent.guardrails import (
    attempt_number,
    eligible_action_types,
    load_event,
    payment_recovered,
    prior_action_types,
)
from app.agent.tools.base import Tool, ToolContext, obj
from app.services.recovery_config import ACTION_TYPES, MAX_INTERVENTION_ATTEMPTS


def _event_or_error(ctx: ToolContext):
    event = load_event(ctx.db, ctx.state.recovery_event_id)
    if event is None:
        return None, {"error": "event_not_found",
                      "recovery_event_id": ctx.state.recovery_event_id}
    return event, None


def _hours_since(ts: datetime | None) -> float | None:
    if ts is None:
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return round((datetime.now(timezone.utc) - ts).total_seconds() / 3600.0, 1)


def _customer_ref(customer) -> str:
    if customer.email and "@" in customer.email:
        return customer.email.split("@", 1)[0]
    return customer.external_customer_id


class GetRecoveryEventContext(Tool):
    name = "get_recovery_event_context"
    description = (
        "Core facts about this recovery event: status, control flag, priority, "
        "failure reason, amount, and which actions were already attempted."
    )
    parameters = obj({})
    output_schema = obj({
        "status": {"type": "string"},
        "is_control": {"type": "boolean"},
        "priority": {"type": "integer"},
        "failure_reason": {"type": "string"},
        "amount_paise": {"type": "integer"},
        "attempt_number": {"type": "integer"},
        "actions_already_attempted": {"type": "array", "items": {"type": "string"}},
        "payment_recovered": {"type": "boolean"},
    })

    def run(self, ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
        event, err = _event_or_error(ctx)
        if err:
            return err
        p = event.payment
        payload = {
            "recovery_event_id": event.id,
            "status": event.status,
            "is_control": event.is_control,
            "priority": event.priority,
            "variant": event.variant,
            "failure_reason": p.failure_reason,
            "amount_paise": p.amount,
            "currency": p.currency,
            "attempt_number": attempt_number(event),
            "max_attempts": MAX_INTERVENTION_ATTEMPTS,
            "actions_already_attempted": prior_action_types(event),
            "payment_status": p.status,
            "payment_recovered": payment_recovered(event),
        }
        ctx.state.event_context = payload
        ctx.state.actions_attempted = list(payload["actions_already_attempted"])
        ctx.state.current_attempt = payload["attempt_number"]
        return payload


class GetPaymentContext(Tool):
    name = "get_payment_context"
    description = (
        "The failed payment: amount, currency, failure reason, how long ago it "
        "failed, retry count, and whether it has since been recovered."
    )
    parameters = obj({})
    output_schema = obj({
        "amount_paise": {"type": "integer"},
        "failure_reason": {"type": "string"},
        "hours_since_failure": {"type": "number"},
        "retry_count": {"type": "integer"},
        "status": {"type": "string"},
        "recovered": {"type": "boolean"},
    })

    def run(self, ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
        event, err = _event_or_error(ctx)
        if err:
            return err
        p = event.payment
        payload = {
            "amount_paise": p.amount,
            "currency": p.currency,
            "failure_reason": p.failure_reason,
            "failed_at": p.failed_at.isoformat() if p.failed_at else None,
            "hours_since_failure": _hours_since(p.failed_at),
            "retry_count": p.retry_count,
            "status": p.status,
            "recovered": p.recovered_at is not None or p.status == "success",
        }
        ctx.state.payment_context = payload
        return payload


class GetSubscriptionContext(Tool):
    name = "get_subscription_context"
    description = (
        "The subscription behind the failed payment: recurring amount, status, "
        "tenure, and when the next payment is due."
    )
    parameters = obj({})
    output_schema = obj({
        "amount_paise": {"type": "integer"},
        "status": {"type": "string"},
        "tenure_days": {"type": "number"},
    })

    def run(self, ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
        event, err = _event_or_error(ctx)
        if err:
            return err
        sub = event.payment.subscription
        payload = {
            "amount_paise": sub.amount,
            "currency": sub.currency,
            "status": sub.status,
            "tenure_days": _hours_since(sub.started_at) and round(
                _hours_since(sub.started_at) / 24.0, 1
            ),
            "next_payment_at": (
                sub.next_payment_at.isoformat() if sub.next_payment_at else None
            ),
        }
        ctx.state.subscription_context = payload
        return payload


class GetCustomerRecoveryHistory(Tool):
    name = "get_customer_recovery_history"
    description = (
        "The customer's payment reliability and prior recovery experience: "
        "lifetime success/failure counts, how many past recovery events they "
        "had and how many recovered, and the interventions already tried on "
        "THIS event with their outcomes."
    )
    parameters = obj({})
    output_schema = obj({
        "customer_ref": {"type": "string"},
        "total_successful_payments": {"type": "integer"},
        "total_failed_payments": {"type": "integer"},
        "reliability_ratio": {"type": "number"},
        "prior_recovery_events": {"type": "integer"},
        "prior_recovery_events_recovered": {"type": "integer"},
        "this_event_interventions": {"type": "array"},
    })

    def run(self, ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
        from sqlalchemy import func, select

        from app.models.outcome import Outcome
        from app.models.payment import Payment
        from app.models.recovery_events import RecoveryEvent
        from app.models.subscription import Subscription

        event, err = _event_or_error(ctx)
        if err:
            return err
        customer = event.payment.subscription.customer

        total = customer.total_successful_payments + customer.total_failed_payments
        reliability = (
            round(customer.total_successful_payments / total, 3) if total else None
        )

        prior_events = ctx.db.execute(
            select(
                func.count(RecoveryEvent.id),
                func.count(RecoveryEvent.id).filter(
                    RecoveryEvent.status == "closed"
                ),
            )
            .select_from(RecoveryEvent)
            .join(Payment, Payment.id == RecoveryEvent.payment_id)
            .join(Subscription, Subscription.id == Payment.subscription_id)
            .where(
                Subscription.customer_id == customer.id,
                RecoveryEvent.id != event.id,
            )
        ).one()

        this_event = [
            {
                "action": i.action_type,
                "status": i.status,
                "recovered": bool(i.outcome and i.outcome.payment_recovered),
            }
            for i in sorted(event.interventions, key=lambda i: i.id)
        ]

        payload = {
            "customer_ref": _customer_ref(customer),
            "total_successful_payments": customer.total_successful_payments,
            "total_failed_payments": customer.total_failed_payments,
            "reliability_ratio": reliability,
            "prior_recovery_events": int(prior_events[0]),
            "prior_recovery_events_recovered": int(prior_events[1]),
            "this_event_interventions": this_event,
            "contacted_on_this_event": len(this_event),
        }
        ctx.state.customer_context = {
            "customer_ref": payload["customer_ref"],
            "reliability_ratio": reliability,
        }
        ctx.state.recovery_history = payload
        return payload


class GetAvailableRecoveryActions(Tool):
    name = "get_available_recovery_actions"
    description = (
        "The recovery actions available RIGHT NOW: which are eligible (untried "
        "and allowed by the rules policy, including premium gating) and which "
        "are not, with the reason."
    )
    parameters = obj({})
    output_schema = obj({
        "eligible_actions": {"type": "array"},
        "ineligible_actions": {"type": "array"},
        "attempts_used": {"type": "integer"},
        "max_attempts": {"type": "integer"},
    })

    def run(self, ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
        event, err = _event_or_error(ctx)
        if err:
            return err

        tried = set(prior_action_types(event))
        eligible = eligible_action_types(event)
        ineligible = []
        for a in ACTION_TYPES:
            if a in eligible:
                continue
            if a in tried:
                reason = "already attempted"
            elif event.is_control:
                reason = "control event"
            else:
                reason = "not offered by rules policy (premium gating / low value)"
            ineligible.append({"action_type": a, "reason": reason})

        payload = {
            "eligible_actions": [
                {"action_type": a, "cost_paise": int(ACTION_TYPES[a]["cost_paise"])}
                for a in eligible
            ],
            "ineligible_actions": ineligible,
            "attempts_used": len(event.interventions),
            "max_attempts": MAX_INTERVENTION_ATTEMPTS,
        }
        ctx.state.eligible_actions = payload["eligible_actions"]
        return payload


CONTEXT_TOOLS = [
    GetRecoveryEventContext(),
    GetPaymentContext(),
    GetSubscriptionContext(),
    GetCustomerRecoveryHistory(),
    GetAvailableRecoveryActions(),
]
