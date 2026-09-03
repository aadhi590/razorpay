"""Action + terminal tools.

``execute_recovery_action`` is the only mutating tool. In LIVE mode it now:

  1. persists an Intervention row, then
  2. asks :class:`~app.services.recovery_execution.RecoveryExecutionService` to
     idempotently create a **Razorpay Test Mode Payment Link** for it.

It still never sends a real SMS/WhatsApp (no verified provider exists) and never
marks a payment recovered -- a created Payment Link is *execution success*, not
recovery. In DRY-RUN mode it makes zero external calls and persists nothing.

After execution the loop hands control back to Gemini: it decides, on its own
next turn, whether to observe the outcome, try another eligible action, stop, or
escalate. The application never assumes success.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from app.agent.guardrails import (
    eligible_action_types,
    load_event,
    payment_recovered,
)
from app.agent.schemas import ESCALATION_TYPES, STOP_REASONS
from app.agent.tools.base import Tool, ToolContext, obj
from app.services.recovery_config import ACTION_TYPES, MAX_INTERVENTION_ATTEMPTS

_MAX_MSG = 320


def _attach_voice(ctx: ToolContext, message: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Synthesize the agent's Hinglish message into an audio FILE and merge the
    voice fields into a tool result. Never a phone call; never delivered to a
    customer. Fails safe -- a TTS failure just yields
    ``voice_generated: false, voice_reason: ...`` and the recovery flow is
    unaffected.

    TTS is a local, cheap, reversible rendering of text the agent already
    produced, so -- unlike the Razorpay Payment Link -- it also runs in dry-run
    (it touches nothing in the recovery workflow).
    """
    from app.services.voice import REASON_DISABLED, VoiceResult

    svc = ctx.voice_service
    if svc is None:
        result = VoiceResult(False, reason=REASON_DISABLED)
    else:
        result = svc.synthesize(
            message, key=f"re{ctx.state.recovery_event_id}"
        )

    ctx.state.voice_generated = result.generated
    ctx.state.voice_reason = result.reason
    ctx.state.audio_url = result.audio_url
    ctx.state.audio_path = result.audio_path
    ctx.state.voice_engine = result.engine
    payload.update(result.as_tool_fields())
    return payload


class ExecuteRecoveryAction(Tool):
    name = "execute_recovery_action"
    description = (
        "Execute one eligible recovery action. Provide a concise, natural "
        "Hinglish customer message personalised from the context you have "
        "gathered. Guardrails are enforced by the application; an ineligible "
        "action is rejected and you may reconsider."
    )
    parameters = obj(
        {
            "action_type": {
                "type": "string",
                "enum": list(ACTION_TYPES.keys()),
                "description": "One of the eligible actions.",
            },
            "customer_message": {
                "type": "string",
                "description": (
                    "Concise natural Hinglish message to the customer "
                    f"(<= {_MAX_MSG} chars). No card numbers, no email, no phone."
                ),
            },
            "reason": {
                "type": "string",
                "description": "One-sentence rationale for choosing this action.",
            },
        },
        required=["action_type", "customer_message", "reason"],
    )
    output_schema = obj({
        "executed": {"type": "boolean"},
        "simulated": {"type": "boolean"},
        "intervention_id": {"type": "integer"},
        "action_type": {"type": "string"},
        "razorpay_payment_link_id": {"type": "string"},
        "payment_link_url": {"type": "string"},
        "razorpay_status": {"type": "string"},
        "reused_existing_link": {"type": "boolean"},
        "communication_sent": {"type": "boolean"},
        "payment_recovered": {"type": "boolean"},
        "attempt_number": {"type": "integer"},
        "dry_run": {"type": "boolean"},
        "note": {"type": "string"},
    })
    mutating = True
    terminal = False

    def run(self, ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
        action_type = args["action_type"]
        reason = str(args.get("reason", ""))[:480]
        message = str(args.get("customer_message", ""))[:_MAX_MSG]

        event = load_event(ctx.db, ctx.state.recovery_event_id)
        # guardrails already ran in the loop; this is a defensive re-check
        assert event is not None and not event.is_control

        attempt_number = len(event.interventions) + 1
        record = {
            "action_type": action_type,
            "customer_message": message,
            "reason": reason,
            "at": datetime.now(timezone.utc).isoformat(),
        }

        if ctx.dry_run:
            record["simulated"] = True
            ctx.state.executed_actions.append(record)
            if action_type not in ctx.state.actions_attempted:
                ctx.state.actions_attempted.append(action_type)
            ctx.state.chosen_action = action_type
            ctx.state.customer_message = message
            return _attach_voice(ctx, message, {
                "executed": False,
                "simulated": True,
                "dry_run": True,
                "action_type": action_type,
                "attempt_number": attempt_number,
                "communication_sent": False,
                "payment_recovered": False,
                "would_persist": "pending Intervention + Razorpay Test Mode Payment Link",
                "customer_message_preview": message,
                "note": (
                    "DRY RUN: no Intervention persisted, NO Razorpay call, no "
                    "message sent. A TTS audio file may still be generated from "
                    "the message (file only -- not a call, not delivered)."
                ),
            })

        from app.models.interventions import Intervention
        from app.services.recovery_execution import RecoveryExecutionService

        intervention = Intervention(
            recovery_event_id=event.id,
            action_type=action_type,
            status="pending",
            cost_paise=int(ACTION_TYPES[action_type]["cost_paise"]),
            agent_reason=f"[gemini] {reason}"[:500],
            executed_at=None,
        )
        ctx.db.add(intervention)
        ctx.db.flush()

        service = RecoveryExecutionService(ctx.db, client=ctx.razorpay_client)
        result = service.create_payment_link(intervention, reason=reason)

        if not result.success:
            # Roll the intervention back so a transient Razorpay failure does
            # not burn an attempt or permanently block this action_type.
            ctx.db.delete(intervention)
            ctx.db.flush()
            ctx.state.errors.append(
                f"execute {action_type}: {result.error_code}"
            )
            return {
                "executed": False,
                "simulated": False,
                "dry_run": False,
                "action_type": action_type,
                "attempt_number": attempt_number,
                "payment_recovered": False,
                "error": result.error_code,
                "error_detail": result.error_summary,
                "note": (
                    "Payment Link creation failed; no attempt consumed. "
                    "You may retry, choose another eligible action, or escalate."
                ),
            }

        event.payment.retry_count += 1
        ctx.db.flush()

        record["simulated"] = False
        record["intervention_id"] = intervention.id
        record["razorpay_payment_link_id"] = result.payment_link_id
        ctx.state.executed_actions.append(record)
        if action_type not in ctx.state.actions_attempted:
            ctx.state.actions_attempted.append(action_type)
        ctx.state.chosen_action = action_type
        ctx.state.customer_message = message

        payload = result.as_tool_payload()
        payload.update(
            {
                "executed": True,
                "simulated": False,
                "dry_run": False,
                "intervention_id": intervention.id,
                "attempt_number": attempt_number,
                "note": (
                    "Razorpay Test Mode Payment Link "
                    f"{'reused' if result.reused else 'created'}; NO SMS/WhatsApp "
                    "sent (no verified provider). Link created != payment "
                    "recovered -- observe the outcome before concluding."
                ),
            }
        )
        return _attach_voice(ctx, message, payload)


class ObserveRecoveryOutcome(Tool):
    name = "observe_recovery_outcome"
    description = (
        "Observe the real recovery state: whether the Razorpay Payment Link was "
        "created, whether it has been PAID (verified webhook), how much was "
        "recovered, whether the event is terminal, and whether another action "
        "is still allowed. A created link is NOT a recovered payment."
    )
    parameters = obj({})
    output_schema = obj({
        "payment_recovered": {"type": "boolean"},
        "recovered_amount_paise": {"type": "integer"},
        "event_status": {"type": "string"},
        "terminal": {"type": "boolean"},
        "another_action_allowed": {"type": "boolean"},
        "interventions": {"type": "array"},
        "note": {"type": "string"},
    })

    def run(self, ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
        event = load_event(ctx.db, ctx.state.recovery_event_id)
        if event is None:
            return {"error": "event_not_found"}

        if ctx.dry_run:
            return {
                "payment_recovered": False,
                "dry_run": True,
                "note": (
                    "DRY RUN: nothing was really executed, so there is no "
                    "external state to observe. Treat as 'unknown' and decide "
                    "your next step."
                ),
            }

        recovered = payment_recovered(event)
        interventions = []
        recovered_amount = None
        for i in sorted(event.interventions, key=lambda x: x.id):
            link_created = i.razorpay_payment_link_id is not None
            paid = i.last_razorpay_status == "paid"
            if i.outcome is not None and i.outcome.payment_recovered:
                recovered_amount = i.outcome.recovered_amount_paise
            interventions.append(
                {
                    "intervention_id": i.id,
                    "action": i.action_type,
                    "payment_link_created": link_created,
                    "razorpay_status": i.last_razorpay_status,
                    "payment_link_paid": paid,
                    "outcome_recorded": i.outcome is not None,
                    "payment_recovered": bool(
                        i.outcome and i.outcome.payment_recovered
                    ),
                }
            )

        terminal = event.status in ("closed", "abandoned")
        attempts = len(event.interventions)
        another_allowed = (
            event.status == "open"
            and not recovered
            and attempts < MAX_INTERVENTION_ATTEMPTS
            and bool(eligible_action_types(event))
        )

        payload = {
            "payment_recovered": recovered,
            "recovered_amount_paise": recovered_amount,
            "event_status": event.status,
            "terminal": terminal,
            "attempts_used": attempts,
            "max_attempts": MAX_INTERVENTION_ATTEMPTS,
            "another_action_allowed": another_allowed,
            "interventions": interventions,
            "note": (
                "payment RECOVERED (verified webhook)" if recovered
                else "payment link(s) created but NOT paid yet; not recovered"
                if any(iv["payment_link_created"] for iv in interventions)
                else "no payment link created yet"
            ),
        }
        ctx.state.outcomes.append(payload)
        return payload


class EscalateRecovery(Tool):
    name = "escalate_recovery"
    description = (
        "Hand this recovery event to a human/team. Use only when automated "
        "actions are exhausted or inappropriate."
    )
    parameters = obj(
        {
            "escalation_type": {
                "type": "string",
                "enum": list(ESCALATION_TYPES),
            },
            "reasoning_summary": {
                "type": "string",
                "description": "Concise rationale (decision rationale, not chain-of-thought).",
            },
        },
        required=["escalation_type", "reasoning_summary"],
    )
    output_schema = obj({
        "escalated": {"type": "boolean"},
        "escalation_type": {"type": "string"},
    })
    terminal = True

    def run(self, ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
        etype = args["escalation_type"]
        summary = str(args.get("reasoning_summary", ""))[:480]
        ctx.state.escalation_required = True
        ctx.state.escalation_type = etype
        ctx.state.final_status = "escalated"
        ctx.state.stop_reason = "escalation_required"
        ctx.state.reasoning_summary = summary
        return {"escalated": True, "escalation_type": etype, "note": "run ends"}


class StopRecovery(Tool):
    name = "stop_recovery"
    description = (
        "End the run. Use for a clean finish (payment recovered, no eligible "
        "actions, attempt cap, expected value too low, action executed and "
        "awaiting outcome, ...). Include the final customer message if one is "
        "warranted."
    )
    parameters = obj(
        {
            "stop_reason": {
                "type": "string",
                "enum": list(STOP_REASONS),
            },
            "reasoning_summary": {
                "type": "string",
                "description": "Concise decision rationale (not hidden chain-of-thought).",
            },
            "customer_message": {
                "type": "string",
                "description": (
                    "Optional final Hinglish message to the customer "
                    f"(<= {_MAX_MSG} chars)."
                ),
            },
        },
        required=["stop_reason", "reasoning_summary"],
    )
    output_schema = obj({
        "stopped": {"type": "boolean"},
        "stop_reason": {"type": "string"},
    })
    terminal = True

    def run(self, ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
        reason = args["stop_reason"]
        if reason not in STOP_REASONS:
            reason = "other"
        ctx.state.final_status = "completed"
        ctx.state.stop_reason = reason
        ctx.state.reasoning_summary = str(args.get("reasoning_summary", ""))[:480]
        payload = {"stopped": True, "stop_reason": reason, "note": "run ends"}
        msg = args.get("customer_message")
        if msg:
            ctx.state.customer_message = str(msg)[:_MAX_MSG]
            payload = _attach_voice(ctx, ctx.state.customer_message, payload)
        return payload


ACTION_TOOLS = [
    ExecuteRecoveryAction(),
    ObserveRecoveryOutcome(),
    EscalateRecovery(),
    StopRecovery(),
]
