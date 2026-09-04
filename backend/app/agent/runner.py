"""Run the agent for one recovery event and persist an auditable trace.

Persistence reuses the existing tables (Section 21 -- no new tables, no
migration):

* one ``AgentEvent`` row (``event_type="agent_recovery_run"``) whose
  ``input_context`` JSON holds the whole turn-by-turn trace, quantitative
  scores, token usage and errors;
* ``AuditLog`` rows (``actor="gemini_recovery_agent"``) for the run outcome and
  for any executed action.

``GEMINI_API_KEY`` is never written to either. Raw chain-of-thought is never
stored -- only the concise decision rationale the model returned via
``stop_recovery`` / ``escalate_recovery``.
"""
from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy.orm import Session

from app.agent.agent import RecoveryAgent
from app.agent.config import AgentConfig
from app.agent.guardrails import load_event
from app.agent.providers import make_provider
from app.agent.providers.base import LLMProvider, ProviderUnavailable
from app.agent.schemas import AgentRunResult
from app.agent.state import RecoveryAgentState
from app.models.agent_events import AgentEvent
from app.models.audit_log import AuditLog

AGENT_ACTOR = "gemini_recovery_agent"


class AgentRunError(Exception):
    """The run could not even start (e.g. event not found)."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def run_recovery_agent(
    db: Session,
    recovery_event_id: int,
    *,
    dry_run: bool = True,
    provider: LLMProvider | None = None,
    config: AgentConfig | None = None,
    persist: bool = True,
    razorpay_client: object | None = None,
    voice_service: object | None = None,
    triggered_by: str = "manual",
) -> AgentRunResult:
    """``triggered_by`` is recorded verbatim in the persisted trace
    (``AgentEvent.input_context`` and every ``AuditLog`` row this run writes) so
    a scheduler-driven run is always distinguishable from a manual one. It is
    metadata only: it never changes what the agent does."""
    config = config or AgentConfig.from_settings()

    event = load_event(db, recovery_event_id)
    if event is None:
        raise AgentRunError("event_not_found",
                            f"recovery event {recovery_event_id} not found")

    run_id = uuid.uuid4().hex

    if provider is None:
        try:
            provider = make_provider(config)
        except ProviderUnavailable as exc:
            state = RecoveryAgentState(recovery_event_id, dry_run)
            state.final_status = "failed_safe"
            state.stop_reason = "quota_or_api_failure"
            state.reasoning_summary = str(exc)
            state.errors.append(str(exc))
            result = _to_result(state, config.model, run_id, config.provider)
            if persist:
                _persist(db, event.id, run_id, result, dry_run, triggered_by)
            return result

    agent = RecoveryAgent(
        db, provider, config,
        razorpay_client=razorpay_client, voice_service=voice_service,
    )
    state = agent.run(recovery_event_id, dry_run=dry_run)
    result = _to_result(
        state,
        provider.model if hasattr(provider, "model") else config.model,
        run_id,
        getattr(config, "provider", "gemini"),
    )

    if persist:
        _persist(db, event.id, run_id, result, dry_run, triggered_by)
        db.commit()
    return result


# --- result assembly -------------------------------------------------

def _to_result(
    state: RecoveryAgentState, model: str, run_id: str, agent: str = "gemini"
) -> AgentRunResult:
    chosen = state.chosen_action
    decision = _decision_label(state)
    outcome = state.outcomes[-1] if state.outcomes else None

    return AgentRunResult(
        recovery_event_id=state.recovery_event_id,
        model=model,
        dry_run=state.dry_run,
        status=state.final_status or "failed_safe",
        stop_reason=state.stop_reason or "other",
        decision=decision,
        chosen_action=chosen,
        customer_message=state.customer_message,
        voice_generated=state.voice_generated,
        voice_reason=state.voice_reason,
        audio_url=state.audio_url,
        voice_engine=state.voice_engine,
        reasoning_summary=state.reasoning_summary,
        escalation_required=state.escalation_required,
        escalation_type=state.escalation_type,
        turns_used=state.turns_used,
        actions_attempted=list(state.actions_attempted),
        actions_executed=[
            r["action_type"] for r in state.executed_actions
        ],
        outcome=outcome,
        latency_ms=state.latency_ms,
        token_usage=state.token_usage(),
        quantitative_scores=state.quantitative_scores,
        action_incrementality=state.action_incrementality,
        action_lift_trend=state.action_lift_trend,
        tool_trace=list(state.trace),
        errors=list(state.errors),
        agent=agent,
    )


def _decision_label(state: RecoveryAgentState) -> str:
    if state.escalation_required:
        return f"escalate:{state.escalation_type}"
    if state.executed_actions:
        return f"execute:{state.executed_actions[-1]['action_type']}"
    return f"stop:{state.stop_reason or 'other'}"


# --- persistence ---------------------------------------------------

def _persist(
    db: Session,
    recovery_event_id: int,
    run_id: str,
    result: AgentRunResult,
    dry_run: bool,
    triggered_by: str = "manual",
) -> None:
    trace_ctx: dict[str, Any] = {
        "agent_run_id": run_id,
        "agent": result.agent,
        "model": result.model,
        "dry_run": dry_run,
        "triggered_by": triggered_by,
        "status": result.status,
        "stop_reason": result.stop_reason,
        "decision": result.decision,
        "chosen_action": result.chosen_action,
        "escalation_required": result.escalation_required,
        "escalation_type": result.escalation_type,
        "turns_used": result.turns_used,
        "latency_ms": result.latency_ms,
        "token_usage": result.token_usage,
        "actions_attempted": result.actions_attempted,
        "actions_executed": result.actions_executed,
        "reasoning_summary": result.reasoning_summary,
        "voice": {
            "voice_generated": result.voice_generated,
            "voice_reason": result.voice_reason,
            "audio_url": result.audio_url,
            "voice_engine": result.voice_engine,
            "note": "audio FILE only -- no phone call, not delivered to the customer",
        },
        "quantitative_scores": result.quantitative_scores,
        "action_incrementality": result.action_incrementality,
        "action_lift_trend": result.action_lift_trend,
        "tool_trace": [t.model_dump() for t in result.tool_trace],
        "tools_requested_in_order": [t.tool for t in result.tool_trace],
        "outcome": result.outcome,
        "errors": result.errors,
    }

    confidence = _chosen_probability(result)

    db.add(
        AgentEvent(
            recovery_event_id=recovery_event_id,
            event_type="agent_recovery_run",
            input_context=trace_ctx,
            decision=result.decision[:500],
            confidence=confidence,
        )
    )
    db.add(
        AuditLog(
            recovery_event_id=recovery_event_id,
            actor=AGENT_ACTOR,
            action=f"agent_run_{result.status}",
            reason=(result.reasoning_summary or result.stop_reason)[:500],
            event_metadata={
                "agent_run_id": run_id,
                "triggered_by": triggered_by,
                "dry_run": dry_run,
                "decision": result.decision,
                "stop_reason": result.stop_reason,
                "turns_used": result.turns_used,
                "tools_requested_in_order": [t.tool for t in result.tool_trace],
            },
        )
    )
    for rec in result.actions_executed:
        db.add(
            AuditLog(
                recovery_event_id=recovery_event_id,
                actor=AGENT_ACTOR,
                action=f"agent_action_{rec}_{'simulated' if dry_run else 'executed'}",
                reason=result.reasoning_summary[:500],
                event_metadata={
                    "agent_run_id": run_id,
                    "triggered_by": triggered_by,
                    "dry_run": dry_run,
                },
            )
        )


def _chosen_probability(result: AgentRunResult) -> float | None:
    """A real number from the quantitative layer (never invented): the chosen
    action's model recovery probability, if scores were fetched."""
    if not result.chosen_action or not result.quantitative_scores:
        return None
    for s in result.quantitative_scores:
        if s.get("action") == result.chosen_action:
            p = s.get("recovery_probability")
            return float(p) if p is not None else None
    return None
