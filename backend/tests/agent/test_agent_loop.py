"""Agent-loop behaviour. Gemini is always mocked (see fakes.py)."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest
from sqlalchemy import select

from app.agent.agent import RecoveryAgent
from app.agent.config import AgentConfig
from app.agent.providers.base import (
    AuthError,
    MalformedResponseError,
    RateLimitedError,
    TransientError,
)
from app.agent.runner import run_recovery_agent
from app.models.interventions import Intervention

from tests.agent.fakes import RaisingProvider, ReactiveProvider, ScriptedProvider


def cfg(**kw) -> AgentConfig:
    base = dict(
        api_key="TEST_KEY_do_not_log", model="fake-model",
        max_turns=6, timeout_seconds=5.0,
        max_invalid_requests=3, max_transient_retries=2,
        max_rate_limit_retries=0,
    )
    base.update(kw)
    return AgentConfig(**base)


def agent(db, provider, **kw) -> RecoveryAgent:
    return RecoveryAgent(db, provider, cfg(**kw))


# 1 ---------------------------------------------------------------
def test_agent_starts_and_completes(db_session_agent, fresh_event):
    prov = ScriptedProvider([
        ("stop_recovery", {"stop_reason": "other", "reasoning_summary": "nothing to do"}),
    ])
    state = agent(db_session_agent, prov).run(fresh_event, dry_run=True)
    assert state.final_status == "completed"
    assert state.turns_used == 1
    assert state.trace[0].tool == "stop_recovery"


# 2 + 3 + 4 -----------------------------------------------------
def test_read_tool_result_flows_back_and_drives_next_turn(db_session_agent, fresh_event):
    prov = ReactiveProvider()
    state = agent(db_session_agent, prov).run(fresh_event, dry_run=True)

    assert state.trace[0].tool == "get_recovery_event_context"
    assert state.trace[0].ok is True
    assert state.event_context is not None            # tool populated state
    # the 2nd model turn saw the 1st tool's result in its conversation
    assert prov.saw_tool_result_before_second_call is True
    # and it chose a *different* next tool, conditioned on that result
    assert state.trace[1].tool != state.trace[0].tool


# 5 -----------------------------------------------------------
def test_agent_chooses_an_eligible_action(db_session_agent, fresh_event):
    prov = ReactiveProvider()
    state = agent(db_session_agent, prov).run(fresh_event, dry_run=True)
    executed = [t for t in state.trace if t.tool == "execute_recovery_action"]
    assert executed, "reactive agent should execute an action on a fresh event"
    assert executed[0].ok is True
    assert state.chosen_action in {
        "retry", "sms_nudge", "whatsapp_nudge", "method_switch_prompt"
    }


# 6 ----------------------------------------------------------
def test_invalid_action_is_rejected(db_session_agent, fresh_event):
    prov = ScriptedProvider([
        ("get_recovery_event_context", {}),
        ("execute_recovery_action", {"action_type": "not_a_real_action",
                                     "customer_message": "hi", "reason": "x"}),
        ("stop_recovery", {"stop_reason": "other", "reasoning_summary": "done"}),
    ])
    state = agent(db_session_agent, prov).run(fresh_event, dry_run=True)
    bad = [t for t in state.trace if t.tool == "execute_recovery_action"][0]
    assert bad.ok is False
    assert bad.guardrail_code == "argument_validation_failed"
    assert state.chosen_action is None


# 7 ----------------------------------------------------------
def test_control_event_cannot_execute(db_session_agent, control_event):
    prov = ScriptedProvider([
        ("execute_recovery_action", {"action_type": "sms_nudge",
                                     "customer_message": "hi", "reason": "x"}),
        ("stop_recovery", {"stop_reason": "control_event",
                           "reasoning_summary": "control"}),
    ])
    state = agent(db_session_agent, prov).run(control_event, dry_run=False)
    exec_trace = [t for t in state.trace if t.tool == "execute_recovery_action"][0]
    assert exec_trace.ok is False
    assert exec_trace.guardrail_code == "control_event"
    assert _intervention_count(db_session_agent, control_event) == 0


# 8 ----------------------------------------------------------
def test_guardrail_blocks_already_attempted_action(db_session_agent, twice_attempted_event):
    prov = ScriptedProvider([
        ("execute_recovery_action", {"action_type": "retry",     # already tried
                                     "customer_message": "hi", "reason": "x"}),
        ("stop_recovery", {"stop_reason": "guardrail_violation",
                           "reasoning_summary": "blocked"}),
    ])
    state = agent(db_session_agent, prov).run(twice_attempted_event, dry_run=False)
    t = [t for t in state.trace if t.tool == "execute_recovery_action"][0]
    assert t.ok is False and t.guardrail_code == "action_already_attempted"
    assert _intervention_count(db_session_agent, twice_attempted_event) == 2  # unchanged


# 9 ----------------------------------------------------------
def test_max_turns_stops_agent(db_session_agent, fresh_event):
    prov = ReactiveProvider()
    state = agent(db_session_agent, prov, max_turns=2).run(fresh_event, dry_run=True)
    assert state.turns_used == 2
    assert state.stop_reason == "max_turns_reached"
    assert state.final_status == "completed"


# 10 ---------------------------------------------------------
def test_payment_recovery_causes_stop(db_session_agent, recovered_event):
    prov = ReactiveProvider()
    state = agent(db_session_agent, prov).run(recovered_event, dry_run=True)
    assert state.stop_reason == "payment_recovered"
    assert state.trace[-1].tool == "stop_recovery"


# 11 ---------------------------------------------------------
def test_no_eligible_action_causes_stop(db_session_agent, all_actions_tried_event):
    prov = ReactiveProvider()
    state = agent(db_session_agent, prov).run(all_actions_tried_event, dry_run=True)
    assert state.stop_reason in {"no_eligible_actions", "escalation_required"}
    assert state.final_status in {"completed", "escalated"}


# 12 ---------------------------------------------------------
def test_escalation_works(db_session_agent, twice_attempted_event):
    # attempt_number == 3 == max_attempts -> reactive escalates
    prov = ReactiveProvider()
    state = agent(db_session_agent, prov).run(twice_attempted_event, dry_run=True)
    assert state.final_status == "escalated"
    assert state.escalation_required is True
    assert state.escalation_type == "manual_review"


# 13 ---------------------------------------------------------
def test_gemini_timeout_handled(db_session_agent, fresh_event):
    prov = RaisingProvider(TransientError("timed out"))
    state = agent(db_session_agent, prov).run(fresh_event, dry_run=True)
    assert state.final_status == "failed_safe"
    assert state.stop_reason == "quota_or_api_failure"
    assert prov.calls == cfg().max_transient_retries + 1  # bounded, then give up


# 14 ---------------------------------------------------------
def test_gemini_429_handled_without_retry_storm(db_session_agent, fresh_event):
    prov = RaisingProvider(RateLimitedError("429 quota"))
    state = agent(db_session_agent, prov).run(fresh_event, dry_run=True)
    assert state.final_status == "failed_safe"
    assert state.stop_reason == "quota_or_api_failure"
    assert prov.calls == 1  # the agent does NOT re-drive a 429


# 15 ---------------------------------------------------------
def test_gemini_malformed_response_handled(db_session_agent, fresh_event):
    prov = RaisingProvider(MalformedResponseError("no candidates"))
    state = agent(db_session_agent, prov).run(fresh_event, dry_run=True)
    assert state.final_status == "failed_safe"
    assert "MalformedResponseError" in " ".join(state.errors)


# 16 ---------------------------------------------------------
def test_api_key_never_appears_in_trace_or_errors(db_session_agent, fresh_event):
    secret = "SUPER_SECRET_KEY_abc123"
    prov = ReactiveProvider()
    result = run_recovery_agent(
        db_session_agent, fresh_event, dry_run=True, provider=prov,
        config=cfg(api_key=secret), persist=True,
    )
    assert secret not in result.model_dump_json()

    from app.models.agent_events import AgentEvent
    ev = db_session_agent.scalars(
        select(AgentEvent).where(AgentEvent.recovery_event_id == fresh_event)
    ).first()
    assert ev is not None                       # trace was persisted
    assert secret not in str(ev.input_context)  # and it is clean


def test_auth_error_message_has_no_key():
    from app.agent.providers.gemini import GeminiProvider
    import urllib.error

    prov = GeminiProvider(cfg(api_key="KEYKEYKEY"))
    err = urllib.error.HTTPError(
        url="x", code=401, msg="Unauthorized", hdrs=None,
        fp=__import__("io").BytesIO(b'{"error":{"message":"API key not valid"}}'),
    )
    with pytest.raises(AuthError) as ei:
        GeminiProvider._raise_for_http_error(err)
    assert "KEYKEYKEY" not in str(ei.value)


# 17 ---------------------------------------------------------
def test_dry_run_never_executes_action(db_session_agent, fresh_event):
    prov = ReactiveProvider()
    state = agent(db_session_agent, prov).run(fresh_event, dry_run=True)
    assert any(t.tool == "execute_recovery_action" and t.ok for t in state.trace)
    assert _intervention_count(db_session_agent, fresh_event) == 0
    assert all(r.get("simulated") for r in state.executed_actions)


def test_live_run_persists_intervention_and_payment_link(db_session_agent, fresh_event):
    """Live execution now creates an Intervention + a Razorpay Test Mode Payment
    Link (via the real client + fake transport -- no network)."""
    from app.agent.agent import RecoveryAgent
    from app.integrations.razorpay.client import RazorpayClient
    from tests.razorpay.fakes import FakeRazorpayTransport, make_config

    transport = FakeRazorpayTransport()
    client = RazorpayClient(make_config(), transport=transport)
    prov = ScriptedProvider([
        ("get_recovery_event_context", {}),
        ("execute_recovery_action", {"action_type": "sms_nudge",
                                     "customer_message": "Hi, dobara try karein.",
                                     "reason": "cheap eligible action"}),
        ("stop_recovery", {"stop_reason": "action_executed_awaiting_outcome",
                           "reasoning_summary": "done"}),
    ])
    state = RecoveryAgent(
        db_session_agent, prov, cfg(), razorpay_client=client
    ).run(fresh_event, dry_run=False)

    assert state.chosen_action == "sms_nudge"
    assert len(transport.create_calls) == 1
    ivs = db_session_agent.scalars(
        select(Intervention).where(Intervention.recovery_event_id == fresh_event)
    ).all()
    assert len(ivs) == 1
    assert ivs[0].razorpay_payment_link_id is not None
    assert ivs[0].outcome is None                       # link != recovery
    for iv in ivs:
        db_session_agent.delete(iv)
    db_session_agent.commit()


# 19 ---------------------------------------------------------
def test_structurally_different_events_produce_different_sequences(
    db_session_agent, fresh_event, twice_attempted_event
):
    a = agent(db_session_agent, ReactiveProvider()).run(fresh_event, dry_run=True)
    b = agent(db_session_agent, ReactiveProvider()).run(twice_attempted_event, dry_run=True)

    seq_a = [t.tool for t in a.trace]
    seq_b = [t.tool for t in b.trace]

    assert seq_a != seq_b, (seq_a, seq_b)
    assert a.turns_used != b.turns_used, (a.turns_used, b.turns_used)
    # fresh event reasons about scores + executes; twice-attempted escalates early
    assert "get_action_scores" in seq_a
    assert seq_b[-1] == "escalate_recovery"


def test_loop_executes_provider_choices_verbatim(db_session_agent, fresh_event):
    """An unusual scripted order is followed exactly -- the loop imposes none
    of its own sequencing."""
    script = [
        ("get_subscription_context", {}),
        ("get_customer_recovery_history", {}),
        ("get_payment_context", {}),
        ("stop_recovery", {"stop_reason": "expected_value_below_threshold",
                           "reasoning_summary": "not worth it"}),
    ]
    prov = ScriptedProvider(list(script))
    state = agent(db_session_agent, prov).run(fresh_event, dry_run=True)
    assert [t.tool for t in state.trace] == [s[0] for s in script]


def test_repeated_invalid_tool_calls_terminate_safely(db_session_agent, fresh_event):
    prov = ScriptedProvider([("no_such_tool", {})] * 5)
    state = agent(db_session_agent, prov).run(fresh_event, dry_run=True)
    assert state.final_status == "failed_safe"
    assert state.invalid_requests >= 3


def test_token_usage_is_accumulated(db_session_agent, fresh_event):
    prov = ReactiveProvider()
    state = agent(db_session_agent, prov).run(fresh_event, dry_run=True)
    assert state.total_tokens == state.turns_used * 18


# --- helpers ------------------------------------------------
def _intervention_count(db, recovery_event_id: int) -> int:
    return len(db.scalars(
        select(Intervention).where(Intervention.recovery_event_id == recovery_event_id)
    ).all())
