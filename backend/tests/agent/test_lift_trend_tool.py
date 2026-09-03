"""get_action_lift_trend -- the read-only tool that lets the agent check whether
an action type's REAL observed effectiveness is trending over time.

Gemini is never called (ScriptedProvider). Mirrors test_insight_tool.py: the
tool must be a normal read tool, return the windowed trend for a real action,
reject an unknown action_type, accumulate onto RecoveryAgentState and the
persisted trace, never influence a guardrail, and be ONE available option on a
turn (not forced).
"""
from __future__ import annotations

from sqlalchemy import select

from app.agent.agent import RecoveryAgent
from app.agent.config import AgentConfig
from app.agent.runner import run_recovery_agent
from app.agent.state import RecoveryAgentState
from app.agent.tools import TOOLS
from app.agent.tools.base import ToolContext
from app.models.agent_events import AgentEvent
from app.models.interventions import Intervention

from tests.agent.fakes import ScriptedProvider

_TOOL = "get_action_lift_trend"
_ACTIONS = {"retry", "sms_nudge", "whatsapp_nudge", "method_switch_prompt"}


def _cfg(**kw) -> AgentConfig:
    base = dict(
        api_key="TEST_KEY_do_not_log", model="fake-model",
        max_turns=8, timeout_seconds=5.0,
        max_invalid_requests=3, max_transient_retries=2, max_rate_limit_retries=0,
    )
    base.update(kw)
    return AgentConfig(**base)


# --- registration -----------------------------------------------------
def test_tool_is_registered_read_only():
    assert _TOOL in TOOLS
    t = TOOLS[_TOOL]
    assert t.mutating is False
    assert t.terminal is False
    assert set(t.parameters["properties"]["action_type"]["enum"]) == _ACTIONS


# --- direct run: windowed trend for a real action -------------------
def test_returns_windowed_trend(db_session_agent, fresh_event):
    ctx = ToolContext(
        db=db_session_agent,
        state=RecoveryAgentState(recovery_event_id=fresh_event, dry_run=True),
        dry_run=True,
    )
    r = TOOLS[_TOOL].run(ctx, {"action_type": "whatsapp_nudge"})

    assert r["action_type"] == "whatsapp_nudge"
    assert r["computable"] is True
    assert r["trend_direction"] in {
        "improving", "declining", "stable_or_insufficient_data"
    }
    assert r["recent_window_size"] > 0
    assert r["baseline_window_size"] > 0
    assert isinstance(r["trend_confidence_interval"], list)
    lo, hi = r["trend_confidence_interval"]
    assert lo <= hi
    # the interval-vs-zero relationship matches the reported direction
    if r["trend_direction"] == "stable_or_insufficient_data":
        assert lo <= 0.0 <= hi
    elif r["trend_direction"] == "declining":
        assert hi < 0.0
    else:
        assert lo > 0.0
    assert "Newcombe/Wilson" in r["note"]
    # accumulated on state, keyed by action_type
    assert ctx.state.action_lift_trend["whatsapp_nudge"] == r


def test_unknown_action_type_is_a_structured_error(db_session_agent, fresh_event):
    ctx = ToolContext(
        db=db_session_agent,
        state=RecoveryAgentState(recovery_event_id=fresh_event, dry_run=True),
        dry_run=True,
    )
    r = TOOLS[_TOOL].run(ctx, {"action_type": "carrier_pigeon"})
    assert r["error"] == "unknown_action_type"
    assert "carrier_pigeon" in r["message"]
    assert set(r["supported"]) == _ACTIONS
    assert ctx.state.action_lift_trend is None


# --- loop: invalid enum rejected BEFORE the tool runs --------------
def test_loop_rejects_invalid_action_type_argument(db_session_agent, fresh_event):
    prov = ScriptedProvider([
        (_TOOL, {"action_type": "not_a_real_action"}),
        ("stop_recovery", {"stop_reason": "other", "reasoning_summary": "done"}),
    ])
    state = RecoveryAgent(db_session_agent, prov, _cfg()).run(fresh_event, dry_run=True)
    entry = [t for t in state.trace if t.tool == _TOOL][0]
    assert entry.ok is False
    assert entry.guardrail_code == "argument_validation_failed"
    assert state.final_status == "completed"


# --- loop: it is ONE available option, not forced -----------------
def test_agent_may_use_it_among_other_tools(db_session_agent, fresh_event):
    used = ScriptedProvider([
        ("get_recovery_event_context", {}),
        ("get_action_scores", {}),
        (_TOOL, {"action_type": "method_switch_prompt"}),
        ("stop_recovery", {
            "stop_reason": "expected_value_below_threshold",
            "reasoning_summary": (
                "Recent-vs-baseline trend shows no erosion; predicted and "
                "observed still agree; not worth an attempt on this amount."
            ),
        }),
    ])
    r_used = run_recovery_agent(
        db_session_agent, fresh_event, dry_run=True, provider=used, config=_cfg(),
        persist=True,
    )
    assert _TOOL in [t.tool for t in r_used.tool_trace]
    assert r_used.action_lift_trend is not None
    assert "method_switch_prompt" in r_used.action_lift_trend
    assert r_used.action_lift_trend["method_switch_prompt"]["computable"] is True

    ev = db_session_agent.scalars(
        select(AgentEvent).where(
            AgentEvent.recovery_event_id == fresh_event,
            AgentEvent.event_type == "agent_recovery_run",
        )
    ).first()
    assert ev is not None
    assert "method_switch_prompt" in ev.input_context["action_lift_trend"]

    # a run that never calls it is equally fine
    skipped = ScriptedProvider([
        ("get_recovery_event_context", {}),
        ("stop_recovery", {"stop_reason": "other", "reasoning_summary": "clear"}),
    ])
    r_skip = run_recovery_agent(
        db_session_agent, fresh_event, dry_run=True, provider=skipped, config=_cfg(),
        persist=False,
    )
    assert _TOOL not in [t.tool for t in r_skip.tool_trace]
    assert r_skip.action_lift_trend is None


# --- it cannot bypass a guardrail --------------------------------
def test_tool_cannot_execute_or_bypass_a_guardrail(
    db_session_agent, all_actions_tried_event
):
    prov = ScriptedProvider([
        (_TOOL, {"action_type": "method_switch_prompt"}),
        ("execute_recovery_action", {
            "action_type": "method_switch_prompt",
            "customer_message": "hi", "reason": "trend still fine",
        }),
        ("stop_recovery", {"stop_reason": "max_attempts_reached", "reasoning_summary": "cap"}),
    ])
    state = RecoveryAgent(db_session_agent, prov, _cfg()).run(
        all_actions_tried_event, dry_run=False
    )
    trend_entry = [t for t in state.trace if t.tool == _TOOL][0]
    exec_entry = [t for t in state.trace if t.tool == "execute_recovery_action"][0]
    assert trend_entry.ok is True
    assert exec_entry.ok is False
    assert exec_entry.guardrail_code == "max_attempts_reached"
    assert state.chosen_action is None
    assert len(
        db_session_agent.scalars(
            select(Intervention).where(
                Intervention.recovery_event_id == all_actions_tried_event
            )
        ).all()
    ) == 3
