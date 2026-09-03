"""get_historical_incrementality_for_action -- the read-only tool that lets the
agent check an action type's REAL observed incremental lift.

Gemini is never called (ScriptedProvider). The tool must:
  * be registered as a normal read tool (not mutating, not terminal),
  * return the observed figures for a well-sampled action,
  * reject an unknown action_type with a structured error,
  * echo the model's predicted uplift ONLY when get_action_scores already ran,
  * accumulate onto RecoveryAgentState and the persisted trace,
  * never influence a guardrail,
  * be available as ONE option among several on a turn (not forced).
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

_TOOL = "get_historical_incrementality_for_action"


def _cfg(**kw) -> AgentConfig:
    base = dict(
        api_key="TEST_KEY_do_not_log", model="fake-model",
        max_turns=8, timeout_seconds=5.0,
        max_invalid_requests=3, max_transient_retries=2, max_rate_limit_retries=0,
    )
    base.update(kw)
    return AgentConfig(**base)


# --- registration -------------------------------------------------------
def test_tool_is_registered_read_only():
    assert _TOOL in TOOLS
    t = TOOLS[_TOOL]
    assert t.mutating is False
    assert t.terminal is False
    enum = t.parameters["properties"]["action_type"]["enum"]
    assert set(enum) == {"retry", "sms_nudge", "whatsapp_nudge", "method_switch_prompt"}


# --- direct run: observed figures for a real action -------------------
def test_returns_observed_incrementality(db_session_agent, fresh_event):
    ctx = ToolContext(
        db=db_session_agent,
        state=RecoveryAgentState(recovery_event_id=fresh_event, dry_run=True),
        dry_run=True,
    )
    r = TOOLS[_TOOL].run(ctx, {"action_type": "whatsapp_nudge"})

    assert r["action_type"] == "whatsapp_nudge"
    assert r["computable"] is True
    assert r["treated_group_size"] >= 30
    assert 0.0 <= r["observed_recovery_rate_for_action"] <= 1.0
    assert 0.0 <= r["baseline_control_recovery_rate"] <= 1.0
    assert r["observed_incremental_lift"] == round(
        r["observed_recovery_rate_for_action"] - r["baseline_control_recovery_rate"], 6
    )
    lo, hi = r["observed_incremental_lift_ci_95"]
    assert lo <= r["observed_incremental_lift"] <= hi
    assert "Newcombe/Wilson" in r["note"]
    # predicted uplift NOT present -- get_action_scores was not called
    assert "model_predicted_uplift_for_context" not in r
    # accumulated on state
    assert ctx.state.action_incrementality["whatsapp_nudge"] == r


def test_unknown_action_type_is_a_structured_error(db_session_agent, fresh_event):
    ctx = ToolContext(
        db=db_session_agent,
        state=RecoveryAgentState(recovery_event_id=fresh_event, dry_run=True),
        dry_run=True,
    )
    r = TOOLS[_TOOL].run(ctx, {"action_type": "carrier_pigeon"})
    assert r["error"] == "unknown_action_type"
    assert "carrier_pigeon" in r["message"]
    assert set(r["supported"]) == {"retry", "sms_nudge", "whatsapp_nudge", "method_switch_prompt"}
    assert ctx.state.action_incrementality is None


def test_predicted_uplift_echoed_only_when_scores_present(db_session_agent, fresh_event):
    state = RecoveryAgentState(recovery_event_id=fresh_event, dry_run=True)
    state.quantitative_scores = [
        {"action": "sms_nudge", "uplift": 0.0731, "cost_paise": 20},
        {"action": "retry", "uplift": 0.0402, "cost_paise": 50},
    ]
    ctx = ToolContext(db=db_session_agent, state=state, dry_run=True)

    with_pred = TOOLS[_TOOL].run(ctx, {"action_type": "sms_nudge"})
    assert with_pred["model_predicted_uplift_for_context"] == 0.0731

    # an action the scores don't cover -> no echo, no crash
    without_pred = TOOLS[_TOOL].run(ctx, {"action_type": "whatsapp_nudge"})
    assert "model_predicted_uplift_for_context" not in without_pred

    assert set(ctx.state.action_incrementality) == {"sms_nudge", "whatsapp_nudge"}


# --- loop: invalid enum rejected BEFORE the tool runs ----------------
def test_loop_rejects_invalid_action_type_argument(db_session_agent, fresh_event):
    prov = ScriptedProvider([
        (_TOOL, {"action_type": "not_a_real_action"}),
        ("stop_recovery", {"stop_reason": "other", "reasoning_summary": "done"}),
    ])
    state = RecoveryAgent(db_session_agent, prov, _cfg()).run(fresh_event, dry_run=True)
    entry = [t for t in state.trace if t.tool == _TOOL][0]
    assert entry.ok is False
    assert entry.guardrail_code == "argument_validation_failed"
    assert state.final_status == "completed"          # ran on, no crash


# --- loop: it is ONE available option, not forced -------------------
def test_agent_may_use_it_among_other_tools(db_session_agent, fresh_event):
    """A run where the agent chooses to consult it after scores -- and one where
    it never does. Both are valid; the tool is available, not mandatory."""
    used = ScriptedProvider([
        ("get_recovery_event_context", {}),
        ("get_action_scores", {}),
        (_TOOL, {"action_type": "method_switch_prompt"}),
        ("stop_recovery", {
            "stop_reason": "expected_value_below_threshold",
            "reasoning_summary": (
                "Predicted uplift and observed historical incrementality both "
                "point the same way; not worth an attempt on this amount."
            ),
        }),
    ])
    r_used = run_recovery_agent(
        db_session_agent, fresh_event, dry_run=True, provider=used, config=_cfg(),
        persist=True,
    )
    assert _TOOL in [t.tool for t in r_used.tool_trace]
    assert r_used.action_incrementality is not None
    assert "method_switch_prompt" in r_used.action_incrementality
    inc = r_used.action_incrementality["method_switch_prompt"]
    assert inc["computable"] is True
    # predicted uplift IS echoed here (scores were fetched first)
    assert "model_predicted_uplift_for_context" in inc

    # persisted trace carries it
    ev = db_session_agent.scalars(
        select(AgentEvent).where(
            AgentEvent.recovery_event_id == fresh_event,
            AgentEvent.event_type == "agent_recovery_run",
        )
    ).first()
    assert ev is not None
    assert "method_switch_prompt" in ev.input_context["action_incrementality"]

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
    assert r_skip.action_incrementality is None


# --- it cannot bypass a guardrail ----------------------------------
def test_tool_cannot_execute_or_bypass_a_guardrail(db_session_agent, all_actions_tried_event):
    """Consulting incrementality for an already-exhausted action changes nothing
    about what the agent is allowed to execute."""
    prov = ScriptedProvider([
        (_TOOL, {"action_type": "method_switch_prompt"}),
        ("execute_recovery_action", {
            "action_type": "method_switch_prompt",
            "customer_message": "hi", "reason": "history looked good",
        }),
        ("stop_recovery", {"stop_reason": "max_attempts_reached", "reasoning_summary": "cap"}),
    ])
    state = RecoveryAgent(db_session_agent, prov, _cfg()).run(
        all_actions_tried_event, dry_run=False
    )
    inc_entry = [t for t in state.trace if t.tool == _TOOL][0]
    exec_entry = [t for t in state.trace if t.tool == "execute_recovery_action"][0]
    assert inc_entry.ok is True                       # the lookup itself is fine
    assert exec_entry.ok is False                     # guardrail still blocks execution
    assert exec_entry.guardrail_code == "max_attempts_reached"
    assert state.chosen_action is None
    # the 3 fixture interventions are unchanged -- no new one was created
    assert len(
        db_session_agent.scalars(
            select(Intervention).where(
                Intervention.recovery_event_id == all_actions_tried_event
            )
        ).all()
    ) == 3
