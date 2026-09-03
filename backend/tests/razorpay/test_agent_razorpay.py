"""Agent <-> Razorpay integration. Gemini is mocked; Razorpay uses the real
client + fake transport, so no network and no real link is ever created."""
from __future__ import annotations

from sqlalchemy import select

from app.agent.agent import RecoveryAgent
from app.agent.config import AgentConfig
from app.integrations.razorpay.client import RazorpayClient
from app.models.interventions import Intervention
from app.models.outcome import Outcome
from app.services.razorpay_webhook import RazorpayWebhookService
from tests.agent.conftest import _make_event
from tests.agent.fakes import ReactiveProvider, ScriptedProvider
from tests.razorpay.fakes import (
    FakeRazorpayTransport,
    make_config,
    payment_link_paid_event,
    signed_webhook,
)

_EXEC = ("execute_recovery_action", {
    "action_type": "whatsapp_nudge",
    "customer_message": "Hi, aapka payment complete nahi hua. Yeh secure link use karein.",
    "reason": "highest expected value eligible action",
})
_STOP = ("stop_recovery", {"stop_reason": "action_executed_awaiting_outcome",
                           "reasoning_summary": "link created, awaiting payment"})


def _cfg(**kw):
    base = dict(api_key="k", model="mock", max_turns=6, timeout_seconds=5.0,
               max_rate_limit_retries=0)
    base.update(kw)
    return AgentConfig(**base)


def _agent(db, provider, transport=None, client=None):
    if client is None and transport is not None:
        client = RazorpayClient(make_config(), transport=transport)
    return RecoveryAgent(db, provider, _cfg(), razorpay_client=client)


def _interventions(db, event_id):
    return db.scalars(
        select(Intervention).where(Intervention.recovery_event_id == event_id)
    ).all()


# --- Section 11: dry-run makes ZERO external calls --------------------
def test_dry_run_makes_no_razorpay_call(rzp_db, fresh_event_id):
    t = FakeRazorpayTransport()
    prov = ScriptedProvider([
        ("get_recovery_event_context", {}), _EXEC, _STOP,
    ])
    state = _agent(rzp_db, prov, transport=t).run(fresh_event_id, dry_run=True)

    assert t.calls == []                              # would fail on any real call
    assert _interventions(rzp_db, fresh_event_id) == []
    exec_trace = [x for x in state.trace if x.tool == "execute_recovery_action"][0]
    assert exec_trace.result_summary  # simulated result was returned


# --- live mode calls the adapter -----------------------------------
def test_live_mode_creates_test_mode_payment_link(rzp_db, fresh_event_id):
    t = FakeRazorpayTransport()
    prov = ScriptedProvider([("get_recovery_event_context", {}), _EXEC, _STOP])
    state = _agent(rzp_db, prov, transport=t).run(fresh_event_id, dry_run=False)

    assert len(t.create_calls) == 1
    ivs = _interventions(rzp_db, fresh_event_id)
    assert len(ivs) == 1
    assert ivs[0].razorpay_payment_link_id and ivs[0].razorpay_payment_link_id.startswith("plink_TEST")
    assert ivs[0].razorpay_reference_id == f"recovery-{fresh_event_id}-{ivs[0].id}"
    assert state.chosen_action == "whatsapp_nudge"


# --- Section 15: link creation != payment recovered ---------------
def test_link_creation_does_not_mark_recovered(rzp_db, fresh_event_id):
    t = FakeRazorpayTransport()
    prov = ScriptedProvider([("get_recovery_event_context", {}), _EXEC,
                             ("observe_recovery_outcome", {}), _STOP])
    state = _agent(rzp_db, prov, transport=t).run(fresh_event_id, dry_run=False)

    exec_res = [x for x in state.trace if x.tool == "execute_recovery_action"][0]
    obs_res = [x for x in state.trace if x.tool == "observe_recovery_outcome"][0]
    assert "payment_recovered=False" in exec_res.result_summary
    assert "payment_recovered=False" in obs_res.result_summary

    from app.agent.guardrails import load_event, payment_recovered
    ev = load_event(rzp_db, fresh_event_id)
    assert payment_recovered(ev) is False
    assert not rzp_db.scalars(
        select(Outcome).join(Intervention).where(
            Intervention.recovery_event_id == fresh_event_id
        )
    ).all()


# --- webhook then observe sees recovered --------------------------
def test_webhook_recovery_then_observe_sees_it(rzp_db, fresh_event_id):
    t = FakeRazorpayTransport()
    # 1. agent executes (creates link)
    _agent(rzp_db, ScriptedProvider([("get_recovery_event_context", {}), _EXEC, _STOP]),
           transport=t).run(fresh_event_id, dry_run=False)
    iv = _interventions(rzp_db, fresh_event_id)[0]

    # 2. a verified payment_link.paid webhook arrives
    raw, sig = signed_webhook(payment_link_paid_event(
        payment_link_id=iv.razorpay_payment_link_id,
        reference_id=iv.razorpay_reference_id, amount=99900,
    ))
    wr = RazorpayWebhookService(rzp_db, config=make_config()).process(
        raw_body=raw, signature=sig, event_id="evt_pytest_agent_paid"
    )
    assert wr.payment_recovered is True

    # 3. a fresh agent run observes the real recovered state
    prov = ScriptedProvider([("observe_recovery_outcome", {}),
                             ("stop_recovery", {"stop_reason": "payment_recovered",
                                                "reasoning_summary": "recovered"})])
    state = _agent(rzp_db, prov, transport=FakeRazorpayTransport()).run(
        fresh_event_id, dry_run=False
    )
    obs = [x for x in state.trace if x.tool == "observe_recovery_outcome"][0]
    assert "payment_recovered=True" in obs.result_summary
    assert state.stop_reason == "payment_recovered"


# --- Gemini stays the decision-maker -----------------------------
def test_agent_can_continue_or_escalate_when_not_recovered(rzp_db):
    """Reactive agent: link created, unpaid -> it does NOT conclude recovery."""
    event_id = _make_event(rzp_db)
    t = FakeRazorpayTransport()
    state = _agent(rzp_db, ReactiveProvider(), transport=t).run(event_id, dry_run=False)
    assert state.stop_reason in {"action_executed_awaiting_outcome",
                                 "max_turns_reached"}
    assert state.final_status in {"completed"}
    # it executed but never claimed recovery
    assert state.chosen_action is not None


# --- guardrails remain authoritative ----------------------------
def test_control_event_cannot_execute_live(rzp_db):
    event_id = _make_event(rzp_db, is_control=True)
    t = FakeRazorpayTransport()
    prov = ScriptedProvider([_EXEC, ("stop_recovery", {"stop_reason": "control_event",
                                                       "reasoning_summary": "control"})])
    state = _agent(rzp_db, prov, transport=t).run(event_id, dry_run=False)
    exec_trace = [x for x in state.trace if x.tool == "execute_recovery_action"][0]
    assert exec_trace.ok is False and exec_trace.guardrail_code == "control_event"
    assert t.create_calls == []


def test_already_recovered_event_cannot_execute(rzp_db):
    event_id = _make_event(rzp_db, prior_actions=["retry"], prior_recovered=True,
                           payment_recovered=True)
    t = FakeRazorpayTransport()
    prov = ScriptedProvider([_EXEC, ("stop_recovery", {"stop_reason": "customer_already_recovered",
                                                       "reasoning_summary": "done"})])
    state = _agent(rzp_db, prov, transport=t).run(event_id, dry_run=False)
    exec_trace = [x for x in state.trace if x.tool == "execute_recovery_action"][0]
    assert exec_trace.ok is False
    assert exec_trace.guardrail_code == "already_recovered"
    assert t.create_calls == []


def test_max_attempts_blocks_execute(rzp_db):
    event_id = _make_event(rzp_db, prior_actions=["retry", "sms_nudge", "whatsapp_nudge"])
    t = FakeRazorpayTransport()
    prov = ScriptedProvider([
        ("execute_recovery_action", {"action_type": "method_switch_prompt",
                                     "customer_message": "x", "reason": "y"}),
        ("stop_recovery", {"stop_reason": "max_attempts_reached", "reasoning_summary": "cap"}),
    ])
    state = _agent(rzp_db, prov, transport=t).run(event_id, dry_run=False)
    exec_trace = [x for x in state.trace if x.tool == "execute_recovery_action"][0]
    assert exec_trace.ok is False
    assert exec_trace.guardrail_code == "max_attempts_reached"
    assert t.create_calls == []


def test_config_missing_surfaces_error_to_agent(rzp_db, fresh_event_id, monkeypatch):
    # Force "no Razorpay credentials" regardless of what .env holds, so this
    # test is deterministic AND makes no real API call.
    from app.integrations.razorpay import config as rzp_config
    monkeypatch.setattr(rzp_config.settings, "RAZORPAY_KEY_ID", None, raising=False)
    monkeypatch.setattr(rzp_config.settings, "RAZORPAY_KEY_SECRET", None, raising=False)

    prov = ScriptedProvider([
        ("get_recovery_event_context", {}), _EXEC,
        ("escalate_recovery", {"escalation_type": "manual_review",
                               "reasoning_summary": "razorpay down"}),
    ])
    agent = RecoveryAgent(rzp_db, prov, _cfg(), razorpay_client=None)
    state = agent.run(fresh_event_id, dry_run=False)
    exec_trace = [x for x in state.trace if x.tool == "execute_recovery_action"][0]
    assert exec_trace.ok is False
    assert "RazorpayConfigError" in (exec_trace.result_summary or "")
    assert state.final_status == "escalated"
    assert _interventions(rzp_db, fresh_event_id) == []   # rolled back


# --- Section 18: non-deterministic loop preserved with Razorpay ----
def test_structurally_different_events_differ_with_razorpay(rzp_db):
    fresh = _make_event(rzp_db)
    twice = _make_event(rzp_db, prior_actions=["retry", "sms_nudge"])
    a = _agent(rzp_db, ReactiveProvider(), transport=FakeRazorpayTransport()).run(
        fresh, dry_run=False)
    b = _agent(rzp_db, ReactiveProvider(), transport=FakeRazorpayTransport()).run(
        twice, dry_run=False)
    seq_a = [x.tool for x in a.trace]
    seq_b = [x.tool for x in b.trace]
    assert seq_a != seq_b
    assert a.turns_used != b.turns_used
    assert seq_b[-1] == "escalate_recovery"
