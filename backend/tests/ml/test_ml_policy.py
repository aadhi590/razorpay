from __future__ import annotations

import asyncio
import json
import os

import pytest

from app.services.ml_recovery_policy import MLRecoveryPolicy
from app.services.recovery_policy import PolicyContext, RulesBasedRecoveryPolicy

pytestmark = pytest.mark.needs_data


def _ctx(**kw) -> PolicyContext:
    base = dict(
        failure_reason="insufficient_funds",
        amount_paise=99900,
        priority=2,
        is_control=False,
        attempt_number=1,
        prior_action_types=[],
        customer_successful_payments=9,
        customer_failed_payments=2,
        recovery_event_id=None,
    )
    base.update(kw)
    return PolicyContext(**base)


def test_policy_ranks_candidates_on_real_event(db_session, open_treatment_event, trained_model):
    policy = MLRecoveryPolicy(db_session, model=trained_model)
    decision = policy.decide(_ctx(recovery_event_id=open_treatment_event["treatment_id"]))
    assert decision.should_intervene
    assert len(decision.candidates) == 4
    # ranked by expected value, descending
    scores = [c.score for c in decision.candidates]
    assert scores == sorted(scores, reverse=True)
    for c in decision.candidates:
        assert 0.0 <= c.estimated_recovery_probability <= 1.0
        assert "ml:" in c.reason
    assert policy.name == "ml:ml_v1"


def test_control_event_gets_no_intervention(db_session, open_treatment_event, trained_model):
    policy = MLRecoveryPolicy(db_session, model=trained_model)
    decision = policy.decide(
        _ctx(is_control=True, recovery_event_id=open_treatment_event["control_id"])
    )
    assert decision.should_intervene is False
    assert decision.candidates == []


def test_fallback_when_no_recovery_event_id(db_session, trained_model):
    policy = MLRecoveryPolicy(db_session, model=trained_model)
    decision = policy.decide(_ctx(recovery_event_id=None))
    assert "ML fallback" in decision.rationale
    assert decision.should_intervene  # rules policy still produced candidates


def test_fallback_when_model_unavailable(db_session):
    policy = MLRecoveryPolicy(db_session, model=None)
    # force "unavailable" regardless of a committed artifact
    policy._model = None
    policy.name = "rules_v1+ml_unavailable"
    decision = policy.decide(_ctx(recovery_event_id=1))
    assert "ML fallback" in decision.rationale
    rules = RulesBasedRecoveryPolicy().decide(_ctx(recovery_event_id=1))
    assert [c.action_type for c in decision.candidates] == [
        c.action_type for c in rules.candidates
    ]


def test_ml_policy_never_raises_on_bad_event(db_session, trained_model):
    policy = MLRecoveryPolicy(db_session, model=trained_model)
    decision = policy.decide(_ctx(recovery_event_id=10**9))
    assert "ML fallback" in decision.rationale  # scoring failed -> safe fallback


# --- endpoint integration ------------------------------------------

def _call(method: str, path: str):
    from app.main import app

    p, _, q = path.partition("?")
    scope = {
        "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
        "method": method, "scheme": "http", "path": p, "raw_path": p.encode(),
        "query_string": q.encode(), "root_path": "",
        "headers": [(b"host", b"t")], "client": ("t", 1), "server": ("t", 80),
    }
    inbox = [{"type": "http.request", "body": b"", "more_body": False}]
    out = {"body": b""}

    async def recv():
        return inbox.pop(0)

    async def send(m):
        if m["type"] == "http.response.start":
            out["status"] = m["status"]
        elif m["type"] == "http.response.body":
            out["body"] += m.get("body", b"")

    asyncio.run(app(scope, recv, send))
    return out["status"], json.loads(out["body"] or b"null")


def test_action_scores_endpoint(open_treatment_event, trained_model):
    tid = open_treatment_event["treatment_id"]
    sc, body = _call("GET", f"/api/v1/ml/recovery-events/{tid}/action-scores")
    assert sc == 200
    assert body["model_available"] is True
    assert body["recommended_action"] in {
        "retry", "sms_nudge", "whatsapp_nudge", "method_switch_prompt"
    }
    assert len(body["scores"]) == 4
    assert "SYNTHETIC" in body["note"].upper()


def test_orchestrate_with_ml_policy_creates_intervention(open_treatment_event, trained_model):
    tid = open_treatment_event["treatment_id"]
    sc, body = _call("POST", f"/api/v1/recovery-events/{tid}/orchestrate?policy=ml")
    assert sc == 200
    assert body["disposition"] == "intervention_created"
    assert body["selected_action"] in {
        "retry", "sms_nudge", "whatsapp_nudge", "method_switch_prompt"
    }
    assert body["confidence"] is not None


def test_orchestrate_bad_policy_returns_400(open_treatment_event):
    tid = open_treatment_event["treatment_id"]
    sc, body = _call("POST", f"/api/v1/recovery-events/{tid}/orchestrate?policy=bogus")
    assert sc == 400
