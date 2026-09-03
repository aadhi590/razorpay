"""Idempotent Payment Link creation (Section 7) -- proven by calling the
creation path twice for one intervention and asserting a single link + a single
Razorpay POST."""
from __future__ import annotations

from datetime import datetime, timezone

from app.integrations.razorpay.client import RazorpayClient
from app.models.interventions import Intervention
from app.services.recovery_execution import (
    RecoveryExecutionService,
    reference_id_for,
)
from tests.agent.conftest import _make_event
from tests.razorpay.fakes import FakeRazorpayTransport, make_config


def _service(db, transport):
    client = RazorpayClient(make_config(), transport=transport)
    return RecoveryExecutionService(db, client=client, config=make_config())


def _intervention(db, event_id: int) -> Intervention:
    iv = Intervention(
        recovery_event_id=event_id, action_type="whatsapp_nudge",
        status="pending", cost_paise=80, agent_reason="test",
    )
    db.add(iv)
    db.flush()
    return iv


def test_second_call_reuses_the_existing_link(rzp_db):
    event_id = _make_event(rzp_db)
    iv = _intervention(rzp_db, event_id)
    t = FakeRazorpayTransport()
    svc = _service(rzp_db, t)

    r1 = svc.create_payment_link(iv)
    r2 = svc.create_payment_link(iv)

    assert r1.success and r2.success
    assert r1.payment_link_id == r2.payment_link_id
    assert r1.reused is False
    assert r2.reused is True
    assert len(t.create_calls) == 1                 # only ONE POST /payment_links
    assert iv.razorpay_payment_link_id == r1.payment_link_id
    assert iv.razorpay_reference_id == reference_id_for(event_id, iv.id)


def test_reference_id_is_deterministic_per_intervention(rzp_db):
    event_id = _make_event(rzp_db)
    iv = _intervention(rzp_db, event_id)
    t = FakeRazorpayTransport()
    _service(rzp_db, t).create_payment_link(iv)
    body = t.create_calls[0]["body"]
    assert body["reference_id"] == f"recovery-{event_id}-{iv.id}"
    assert body["amount"] == 99900                  # authoritative payment amount
    assert body["notes"]["recovery_event_id"] == str(event_id)
    assert body["notes"]["intervention_id"] == str(iv.id)


def test_link_creation_does_not_create_outcome_or_mark_recovered(rzp_db):
    event_id = _make_event(rzp_db)
    iv = _intervention(rzp_db, event_id)
    _service(rzp_db, FakeRazorpayTransport()).create_payment_link(iv)
    rzp_db.flush()

    from app.agent.guardrails import payment_recovered, load_event
    ev = load_event(rzp_db, event_id)
    assert iv.outcome is None
    assert payment_recovered(ev) is False
    assert ev.payment.status == "failed"
    assert ev.payment.recovered_at is None


def test_config_not_ready_returns_structured_error_no_call(rzp_db):
    event_id = _make_event(rzp_db)
    iv = _intervention(rzp_db, event_id)
    # no client injected + config missing credentials
    bad_cfg = make_config(key_id=None, key_secret=None)
    svc = RecoveryExecutionService(rzp_db, client=None, config=bad_cfg)
    result = svc.create_payment_link(iv)
    assert result.success is False
    assert result.error_code == "RazorpayConfigError"
    assert iv.razorpay_payment_link_id is None
