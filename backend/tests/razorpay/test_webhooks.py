"""Webhook processing: real HMAC verification, correlation, Outcome, idempotency.

The delivery is locally originated (no tunnel in this env) but the signature is
a real HMAC-SHA256 over the raw body with the real webhook secret, and the same
bytes are POSTed twice to prove idempotency -- not reasoned about.
"""
from __future__ import annotations

import json

from sqlalchemy import select

from app.models.outcome import Outcome
from app.models.webhook_event import ProcessedWebhookEvent
from app.services.razorpay_webhook import RazorpayWebhookService
from tests.razorpay.fakes import (
    TEST_WEBHOOK_SECRET,
    make_config,
    payment_link_paid_event,
    signed_webhook,
)


def _svc(db):
    return RazorpayWebhookService(db, config=make_config())


def _process(db, body: dict, *, event_id: str, signature: str | None = None):
    raw, sig = signed_webhook(body)
    return _svc(db).process(
        raw_body=raw,
        signature=signature if signature is not None else sig,
        event_id=event_id,
    )


def _paid_body(linked, amount=99900, **kw):
    return payment_link_paid_event(
        payment_link_id=linked["payment_link_id"],
        reference_id=linked["reference_id"],
        amount=amount,
        **kw,
    )


# --- signature -----------------------------------------------------
def test_valid_signature_accepted(rzp_db, linked_intervention):
    r = _process(rzp_db, _paid_body(linked_intervention),
                 event_id="evt_pytest_ok1")
    assert r.http_status == 200 and r.status == "ok"


def _outcomes_for(db, intervention_id):
    return db.scalars(
        select(Outcome).where(Outcome.intervention_id == intervention_id)
    ).all()


def test_invalid_signature_rejected(rzp_db, linked_intervention):
    r = _process(rzp_db, _paid_body(linked_intervention),
                 event_id="evt_pytest_badsig", signature="deadbeef")
    assert r.http_status == 400 and r.status == "invalid_signature"
    assert not _outcomes_for(rzp_db, linked_intervention["intervention_id"])


def test_missing_signature_rejected(rzp_db, linked_intervention):
    raw, _ = signed_webhook(_paid_body(linked_intervention))
    r = _svc(rzp_db).process(raw_body=raw, signature=None, event_id="evt_pytest_nosig")
    assert r.http_status == 400 and r.status == "invalid_signature"


def test_malformed_payload_rejected_safely(rzp_db):
    raw = b"{not json"
    from app.integrations.razorpay.webhooks import compute_signature
    sig = compute_signature(raw, TEST_WEBHOOK_SECRET)
    r = _svc(rzp_db).process(raw_body=raw, signature=sig, event_id="evt_pytest_mal")
    assert r.http_status == 400 and r.status == "bad_payload"


# --- events -------------------------------------------------------
def test_payment_link_paid_creates_single_outcome_and_closes_event(
    rzp_db, linked_intervention
):
    r = _process(rzp_db, _paid_body(linked_intervention, amount=99900),
                 event_id="evt_pytest_paid1")
    assert r.status == "ok" and r.payment_recovered is True
    assert r.recovered_amount_paise == 99900

    from app.models.recovery_events import RecoveryEvent
    from app.models.interventions import Intervention

    ev = rzp_db.get(RecoveryEvent, linked_intervention["recovery_event_id"])
    iv = rzp_db.get(Intervention, linked_intervention["intervention_id"])
    outcomes = rzp_db.scalars(
        select(Outcome).where(Outcome.intervention_id == iv.id)
    ).all()
    assert len(outcomes) == 1
    assert outcomes[0].payment_recovered is True
    assert outcomes[0].recovered_amount_paise == 99900
    assert ev.status == "closed" and ev.payment.status == "success"
    assert iv.last_razorpay_status == "paid"
    assert iv.razorpay_payment_id == "pay_TESTpaid0001"


def test_partially_paid_records_status_but_no_outcome(rzp_db, linked_intervention):
    body = _paid_body(linked_intervention, event="payment_link.partially_paid",
                      status="partially_paid")
    r = _process(rzp_db, body, event_id="evt_pytest_partial")
    assert r.status == "ok" and r.payment_recovered is False
    assert not _outcomes_for(rzp_db, linked_intervention["intervention_id"])
    from app.models.interventions import Intervention
    iv = rzp_db.get(Intervention, linked_intervention["intervention_id"])
    assert iv.last_razorpay_status == "partially_paid"


def test_cancelled_and_expired_no_outcome(rzp_db, linked_intervention):
    for ev_name, eid in (("payment_link.cancelled", "evt_pytest_cancel"),
                         ("payment_link.expired", "evt_pytest_expire")):
        body = _paid_body(linked_intervention, event=ev_name, status="cancelled")
        r = _process(rzp_db, body, event_id=eid)
        assert r.status == "ok" and r.payment_recovered is False
    assert not _outcomes_for(rzp_db, linked_intervention["intervention_id"])


def test_unknown_event_ignored(rzp_db, linked_intervention):
    body = _paid_body(linked_intervention, event="payment.dispute.created")
    r = _process(rzp_db, body, event_id="evt_pytest_unknown")
    assert r.http_status == 200 and r.status == "ignored"


def test_uncorrelatable_event(rzp_db):
    body = payment_link_paid_event(
        payment_link_id="plink_nope", reference_id="recovery-999999-999999",
        amount=1000,
    )
    r = _process(rzp_db, body, event_id="evt_pytest_nocorr")
    assert r.status == "correlation_failed"
    # not recorded -> a later (correlatable) redelivery can still be processed
    assert rzp_db.get(ProcessedWebhookEvent, "evt_pytest_nocorr") is None


# --- idempotency: the same delivery twice ------------------------
def test_duplicate_delivery_by_event_id_is_idempotent(rzp_db, linked_intervention):
    raw, sig = signed_webhook(_paid_body(linked_intervention, amount=99900))
    svc = _svc(rzp_db)

    r1 = svc.process(raw_body=raw, signature=sig, event_id="evt_pytest_dup")
    r2 = svc.process(raw_body=raw, signature=sig, event_id="evt_pytest_dup")

    assert r1.status == "ok"
    assert r2.status == "already_processed"
    iv_id = linked_intervention["intervention_id"]
    outcomes = rzp_db.scalars(
        select(Outcome).where(Outcome.intervention_id == iv_id)
    ).all()
    assert len(outcomes) == 1                        # NOT two
    processed = rzp_db.scalars(
        select(ProcessedWebhookEvent).where(
            ProcessedWebhookEvent.event_id == "evt_pytest_dup"
        )
    ).all()
    assert len(processed) == 1


def test_duplicate_by_outcome_guard_when_event_id_differs(rzp_db, linked_intervention):
    """Two deliveries with *different* event ids still produce one Outcome."""
    raw, sig = signed_webhook(_paid_body(linked_intervention, amount=99900))
    svc = _svc(rzp_db)
    r1 = svc.process(raw_body=raw, signature=sig, event_id="evt_pytest_d1")
    r2 = svc.process(raw_body=raw, signature=sig, event_id="evt_pytest_d2")
    assert r1.status == "ok"
    assert r2.status == "already_processed"
    outcomes = rzp_db.scalars(
        select(Outcome).where(
            Outcome.intervention_id == linked_intervention["intervention_id"]
        )
    ).all()
    assert len(outcomes) == 1
