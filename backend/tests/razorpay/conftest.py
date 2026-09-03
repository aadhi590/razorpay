"""DB fixtures for the Razorpay webhook / execution / agent-integration tests.

Reuses the isolated-event factory from ``tests/agent/conftest.py`` (same
``pytest_ml_agent_`` tag + cleanup) rather than inventing a new mechanism.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest
from sqlalchemy import delete, select

from app.database import SessionLocal
from app.models.interventions import Intervention
from app.models.webhook_event import ProcessedWebhookEvent
from app.services.recovery_execution import reference_id_for
from tests.agent.conftest import _cleanup, _make_event


@pytest.fixture
def rzp_db():
    s = SessionLocal()
    _cleanup(s)
    s.execute(
        delete(ProcessedWebhookEvent).where(
            ProcessedWebhookEvent.event_id.like("evt_pytest_%")
        )
    )
    s.commit()
    try:
        yield s
    finally:
        s.execute(
            delete(ProcessedWebhookEvent).where(
                ProcessedWebhookEvent.event_id.like("evt_pytest_%")
            )
        )
        s.commit()
        _cleanup(s)
        s.rollback()
        s.close()


@pytest.fixture
def fresh_event_id(rzp_db) -> int:
    return _make_event(rzp_db)


@pytest.fixture
def linked_intervention(rzp_db):
    """A fresh event + one intervention that already has a Razorpay Payment
    Link recorded (as if execute_recovery_action had run in live mode)."""
    event_id = _make_event(rzp_db)
    iv = Intervention(
        recovery_event_id=event_id,
        action_type="whatsapp_nudge",
        status="executed",
        cost_paise=80,
        agent_reason="[gemini] test",
        executed_at=datetime.now(timezone.utc),
    )
    rzp_db.add(iv)
    rzp_db.flush()
    ref = reference_id_for(event_id, iv.id)
    iv.razorpay_reference_id = ref
    iv.razorpay_payment_link_id = f"plink_pytest_{iv.id}"
    iv.razorpay_short_url = "https://rzp.io/i/pytest"
    iv.last_razorpay_status = "created"
    rzp_db.commit()
    return {
        "recovery_event_id": event_id,
        "intervention_id": iv.id,
        "payment_link_id": iv.razorpay_payment_link_id,
        "reference_id": ref,
    }
