"""Isolated recovery-event fixtures for the agent suite.

Follows the existing ``tests/conftest.py::open_treatment_event`` pattern: create
tagged rows in the real database, yield ids, delete on teardown. No existing row
is touched. Tag ``pytest_ml_`` so the shared cleanup also catches them.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import delete, select

from app.database import SessionLocal
from app.models.agent_events import AgentEvent
from app.models.audit_log import AuditLog
from app.models.customer import Customer
from app.models.interventions import Intervention
from app.models.outcome import Outcome
from app.models.payment import Payment
from app.models.recovery_events import RecoveryEvent
from app.models.subscription import Subscription

_TAG = "pytest_ml_agent_"


def _cleanup(db) -> None:
    cust_ids = [
        c.id for c in db.scalars(
            select(Customer).where(Customer.external_customer_id.like(f"{_TAG}%"))
        )
    ]
    if not cust_ids:
        return
    sub_ids = [s for s in db.scalars(
        select(Subscription.id).where(Subscription.customer_id.in_(cust_ids))
    )] or [-1]
    pay_ids = [p for p in db.scalars(
        select(Payment.id).where(Payment.subscription_id.in_(sub_ids))
    )] or [-1]
    re_ids = [r for r in db.scalars(
        select(RecoveryEvent.id).where(RecoveryEvent.payment_id.in_(pay_ids))
    )] or [-1]
    iv_ids = [i for i in db.scalars(
        select(Intervention.id).where(Intervention.recovery_event_id.in_(re_ids))
    )] or [-1]
    db.execute(delete(Outcome).where(Outcome.intervention_id.in_(iv_ids)))
    db.execute(delete(AuditLog).where(AuditLog.recovery_event_id.in_(re_ids)))
    db.execute(delete(AgentEvent).where(AgentEvent.recovery_event_id.in_(re_ids)))
    db.execute(delete(Intervention).where(Intervention.recovery_event_id.in_(re_ids)))
    db.execute(delete(RecoveryEvent).where(RecoveryEvent.id.in_(re_ids)))
    db.execute(delete(Payment).where(Payment.id.in_(pay_ids)))
    db.execute(delete(Subscription).where(Subscription.id.in_(sub_ids)))
    db.execute(delete(Customer).where(Customer.id.in_(cust_ids)))
    db.commit()


def _make_event(
    db,
    *,
    prior_actions: list[str] | None = None,
    prior_recovered: bool = False,
    is_control: bool = False,
    status: str = "open",
    amount: int = 99900,
    failure_reason: str = "insufficient_funds",
    payment_recovered: bool = False,
) -> int:
    now = datetime.now(timezone.utc)
    cust = Customer(
        external_customer_id=f"{_TAG}{now.timestamp()}_{len(prior_actions or [])}",
        email="rahul@example.com",
        total_successful_payments=8,
        total_failed_payments=3,
        created_at=now - timedelta(days=200),
    )
    db.add(cust)
    db.flush()
    sub = Subscription(
        customer_id=cust.id,
        external_subscription_id=f"{_TAG}sub_{cust.id}",
        amount=amount,
        currency="INR",
        status="active",
        started_at=now - timedelta(days=120),
        next_payment_at=now + timedelta(days=10),
    )
    db.add(sub)
    db.flush()
    pay = Payment(
        subscription_id=sub.id,
        amount=amount,
        currency="INR",
        status="success" if payment_recovered else "failed",
        failure_reason=failure_reason,
        failed_at=now - timedelta(hours=8),
        retry_count=len(prior_actions or []),
        recovered_at=(now - timedelta(hours=1)) if payment_recovered else None,
    )
    db.add(pay)
    db.flush()
    event = RecoveryEvent(
        payment_id=pay.id,
        status=status,
        priority=2,
        is_control=is_control,
        variant="control" if is_control else "treatment",
        created_at=now - timedelta(hours=8),
    )
    db.add(event)
    db.flush()
    for idx, action in enumerate(prior_actions or [], start=1):
        iv = Intervention(
            recovery_event_id=event.id,
            action_type=action,
            status="executed",
            cost_paise=20,
            agent_reason=f"prior attempt {idx}",
            executed_at=now - timedelta(hours=8 - idx),
        )
        db.add(iv)
        db.flush()
        db.add(Outcome(
            intervention_id=iv.id,
            payment_recovered=prior_recovered and idx == len(prior_actions or []),
            recovered_amount_paise=amount if (prior_recovered and idx == len(prior_actions or [])) else 0,
            observed_at=now - timedelta(hours=7 - idx),
        ))
    db.commit()
    return event.id


@pytest.fixture
def db_session_agent():
    s = SessionLocal()
    try:
        yield s
    finally:
        s.rollback()
        s.close()


@pytest.fixture
def make_recovery_event(db_session_agent):
    """Factory: create an isolated recovery event with a given history."""
    db = db_session_agent
    _cleanup(db)
    created: list[int] = []

    def _factory(**kw) -> int:
        eid = _make_event(db, **kw)
        created.append(eid)
        return eid

    try:
        yield _factory
    finally:
        _cleanup(db)


@pytest.fixture
def fresh_event(make_recovery_event) -> int:
    return make_recovery_event()


@pytest.fixture
def twice_attempted_event(make_recovery_event) -> int:
    return make_recovery_event(prior_actions=["retry", "sms_nudge"])


@pytest.fixture
def all_actions_tried_event(make_recovery_event) -> int:
    return make_recovery_event(
        prior_actions=["retry", "sms_nudge", "whatsapp_nudge"]
    )


@pytest.fixture
def recovered_event(make_recovery_event) -> int:
    return make_recovery_event(
        prior_actions=["whatsapp_nudge"], prior_recovered=True,
        payment_recovered=True,
    )


@pytest.fixture
def control_event(make_recovery_event) -> int:
    return make_recovery_event(is_control=True)
