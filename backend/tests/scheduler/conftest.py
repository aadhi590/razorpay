"""Isolated open-recovery-event fixtures for the scheduler suite.

Same approach as ``tests/analytics/test_portfolio_allocation.py``: create tagged
rows in the real database, pin the PortfolioAllocator's population to them via
its one seam (``_open_eligible_events``), delete on teardown. Gemini is never
called live -- a fake provider is injected everywhere.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import delete, select

from app.agent.guardrails import _LOAD_OPTIONS
from app.database import SessionLocal
from app.models.agent_events import AgentEvent
from app.models.audit_log import AuditLog
from app.models.customer import Customer
from app.models.interventions import Intervention
from app.models.outcome import Outcome
from app.models.payment import Payment
from app.models.recovery_events import RecoveryEvent
from app.models.subscription import Subscription

_TAG = "pytest_sched_"


def _cleanup(db) -> None:
    cust_ids = [
        c.id
        for c in db.scalars(
            select(Customer).where(
                Customer.external_customer_id.like(f"{_TAG}%")
            )
        )
    ]
    if not cust_ids:
        return
    sub_ids = [
        s
        for s in db.scalars(
            select(Subscription.id).where(Subscription.customer_id.in_(cust_ids))
        )
    ] or [-1]
    pay_ids = [
        p
        for p in db.scalars(
            select(Payment.id).where(Payment.subscription_id.in_(sub_ids))
        )
    ] or [-1]
    re_ids = [
        r
        for r in db.scalars(
            select(RecoveryEvent.id).where(RecoveryEvent.payment_id.in_(pay_ids))
        )
    ] or [-1]
    iv_ids = [
        i
        for i in db.scalars(
            select(Intervention.id).where(
                Intervention.recovery_event_id.in_(re_ids)
            )
        )
    ] or [-1]
    db.execute(delete(Outcome).where(Outcome.intervention_id.in_(iv_ids)))
    db.execute(delete(AuditLog).where(AuditLog.recovery_event_id.in_(re_ids)))
    db.execute(delete(AgentEvent).where(AgentEvent.recovery_event_id.in_(re_ids)))
    db.execute(
        delete(Intervention).where(Intervention.recovery_event_id.in_(re_ids))
    )
    db.execute(delete(RecoveryEvent).where(RecoveryEvent.id.in_(re_ids)))
    db.execute(delete(Payment).where(Payment.id.in_(pay_ids)))
    db.execute(delete(Subscription).where(Subscription.id.in_(sub_ids)))
    db.execute(delete(Customer).where(Customer.id.in_(cust_ids)))
    db.commit()


@pytest.fixture
def sched_db():
    s = SessionLocal()
    _cleanup(s)
    try:
        yield s
    finally:
        _cleanup(s)
        s.rollback()
        s.close()


@pytest.fixture
def make_open_events(sched_db):
    """Factory -> list[int]. Creates N isolated open, non-control recovery events
    with strictly descending payment amounts, so the rules-policy expected-value
    ranking is exactly the creation order."""
    db = sched_db

    def _factory(
        amounts: list[int] | None = None,
        n: int | None = None,
        failure_reason: str = "insufficient_funds",
    ) -> list[int]:
        if amounts is None:
            base = 500000
            amounts = [base - i * 50000 for i in range(n or 3)]
        now = datetime.now(timezone.utc)
        cust = Customer(
            external_customer_id=f"{_TAG}{now.timestamp()}",
            email="sched@example.com",
            total_successful_payments=8,
            total_failed_payments=2,
            created_at=now - timedelta(days=200),
        )
        db.add(cust)
        db.flush()
        sub = Subscription(
            customer_id=cust.id,
            external_subscription_id=f"{_TAG}sub_{cust.id}",
            amount=max(amounts),
            currency="INR",
            status="active",
            started_at=now - timedelta(days=120),
        )
        db.add(sub)
        db.flush()
        ids: list[int] = []
        for amt in amounts:
            pay = Payment(
                subscription_id=sub.id,
                amount=amt,
                currency="INR",
                status="failed",
                failure_reason=failure_reason,
                failed_at=now - timedelta(hours=6),
                retry_count=0,
            )
            db.add(pay)
            db.flush()
            ev = RecoveryEvent(
                payment_id=pay.id,
                status="open",
                priority=1,
                is_control=False,
                variant="treatment",
                created_at=now - timedelta(hours=6),
            )
            db.add(ev)
            db.flush()
            ids.append(ev.id)
        db.commit()
        return ids

    return _factory


@pytest.fixture
def pin_allocator_population(monkeypatch):
    """Pin every PortfolioAllocator instance's batch to a given id list (or []),
    so shared-DB rows don't perturb the scheduler's deterministic behaviour."""

    def _pin(event_ids: list[int]) -> None:
        from app.services.portfolio_allocator import PortfolioAllocator

        def _pinned(self):
            if not event_ids:
                return []
            rows = self.db.scalars(
                select(RecoveryEvent)
                .where(RecoveryEvent.id.in_(event_ids))
                .options(*_LOAD_OPTIONS)
            ).all()
            return list(rows)

        monkeypatch.setattr(
            PortfolioAllocator, "_open_eligible_events", _pinned, raising=True
        )

    return _pin
