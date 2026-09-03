"""Hand-computable control/treatment datasets for the recovery-impact endpoint.

Every event is tagged to a dedicated throwaway ``Experiment`` so the tests can
isolate their own data with ``?experiment_id=`` regardless of the ~10k events
already in the database. Cleaned up on teardown.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import delete, select

from app.database import SessionLocal
from app.models.customer import Customer
from app.models.experiment import Experiment
from app.models.interventions import Intervention
from app.models.outcome import Outcome
from app.models.payment import Payment
from app.models.recovery_events import RecoveryEvent
from app.models.subscription import Subscription

_TAG = "pytest_impact_"


def _cleanup(db) -> None:
    exp_ids = [
        e.id for e in db.scalars(
            select(Experiment).where(Experiment.name.like(f"{_TAG}%"))
        )
    ]
    cust_ids = [
        c.id for c in db.scalars(
            select(Customer).where(Customer.external_customer_id.like(f"{_TAG}%"))
        )
    ]
    if cust_ids:
        sub_ids = [s for s in db.scalars(
            select(Subscription.id).where(Subscription.customer_id.in_(cust_ids)))] or [-1]
        pay_ids = [p for p in db.scalars(
            select(Payment.id).where(Payment.subscription_id.in_(sub_ids)))] or [-1]
        re_ids = [r for r in db.scalars(
            select(RecoveryEvent.id).where(RecoveryEvent.payment_id.in_(pay_ids)))] or [-1]
        iv_ids = [i for i in db.scalars(
            select(Intervention.id).where(Intervention.recovery_event_id.in_(re_ids)))] or [-1]
        db.execute(delete(Outcome).where(Outcome.intervention_id.in_(iv_ids)))
        db.execute(delete(Intervention).where(Intervention.recovery_event_id.in_(re_ids)))
        db.execute(delete(RecoveryEvent).where(RecoveryEvent.id.in_(re_ids)))
        db.execute(delete(Payment).where(Payment.id.in_(pay_ids)))
        db.execute(delete(Subscription).where(Subscription.id.in_(sub_ids)))
        db.execute(delete(Customer).where(Customer.id.in_(cust_ids)))
    if exp_ids:
        db.execute(delete(Experiment).where(Experiment.id.in_(exp_ids)))
    db.commit()


@pytest.fixture
def impact_db():
    s = SessionLocal()
    _cleanup(s)
    try:
        yield s
    finally:
        _cleanup(s)
        s.rollback()
        s.close()


def build_dataset(
    db,
    *,
    control: list[int],
    control_recovered: list[bool],
    treated: list[int],
    treated_recovered: list[bool],
    treated_action: str = "sms_nudge",
) -> int:
    """Create one experiment's worth of events. ``control`` / ``treated`` are
    lists of payment amounts (paise); the ``*_recovered`` booleans say which are
    recovered. ``treated_action`` is the ``action_type`` of the single
    Intervention created for each treated event (default ``"sms_nudge"``,
    preserving prior behaviour). Returns the experiment_id."""
    now = datetime.now(timezone.utc)
    exp = Experiment(
        name=f"{_TAG}{now.timestamp()}",
        intervention_type="sms_nudge",
        control_percentage=50,
        treatment_percentage=50,
        status="active",
        started_at=now - timedelta(days=30),
    )
    db.add(exp)
    db.flush()

    cust = Customer(
        external_customer_id=f"{_TAG}{now.timestamp()}",
        email="impact@example.com",
        total_successful_payments=5,
        total_failed_payments=5,
        created_at=now - timedelta(days=90),
    )
    db.add(cust)
    db.flush()
    sub = Subscription(
        customer_id=cust.id,
        external_subscription_id=f"{_TAG}sub_{cust.id}",
        amount=max(control + treated) if (control or treated) else 10000,
        currency="INR",
        status="active",
        started_at=now - timedelta(days=60),
    )
    db.add(sub)
    db.flush()

    def _event(amount: int, is_control: bool, recovered: bool) -> None:
        pay = Payment(
            subscription_id=sub.id,
            amount=amount,
            currency="INR",
            status="success" if (recovered and is_control) else "failed",
            failure_reason="insufficient_funds",
            failed_at=now - timedelta(hours=10),
            retry_count=0,
            recovered_at=(now - timedelta(hours=1)) if (recovered and is_control) else None,
        )
        db.add(pay)
        db.flush()
        ev = RecoveryEvent(
            payment_id=pay.id,
            status="closed" if recovered else "open",
            priority=1,
            is_control=is_control,
            experiment_id=exp.id,
            variant="control" if is_control else "treatment",
            created_at=now - timedelta(hours=10),
        )
        db.add(ev)
        db.flush()
        if not is_control:
            iv = Intervention(
                recovery_event_id=ev.id,
                action_type=treated_action,
                status="executed",
                cost_paise=20,
                executed_at=now - timedelta(hours=8),
            )
            db.add(iv)
            db.flush()
            db.add(Outcome(
                intervention_id=iv.id,
                payment_recovered=recovered,
                recovered_amount_paise=amount if recovered else 0,
                observed_at=now - timedelta(hours=6),
            ))

    for amt, rec in zip(control, control_recovered):
        _event(amt, is_control=True, recovered=rec)
    for amt, rec in zip(treated, treated_recovered):
        _event(amt, is_control=False, recovered=rec)
    db.commit()
    return exp.id
