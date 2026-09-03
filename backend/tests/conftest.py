from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import delete, select, text

from app.database import SessionLocal, engine
from app.models.customer import Customer
from app.models.payment import Payment
from app.models.recovery_events import RecoveryEvent
from app.models.subscription import Subscription

_FIXTURE_TAG = "pytest_ml_"


@pytest.fixture
def db_session():
    s = SessionLocal()
    try:
        yield s
    finally:
        s.rollback()
        s.close()


@pytest.fixture
def conn():
    with engine.connect() as c:
        yield c


@pytest.fixture(scope="session")
def _table_counts() -> dict:
    with engine.connect() as c:
        return {
            t: c.execute(text(f"select count(*) from {t}")).scalar()
            for t in ("interventions", "outcomes", "recovery_events")
        }


@pytest.fixture
def require_training_data(_table_counts):
    if _table_counts["interventions"] < 100:
        pytest.skip(
            "needs full-pipeline data: "
            "python -m app.scripts.generate_data --reset --customers 1000 --seed 42"
        )


@pytest.fixture(scope="session")
def trained_model(tmp_path_factory):
    """A loaded RecoveryModel. Uses the committed artifact if present, else
    trains a fast one into a temp dir."""
    from app.ml.inference.predictor import RecoveryModel
    from app.ml.models.artifact import ModelUnavailable

    try:
        return RecoveryModel.load()
    except ModelUnavailable:
        pass
    out = tmp_path_factory.mktemp("ml_artifacts")
    from app.ml.training.train import run_training

    with engine.connect() as c:
        if c.execute(text("select count(*) from interventions")).scalar() < 100:
            pytest.skip("no artifact and no training data")
        run = run_training(c, out_dir=out, fast=True)
    return RecoveryModel.load(run.artifact_path)


@pytest.fixture
def open_treatment_event(db_session):
    """Create an isolated open treatment recovery event (+ a control one),
    yield their ids, then delete everything."""
    now = datetime.now(timezone.utc)
    _cleanup(db_session)
    cust = Customer(
        external_customer_id=f"{_FIXTURE_TAG}{now.timestamp()}",
        email="t@example.com",
        total_successful_payments=9,
        total_failed_payments=2,
        created_at=now - timedelta(days=180),
    )
    db_session.add(cust)
    db_session.flush()
    sub = Subscription(
        customer_id=cust.id,
        external_subscription_id=f"{_FIXTURE_TAG}sub_{cust.id}",
        amount=99900,
        currency="INR",
        status="active",
        started_at=now - timedelta(days=100),
    )
    db_session.add(sub)
    db_session.flush()

    def _event(reason: str, is_control: bool) -> int:
        p = Payment(
            subscription_id=sub.id,
            amount=99900,
            currency="INR",
            status="failed",
            failure_reason=reason,
            failed_at=now - timedelta(hours=6),
            retry_count=0,
        )
        db_session.add(p)
        db_session.flush()
        re = RecoveryEvent(
            payment_id=p.id,
            status="open",
            priority=2,
            is_control=is_control,
            variant="control" if is_control else "treatment",
            experiment_id=None,
            created_at=now - timedelta(hours=6),
        )
        db_session.add(re)
        db_session.flush()
        return re.id

    treatment_id = _event("insufficient_funds", False)
    control_id = _event("card_expired", True)
    db_session.commit()
    try:
        yield {"treatment_id": treatment_id, "control_id": control_id}
    finally:
        _cleanup(db_session)


def _cleanup(session) -> None:
    from app.models.agent_events import AgentEvent
    from app.models.audit_log import AuditLog
    from app.models.interventions import Intervention
    from app.models.outcome import Outcome

    cust_ids = [
        c.id
        for c in session.scalars(
            select(Customer).where(
                Customer.external_customer_id.like(f"{_FIXTURE_TAG}%")
            )
        )
    ]
    if not cust_ids:
        return
    sub_ids = [s for s in session.scalars(select(Subscription.id).where(Subscription.customer_id.in_(cust_ids)))] or [-1]
    pay_ids = [p for p in session.scalars(select(Payment.id).where(Payment.subscription_id.in_(sub_ids)))] or [-1]
    re_ids = [r for r in session.scalars(select(RecoveryEvent.id).where(RecoveryEvent.payment_id.in_(pay_ids)))] or [-1]
    iv_ids = [i for i in session.scalars(select(Intervention.id).where(Intervention.recovery_event_id.in_(re_ids)))] or [-1]
    session.execute(delete(Outcome).where(Outcome.intervention_id.in_(iv_ids)))
    session.execute(delete(AuditLog).where(AuditLog.recovery_event_id.in_(re_ids)))
    session.execute(delete(AgentEvent).where(AgentEvent.recovery_event_id.in_(re_ids)))
    session.execute(delete(Intervention).where(Intervention.recovery_event_id.in_(re_ids)))
    session.execute(delete(RecoveryEvent).where(RecoveryEvent.id.in_(re_ids)))
    session.execute(delete(Payment).where(Payment.id.in_(pay_ids)))
    session.execute(delete(Subscription).where(Subscription.id.in_(sub_ids)))
    session.execute(delete(Customer).where(Customer.id.in_(cust_ids)))
    session.commit()
