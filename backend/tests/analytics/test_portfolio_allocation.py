"""PortfolioAllocator / GET /api/v1/analytics/portfolio-allocation.

A hand-constructed batch of isolated open recovery events with monotone
payment amounts, so the rules-policy expected-value ranking is exactly the
descending-amount order and every cutoff is hand-checkable.

The shared test database already carries other open recovery events, so the
deterministic ranking tests pin the allocator's *population* to a known id list
via its one seam -- ``_open_eligible_events`` -- exactly the way the agent tests
swap in a fake provider. One test deliberately leaves that seam alone to prove
the real eligibility filter (control / max-attempts events excluded) still holds.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import delete, func, select

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
from app.services.portfolio_allocator import (
    DEFAULT_BATCH_CAPACITY,
    PortfolioAllocator,
)
from app.services.recovery_config import MAX_INTERVENTION_ATTEMPTS

_TAG = "pytest_alloc_"


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
    db.execute(delete(Intervention).where(Intervention.recovery_event_id.in_(re_ids)))
    db.execute(delete(RecoveryEvent).where(RecoveryEvent.id.in_(re_ids)))
    db.execute(delete(Payment).where(Payment.id.in_(pay_ids)))
    db.execute(delete(Subscription).where(Subscription.id.in_(sub_ids)))
    db.execute(delete(Customer).where(Customer.id.in_(cust_ids)))
    db.commit()


@pytest.fixture
def alloc_db():
    s = SessionLocal()
    _cleanup(s)
    try:
        yield s
    finally:
        _cleanup(s)
        s.rollback()
        s.close()


def _mk_subscription(db) -> Subscription:
    now = datetime.now(timezone.utc)
    cust = Customer(
        external_customer_id=f"{_TAG}{now.timestamp()}",
        email="alloc@example.com",
        total_successful_payments=8,
        total_failed_payments=2,
        created_at=now - timedelta(days=200),
    )
    db.add(cust)
    db.flush()
    sub = Subscription(
        customer_id=cust.id,
        external_subscription_id=f"{_TAG}sub_{cust.id}",
        amount=500000,
        currency="INR",
        status="active",
        started_at=now - timedelta(days=120),
    )
    db.add(sub)
    db.flush()
    return sub


def _open_event(
    db,
    sub: Subscription,
    *,
    amount: int,
    is_control: bool = False,
    interventions: int = 0,
    failure_reason: str = "insufficient_funds",
) -> int:
    now = datetime.now(timezone.utc)
    pay = Payment(
        subscription_id=sub.id,
        amount=amount,
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
        is_control=is_control,
        variant="control" if is_control else "treatment",
        created_at=now - timedelta(hours=6),
    )
    db.add(ev)
    db.flush()
    for i in range(interventions):
        db.add(
            Intervention(
                recovery_event_id=ev.id,
                action_type=["retry", "sms_nudge", "whatsapp_nudge"][i % 3],
                status="executed",
                cost_paise=20,
                executed_at=now - timedelta(hours=5 - i),
            )
        )
    db.flush()
    return ev.id


def _allocator_over(db, event_ids: list[int]) -> PortfolioAllocator:
    """An allocator whose population is pinned to ``event_ids`` (order preserved
    as loaded, then the allocator sorts by expected value itself)."""
    alloc = PortfolioAllocator(db)

    def _pinned() -> list[RecoveryEvent]:
        rows = db.scalars(
            select(RecoveryEvent)
            .where(RecoveryEvent.id.in_(event_ids))
            .options(*_LOAD_OPTIONS)
        ).all()
        return list(rows)

    alloc._open_eligible_events = _pinned  # type: ignore[method-assign]
    return alloc


def _write_counts(db) -> dict:
    return {
        "interventions": db.scalar(select(func.count()).select_from(Intervention)),
        "outcomes": db.scalar(select(func.count()).select_from(Outcome)),
        "agent_events": db.scalar(select(func.count()).select_from(AgentEvent)),
        "audit_logs": db.scalar(select(func.count()).select_from(AuditLog)),
        "recovery_events": db.scalar(
            select(func.count()).select_from(RecoveryEvent)
        ),
    }


# --- 1. ranking + cutoff on a hand-constructed batch -------------------
def test_ranking_and_cutoff_match_expected_value_order(alloc_db):
    sub = _mk_subscription(alloc_db)
    # monotone amounts -> monotone rules-policy expected value (same p, same
    # cost, EV = p*amount - cost), so rank order == descending amount order.
    amounts = [500000, 400000, 300000, 200000, 100000]
    ids = [_open_event(alloc_db, sub, amount=a) for a in amounts]
    alloc_db.commit()

    result = _allocator_over(alloc_db, ids).allocate(capacity=2)

    assert result.computable is True
    assert result.policy == "rules_v1"
    assert result.ranking_basis == "raw_expected_value"
    assert result.total_open_eligible_events == 5
    assert result.events_ranked == 5
    assert result.events_without_actionable_option == 0
    assert result.capacity == 2
    assert result.capacity_used == 2

    assert [e.recovery_event_id for e in result.act] == ids[:2]
    assert [e.rank for e in result.act] == [1, 2]
    assert [e.recovery_event_id for e in result.skip] == ids[2:]
    assert [e.rank for e in result.skip] == [3, 4, 5]

    evs = [e.expected_value_paise for e in result.act + result.skip]
    assert evs == sorted(evs, reverse=True)

    assert result.expected_value_captured_paise == pytest.approx(
        sum(e.expected_value_paise for e in result.act)
    )
    assert result.expected_value_if_unlimited_paise == pytest.approx(sum(evs))
    assert result.expected_value_forgone_to_capacity_paise == pytest.approx(
        result.expected_value_if_unlimited_paise
        - result.expected_value_captured_paise
    )
    assert result.expected_value_forgone_to_capacity_paise > 0


# --- 2. skip reasons are specific, not generic ------------------------
def test_skip_reason_cites_the_actual_rank_cutoff(alloc_db):
    sub = _mk_subscription(alloc_db)
    ids = [
        _open_event(alloc_db, sub, amount=a)
        for a in [500000, 400000, 300000, 200000]
    ]
    alloc_db.commit()

    result = _allocator_over(alloc_db, ids).allocate(capacity=1)
    assert result.capacity_used == 1
    assert [e.recovery_event_id for e in result.act] == [ids[0]]

    cutoff_ev = result.act[0].expected_value_paise
    for e in result.skip:
        assert "below the capacity cutoff at rank #1" in e.reason
        assert f"{cutoff_ev:.0f} paise" in e.reason
        assert "not enough budget" not in e.reason.lower()
    assert "1 place(s) below" in result.skip[0].reason
    assert "3 place(s) below" in result.skip[2].reason


# --- 3. empty batch -> computable=false, no fabricated ranking --------
def test_empty_batch_returns_computable_false(alloc_db):
    result = _allocator_over(alloc_db, []).allocate(capacity=3)
    assert result.computable is False
    assert result.reason is not None
    assert "nothing to allocate" in result.reason
    assert result.act == [] and result.skip == []
    assert result.expected_value_captured_paise == 0.0
    assert result.expected_value_if_unlimited_paise == 0.0
    assert result.capacity_used == 0
    assert result.total_open_eligible_events == 0


# --- 4. capacity larger than the population -> everyone acts ----------
def test_capacity_exceeding_population_acts_on_everyone(alloc_db):
    sub = _mk_subscription(alloc_db)
    ids = [_open_event(alloc_db, sub, amount=a) for a in [300000, 200000, 100000]]
    alloc_db.commit()

    result = _allocator_over(alloc_db, ids).allocate(capacity=99)
    assert {e.recovery_event_id for e in result.act} == set(ids)
    assert result.skip == []
    assert result.capacity_used == 3
    assert result.expected_value_forgone_to_capacity_paise == pytest.approx(0.0)
    assert result.expected_value_captured_paise == pytest.approx(
        result.expected_value_if_unlimited_paise
    )


# --- 5. reuses the existing eligibility definition (real population) --
def test_control_and_max_attempts_events_are_never_selected(alloc_db):
    sub = _mk_subscription(alloc_db)
    treated = _open_event(alloc_db, sub, amount=500000)
    control = _open_event(alloc_db, sub, amount=500000, is_control=True)
    maxed = _open_event(
        alloc_db, sub, amount=500000, interventions=MAX_INTERVENTION_ATTEMPTS
    )
    alloc_db.commit()

    # real _open_eligible_events, not the pinned seam
    result = PortfolioAllocator(alloc_db).allocate(capacity=50)
    seen = {e.recovery_event_id for e in result.act + result.skip}
    assert treated in seen
    assert control not in seen
    assert maxed not in seen


# --- 6. strictly read-only: no writes, no state mutation ------------
def test_allocation_creates_nothing_and_mutates_nothing(alloc_db):
    sub = _mk_subscription(alloc_db)
    ids = [_open_event(alloc_db, sub, amount=a) for a in [500000, 300000, 100000]]
    alloc_db.commit()

    before = _write_counts(alloc_db)
    _allocator_over(alloc_db, ids).allocate(capacity=1)
    _allocator_over(alloc_db, ids).allocate(capacity=2)
    alloc_db.expire_all()
    after = _write_counts(alloc_db)
    assert before == after


# --- 7. default capacity is applied when omitted -------------------
def test_default_capacity_is_used_when_omitted(alloc_db):
    sub = _mk_subscription(alloc_db)
    ids = [
        _open_event(alloc_db, sub, amount=a)
        for a in [500000, 400000, 300000, 200000, 100000, 50000]
    ]
    alloc_db.commit()

    result = _allocator_over(alloc_db, ids).allocate()
    assert result.capacity == DEFAULT_BATCH_CAPACITY
    assert result.capacity_used == DEFAULT_BATCH_CAPACITY


# --- 8. unknown policy name is rejected ---------------------------
def test_unknown_policy_name_raises_value_error(alloc_db):
    with pytest.raises(ValueError):
        PortfolioAllocator(alloc_db, policy_name="not_a_policy")


# --- 9. HTTP endpoint: shape + read-only + bad policy ---------------
def _call(method: str, path: str):
    import asyncio
    import json

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


def test_endpoint_returns_allocation_shape_and_writes_nothing(alloc_db):
    before = _write_counts(alloc_db)
    status, body = _call(
        "GET", "/api/v1/analytics/portfolio-allocation?capacity=2"
    )
    assert status == 200
    assert body["ranking_basis"] == "raw_expected_value"
    assert body["capacity"] == 2
    assert isinstance(body["act"], list) and isinstance(body["skip"], list)
    assert "expected_value_forgone_to_capacity_paise" in body
    for row in body["act"] + body["skip"]:
        assert set(row) >= {
            "recovery_event_id", "best_action", "expected_value_paise",
            "rank", "decision", "reason",
        }
    alloc_db.expire_all()
    assert _write_counts(alloc_db) == before


def test_endpoint_rejects_unknown_policy_with_400(alloc_db):
    status, body = _call(
        "GET", "/api/v1/analytics/portfolio-allocation?policy=bogus"
    )
    assert status == 400
    assert "bogus" in (body.get("detail") or "")
