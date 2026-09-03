"""AnalyticsService.action_lift_trend -- the recent-vs-baseline effectiveness
trend consumed by the get_action_lift_trend agent tool.

It is the time-window extension of action_incrementality and must reuse the same
Newcombe/Wilson machinery, so these tests mirror test_action_incrementality.py:
hand-computable constructed datasets, ?experiment_id isolation, a pinned ``now``
so the 90-day window boundary is deterministic, plus the real global dataset.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from sqlalchemy import select

from app.models.customer import Customer
from app.models.experiment import Experiment
from app.models.interventions import Intervention
from app.models.outcome import Outcome
from app.models.payment import Payment
from app.models.recovery_events import RecoveryEvent
from app.models.subscription import Subscription
from app.services.analytics_service import RECENT_WINDOW_DAYS, AnalyticsService
from app.services.proportion_stats import newcombe_difference_interval

# tests/analytics/conftest.py cleans up everything tagged "pytest_impact_"
from tests.analytics.conftest import _TAG

_NOW = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)
_CUTOFF = _NOW - timedelta(days=RECENT_WINDOW_DAYS)
_RECENT_AT = _CUTOFF + timedelta(days=10)   # inside the recent window
_PRIOR_AT = _CUTOFF - timedelta(days=40)    # inside the disjoint prior window


def _build_windowed(
    db,
    *,
    action: str,
    recent_treated: int,
    recent_recovered: int,
    prior_treated: int,
    prior_recovered: int,
    recent_control: int = 120,
    recent_control_recovered: int = 4,
    prior_control: int = 120,
    prior_control_recovered: int = 4,
) -> int:
    """One experiment: treated events using ``action`` and control events, split
    across the recent window (``_RECENT_AT``) and the disjoint prior window
    (``_PRIOR_AT``). Returns the experiment_id."""
    exp = Experiment(
        name=f"{_TAG}{datetime.now(timezone.utc).timestamp()}",
        intervention_type=action,
        control_percentage=50,
        treatment_percentage=50,
        status="active",
        started_at=_PRIOR_AT - timedelta(days=10),
    )
    db.add(exp)
    db.flush()
    cust = Customer(
        external_customer_id=f"{_TAG}{datetime.now(timezone.utc).timestamp()}",
        email="trend@example.com",
        total_successful_payments=5,
        total_failed_payments=5,
        created_at=_PRIOR_AT - timedelta(days=30),
    )
    db.add(cust)
    db.flush()
    sub = Subscription(
        customer_id=cust.id,
        external_subscription_id=f"{_TAG}sub_{cust.id}",
        amount=10000,
        currency="INR",
        status="active",
        started_at=_PRIOR_AT - timedelta(days=20),
    )
    db.add(sub)
    db.flush()

    def _event(created_at, is_control: bool, recovered: bool) -> None:
        pay = Payment(
            subscription_id=sub.id,
            amount=10000,
            currency="INR",
            status="success" if (recovered and is_control) else "failed",
            failure_reason="insufficient_funds",
            failed_at=created_at,
            retry_count=0,
            recovered_at=(created_at + timedelta(hours=6))
            if (recovered and is_control)
            else None,
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
            created_at=created_at,
        )
        db.add(ev)
        db.flush()
        if not is_control:
            iv = Intervention(
                recovery_event_id=ev.id,
                action_type=action,
                status="executed",
                cost_paise=20,
                executed_at=created_at + timedelta(hours=2),
            )
            db.add(iv)
            db.flush()
            db.add(
                Outcome(
                    intervention_id=iv.id,
                    payment_recovered=recovered,
                    recovered_amount_paise=10000 if recovered else 0,
                    observed_at=created_at + timedelta(hours=4),
                )
            )

    for at, n, rec, is_ctrl in (
        (_RECENT_AT, recent_treated, recent_recovered, False),
        (_PRIOR_AT, prior_treated, prior_recovered, False),
        (_RECENT_AT, recent_control, recent_control_recovered, True),
        (_PRIOR_AT, prior_control, prior_control_recovered, True),
    ):
        for i in range(n):
            _event(at, is_ctrl, recovered=(i < rec))
    db.commit()
    return exp.id


def _trend(db, action, exp_id):
    return AnalyticsService(db).action_lift_trend(
        action, experiment_id=exp_id, now=_NOW
    )


# --- 1. engineered decline -------------------------------------------
def test_detects_a_declining_trend(impact_db):
    # prior action recovery rate 60/200 = 0.30 ; recent 20/200 = 0.10
    exp_id = _build_windowed(
        impact_db,
        action="whatsapp_nudge",
        recent_treated=200, recent_recovered=20,
        prior_treated=200, prior_recovered=60,
    )
    r = _trend(impact_db, "whatsapp_nudge", exp_id)

    assert r.computable is True
    assert r.trend_direction == "declining"
    assert r.recent_window_action_recovery_rate == 0.1
    assert r.baseline_window_action_recovery_rate == 0.3
    lo, hi = r.trend_confidence_interval
    assert hi < 0.0  # interval entirely below zero
    # reuses the exact Newcombe helper: recent = a, prior = b
    ci = newcombe_difference_interval(20, 200, 60, 200)
    assert r.trend_confidence_interval == ci.as_list()
    assert r.confidence_method == "newcombe_wilson_95_difference"
    assert "declining" in r.sample_size_note
    assert "Newcombe/Wilson" in r.sample_size_note


# --- 2. engineered improvement -------------------------------------
def test_detects_an_improving_trend(impact_db):
    # prior 20/200 = 0.10 ; recent 70/200 = 0.35
    exp_id = _build_windowed(
        impact_db,
        action="retry",
        recent_treated=200, recent_recovered=70,
        prior_treated=200, prior_recovered=20,
    )
    r = _trend(impact_db, "retry", exp_id)

    assert r.computable is True
    assert r.trend_direction == "improving"
    lo, hi = r.trend_confidence_interval
    assert lo > 0.0
    assert r.trend_confidence_interval == newcombe_difference_interval(
        70, 200, 20, 200
    ).as_list()


# --- 3. genuinely flat -> stable, interval includes zero -----------
def test_flat_data_is_reported_as_stable_not_a_trend(impact_db):
    # prior 40/200 = 0.20 ; recent 44/200 = 0.22 -- a 2pp wobble
    exp_id = _build_windowed(
        impact_db,
        action="sms_nudge",
        recent_treated=200, recent_recovered=44,
        prior_treated=200, prior_recovered=40,
    )
    r = _trend(impact_db, "sms_nudge", exp_id)

    assert r.computable is True
    assert r.trend_direction == "stable_or_insufficient_data"
    lo, hi = r.trend_confidence_interval
    assert lo < 0.0 < hi
    assert "no trend distinguishable from sampling noise" in r.sample_size_note


# --- 4. thin recent window -> explicit, no fabricated trend --------
def test_insufficient_recent_window_is_explicit(impact_db):
    exp_id = _build_windowed(
        impact_db,
        action="method_switch_prompt",
        recent_treated=10, recent_recovered=3,   # < MIN_DISTINCT_EVENTS_PER_ACTION
        prior_treated=200, prior_recovered=40,
    )
    r = _trend(impact_db, "method_switch_prompt", exp_id)

    assert r.computable is False
    assert r.reason == "insufficient_recent_data"
    assert r.trend_direction == "stable_or_insufficient_data"
    assert r.trend_confidence_interval is None
    assert r.confidence_method == "not_computed"
    assert r.recent_window_size == 10


# --- 5. thin prior window -> explicit baseline reason --------------
def test_insufficient_baseline_window_is_explicit(impact_db):
    exp_id = _build_windowed(
        impact_db,
        action="whatsapp_nudge",
        recent_treated=200, recent_recovered=40,
        prior_treated=8, prior_recovered=2,
    )
    r = _trend(impact_db, "whatsapp_nudge", exp_id)

    assert r.computable is False
    assert r.reason == "insufficient_baseline_data"
    assert r.trend_direction == "stable_or_insufficient_data"
    # the computable recent side is still reported
    assert r.recent_window_action_recovery_rate == 0.2


# --- 6. control-arm shift is disclosed in the note ----------------
def test_control_arm_shift_is_disclosed(impact_db):
    exp_id = _build_windowed(
        impact_db,
        action="retry",
        recent_treated=200, recent_recovered=30,
        prior_treated=200, prior_recovered=32,
        recent_control=200, recent_control_recovered=40,   # 0.20
        prior_control=200, prior_control_recovered=8,       # 0.04
    )
    r = _trend(impact_db, "retry", exp_id)
    assert r.recent_window_control_recovery_rate == 0.2
    assert r.baseline_window_control_recovery_rate == 0.04
    assert "Control-arm recovery rate over the same windows" in r.sample_size_note
    assert "system-wide change" in r.sample_size_note


# --- 7. against the real global dataset ---------------------------
def test_global_call_is_computable_and_reuses_incrementality(impact_db):
    svc = AnalyticsService(impact_db)
    r = svc.action_lift_trend("whatsapp_nudge")
    # all real action types are well-sampled in both windows
    assert r.computable is True
    assert r.trend_direction in {
        "improving", "declining", "stable_or_insufficient_data"
    }
    # all-time lift matches the standalone incrementality tool exactly
    inc = svc.action_incrementality("whatsapp_nudge")
    assert r.all_time_lift == inc.observed_incremental_lift
    assert r.all_time_window_size == inc.treated_group_size


def test_windows_are_disjoint_and_partition_the_action_uses(impact_db):
    """recent_window_size + baseline_window_size should account for every use of
    the action (the two windows are created_at >= cutoff and < cutoff)."""
    svc = AnalyticsService(impact_db)
    r = svc.action_lift_trend("sms_nudge")
    inc = svc.action_incrementality("sms_nudge")
    assert r.recent_window_size + r.baseline_window_size == inc.treated_group_size
