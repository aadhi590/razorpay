"""Recovery Scheduler -- bounded in-process auto-trigger.

Gemini is NEVER called live: a fake provider is injected into every cycle, the
same fakes the agent suite uses. The PortfolioAllocator population is pinned to
isolated tagged events so shared-DB rows can't perturb the deterministic
assertions.
"""
from __future__ import annotations

import asyncio
import json
import time
from collections import deque

import pytest
from sqlalchemy import select

from app.models.agent_events import AgentEvent
from app.models.audit_log import AuditLog
from app.models.interventions import Intervention
from app.services.recovery_scheduler import (
    RecoveryScheduler,
    SchedulerConfig,
)
from tests.agent.fakes import RaisingProvider, ReactiveProvider


def _cfg(**kw) -> SchedulerConfig:
    base = dict(
        enabled=False,
        interval_seconds=1.0,
        max_auto_runs_per_cycle=2,
        dry_run=True,
        policy="rules",
        history_size=20,
    )
    base.update(kw)
    return SchedulerConfig(**base)


def _fake_factory():
    # a fresh reactive provider per run (the fakes are stateful)
    return ReactiveProvider()


# ---------------------------------------------------------------------------
# 1. disabled by default -> the timer never starts
# ---------------------------------------------------------------------------


def test_disabled_scheduler_never_starts_a_thread():
    sched = RecoveryScheduler(config=_cfg(enabled=False))
    assert sched.start() is False
    assert sched.running is False
    st = sched.status()
    assert st["enabled"] is False and st["running"] is False
    assert st["cycles_run"] == 0


def test_lifespan_start_is_noop_when_disabled(monkeypatch):
    from app.services import recovery_scheduler as mod

    monkeypatch.setattr(mod.scheduler, "_config", _cfg(enabled=False))
    assert mod.scheduler.start() is False
    assert mod.scheduler.running is False


# ---------------------------------------------------------------------------
# 2. one cycle triggers real agent runs on the allocator's ranked "act" set
# ---------------------------------------------------------------------------


def test_cycle_triggers_agent_runs_on_the_ranked_act_set(
    sched_db, make_open_events, pin_allocator_population
):
    ids = make_open_events(amounts=[500000, 400000, 300000, 200000, 100000])
    pin_allocator_population(ids)

    sched = RecoveryScheduler(config=_cfg(max_auto_runs_per_cycle=2),
                              provider_factory=_fake_factory)
    rec = sched.run_cycle(db=sched_db, trigger="run_once")

    assert rec.allocation_computable is True
    assert rec.events_considered == 5
    assert rec.events_ranked == 5
    # exactly the top 2 by expected value (== descending amount here) were run
    assert rec.events_triggered == 2
    assert [t.recovery_event_id for t in rec.triggered] == ids[:2]
    assert [t.rank for t in rec.triggered] == [1, 2]
    for t in rec.triggered:
        assert t.error is None
        assert t.run_status in {"completed", "escalated", "failed_safe"}

    # an AgentEvent trace exists for each triggered event
    traced = sched_db.scalars(
        select(AgentEvent).where(AgentEvent.recovery_event_id.in_(ids[:2]))
    ).all()
    assert {a.recovery_event_id for a in traced} == set(ids[:2])


# ---------------------------------------------------------------------------
# 3. hard cap -- never exceeds SCHEDULER_MAX_AUTO_RUNS_PER_CYCLE
# ---------------------------------------------------------------------------


def test_hard_cap_is_never_exceeded_even_with_more_eligible_events(
    sched_db, make_open_events, pin_allocator_population
):
    ids = make_open_events(amounts=[500000, 450000, 400000, 350000, 300000, 250000])
    pin_allocator_population(ids)

    sched = RecoveryScheduler(config=_cfg(max_auto_runs_per_cycle=2),
                              provider_factory=_fake_factory)
    rec = sched.run_cycle(db=sched_db, trigger="run_once")

    assert rec.hard_cap == 2
    assert len(rec.triggered) <= 2
    assert rec.events_triggered == 2
    assert rec.act_set_size == 2
    # 4 positive-EV ranked events were left unacted purely because of the cap
    assert rec.eligible_beyond_cap == 4
    assert rec.hard_cap_enforced is True
    assert rec.expected_value_forgone_to_capacity_paise > 0

    # every not-run event is in skipped with a concrete reason (never generic)
    skipped_ids = {s.recovery_event_id for s in rec.skipped}
    assert set(ids[2:]).issubset(skipped_ids)
    for s in rec.skipped:
        assert s.reason and "not enough budget" not in s.reason.lower()

    # only 2 agent traces were actually written
    traced = sched_db.scalars(
        select(AgentEvent).where(AgentEvent.recovery_event_id.in_(ids))
    ).all()
    assert len(traced) == 2


def test_cap_of_zero_triggers_nothing(
    sched_db, make_open_events, pin_allocator_population
):
    ids = make_open_events(n=3)
    pin_allocator_population(ids)
    sched = RecoveryScheduler(config=_cfg(max_auto_runs_per_cycle=0),
                              provider_factory=_fake_factory)
    rec = sched.run_cycle(db=sched_db, trigger="run_once")
    assert rec.events_triggered == 0
    assert rec.triggered == []
    assert any("SCHEDULER_MAX_AUTO_RUNS_PER_CYCLE is 0" in e for e in rec.errors)
    assert not sched_db.scalars(
        select(AgentEvent).where(AgentEvent.recovery_event_id.in_(ids))
    ).all()


# ---------------------------------------------------------------------------
# 4. scheduler-triggered runs are tagged triggered_by="scheduler"
# ---------------------------------------------------------------------------


def test_scheduler_runs_are_tagged_triggered_by_scheduler(
    sched_db, make_open_events, pin_allocator_population
):
    ids = make_open_events(n=1)
    pin_allocator_population(ids)
    sched = RecoveryScheduler(config=_cfg(max_auto_runs_per_cycle=1),
                              provider_factory=_fake_factory)
    sched.run_cycle(db=sched_db, trigger="run_once")

    ev = sched_db.scalars(
        select(AgentEvent).where(AgentEvent.recovery_event_id == ids[0])
    ).one()
    assert ev.input_context["triggered_by"] == "scheduler"

    logs = sched_db.scalars(
        select(AuditLog).where(AuditLog.recovery_event_id == ids[0])
    ).all()
    assert logs
    assert all(l.event_metadata.get("triggered_by") == "scheduler" for l in logs)


def test_manual_agent_run_is_tagged_manual(
    sched_db, make_open_events, monkeypatch
):
    from app.agent import runner

    eid = make_open_events(n=1)[0]
    monkeypatch.setattr(
        runner, "GeminiProvider", lambda config=None: ReactiveProvider()
    )
    runner.run_recovery_agent(sched_db, eid, dry_run=True, triggered_by="manual")

    ev = sched_db.scalars(
        select(AgentEvent).where(AgentEvent.recovery_event_id == eid)
    ).one()
    assert ev.input_context["triggered_by"] == "manual"


# ---------------------------------------------------------------------------
# 5. dry-run discipline
# ---------------------------------------------------------------------------


def test_dry_run_cycle_creates_no_intervention(
    sched_db, make_open_events, pin_allocator_population
):
    ids = make_open_events(n=2)
    pin_allocator_population(ids)
    sched = RecoveryScheduler(config=_cfg(dry_run=True, max_auto_runs_per_cycle=2),
                              provider_factory=_fake_factory)
    rec = sched.run_cycle(db=sched_db, trigger="run_once")
    assert rec.dry_run is True
    assert not sched_db.scalars(
        select(Intervention).where(Intervention.recovery_event_id.in_(ids))
    ).all()


def test_wet_run_cycle_can_create_a_pending_intervention(
    sched_db, make_open_events, pin_allocator_population
):
    ids = make_open_events(n=1)
    pin_allocator_population(ids)
    sched = RecoveryScheduler(config=_cfg(dry_run=False, max_auto_runs_per_cycle=1),
                              provider_factory=_fake_factory)
    rec = sched.run_cycle(db=sched_db, trigger="run_once")
    assert rec.dry_run is False
    ivs = sched_db.scalars(
        select(Intervention).where(Intervention.recovery_event_id == ids[0])
    ).all()
    # the reactive fake executes the best-scored action for a fresh event
    assert len(ivs) == 1
    assert ivs[0].status in {"pending", "executed"}


# ---------------------------------------------------------------------------
# 6. nothing to allocate -> zero runs, recorded, no crash
# ---------------------------------------------------------------------------


def test_no_open_events_is_a_clean_empty_cycle(
    sched_db, pin_allocator_population
):
    pin_allocator_population([])
    sched = RecoveryScheduler(config=_cfg(), provider_factory=_fake_factory)
    rec = sched.run_cycle(db=sched_db, trigger="run_once")
    assert rec.allocation_computable is False
    assert rec.allocation_reason is not None
    assert rec.events_triggered == 0
    assert rec.triggered == []
    # still recorded in history
    assert sched.status()["cycle_history"][0]["cycle_id"] == rec.cycle_id


# ---------------------------------------------------------------------------
# 7. a failing agent run is recorded; the cycle keeps going
# ---------------------------------------------------------------------------


def test_one_failing_run_does_not_abort_the_cycle(
    sched_db, make_open_events, pin_allocator_population, monkeypatch
):
    ids = make_open_events(amounts=[500000, 400000])
    pin_allocator_population(ids)

    from app.services import recovery_scheduler as mod

    real = mod.run_recovery_agent

    def _flaky(session, event_id, **kw):
        if event_id == ids[0]:
            raise RuntimeError("boom in agent run")
        return real(session, event_id, **kw)

    monkeypatch.setattr(mod, "run_recovery_agent", _flaky)

    sched = RecoveryScheduler(config=_cfg(max_auto_runs_per_cycle=2),
                              provider_factory=_fake_factory)
    rec = sched.run_cycle(db=sched_db, trigger="run_once")

    assert len(rec.triggered) == 2
    failed = [t for t in rec.triggered if t.recovery_event_id == ids[0]][0]
    ok = [t for t in rec.triggered if t.recovery_event_id == ids[1]][0]
    assert failed.run_status == "error"
    assert "boom in agent run" in failed.error
    assert ok.run_status in {"completed", "escalated", "failed_safe"}
    # events_triggered counts only the ones that actually ran
    assert rec.events_triggered == 1


def test_provider_quota_failure_is_a_failed_safe_run_not_a_crash(
    sched_db, make_open_events, pin_allocator_population
):
    from app.agent.providers.base import RateLimitedError

    ids = make_open_events(n=1)
    pin_allocator_population(ids)
    sched = RecoveryScheduler(
        config=_cfg(max_auto_runs_per_cycle=1),
        provider_factory=lambda: RaisingProvider(RateLimitedError("429 quota")),
    )
    rec = sched.run_cycle(db=sched_db, trigger="run_once")
    assert len(rec.triggered) == 1
    assert rec.triggered[0].run_status == "failed_safe"
    assert rec.triggered[0].error is None  # a clean fail-safe, not an exception


# ---------------------------------------------------------------------------
# 8. cycle history is bounded
# ---------------------------------------------------------------------------


def test_cycle_history_is_bounded(sched_db, pin_allocator_population):
    pin_allocator_population([])
    sched = RecoveryScheduler(config=_cfg(history_size=3),
                              provider_factory=_fake_factory)
    for _ in range(6):
        sched.run_cycle(db=sched_db, trigger="run_once")
    hist = sched.status()["cycle_history"]
    assert len(hist) == 3
    # newest first, contiguous ids
    ids = [c["cycle_id"] for c in hist]
    assert ids == [6, 5, 4]


# ---------------------------------------------------------------------------
# 9. the background timer actually fires cycles, then stops cleanly
# ---------------------------------------------------------------------------


def test_timer_thread_runs_cycles_then_stops(
    sched_db, make_open_events, pin_allocator_population
):
    ids = make_open_events(n=1)
    pin_allocator_population(ids)
    sched = RecoveryScheduler(
        config=_cfg(enabled=True, interval_seconds=1.0, max_auto_runs_per_cycle=1),
        provider_factory=_fake_factory,
    )
    assert sched.start() is True
    assert sched.running is True
    try:
        deadline = time.monotonic() + 12.0
        while (
            time.monotonic() < deadline
            and len(sched.status()["cycle_history"]) < 1
        ):
            time.sleep(0.2)
        st = sched.status()
        assert st["cycles_run"] >= 1
        assert st["cycle_history"][0]["trigger"] == "timer"
    finally:
        sched.stop()
    assert sched.running is False


# ---------------------------------------------------------------------------
# 10. HTTP surface: GET /status and POST /run-once
# ---------------------------------------------------------------------------


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


@pytest.fixture
def singleton_scheduler(monkeypatch):
    """Point the process-wide scheduler at a fake provider + fresh history."""
    from app.agent import runner
    from app.services import recovery_scheduler as mod

    monkeypatch.setattr(
        runner, "GeminiProvider", lambda config=None: ReactiveProvider()
    )
    monkeypatch.setattr(mod.scheduler, "_history", deque(maxlen=20))
    monkeypatch.setattr(mod.scheduler, "_cycle_seq", 0)
    monkeypatch.setattr(mod.scheduler, "_cycles_completed", 0)
    monkeypatch.setattr(mod.scheduler, "_provider_factory", None)
    return mod.scheduler


def test_status_endpoint_reports_disabled_state_and_config(
    singleton_scheduler, monkeypatch
):
    monkeypatch.setattr(singleton_scheduler, "_config", _cfg(enabled=False))
    sc, body = _call("GET", "/api/v1/scheduler/status")
    assert sc == 200
    assert body["enabled"] is False
    assert body["running"] is False
    assert body["config"]["max_auto_runs_per_cycle"] == 2
    assert body["cycle_history"] == []


def test_run_once_endpoint_runs_one_cycle_and_it_shows_in_status(
    singleton_scheduler, monkeypatch, sched_db, make_open_events,
    pin_allocator_population,
):
    ids = make_open_events(amounts=[500000, 400000, 300000])
    pin_allocator_population(ids)
    monkeypatch.setattr(
        singleton_scheduler, "_config",
        _cfg(enabled=False, max_auto_runs_per_cycle=2),
    )

    sc, body = _call("POST", "/api/v1/scheduler/run-once")
    assert sc == 200
    assert body["ran"] is True
    cyc = body["cycle"]
    assert cyc["trigger"] == "run_once"
    assert cyc["events_considered"] == 3
    assert cyc["events_triggered"] == 2
    assert [t["recovery_event_id"] for t in cyc["triggered"]] == ids[:2]
    assert cyc["eligible_beyond_cap"] == 1
    assert cyc["hard_cap_enforced"] is True

    # the same cycle, decision + skip reasons included, is in /status history
    sc2, st = _call("GET", "/api/v1/scheduler/status")
    assert sc2 == 200
    assert st["cycle_history"][0]["cycle_id"] == cyc["cycle_id"]
    assert st["cycle_history"][0]["skipped"][0]["reason"]
    assert st["last_cycle_at"] is not None

    # run-once respected dry-run: no interventions
    assert not sched_db.scalars(
        select(Intervention).where(Intervention.recovery_event_id.in_(ids))
    ).all()


def test_run_once_on_empty_batch_is_200_and_clean(
    singleton_scheduler, monkeypatch, pin_allocator_population
):
    pin_allocator_population([])
    monkeypatch.setattr(singleton_scheduler, "_config", _cfg(enabled=False))
    sc, body = _call("POST", "/api/v1/scheduler/run-once")
    assert sc == 200
    assert body["cycle"]["allocation_computable"] is False
    assert body["cycle"]["events_triggered"] == 0


def test_scheduler_routes_are_registered():
    from app.main import app

    paths = set(app.openapi()["paths"])
    assert "/api/v1/scheduler/status" in paths
    assert "/api/v1/scheduler/run-once" in paths
