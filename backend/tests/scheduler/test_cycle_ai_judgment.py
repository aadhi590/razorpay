"""AI cycle judgment -- one bounded Gemini call inserted into run_cycle,
immediately before the PortfolioAllocator.

Gemini is NEVER called live here: a fake judgment provider is injected via
``RecoveryScheduler(judgment_provider_factory=...)``. The agent runs still use
the agent suite's fake provider. The allocator population is pinned to isolated
tagged events.
"""
from __future__ import annotations

import asyncio
import json
from collections import deque

import pytest
from sqlalchemy import select

from app.agent.providers.base import MalformedResponseError, RateLimitedError
from app.models.interventions import Intervention
from app.services.recovery_scheduler import RecoveryScheduler, SchedulerConfig
from tests.agent.fakes import ReactiveProvider


def _cfg(**kw) -> SchedulerConfig:
    base = dict(
        enabled=False,
        interval_seconds=1.0,
        max_auto_runs_per_cycle=3,
        dry_run=True,
        policy="rules",
        history_size=20,
        ai_judgment_enabled=True,
    )
    base.update(kw)
    return SchedulerConfig(**base)


def _agent_factory():
    return ReactiveProvider()


class FakeJudge:
    """Stands in for GeminiProvider.generate_json -- one structured call."""

    def __init__(self, response: dict | None = None, exc: BaseException | None = None):
        self.calls: list[dict] = []
        self._response = response
        self._exc = exc

    def generate_json(
        self, *, system_prompt, user_prompt, response_schema=None,
        max_output_tokens=None,
    ) -> dict:
        self.calls.append(
            {"system_prompt": system_prompt, "user_prompt": user_prompt,
             "response_schema": response_schema}
        )
        if self._exc is not None:
            raise self._exc
        return dict(self._response or {})


@pytest.fixture
def allocate_spy(monkeypatch):
    """Record every capacity the scheduler hands to PortfolioAllocator.allocate."""
    from app.services import recovery_scheduler as mod

    calls: list = []
    real = mod.PortfolioAllocator.allocate

    def spy(self, capacity=None):
        calls.append(capacity)
        return real(self, capacity)

    monkeypatch.setattr(mod.PortfolioAllocator, "allocate", spy)
    return calls


def _build(judge, **cfg_kw):
    return RecoveryScheduler(
        config=_cfg(**cfg_kw),
        provider_factory=_agent_factory,
        judgment_provider_factory=(lambda: judge) if judge is not None else None,
    )


# ---------------------------------------------------------------------------
# 1. disabled -> no Gemini call at all, behaviour unchanged
# ---------------------------------------------------------------------------


def test_judgment_disabled_makes_no_gemini_call(
    sched_db, make_open_events, pin_allocator_population, allocate_spy
):
    ids = make_open_events(amounts=[500000, 400000, 300000, 200000])
    pin_allocator_population(ids)
    judge = FakeJudge(response={"decision": "skip_this_cycle", "reason": "no"})

    sched = _build(judge, ai_judgment_enabled=False, max_auto_runs_per_cycle=2)
    rec = sched.run_cycle(db=sched_db, trigger="run_once")

    assert judge.calls == []                      # provider never invoked
    assert rec.ai_judgment["source"] == "disabled"
    assert rec.effective_cap == 2
    assert rec.events_triggered == 2              # exactly the pre-feature result
    assert allocate_spy == [2]


# ---------------------------------------------------------------------------
# 2. proceed_full_capacity -> identical to the disabled path
# ---------------------------------------------------------------------------


def test_proceed_full_capacity_runs_at_configured_cap(
    sched_db, make_open_events, pin_allocator_population, allocate_spy
):
    ids = make_open_events(amounts=[500000, 400000, 300000, 200000])
    pin_allocator_population(ids)
    judge = FakeJudge(response={"decision": "proceed_full_capacity",
                                "reason": "recent cycles look healthy"})

    rec = _build(judge, max_auto_runs_per_cycle=3).run_cycle(
        db=sched_db, trigger="run_once"
    )

    assert len(judge.calls) == 1
    assert rec.ai_judgment["decision"] == "proceed_full_capacity"
    assert rec.ai_judgment["source"] == "gemini"
    assert rec.ai_judgment["applied"] is False
    assert rec.effective_cap == 3
    assert rec.events_triggered == 3
    assert [t.recovery_event_id for t in rec.triggered] == ids[:3]
    assert allocate_spy == [3]


# ---------------------------------------------------------------------------
# 3. proceed_reduced_capacity below the cap -> effective = suggested
# ---------------------------------------------------------------------------


def test_reduced_capacity_below_cap_is_applied(
    sched_db, make_open_events, pin_allocator_population, allocate_spy
):
    ids = make_open_events(amounts=[500000, 400000, 300000, 200000, 100000])
    pin_allocator_population(ids)
    judge = FakeJudge(response={"decision": "proceed_reduced_capacity",
                                "suggested_capacity": 1,
                                "reason": "one recent cycle hit quota errors"})

    rec = _build(judge, max_auto_runs_per_cycle=3).run_cycle(
        db=sched_db, trigger="run_once"
    )

    assert rec.ai_judgment["decision"] == "proceed_reduced_capacity"
    assert rec.ai_judgment["suggested_capacity"] == 1
    assert rec.ai_judgment["effective_capacity"] == 1
    assert rec.ai_judgment["applied"] is True
    assert rec.effective_cap == 1
    assert rec.hard_cap == 3                       # the real ceiling is unchanged
    assert rec.events_triggered == 1
    assert [t.recovery_event_id for t in rec.triggered] == ids[:1]
    # the judgment only scales the number handed to the allocator; the allocator
    # then produces its own specific per-event skip reasons for the rest
    assert allocate_spy == [1]
    skipped_ids = {s.recovery_event_id for s in rec.skipped}
    assert set(ids[1:]).issubset(skipped_ids)
    for s in rec.skipped:
        assert s.reason and "not enough budget" not in s.reason.lower()
    # AI attribution is recorded on the cycle, transparently
    assert rec.ai_judgment["applied"] is True
    assert rec.ai_judgment["reason"] == "one recent cycle hit quota errors"


# ---------------------------------------------------------------------------
# 4. THE key test: a suggested number ABOVE the cap is clamped, never exceeds
# ---------------------------------------------------------------------------


def test_reduced_capacity_above_cap_is_clamped_never_exceeds(
    sched_db, make_open_events, pin_allocator_population, allocate_spy
):
    ids = make_open_events(amounts=[500000, 400000, 300000, 200000, 100000])
    pin_allocator_population(ids)
    judge = FakeJudge(response={"decision": "proceed_reduced_capacity",
                                "suggested_capacity": 99,
                                "reason": "adversarial-looking oversize suggestion"})

    rec = _build(judge, max_auto_runs_per_cycle=3).run_cycle(
        db=sched_db, trigger="run_once"
    )

    assert rec.ai_judgment["suggested_capacity"] == 99
    assert rec.ai_judgment["effective_capacity"] == 3   # clamped to the hard cap
    assert rec.effective_cap == 3
    assert rec.events_triggered == 3                    # never 99, never > cap
    assert len(rec.triggered) <= 3
    assert allocate_spy == [3]                          # allocator got the cap, not 99


def test_reduced_capacity_negative_suggestion_floors_at_one(
    sched_db, make_open_events, pin_allocator_population
):
    ids = make_open_events(amounts=[500000, 400000, 300000])
    pin_allocator_population(ids)
    judge = FakeJudge(response={"decision": "proceed_reduced_capacity",
                                "suggested_capacity": -5, "reason": "x"})
    rec = _build(judge, max_auto_runs_per_cycle=3).run_cycle(
        db=sched_db, trigger="run_once"
    )
    assert rec.effective_cap == 1
    assert rec.events_triggered == 1


# ---------------------------------------------------------------------------
# 5. skip_this_cycle -> allocator never called, zero runs, recorded
# ---------------------------------------------------------------------------


def test_skip_this_cycle_does_not_call_the_allocator(
    sched_db, make_open_events, pin_allocator_population, allocate_spy
):
    ids = make_open_events(amounts=[500000, 400000, 300000])
    pin_allocator_population(ids)
    judge = FakeJudge(response={"decision": "skip_this_cycle",
                                "reason": "every recent run failed on provider quota"})

    rec = _build(judge, max_auto_runs_per_cycle=3).run_cycle(
        db=sched_db, trigger="run_once"
    )

    assert allocate_spy == []                      # allocator.allocate NOT called
    assert rec.ai_judgment["decision"] == "skip_this_cycle"
    assert rec.ai_judgment["applied"] is True
    assert rec.effective_cap == 0
    assert rec.events_triggered == 0
    assert rec.triggered == []
    assert rec.events_considered == 0
    assert "skipped before allocation by AI judgment" in rec.allocation_reason
    assert "quota" in rec.allocation_reason
    # no agent runs, no interventions
    assert not sched_db.scalars(
        select(Intervention).where(Intervention.recovery_event_id.in_(ids))
    ).all()


# ---------------------------------------------------------------------------
# 6. fail-safe: call fails / times out / rate-limited -> proceed_full_capacity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "exc",
    [
        RateLimitedError("HTTP 429 quota exceeded"),
        MalformedResponseError("Gemini response was not valid JSON"),
        TimeoutError("read timed out"),
        RuntimeError("something unexpected"),
    ],
)
def test_failed_judgment_call_fails_safe_to_full_capacity(
    sched_db, make_open_events, pin_allocator_population, allocate_spy, exc
):
    ids = make_open_events(amounts=[500000, 400000, 300000, 200000])
    pin_allocator_population(ids)
    judge = FakeJudge(exc=exc)

    rec = _build(judge, max_auto_runs_per_cycle=3).run_cycle(
        db=sched_db, trigger="run_once"
    )

    assert rec.ai_judgment["source"] == "fail_safe_default"
    assert rec.ai_judgment["decision"] == "proceed_full_capacity"
    assert rec.ai_judgment["error"] is not None
    assert rec.effective_cap == 3
    assert rec.events_triggered == 3              # legitimate recovery NOT blocked
    assert allocate_spy == [3]
    assert rec.errors == []                       # no crash, cycle completed


# ---------------------------------------------------------------------------
# 7. malformed / unexpected output -> same fail-safe default
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "response",
    [
        {"decision": "banana", "reason": "unknown decision value"},
        {"reason": "no decision key at all"},
        {"decision": "proceed_reduced_capacity", "reason": "reduce but no number"},
        {"decision": "proceed_reduced_capacity", "suggested_capacity": "lots",
         "reason": "non-integer suggested capacity"},
        {},
    ],
)
def test_malformed_output_fails_safe_never_crashes(
    sched_db, make_open_events, pin_allocator_population, response
):
    ids = make_open_events(amounts=[500000, 400000, 300000])
    pin_allocator_population(ids)
    judge = FakeJudge(response=response)

    rec = _build(judge, max_auto_runs_per_cycle=3).run_cycle(
        db=sched_db, trigger="run_once"
    )

    assert rec.ai_judgment["source"] == "fail_safe_default"
    assert rec.ai_judgment["effective_capacity"] == 3
    assert rec.effective_cap == 3
    assert rec.events_triggered == 3


# ---------------------------------------------------------------------------
# 8. exactly one call per cycle; summary is built from cycle history
# ---------------------------------------------------------------------------


def test_exactly_one_call_per_cycle_and_summary_uses_history(
    sched_db, make_open_events, pin_allocator_population
):
    ids = make_open_events(amounts=[500000, 400000, 300000])
    pin_allocator_population(ids)
    judge = FakeJudge(response={"decision": "proceed_full_capacity", "reason": "ok"})
    sched = _build(judge, max_auto_runs_per_cycle=2)

    sched.run_cycle(db=sched_db, trigger="run_once")
    sched.run_cycle(db=sched_db, trigger="run_once")

    assert len(judge.calls) == 2                  # one bounded call per cycle
    # the second call's prompt carries the first cycle's outcome
    second_prompt = judge.calls[1]["user_prompt"]
    assert "recent_cycles" in second_prompt
    assert "run_status_breakdown" in second_prompt
    assert '"cycle_id": 1' in second_prompt
    # the structured-output schema is passed
    assert judge.calls[0]["response_schema"]["properties"]["decision"]["enum"]


# ---------------------------------------------------------------------------
# 9. decision + reason are visible in /status and run-once
# ---------------------------------------------------------------------------


def _http(method: str, path: str):
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
def singleton_with_judge(monkeypatch):
    from app.agent import runner
    from app.services import recovery_scheduler as mod

    monkeypatch.setattr(
        runner, "make_provider", lambda config=None: ReactiveProvider()
    )
    monkeypatch.setattr(mod.scheduler, "_history", deque(maxlen=20))
    monkeypatch.setattr(mod.scheduler, "_cycle_seq", 0)
    monkeypatch.setattr(mod.scheduler, "_cycles_completed", 0)
    monkeypatch.setattr(mod.scheduler, "_provider_factory", None)
    return mod.scheduler


def test_judgment_shows_in_status_and_run_once(
    singleton_with_judge, monkeypatch, sched_db, make_open_events,
    pin_allocator_population,
):
    ids = make_open_events(amounts=[500000, 400000, 300000, 200000])
    pin_allocator_population(ids)
    judge = FakeJudge(response={"decision": "proceed_reduced_capacity",
                                "suggested_capacity": 2,
                                "reason": "throttling after repeated 429s"})
    monkeypatch.setattr(
        singleton_with_judge, "_config",
        _cfg(enabled=False, ai_judgment_enabled=True, max_auto_runs_per_cycle=3),
    )
    monkeypatch.setattr(
        singleton_with_judge, "_judgment_provider_factory", lambda: judge
    )

    sc, body = _http("POST", "/api/v1/scheduler/run-once")
    assert sc == 200
    cyc = body["cycle"]
    assert cyc["ai_judgment"]["decision"] == "proceed_reduced_capacity"
    assert cyc["ai_judgment"]["effective_capacity"] == 2
    assert cyc["ai_judgment"]["reason"] == "throttling after repeated 429s"
    assert cyc["effective_cap"] == 2
    assert cyc["events_triggered"] == 2

    sc2, st = _http("GET", "/api/v1/scheduler/status")
    assert sc2 == 200
    assert st["config"]["ai_judgment_enabled"] is True
    assert st["cycle_history"][0]["ai_judgment"]["reason"] == (
        "throttling after repeated 429s"
    )


def test_status_config_exposes_judgment_flag_off_by_default(singleton_with_judge):
    sc, body = _http("GET", "/api/v1/scheduler/status")
    assert sc == 200
    assert "ai_judgment_enabled" in body["config"]


# ---------------------------------------------------------------------------
# 10. GeminiProvider.generate_json -- the single structured request
# ---------------------------------------------------------------------------


def test_generate_json_parses_a_structured_response(monkeypatch):
    from app.agent.config import AgentConfig
    from app.agent.providers.gemini import GeminiProvider

    prov = GeminiProvider(AgentConfig(
        api_key="k", model="m", max_turns=1, timeout_seconds=5.0,
    ))
    captured = {}

    def fake_post_once(payload: bytes):
        captured["body"] = json.loads(payload)
        return {"candidates": [{"content": {"parts": [
            {"text": '{"decision": "skip_this_cycle", "reason": "quiet"}'}
        ]}}]}

    monkeypatch.setattr(prov, "_post_once", fake_post_once)
    out = prov.generate_json(
        system_prompt="sys", user_prompt="hi",
        response_schema={"type": "object"},
    )
    assert out == {"decision": "skip_this_cycle", "reason": "quiet"}
    # non-tool-calling: no tools / toolConfig in the request body
    assert "tools" not in captured["body"]
    assert "toolConfig" not in captured["body"]
    assert captured["body"]["generationConfig"]["responseMimeType"] == (
        "application/json"
    )


def test_generate_json_rejects_non_json_text(monkeypatch):
    from app.agent.config import AgentConfig
    from app.agent.providers.gemini import GeminiProvider

    prov = GeminiProvider(AgentConfig(
        api_key="k", model="m", max_turns=1, timeout_seconds=5.0,
    ))
    monkeypatch.setattr(
        prov, "_post_once",
        lambda payload: {"candidates": [{"content": {"parts": [
            {"text": "sorry, I cannot help with that"}
        ]}}]},
    )
    with pytest.raises(MalformedResponseError):
        prov.generate_json(system_prompt="s", user_prompt="u")
