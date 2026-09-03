"""The ML work must not break anything that already worked."""
from __future__ import annotations

import asyncio
import json


def _get(path: str):
    from app.main import app

    p, _, q = path.partition("?")
    scope = {
        "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
        "method": "GET", "scheme": "http", "path": p, "raw_path": p.encode(),
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


def test_app_imports_and_routes_present():
    from app.main import app

    paths = set(app.openapi()["paths"])
    for p in (
        "/health",
        "/api/v1/analytics/summary",
        "/api/v1/recovery-events/{recovery_event_id}/orchestrate",
        "/api/v1/orchestrator/run",
        "/api/v1/ml/model",
        "/api/v1/ml/recovery-events/{recovery_event_id}/action-scores",
    ):
        assert p in paths, p


def test_health_ok():
    sc, body = _get("/health")
    assert sc == 200 and body["status"] == "healthy"


def test_analytics_summary_still_works():
    sc, body = _get("/api/v1/analytics/summary")
    assert sc == 200
    assert "overall_recovery_rate" in body
    assert body["total_recovery_events"] >= 0


def test_rules_policy_unchanged_by_context_field():
    """Adding PolicyContext.recovery_event_id must not change rules behaviour."""
    from app.services.recovery_policy import PolicyContext, RulesBasedRecoveryPolicy

    ctx_kwargs = dict(
        failure_reason="insufficient_funds",
        amount_paise=50000,
        priority=1,
        is_control=False,
        attempt_number=1,
        prior_action_types=[],
        customer_successful_payments=8,
        customer_failed_payments=2,
    )
    a = RulesBasedRecoveryPolicy().decide(PolicyContext(**ctx_kwargs))
    b = RulesBasedRecoveryPolicy().decide(
        PolicyContext(**ctx_kwargs, recovery_event_id=123)
    )
    assert [c.action_type for c in a.candidates] == [c.action_type for c in b.candidates]
    assert a.candidates[0].score == b.candidates[0].score
