"""The /api/v1/agent endpoint + proof the existing endpoints are untouched.

Gemini is mocked by monkeypatching the provider the runner constructs.
"""
from __future__ import annotations

import asyncio
import json

import pytest
from sqlalchemy import select

from app.models.interventions import Intervention
from tests.agent.fakes import ReactiveProvider


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
def mock_gemini(monkeypatch):
    monkeypatch.setattr(
        "app.agent.runner.GeminiProvider", lambda config=None: ReactiveProvider()
    )


def test_agent_endpoint_dry_run(mock_gemini, fresh_event, db_session_agent):
    sc, body = _call(
        "POST", f"/api/v1/agent/recovery-events/{fresh_event}/run?dry_run=true"
    )
    assert sc == 200
    assert body["agent"] == "gemini"
    assert body["dry_run"] is True
    assert body["turns_used"] >= 2
    # full turn-by-turn trace is in the response, not just the final answer
    tools = [t["tool"] for t in body["tool_trace"]]
    assert tools[0] == "get_recovery_event_context"
    assert body["status"] in {"completed", "escalated"}
    # dry run created no intervention
    assert not db_session_agent.scalars(
        select(Intervention).where(Intervention.recovery_event_id == fresh_event)
    ).all()


def test_agent_endpoint_404_for_missing_event(mock_gemini):
    sc, body = _call("POST", "/api/v1/agent/recovery-events/999999999/run?dry_run=true")
    assert sc == 404


def test_agent_endpoint_fails_safe_without_api_key(monkeypatch, fresh_event):
    from app.agent import config as agent_config

    monkeypatch.setattr(agent_config.settings, "GEMINI_API_KEY", None, raising=False)
    sc, body = _call(
        "POST", f"/api/v1/agent/recovery-events/{fresh_event}/run?dry_run=true"
    )
    assert sc == 200
    assert body["status"] == "failed_safe"
    assert body["stop_reason"] == "quota_or_api_failure"


def test_existing_orchestrator_endpoints_unchanged(mock_gemini):
    from app.main import app

    paths = set(app.openapi()["paths"])
    for p in (
        "/api/v1/recovery-events/{recovery_event_id}/orchestrate",
        "/api/v1/orchestrator/run",
        "/api/v1/ml/recovery-events/{recovery_event_id}/action-scores",
        "/api/v1/uplift/recovery-events/{recovery_event_id}/uplift-scores",
        "/api/v1/agent/recovery-events/{recovery_event_id}/run",
    ):
        assert p in paths

    sc, body = _call("GET", "/api/v1/analytics/summary")
    assert sc == 200 and "overall_recovery_rate" in body
