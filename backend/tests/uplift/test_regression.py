"""Phase 10: the uplift work must not change any existing behaviour."""
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


def test_all_prior_routes_still_present_plus_uplift():
    from app.main import app

    paths = set(app.openapi()["paths"])
    for p in (
        "/health",
        "/api/v1/analytics/summary",
        "/api/v1/orchestrator/run",
        "/api/v1/ml/model",
        "/api/v1/ml/recovery-events/{recovery_event_id}/action-scores",
        "/api/v1/uplift/model",
        "/api/v1/uplift/recovery-events/{recovery_event_id}/uplift-scores",
    ):
        assert p in paths, p


def test_rules_policy_output_unchanged():
    """The uplift work adds PolicyContext nothing and touches no rules code:
    the rules policy must still rank purely by expected value, descending."""
    from app.services.recovery_policy import PolicyContext, RulesBasedRecoveryPolicy

    ctx = PolicyContext(
        failure_reason="bank_timeout", amount_paise=49900, priority=1,
        is_control=False, attempt_number=1, prior_action_types=[],
        customer_successful_payments=7, customer_failed_payments=3,
    )
    d = RulesBasedRecoveryPolicy().decide(ctx)
    assert len(d.candidates) == 4
    assert [c.score for c in d.candidates] == sorted(
        (c.score for c in d.candidates), reverse=True
    )
    # adding recovery_event_id to the context must not change the ranking
    d2 = RulesBasedRecoveryPolicy().decide(
        PolicyContext(
            failure_reason="bank_timeout", amount_paise=49900, priority=1,
            is_control=False, attempt_number=1, prior_action_types=[],
            customer_successful_payments=7, customer_failed_payments=3,
            recovery_event_id=42,
        )
    )
    assert [c.action_type for c in d.candidates] == [
        c.action_type for c in d2.candidates
    ]


def test_default_policy_is_still_rules():
    from app.services.policy_factory import DEFAULT_POLICY, resolve_policy
    from app.database import SessionLocal
    from app.services.recovery_policy import RulesBasedRecoveryPolicy

    assert DEFAULT_POLICY == "rules"
    s = SessionLocal()
    try:
        assert isinstance(resolve_policy(None, s), RulesBasedRecoveryPolicy)
    finally:
        s.close()


def test_ml_policy_still_resolves():
    from app.database import SessionLocal
    from app.services.ml_recovery_policy import MLRecoveryPolicy
    from app.services.policy_factory import resolve_policy

    s = SessionLocal()
    try:
        assert isinstance(resolve_policy("ml", s), MLRecoveryPolicy)
    finally:
        s.close()


def test_bad_policy_still_400_via_factory():
    import pytest
    from app.database import SessionLocal
    from app.services.policy_factory import resolve_policy

    s = SessionLocal()
    try:
        with pytest.raises(ValueError):
            resolve_policy("nonsense", s)
    finally:
        s.close()


def test_analytics_summary_unaffected():
    sc, body = _get("/api/v1/analytics/summary")
    assert sc == 200
    assert "overall_recovery_rate" in body


def test_predictive_feature_sql_still_leak_free():
    from app.ml.features.point_in_time import PointInTimeFeatureExtractor
    from app.ml.features.schema import FORBIDDEN_LEAKAGE_SOURCES

    for mode in ("training", "inference"):
        sql = PointInTimeFeatureExtractor.feature_sql(mode).lower()
        for tok in FORBIDDEN_LEAKAGE_SOURCES:
            assert tok.lower() not in sql
