"""POST /api/v1/webhooks/razorpay through the real FastAPI ASGI app."""
from __future__ import annotations

import asyncio
import json

import pytest
from sqlalchemy import select

from app.models.outcome import Outcome
from tests.razorpay.fakes import (
    make_config,
    payment_link_paid_event,
    signed_webhook,
)


@pytest.fixture(autouse=True)
def _webhook_secret(monkeypatch):
    # give the endpoint a webhook secret without touching .env
    from app.integrations.razorpay import config as rzp_config
    from tests.razorpay.fakes import TEST_WEBHOOK_SECRET

    monkeypatch.setattr(
        rzp_config.settings, "RAZORPAY_WEBHOOK_SECRET", TEST_WEBHOOK_SECRET,
        raising=False,
    )


def _post(path: str, body: bytes, headers: dict[str, str]):
    from app.main import app

    hdrs = [(k.lower().encode(), v.encode()) for k, v in headers.items()]
    hdrs.append((b"host", b"t"))
    scope = {
        "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
        "method": "POST", "scheme": "http", "path": path, "raw_path": path.encode(),
        "query_string": b"", "root_path": "", "headers": hdrs,
        "client": ("t", 1), "server": ("t", 80),
    }
    inbox = [{"type": "http.request", "body": body, "more_body": False}]
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


def test_endpoint_rejects_bad_signature(rzp_db, linked_intervention):
    body = payment_link_paid_event(
        payment_link_id=linked_intervention["payment_link_id"],
        reference_id=linked_intervention["reference_id"], amount=99900,
    )
    raw = json.dumps(body).encode()
    sc, resp = _post("/api/v1/webhooks/razorpay", raw, {
        "content-type": "application/json",
        "x-razorpay-signature": "not-a-real-signature",
        "x-razorpay-event-id": "evt_pytest_api_bad",
    })
    assert sc == 400 and resp["status"] == "invalid_signature"


def test_endpoint_processes_paid_and_is_idempotent(rzp_db, linked_intervention):
    body = payment_link_paid_event(
        payment_link_id=linked_intervention["payment_link_id"],
        reference_id=linked_intervention["reference_id"], amount=99900,
    )
    raw, sig = signed_webhook(body)
    headers = {
        "content-type": "application/json",
        "x-razorpay-signature": sig,
        "x-razorpay-event-id": "evt_pytest_api_paid",
    }
    sc1, r1 = _post("/api/v1/webhooks/razorpay", raw, headers)
    sc2, r2 = _post("/api/v1/webhooks/razorpay", raw, headers)

    assert sc1 == 200 and r1["status"] == "ok" and r1["payment_recovered"] is True
    assert sc2 == 200 and r2["status"] == "already_processed"

    outcomes = rzp_db.scalars(
        select(Outcome).where(
            Outcome.intervention_id == linked_intervention["intervention_id"]
        )
    ).all()
    assert len(outcomes) == 1
