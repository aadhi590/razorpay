"""RazorpayClient -- exercised through the real client code, fake transport."""
from __future__ import annotations

import base64
import json

import pytest

from app.integrations.razorpay.client import RazorpayClient
from app.integrations.razorpay.exceptions import (
    RazorpayAuthError,
    RazorpayMalformedResponse,
    RazorpayRateLimitError,
    RazorpayTransientError,
    RazorpayValidationError,
)
from app.integrations.razorpay.schemas import PaymentLinkCreateRequest
from tests.razorpay.fakes import FakeRazorpayTransport, make_config


def _client(transport=None, **cfg):
    return RazorpayClient(make_config(**cfg), transport=transport or FakeRazorpayTransport())


def _req(**kw):
    base = dict(
        amount=99900, currency="INR", reference_id="recovery-1-1",
        description="test recovery",
    )
    base.update(kw)
    return PaymentLinkCreateRequest(**base)


def test_create_payment_link_success():
    t = FakeRazorpayTransport()
    link = _client(t).create_payment_link(_req())
    assert link.id.startswith("plink_TEST")
    assert link.status == "created"
    assert link.amount == 99900
    assert link.short_url


def test_request_carries_correct_amount_currency_reference():
    t = FakeRazorpayTransport()
    _client(t).create_payment_link(_req(amount=49900, reference_id="recovery-7-9"))
    body = t.create_calls[0]["body"]
    assert body["amount"] == 49900          # smallest unit, unchanged
    assert body["currency"] == "INR"
    assert body["reference_id"] == "recovery-7-9"
    assert body["notify"] == {"sms": False, "email": False}


def test_basic_auth_header_built_from_key_pair():
    t = FakeRazorpayTransport()
    _client(t).create_payment_link(_req())
    auth = t.calls[0]["headers"]["Authorization"]
    assert auth.startswith("Basic ")
    decoded = base64.b64decode(auth.split(" ", 1)[1]).decode()
    assert decoded == "rzp_test_fake0000000000:fake_secret_value_not_real"


def test_transient_error_is_retried_then_succeeds():
    t = FakeRazorpayTransport()
    t.queue(503, {"error": {"code": "SERVER", "description": "try later"}})
    t.queue(200, {"id": "plink_ok", "status": "created", "amount": 100,
                  "amount_paid": 0, "currency": "INR", "short_url": "u"})
    link = _client(t).create_payment_link(_req())
    assert link.id == "plink_ok"
    assert len([c for c in t.calls]) == 2      # retried once


def test_transient_error_gives_up_after_bounded_retries():
    t = FakeRazorpayTransport()
    for _ in range(5):
        t.queue(500, {"error": {"code": "X", "description": "down"}})
    with pytest.raises(RazorpayTransientError):
        _client(t).create_payment_link(_req())
    assert len(t.calls) == 3                   # 1 + max_transient_retries(2)


def test_deterministic_4xx_is_not_retried():
    t = FakeRazorpayTransport()
    t.queue(400, {"error": {"code": "BAD_REQUEST_ERROR",
                            "description": "reference id already exists"}})
    with pytest.raises(RazorpayValidationError):
        _client(t).create_payment_link(_req())
    assert len(t.calls) == 1


def test_authentication_failure():
    t = FakeRazorpayTransport()
    t.queue(401, {"error": {"code": "BAD_REQUEST_ERROR",
                            "description": "Authentication failed"}})
    with pytest.raises(RazorpayAuthError):
        _client(t).create_payment_link(_req())
    assert len(t.calls) == 1


def test_rate_limit_retried_once_then_raises():
    t = FakeRazorpayTransport()
    t.queue(429, {"error": {"code": "RATE", "description": "slow down"}})
    t.queue(429, {"error": {"code": "RATE", "description": "slow down"}})
    with pytest.raises(RazorpayRateLimitError):
        _client(t).create_payment_link(_req())
    assert len(t.calls) == 2                   # 1 + max_rate_limit_retries(1)


def test_malformed_success_body():
    t = FakeRazorpayTransport()
    t.queue_raw(200, b"<html>not json</html>")
    with pytest.raises(RazorpayMalformedResponse):
        _client(t).create_payment_link(_req())


def test_secret_never_appears_in_exception():
    t = FakeRazorpayTransport()
    t.queue(401, {"error": {"code": "X", "description": "Authentication failed"}})
    try:
        _client(t).create_payment_link(_req())
        assert False
    except RazorpayAuthError as exc:
        blob = str(exc) + exc.summary()
        assert "fake_secret_value_not_real" not in blob
        assert "Basic " not in blob


def test_https_is_enforced():
    from app.integrations.razorpay.exceptions import RazorpayValidationError as V
    with pytest.raises(V):
        RazorpayClient(make_config(base_url="http://api.razorpay.com/v1"))


def test_request_id_preserved_on_error():
    t = FakeRazorpayTransport()
    t.queue(400, {"error": {"code": "BAD", "description": "nope"}},
            headers={"X-Razorpay-Request-Id": "req_abc123"})
    try:
        _client(t).create_payment_link(_req())
        assert False
    except RazorpayValidationError as exc:
        assert exc.request_id == "req_abc123"
