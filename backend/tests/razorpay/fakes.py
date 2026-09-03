"""Test doubles for the Razorpay integration -- no network, ever.

* :class:`FakeRazorpayTransport` plugs into the *real* ``RazorpayClient`` via its
  ``transport=`` hook, so auth-header construction, retry logic and response
  parsing are all exercised for real; only the socket is replaced.
* :func:`signed_webhook` builds a body + a real HMAC-SHA256 signature so the
  webhook signature path is exercised with real crypto.
"""
from __future__ import annotations

import json
from typing import Any

from app.integrations.razorpay.config import RazorpayConfig
from app.integrations.razorpay.webhooks import compute_signature

TEST_WEBHOOK_SECRET = "whsec_test_only_not_real"


def make_config(**overrides: Any) -> RazorpayConfig:
    base = dict(
        key_id="rzp_test_fake0000000000",
        key_secret="fake_secret_value_not_real",
        webhook_secret=TEST_WEBHOOK_SECRET,
        base_url="https://api.razorpay.com/v1",
        test_mode=True,
        timeout_seconds=5.0,
        payment_link_expiry_minutes=60,
        max_transient_retries=2,
        max_rate_limit_retries=1,
        backoff_base_seconds=0.0,
        backoff_max_seconds=0.0,
    )
    base.update(overrides)
    return RazorpayConfig(**base)


class FakeRazorpayTransport:
    """Callable matching ``RazorpayClient`` transport signature."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.responses: list[Any] = []  # queue of (status, headers, bytes) | Exception
        self.auto = True
        self._n = 0
        self.created_links: list[dict[str, Any]] = []

    def queue(self, status: int, body: dict[str, Any], headers: dict | None = None) -> None:
        self.responses.append((status, headers or {}, json.dumps(body).encode()))

    def queue_raw(self, status: int, raw: bytes, headers: dict | None = None) -> None:
        self.responses.append((status, headers or {}, raw))

    def queue_exc(self, exc: Exception) -> None:
        self.responses.append(exc)

    @property
    def create_calls(self) -> list[dict[str, Any]]:
        return [c for c in self.calls
                if c["method"] == "POST" and c["url"].endswith("/payment_links")]

    def __call__(self, method, url, headers, body):
        parsed_body = json.loads(body) if body else None
        self.calls.append(
            {"method": method, "url": url, "headers": dict(headers), "body": parsed_body}
        )
        if self.responses:
            item = self.responses.pop(0)
            if isinstance(item, BaseException):
                raise item
            return item

        if not self.auto:
            return (500, {}, b'{"error":{"code":"SERVER","description":"no canned response"}}')

        if method == "POST" and url.endswith("/payment_links"):
            self._n += 1
            resp = {
                "id": f"plink_TEST{self._n:04d}",
                "entity": "payment_link",
                "status": "created",
                "amount": parsed_body["amount"],
                "amount_paid": 0,
                "currency": parsed_body.get("currency", "INR"),
                "reference_id": parsed_body["reference_id"],
                "short_url": f"https://rzp.io/i/TEST{self._n:04d}",
                "notes": parsed_body.get("notes", {}),
            }
            self.created_links.append(resp)
            return (200, {"X-Razorpay-Request-Id": "req_testfake"},
                    json.dumps(resp).encode())

        if method == "GET" and "/payment_links/" in url:
            plid = url.rsplit("/", 1)[1].split("?")[0]
            existing = next((l for l in self.created_links if l["id"] == plid), None)
            body_out = existing or {
                "id": plid, "status": "created", "amount": 50000,
                "amount_paid": 0, "currency": "INR",
                "short_url": "https://rzp.io/i/x",
            }
            return (200, {}, json.dumps(body_out).encode())

        if method == "GET" and "payment_links?" in url:
            return (200, {}, json.dumps({"payment_links": []}).encode())

        return (404, {}, b'{"error":{"code":"NOT_FOUND","description":"unknown"}}')


def payment_link_paid_event(
    *,
    payment_link_id: str,
    reference_id: str,
    amount: int,
    payment_id: str = "pay_TESTpaid0001",
    event: str = "payment_link.paid",
    status: str = "paid",
) -> dict[str, Any]:
    return {
        "entity": "event",
        "account_id": "acc_TESTfake",
        "event": event,
        "contains": ["payment_link", "payment"],
        "payload": {
            "payment_link": {
                "entity": {
                    "id": payment_link_id,
                    "reference_id": reference_id,
                    "status": status,
                    "amount": amount,
                    "amount_paid": amount if status == "paid" else 0,
                    "currency": "INR",
                }
            },
            "payment": {
                "entity": {
                    "id": payment_id,
                    "amount": amount,
                    "currency": "INR",
                    "status": "captured",
                }
            },
        },
        "created_at": 1_759_000_000,
    }


def signed_webhook(
    body: dict[str, Any], secret: str = TEST_WEBHOOK_SECRET
) -> tuple[bytes, str]:
    """Return ``(raw_body_bytes, signature_hex)`` for a webhook body."""
    raw = json.dumps(body).encode("utf-8")
    return raw, compute_signature(raw, secret)
