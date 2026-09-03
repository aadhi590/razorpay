"""Razorpay webhook signature verification.

Razorpay signs each webhook with::

    signature = HMAC_SHA256(key=webhook_secret, msg=raw_request_body).hexdigest()

and sends it in the ``X-Razorpay-Signature`` header (plus ``X-Razorpay-Event-Id``
for idempotency). This is the exact scheme Razorpay's own SDK
(``Utility.verify_webhook_signature``) implements.

The raw request body MUST be used byte-for-byte -- re-serialising the parsed
JSON would change the bytes and break verification.
"""
from __future__ import annotations

import hashlib
import hmac

from app.integrations.razorpay.exceptions import WebhookSignatureError

SIGNATURE_HEADER = "X-Razorpay-Signature"
EVENT_ID_HEADER = "X-Razorpay-Event-Id"


def compute_signature(raw_body: bytes, secret: str) -> str:
    return hmac.new(
        secret.encode("utf-8"), raw_body, hashlib.sha256
    ).hexdigest()


def verify_signature(raw_body: bytes, signature: str | None, secret: str) -> None:
    """Raise :class:`WebhookSignatureError` unless the signature matches."""
    if not signature:
        raise WebhookSignatureError("missing X-Razorpay-Signature header")
    expected = compute_signature(raw_body, secret)
    if not hmac.compare_digest(expected, signature):
        raise WebhookSignatureError("webhook signature mismatch")


def is_valid_signature(raw_body: bytes, signature: str | None, secret: str) -> bool:
    try:
        verify_signature(raw_body, signature, secret)
        return True
    except WebhookSignatureError:
        return False
