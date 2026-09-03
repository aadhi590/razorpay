"""Razorpay Test Mode integration adapter.

Owns authentication, HTTP, retries, error mapping, Payment Link create/fetch,
and webhook signature verification. Contains **no** recovery business logic --
the recovery services call into this package, never the reverse.

TEST MODE ONLY in this stage: :meth:`RazorpayConfig.require_ready` refuses any
non-test-mode configuration before a single byte leaves the process.
"""
from __future__ import annotations

from app.integrations.razorpay.client import RazorpayClient
from app.integrations.razorpay.config import RazorpayConfig
from app.integrations.razorpay.exceptions import (
    RazorpayAuthError,
    RazorpayConfigError,
    RazorpayError,
    RazorpayMalformedResponse,
    RazorpayRateLimitError,
    RazorpayTransientError,
    RazorpayValidationError,
    WebhookSignatureError,
)
from app.integrations.razorpay.schemas import (
    PaymentLink,
    PaymentLinkCreateRequest,
    PaymentLinkCustomer,
    WebhookEnvelope,
)
from app.integrations.razorpay.webhooks import (
    EVENT_ID_HEADER,
    SIGNATURE_HEADER,
    compute_signature,
    verify_signature,
)

__all__ = [
    "RazorpayClient",
    "RazorpayConfig",
    "PaymentLink",
    "PaymentLinkCreateRequest",
    "PaymentLinkCustomer",
    "WebhookEnvelope",
    "verify_signature",
    "compute_signature",
    "SIGNATURE_HEADER",
    "EVENT_ID_HEADER",
    "RazorpayError",
    "RazorpayConfigError",
    "RazorpayAuthError",
    "RazorpayValidationError",
    "RazorpayRateLimitError",
    "RazorpayTransientError",
    "RazorpayMalformedResponse",
    "WebhookSignatureError",
]
