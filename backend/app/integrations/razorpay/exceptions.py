"""Razorpay integration error hierarchy.

Every error the recovery layer might need to branch on has a distinct type. No
error message ever contains the key secret, the webhook secret, or an
Authorization header -- the client redacts those before constructing an
exception.
"""
from __future__ import annotations


class RazorpayError(Exception):
    """Base class for every Razorpay integration failure."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        razorpay_error_code: str | None = None,
        request_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.razorpay_error_code = razorpay_error_code
        self.request_id = request_id

    def summary(self) -> str:
        """A short, secret-free string safe to hand to the agent / audit log."""
        bits = [type(self).__name__]
        if self.status_code is not None:
            bits.append(f"http={self.status_code}")
        if self.razorpay_error_code:
            bits.append(f"code={self.razorpay_error_code}")
        if self.request_id:
            bits.append(f"request_id={self.request_id}")
        return " ".join(bits) + f": {self.args[0]}"


class RazorpayConfigError(RazorpayError):
    """Test-mode configuration is missing or is not a valid test-mode setup.

    Raised before any network call, so a misconfiguration can never reach the
    Razorpay API.
    """


class RazorpayAuthError(RazorpayError):
    """HTTP 401 / 403 -- bad key id/secret. Never retried."""


class RazorpayValidationError(RazorpayError):
    """HTTP 400 / 422 -- a deterministic request error (bad amount, duplicate
    reference id, ...). Never retried."""


class RazorpayRateLimitError(RazorpayError):
    """HTTP 429. Retried at most once, with backoff."""


class RazorpayTransientError(RazorpayError):
    """HTTP 5xx / connection reset / timeout. Retried a small bounded number of
    times."""


class RazorpayMalformedResponse(RazorpayError):
    """The response was not valid JSON or lacked the expected fields."""


class WebhookSignatureError(RazorpayError):
    """The webhook signature header is missing or does not match."""
