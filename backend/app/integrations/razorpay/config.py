"""Razorpay configuration + the hard TEST MODE guard.

``RazorpayConfig.from_settings()`` reads the typed fields already on
``app.config.Settings`` (added the same way the Gemini settings were -- explicit
optional fields, ``extra="forbid"`` untouched). ``require_ready()`` is called by
the execution service before *any* live call and refuses to proceed unless the
configuration is a complete, explicit **test-mode** setup.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.config import settings
from app.integrations.razorpay.exceptions import RazorpayConfigError

_TEST_KEY_PREFIX = "rzp_test_"


@dataclass(frozen=True)
class RazorpayConfig:
    key_id: str | None
    key_secret: str | None
    webhook_secret: str | None
    base_url: str
    test_mode: bool
    timeout_seconds: float
    payment_link_expiry_minutes: int

    # client-internal, conservative
    max_transient_retries: int = 2
    max_rate_limit_retries: int = 1
    backoff_base_seconds: float = 0.5
    backoff_max_seconds: float = 4.0

    @classmethod
    def from_settings(cls) -> "RazorpayConfig":
        return cls(
            key_id=settings.RAZORPAY_KEY_ID,
            key_secret=settings.RAZORPAY_KEY_SECRET,
            webhook_secret=settings.RAZORPAY_WEBHOOK_SECRET,
            base_url=settings.RAZORPAY_BASE_URL.rstrip("/"),
            test_mode=bool(settings.RAZORPAY_TEST_MODE),
            timeout_seconds=float(settings.RAZORPAY_TIMEOUT_SECONDS),
            payment_link_expiry_minutes=max(
                16, int(settings.RAZORPAY_PAYMENT_LINK_EXPIRY_MINUTES)
            ),
        )

    # -- introspection (never returns a secret value) ------------------
    @property
    def has_api_credentials(self) -> bool:
        return bool((self.key_id or "").strip() and (self.key_secret or "").strip())

    @property
    def has_webhook_secret(self) -> bool:
        return bool((self.webhook_secret or "").strip())

    @property
    def looks_like_test_key(self) -> bool:
        return (self.key_id or "").startswith(_TEST_KEY_PREFIX)

    @property
    def is_ready(self) -> bool:
        return (
            self.test_mode
            and self.has_api_credentials
            and self.looks_like_test_key
        )

    def status(self) -> dict[str, bool | str]:
        """Secret-free snapshot for the inspection endpoint / audit trail."""
        return {
            "test_mode": self.test_mode,
            "base_url": self.base_url,
            "key_id_configured": self.has_api_credentials,
            "key_id_is_test_key": self.looks_like_test_key,
            "webhook_secret_configured": self.has_webhook_secret,
            "ready_for_live_calls": self.is_ready,
        }

    # -- the guard --------------------------------------------------
    def require_ready(self) -> None:
        if not self.test_mode:
            raise RazorpayConfigError(
                "RAZORPAY_TEST_MODE is not true; this stage refuses any "
                "non-test-mode Razorpay path"
            )
        if not self.has_api_credentials:
            raise RazorpayConfigError(
                "RAZORPAY_KEY_ID / RAZORPAY_KEY_SECRET are not configured"
            )
        if not self.looks_like_test_key:
            raise RazorpayConfigError(
                "RAZORPAY_KEY_ID is not a test-mode key (expected an "
                f"'{_TEST_KEY_PREFIX}' prefix); refusing to proceed"
            )

    def require_webhook_secret(self) -> str:
        if not self.has_webhook_secret:
            raise RazorpayConfigError("RAZORPAY_WEBHOOK_SECRET is not configured")
        return self.webhook_secret  # type: ignore[return-value]
