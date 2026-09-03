"""Minimal, robust Razorpay REST client (stdlib only).

Same posture as the Gemini provider: no SDK dependency. The official
``razorpay`` Python SDK is not installed and pulls ``requests`` + others; the
handful of calls we need (create / fetch a payment link) are a few lines of
``urllib`` and carry zero dependency risk.

Security:
* HTTPS is enforced -- a non-``https`` base URL raises before any call.
* Basic auth (key id : key secret) is built per-request; the Authorization
  header is **redacted** from every log line and every exception.
* Razorpay's ``X-Razorpay-Request-Id`` is preserved on errors for diagnostics.

Retries: bounded, and only for genuinely transient failures (HTTP 5xx, 429,
connection reset, timeout). Deterministic 4xx (validation, auth) are never
retried.

Testability: pass ``transport=`` -- a callable ``(method, url, headers, body)
-> (status, headers, body_bytes)`` -- to exercise the client with no network.
"""
from __future__ import annotations

import base64
import json
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable

from app.integrations.razorpay.config import RazorpayConfig
from app.integrations.razorpay.exceptions import (
    RazorpayAuthError,
    RazorpayMalformedResponse,
    RazorpayRateLimitError,
    RazorpayTransientError,
    RazorpayValidationError,
)
from app.integrations.razorpay.schemas import PaymentLink, PaymentLinkCreateRequest

Transport = Callable[[str, str, dict[str, str], bytes | None], tuple[int, dict[str, str], bytes]]

_REDACTED = "***redacted***"


def _redact(headers: dict[str, str]) -> dict[str, str]:
    return {
        k: (_REDACTED if k.lower() in {"authorization", "x-razorpay-signature"} else v)
        for k, v in headers.items()
    }


class RazorpayClient:
    def __init__(
        self,
        config: RazorpayConfig | None = None,
        *,
        transport: Transport | None = None,
    ) -> None:
        self.config = config or RazorpayConfig.from_settings()
        self._transport = transport
        if not self.config.base_url.startswith("https://"):
            raise RazorpayValidationError(
                f"RAZORPAY_BASE_URL must be https:// (got {self.config.base_url!r})"
            )

    # -- public API --------------------------------------------------
    def create_payment_link(self, request: PaymentLinkCreateRequest) -> PaymentLink:
        data = self._request("POST", "/payment_links", body=request.to_body())
        return PaymentLink.from_api(data)

    def fetch_payment_link(self, payment_link_id: str) -> PaymentLink:
        data = self._request("GET", f"/payment_links/{payment_link_id}")
        return PaymentLink.from_api(data)

    def find_payment_link_by_reference(self, reference_id: str) -> PaymentLink | None:
        q = urllib.parse.urlencode({"reference_id": reference_id})
        data = self._request("GET", f"/payment_links?{q}")
        items = data.get("payment_links") or []
        return PaymentLink.from_api(items[0]) if items else None

    # -- transport --------------------------------------------------
    def _auth_header(self) -> str:
        raw = f"{self.config.key_id}:{self.config.key_secret}".encode("utf-8")
        return "Basic " + base64.b64encode(raw).decode("ascii")

    def _request(
        self, method: str, path: str, *, body: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        url = f"{self.config.base_url}{path}"
        headers = {
            "Authorization": self._auth_header(),
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        payload = json.dumps(body).encode("utf-8") if body is not None else None

        cfg = self.config
        transient_left = cfg.max_transient_retries
        rate_left = cfg.max_rate_limit_retries
        attempt = 0
        while True:
            attempt += 1
            try:
                status, resp_headers, raw = self._send(method, url, headers, payload)
                return self._parse(status, resp_headers, raw)
            except RazorpayRateLimitError:
                if rate_left <= 0:
                    raise
                rate_left -= 1
                time.sleep(self._backoff(attempt))
            except RazorpayTransientError:
                if transient_left <= 0:
                    raise
                transient_left -= 1
                time.sleep(self._backoff(attempt))

    def _backoff(self, attempt: int) -> float:
        return min(
            self.config.backoff_base_seconds * (2 ** (attempt - 1)),
            self.config.backoff_max_seconds,
        )

    def _send(
        self, method: str, url: str, headers: dict[str, str], payload: bytes | None
    ) -> tuple[int, dict[str, str], bytes]:
        if self._transport is not None:
            return self._transport(method, url, headers, payload)

        req = urllib.request.Request(url, data=payload, method=method, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=self.config.timeout_seconds) as resp:
                return resp.status, dict(resp.headers.items()), resp.read()
        except urllib.error.HTTPError as exc:
            return exc.code, dict(exc.headers.items() if exc.headers else {}), exc.read()
        except (urllib.error.URLError, socket.timeout, TimeoutError) as exc:
            raise RazorpayTransientError(
                f"network error calling Razorpay {method} {self._safe_path(url)}: {exc}"
            ) from None

    # -- response handling ----------------------------------------
    @staticmethod
    def _safe_path(url: str) -> str:
        p = urllib.parse.urlparse(url)
        return p.path

    def _parse(
        self, status: int, headers: dict[str, str], raw: bytes
    ) -> dict[str, Any]:
        request_id = headers.get("X-Razorpay-Request-Id") or headers.get(
            "x-razorpay-request-id"
        )
        try:
            data = json.loads(raw.decode("utf-8")) if raw else {}
        except (json.JSONDecodeError, UnicodeDecodeError):
            if 200 <= status < 300:
                raise RazorpayMalformedResponse(
                    "Razorpay returned a non-JSON success body",
                    status_code=status,
                    request_id=request_id,
                ) from None
            data = {}

        if 200 <= status < 300:
            if not isinstance(data, dict):
                raise RazorpayMalformedResponse(
                    "Razorpay success body was not a JSON object",
                    status_code=status,
                    request_id=request_id,
                )
            return data

        err = (data.get("error") or {}) if isinstance(data, dict) else {}
        code = err.get("code")
        desc = str(err.get("description") or f"HTTP {status}")[:300]

        if status in (401, 403):
            raise RazorpayAuthError(
                "Razorpay rejected the API credentials",
                status_code=status, razorpay_error_code=code, request_id=request_id,
            )
        if status == 429:
            raise RazorpayRateLimitError(
                "Razorpay rate limit exceeded",
                status_code=status, razorpay_error_code=code, request_id=request_id,
            )
        if status in (400, 402, 404, 409, 422):
            raise RazorpayValidationError(
                desc, status_code=status, razorpay_error_code=code, request_id=request_id
            )
        if status >= 500:
            raise RazorpayTransientError(
                f"Razorpay server error (HTTP {status})",
                status_code=status, razorpay_error_code=code, request_id=request_id,
            )
        raise RazorpayValidationError(
            desc, status_code=status, razorpay_error_code=code, request_id=request_id
        )
