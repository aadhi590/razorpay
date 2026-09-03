"""Razorpay request / response / webhook schemas.

Only the fields the recovery flow actually uses are modelled; unknown fields are
ignored so a Razorpay API addition never breaks parsing.

Verified against Razorpay's Payment Links API and Webhooks docs:
* POST /v1/payment_links  -> https://razorpay.com/docs/api/payments/payment-links/create-standard/
* Webhook events          -> https://razorpay.com/docs/webhooks/payloads/payment-links/
* Signature scheme        -> HMAC-SHA256(raw_body, webhook_secret) hex, X-Razorpay-Signature
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class PaymentLinkCustomer(BaseModel):
    model_config = ConfigDict(extra="ignore")
    name: str | None = None
    email: str | None = None
    contact: str | None = None


class PaymentLinkCreateRequest(BaseModel):
    """The body we POST to /v1/payment_links (standard link)."""

    model_config = ConfigDict(extra="forbid")

    amount: int = Field(gt=0, description="smallest currency unit (paise for INR)")
    currency: str = "INR"
    accept_partial: bool = False
    reference_id: str
    description: str
    customer: PaymentLinkCustomer | None = None
    notify: dict[str, bool] = Field(default_factory=lambda: {"sms": False, "email": False})
    reminder_enable: bool = False
    notes: dict[str, str] = Field(default_factory=dict)
    expire_by: int | None = None  # epoch SECONDS

    def to_body(self) -> dict[str, Any]:
        body = self.model_dump(exclude_none=True)
        if self.customer is not None:
            cust = {k: v for k, v in body.get("customer", {}).items() if v}
            if cust:
                body["customer"] = cust
            else:
                body.pop("customer", None)
        return body


class PaymentLink(BaseModel):
    """The subset of a payment_link entity we keep."""

    model_config = ConfigDict(extra="ignore")

    id: str
    status: str
    amount: int
    amount_paid: int = 0
    currency: str = "INR"
    reference_id: str | None = None
    short_url: str | None = None
    expire_by: int | None = None
    order_id: str | None = None
    notes: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> "PaymentLink":
        return cls.model_validate(data)


class RazorpayPayment(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str
    amount: int
    currency: str = "INR"
    status: str
    order_id: str | None = None


class WebhookEnvelope(BaseModel):
    """Top-level webhook body. Parsed ONLY after signature verification."""

    model_config = ConfigDict(extra="ignore")

    entity: str = "event"
    event: str
    contains: list[str] = Field(default_factory=list)
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: int | None = None

    def payment_link_entity(self) -> dict[str, Any] | None:
        return (self.payload.get("payment_link") or {}).get("entity")

    def payment_entity(self) -> dict[str, Any] | None:
        return (self.payload.get("payment") or {}).get("entity")
