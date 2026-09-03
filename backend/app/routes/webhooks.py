"""Razorpay webhook receiver.

    POST /api/v1/webhooks/razorpay

The raw request body is read byte-for-byte (re-serialising would break the HMAC)
and handed to :class:`RazorpayWebhookService`, which verifies the signature
*before* parsing, correlates the event, applies the state change exactly once,
and is idempotent against duplicate deliveries.

The endpoint never logs the body, the signature, or the webhook secret.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, Request, Response
from sqlalchemy.orm import Session

from app.database import get_db
from app.integrations.razorpay import EVENT_ID_HEADER, SIGNATURE_HEADER
from app.integrations.razorpay.exceptions import RazorpayConfigError
from app.services.razorpay_webhook import RazorpayWebhookService

router = APIRouter(prefix="/api/v1/webhooks", tags=["webhooks"])


@router.post("/razorpay")
async def razorpay_webhook(
    request: Request,
    response: Response,
    db: Session = Depends(get_db),
):
    raw_body = await request.body()
    signature = request.headers.get(SIGNATURE_HEADER)
    event_id = request.headers.get(EVENT_ID_HEADER)

    try:
        service = RazorpayWebhookService(db)
        result = service.process(
            raw_body=raw_body, signature=signature, event_id=event_id
        )
    except RazorpayConfigError:
        # Secret not configured: tell Razorpay to retry later, don't 500-loop.
        response.status_code = 503
        return {"status": "webhook_not_configured"}

    response.status_code = result.http_status
    return result.body()
