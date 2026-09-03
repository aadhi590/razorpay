"""Safe, read-only inspection of a recovery event's Razorpay state.

    GET /api/v1/recovery-events/{recovery_event_id}/razorpay

Returns only non-sensitive identifiers (payment link id, short url, status,
payment id) plus the derived recovery state. No secrets, no raw payloads.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.agent.guardrails import load_event, payment_recovered
from app.database import get_db
from app.integrations.razorpay import RazorpayConfig

router = APIRouter(prefix="/api/v1/recovery-events", tags=["recovery-events"])


@router.get("/{recovery_event_id}/razorpay")
def recovery_event_razorpay(
    recovery_event_id: int,
    db: Session = Depends(get_db),
) -> dict:
    event = load_event(db, recovery_event_id)
    if event is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Recovery event not found."
        )

    interventions = []
    for i in sorted(event.interventions, key=lambda x: x.id):
        interventions.append(
            {
                "intervention_id": i.id,
                "action_type": i.action_type,
                "status": i.status,
                "razorpay_payment_link_id": i.razorpay_payment_link_id,
                "razorpay_short_url": i.razorpay_short_url,
                "razorpay_reference_id": i.razorpay_reference_id,
                "razorpay_payment_id": i.razorpay_payment_id,
                "last_razorpay_status": i.last_razorpay_status,
                "payment_link_created": i.razorpay_payment_link_id is not None,
                "payment_link_paid": i.last_razorpay_status == "paid",
                "outcome_recorded": i.outcome is not None,
                "outcome_payment_recovered": bool(
                    i.outcome and i.outcome.payment_recovered
                ),
                "recovered_amount_paise": (
                    i.outcome.recovered_amount_paise if i.outcome else None
                ),
            }
        )

    return {
        "recovery_event_id": event.id,
        "status": event.status,
        "is_control": event.is_control,
        "payment_recovered": payment_recovered(event),
        "payment_status": event.payment.status,
        "amount_paise": event.payment.amount,
        "interventions": interventions,
        "razorpay_config": RazorpayConfig.from_settings().status(),
    }
