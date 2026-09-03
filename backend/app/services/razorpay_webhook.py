"""Razorpay webhook processing.

    POST /api/v1/webhooks/razorpay -> route -> THIS SERVICE

Order of operations (never reordered):

1. verify the ``X-Razorpay-Signature`` HMAC against ``RAZORPAY_WEBHOOK_SECRET``
   using the RAW body -- reject before parsing anything;
2. parse the envelope;
3. idempotency check against ``processed_webhook_events`` (PK = Razorpay's
   ``X-Razorpay-Event-Id``);
4. correlate the payment_link entity to our Intervention / RecoveryEvent;
5. apply the authoritative state change -- **exactly once**:
   * ``payment_link.paid``  -> create the single ``Outcome`` (payment_recovered
     = true, amount from the Razorpay *payment* entity), close the event, mark
     the Payment recovered;
   * other events           -> record ``last_razorpay_status`` only, no Outcome;
6. record the processed event id (its PK makes a duplicate delivery a no-op).

Execution success (a Payment Link exists) is NOT payment recovery. Only this
service, on a verified ``payment_link.paid``, sets ``payment_recovered = true``.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, selectinload

from app.integrations.razorpay import RazorpayConfig, WebhookEnvelope, verify_signature
from app.integrations.razorpay.exceptions import WebhookSignatureError
from app.models.agent_events import AgentEvent
from app.models.audit_log import AuditLog
from app.models.interventions import Intervention
from app.models.outcome import Outcome
from app.models.payment import Payment
from app.models.recovery_events import RecoveryEvent
from app.models.subscription import Subscription
from app.models.webhook_event import ProcessedWebhookEvent

WEBHOOK_ACTOR = "razorpay_webhook"

_HANDLED_EVENTS = {
    "payment_link.paid",
    "payment_link.partially_paid",
    "payment_link.cancelled",
    "payment_link.expired",
}


@dataclass
class WebhookProcessResult:
    http_status: int
    status: str                       # ok | already_processed | ignored | correlation_failed | invalid_signature | bad_payload
    event: str | None = None
    recovery_event_id: int | None = None
    intervention_id: int | None = None
    payment_recovered: bool = False
    recovered_amount_paise: int | None = None

    def body(self) -> dict:
        out = {"status": self.status}
        if self.event:
            out["event"] = self.event
        if self.recovery_event_id is not None:
            out["recovery_event_id"] = self.recovery_event_id
        if self.payment_recovered:
            out["payment_recovered"] = True
        return out


class RazorpayWebhookService:
    def __init__(self, db: Session, config: RazorpayConfig | None = None) -> None:
        self.db = db
        self.config = config or RazorpayConfig.from_settings()

    # ------------------------------------------------------------------
    def process(
        self, *, raw_body: bytes, signature: str | None, event_id: str | None
    ) -> WebhookProcessResult:
        # 1. signature -- before anything else
        secret = self.config.require_webhook_secret()
        try:
            verify_signature(raw_body, signature, secret)
        except WebhookSignatureError:
            return WebhookProcessResult(http_status=400, status="invalid_signature")

        # 2. parse
        try:
            import json

            envelope = WebhookEnvelope.model_validate(json.loads(raw_body))
        except Exception:  # noqa: BLE001 - any parse failure is a bad payload
            return WebhookProcessResult(http_status=400, status="bad_payload")

        key = event_id or self._synthetic_id(raw_body)

        # 3. idempotency: already processed?
        if self.db.get(ProcessedWebhookEvent, key) is not None:
            return WebhookProcessResult(
                http_status=200, status="already_processed", event=envelope.event
            )

        # 4. dispatch
        if envelope.event not in _HANDLED_EVENTS:
            return WebhookProcessResult(
                http_status=200, status="ignored", event=envelope.event
            )

        try:
            result = self._dispatch(envelope)
            if result.status != "correlation_failed":
                # A genuinely uncorrelatable event is left unrecorded so
                # Razorpay's own retries can catch a late-arriving link row.
                self._record(key, envelope.event, result)
            self.db.commit()
        except IntegrityError:
            # concurrent duplicate delivery beat us to the PK insert
            self.db.rollback()
            return WebhookProcessResult(
                http_status=200, status="already_processed", event=envelope.event
            )
        except Exception:
            self.db.rollback()
            raise
        return result

    # ------------------------------------------------------------------
    def _dispatch(self, envelope: WebhookEnvelope) -> WebhookProcessResult:
        pl = envelope.payment_link_entity() or {}
        intervention = self._correlate(pl)
        if intervention is None:
            return WebhookProcessResult(
                http_status=200, status="correlation_failed", event=envelope.event
            )

        if envelope.event == "payment_link.paid":
            return self._handle_paid(envelope, intervention)
        return self._handle_non_terminal(envelope, intervention, pl)

    # -- correlation ------------------------------------------------
    def _correlate(self, pl_entity: dict) -> Intervention | None:
        link_id = pl_entity.get("id")
        ref = pl_entity.get("reference_id")

        stmt_opts = (
            selectinload(Intervention.outcome),
            selectinload(Intervention.recovery_event)
            .selectinload(RecoveryEvent.payment)
            .selectinload(Payment.subscription)
            .selectinload(Subscription.customer),
        )
        if link_id:
            iv = self.db.scalars(
                select(Intervention)
                .where(Intervention.razorpay_payment_link_id == link_id)
                .options(*stmt_opts)
            ).one_or_none()
            if iv is not None:
                return iv
        if ref and ref.startswith("recovery-"):
            parts = ref.split("-")
            if len(parts) == 3 and parts[2].isdigit():
                iv = self.db.scalars(
                    select(Intervention)
                    .where(Intervention.id == int(parts[2]))
                    .options(*stmt_opts)
                ).one_or_none()
                if iv is not None and iv.recovery_event_id == int(parts[1]):
                    return iv
        return None

    # -- handlers -------------------------------------------------
    def _handle_paid(
        self, envelope: WebhookEnvelope, intervention: Intervention
    ) -> WebhookProcessResult:
        event = intervention.recovery_event
        payment = event.payment
        now = datetime.now(timezone.utc)

        pay_entity = envelope.payment_entity() or {}
        pl_entity = envelope.payment_link_entity() or {}
        amount = int(
            pay_entity.get("amount")
            or pl_entity.get("amount_paid")
            or payment.amount
        )
        razorpay_payment_id = pay_entity.get("id")

        # idempotency layer 2: Outcome already exists for this intervention
        if intervention.outcome is not None:
            return WebhookProcessResult(
                http_status=200, status="already_processed", event=envelope.event,
                recovery_event_id=event.id, intervention_id=intervention.id,
                payment_recovered=intervention.outcome.payment_recovered,
                recovered_amount_paise=intervention.outcome.recovered_amount_paise,
            )

        self.db.add(
            Outcome(
                intervention_id=intervention.id,
                payment_recovered=True,
                recovered_amount_paise=amount,
                recovery_time_seconds=self._recovery_seconds(intervention, now),
                observed_at=now,
            )
        )
        intervention.razorpay_payment_id = razorpay_payment_id
        intervention.last_razorpay_status = "paid"
        if intervention.status != "executed":
            intervention.status = "executed"
        if intervention.executed_at is None:
            intervention.executed_at = now

        if payment.status != "success":
            payment.status = "success"
        if payment.recovered_at is None:
            payment.recovered_at = now

        if event.status == "open":
            event.status = "closed"
            event.closed_at = now

        self.db.add(
            AuditLog(
                recovery_event_id=event.id,
                actor=WEBHOOK_ACTOR,
                action="recovery_confirmed_payment_link_paid",
                reason=(
                    f"verified payment_link.paid: {amount} paise recovered via "
                    f"{intervention.action_type}"
                )[:500],
                event_metadata={
                    "intervention_id": intervention.id,
                    "razorpay_payment_link_id": intervention.razorpay_payment_link_id,
                    "razorpay_payment_id": razorpay_payment_id,
                    "recovered_amount_paise": amount,
                },
            )
        )
        self.db.add(
            AgentEvent(
                recovery_event_id=event.id,
                event_type="razorpay_webhook_recovery",
                input_context={
                    "source": "razorpay_webhook",
                    "event": envelope.event,
                    "intervention_id": intervention.id,
                    "action_type": intervention.action_type,
                    "razorpay_payment_link_id": intervention.razorpay_payment_link_id,
                    "razorpay_payment_id": razorpay_payment_id,
                    "recovered_amount_paise": amount,
                },
                decision="payment_recovered",
                confidence=1.0,
            )
        )
        self.db.flush()
        return WebhookProcessResult(
            http_status=200, status="ok", event=envelope.event,
            recovery_event_id=event.id, intervention_id=intervention.id,
            payment_recovered=True, recovered_amount_paise=amount,
        )

    def _handle_non_terminal(
        self, envelope: WebhookEnvelope, intervention: Intervention, pl_entity: dict
    ) -> WebhookProcessResult:
        status_map = {
            "payment_link.partially_paid": "partially_paid",
            "payment_link.cancelled": "cancelled",
            "payment_link.expired": "expired",
        }
        new_status = status_map.get(envelope.event, pl_entity.get("status") or "unknown")
        intervention.last_razorpay_status = new_status
        self.db.add(
            AuditLog(
                recovery_event_id=intervention.recovery_event_id,
                actor=WEBHOOK_ACTOR,
                action=f"razorpay_{envelope.event.replace('.', '_')}",
                reason=(
                    f"payment link now {new_status}; no recovery recorded "
                    f"(execution success != payment recovered)"
                )[:500],
                event_metadata={
                    "intervention_id": intervention.id,
                    "razorpay_payment_link_id": intervention.razorpay_payment_link_id,
                    "razorpay_status": new_status,
                },
            )
        )
        self.db.flush()
        return WebhookProcessResult(
            http_status=200, status="ok", event=envelope.event,
            recovery_event_id=intervention.recovery_event_id,
            intervention_id=intervention.id, payment_recovered=False,
        )

    # -- helpers -------------------------------------------------
    @staticmethod
    def _recovery_seconds(intervention: Intervention, now: datetime) -> int | None:
        ref = intervention.executed_at or intervention.recovery_event.created_at
        if ref is None:
            return None
        if ref.tzinfo is None:
            ref = ref.replace(tzinfo=timezone.utc)
        return max(0, int((now - ref).total_seconds()))

    @staticmethod
    def _synthetic_id(raw_body: bytes) -> str:
        return "synthetic-" + hashlib.sha256(raw_body).hexdigest()[:48]

    def _record(
        self, key: str, event_type: str, result: WebhookProcessResult
    ) -> None:
        self.db.add(
            ProcessedWebhookEvent(
                event_id=key,
                event_type=event_type,
                recovery_event_id=result.recovery_event_id,
                intervention_id=result.intervention_id,
                result=result.status,
            )
        )
        self.db.flush()
