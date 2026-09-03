from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class ProcessedWebhookEvent(Base):
    """One row per Razorpay webhook delivery we have fully processed.

    ``event_id`` is Razorpay's ``X-Razorpay-Event-Id`` header. Its primary-key
    role is what makes webhook handling idempotent: a duplicate delivery hits a
    ``PRIMARY KEY`` violation and is acknowledged without re-applying any state
    change. Delivery with no event id falls back to a synthetic key derived
    from the payload signature.
    """

    __tablename__ = "processed_webhook_events"

    event_id: Mapped[str] = mapped_column(String(120), primary_key=True)

    event_type: Mapped[str] = mapped_column(String(60), nullable=False)

    recovery_event_id: Mapped[int | None] = mapped_column(
        ForeignKey("recovery_events.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    intervention_id: Mapped[int | None] = mapped_column(
        ForeignKey("interventions.id", ondelete="SET NULL"),
        nullable=True,
    )

    result: Mapped[str] = mapped_column(String(40), nullable=False)

    received_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.utcnow,
        nullable=False,
    )
