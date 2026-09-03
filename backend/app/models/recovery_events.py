from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base

if TYPE_CHECKING:
    from app.models.agent_events import AgentEvent
    from app.models.interventions import Intervention
    from app.models.payment import Payment


class RecoveryEvent(Base):
    __tablename__ = "recovery_events"

    id: Mapped[int] = mapped_column(
        primary_key=True,
        index=True,
    )

    payment_id: Mapped[int] = mapped_column(
        ForeignKey("payments.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )

    status: Mapped[str] = mapped_column(
        String(30),
        default="open",
        nullable=False,
    )

    priority: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    is_control: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        server_default="false",
        nullable=False,
    )

    experiment_id: Mapped[int | None] = mapped_column(
        ForeignKey("experiments.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    variant: Mapped[str | None] = mapped_column(
        String(20),
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.utcnow,
        nullable=False,
    )

    closed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    payment: Mapped["Payment"] = relationship(
        back_populates="recovery_events",
    )

    interventions: Mapped[list["Intervention"]] = relationship(
        back_populates="recovery_event",
        cascade="all, delete-orphan",
    )

    agent_events: Mapped[list["AgentEvent"]] = relationship(
        back_populates="recovery_event",
        cascade="all, delete-orphan",
    )