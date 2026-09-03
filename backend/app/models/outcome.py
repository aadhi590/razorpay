from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base
from app.models.interventions import Intervention


class Outcome(Base):
    __tablename__ = "outcomes"

    id: Mapped[int] = mapped_column(
        primary_key=True,
        index=True,
    )

    intervention_id: Mapped[int] = mapped_column(
        ForeignKey("interventions.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )

    payment_recovered: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    recovered_amount_paise: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    recovery_time_seconds: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    observed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.utcnow,
        nullable=False,
    )

    intervention: Mapped["Intervention"] = relationship(
        back_populates="outcome",
    )