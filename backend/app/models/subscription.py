from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import (  # pyright: ignore[reportMissingImports]
    DateTime,
    ForeignKey,
    Integer,
    String,
)
from sqlalchemy.orm import (  # pyright: ignore[reportMissingImports]
    Mapped,
    mapped_column,
    relationship,
)

from app.database import Base

if TYPE_CHECKING:
    from app.models.customer import Customer
    from app.models.payment import Payment


class Subscription(Base):
    __tablename__ = "subscriptions"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)

    customer_id: Mapped[int] = mapped_column(
        ForeignKey("customers.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    external_subscription_id: Mapped[str] = mapped_column(
        String(100),
        unique=True,
        index=True,
        nullable=False,
    )

    amount: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    currency: Mapped[str] = mapped_column(
        String(3),
        default="INR",
        nullable=False,
    )

    status: Mapped[str] = mapped_column(
        String(30),
        default="active",
        nullable=False,
    )

    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )

    next_payment_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    customer: Mapped["Customer"] = relationship(
        back_populates="subscriptions",
    )

    payments: Mapped[list["Payment"]] = relationship(
        back_populates="subscription",
        cascade="all, delete-orphan",
    )