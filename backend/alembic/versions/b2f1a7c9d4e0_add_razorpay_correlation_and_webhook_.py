"""add razorpay correlation fields and processed webhook events table

Revision ID: b2f1a7c9d4e0
Revises: 361460832735
Create Date: 2026-09-03

Additive only. Every new column on ``interventions`` is nullable, so existing
rows and the rules / ML / uplift / dry-run paths are unaffected. The
``processed_webhook_events`` table exists solely to make Razorpay webhook
handling idempotent at the database level (its primary key is Razorpay's
event id).
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "b2f1a7c9d4e0"
down_revision: Union[str, Sequence[str], None] = "361460832735"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "interventions",
        sa.Column("razorpay_reference_id", sa.String(length=120), nullable=True),
    )
    op.add_column(
        "interventions",
        sa.Column("razorpay_payment_link_id", sa.String(length=64), nullable=True),
    )
    op.add_column(
        "interventions",
        sa.Column("razorpay_short_url", sa.String(length=255), nullable=True),
    )
    op.add_column(
        "interventions",
        sa.Column("razorpay_payment_id", sa.String(length=64), nullable=True),
    )
    op.add_column(
        "interventions",
        sa.Column("last_razorpay_status", sa.String(length=30), nullable=True),
    )
    op.create_unique_constraint(
        "uq_interventions_razorpay_reference_id",
        "interventions",
        ["razorpay_reference_id"],
    )
    op.create_unique_constraint(
        "uq_interventions_razorpay_payment_link_id",
        "interventions",
        ["razorpay_payment_link_id"],
    )
    op.create_index(
        op.f("ix_interventions_razorpay_payment_link_id"),
        "interventions",
        ["razorpay_payment_link_id"],
        unique=False,
    )

    op.create_table(
        "processed_webhook_events",
        sa.Column("event_id", sa.String(length=120), primary_key=True),
        sa.Column("event_type", sa.String(length=60), nullable=False),
        sa.Column("recovery_event_id", sa.Integer(), nullable=True),
        sa.Column("intervention_id", sa.Integer(), nullable=True),
        sa.Column("result", sa.String(length=40), nullable=False),
        sa.Column(
            "received_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["recovery_event_id"], ["recovery_events.id"], ondelete="SET NULL"
        ),
        sa.ForeignKeyConstraint(
            ["intervention_id"], ["interventions.id"], ondelete="SET NULL"
        ),
    )
    op.create_index(
        op.f("ix_processed_webhook_events_recovery_event_id"),
        "processed_webhook_events",
        ["recovery_event_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        op.f("ix_processed_webhook_events_recovery_event_id"),
        table_name="processed_webhook_events",
    )
    op.drop_table("processed_webhook_events")
    op.drop_index(
        op.f("ix_interventions_razorpay_payment_link_id"),
        table_name="interventions",
    )
    op.drop_constraint(
        "uq_interventions_razorpay_payment_link_id", "interventions", type_="unique"
    )
    op.drop_constraint(
        "uq_interventions_razorpay_reference_id", "interventions", type_="unique"
    )
    op.drop_column("interventions", "last_razorpay_status")
    op.drop_column("interventions", "razorpay_payment_id")
    op.drop_column("interventions", "razorpay_short_url")
    op.drop_column("interventions", "razorpay_payment_link_id")
    op.drop_column("interventions", "razorpay_reference_id")
