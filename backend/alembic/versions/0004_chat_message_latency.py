"""Persist end-to-end RAG response latency on assistant messages.

Revision ID: 0004_chat_message_latency
Revises: 0003_ingestion_page_progress
Create Date: 2026-07-13
"""

from typing import Sequence

from alembic import op
import sqlalchemy as sa

revision: str = "0004_chat_message_latency"
down_revision: str | None = "0003_ingestion_page_progress"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "chat_messages",
        sa.Column("latency_ms", sa.Integer(), nullable=True),
    )
    op.create_check_constraint(
        "ck_chat_messages_latency_ms_nonnegative",
        "chat_messages",
        "latency_ms IS NULL OR latency_ms >= 0",
    )


def downgrade() -> None:
    op.drop_constraint(
        "ck_chat_messages_latency_ms_nonnegative",
        "chat_messages",
        type_="check",
    )
    op.drop_column("chat_messages", "latency_ms")
