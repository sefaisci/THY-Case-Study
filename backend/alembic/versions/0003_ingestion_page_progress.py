"""Persist page-based ingestion progress.

Revision ID: 0003_ingestion_page_progress
Revises: 0002_one_active_ingestion_job
Create Date: 2026-07-11
"""

from typing import Sequence

from alembic import op
import sqlalchemy as sa

revision: str = "0003_ingestion_page_progress"
down_revision: str | None = "0002_one_active_ingestion_job"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "ingestion_jobs",
        sa.Column("total_pages", sa.Integer(), nullable=False, server_default="0"),
    )
    op.add_column(
        "ingestion_jobs",
        sa.Column("processed_pages", sa.Integer(), nullable=False, server_default="0"),
    )


def downgrade() -> None:
    op.drop_column("ingestion_jobs", "processed_pages")
    op.drop_column("ingestion_jobs", "total_pages")
