"""Enforce one active ingestion job per document in PostgreSQL.

Revision ID: 0002_one_active_ingestion_job
Revises: 0001_initial
Create Date: 2026-07-10
"""

from typing import Sequence

from alembic import op
import sqlalchemy as sa

revision: str = "0002_one_active_ingestion_job"
down_revision: str | None = "0001_initial"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

INDEX_NAME = "uq_ingestion_jobs_active_document"


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        # SQLite tests create their schema from SQLAlchemy metadata, which
        # declares the equivalent sqlite_where predicate. Production Alembic
        # migrations target PostgreSQL and install the invariant below.
        return

    # Reconcile legacy duplicates deterministically before adding the index.
    # Prefer a processing worker, then retain the oldest active job.  Other
    # active rows become explicit retryable failures instead of silently
    # disappearing from job history.
    op.execute(
        sa.text(
            """
            WITH ranked_active_jobs AS (
                SELECT
                    id,
                    ROW_NUMBER() OVER (
                        PARTITION BY document_id
                        ORDER BY
                            CASE status WHEN 'processing' THEN 0 ELSE 1 END,
                            created_at,
                            id
                    ) AS active_rank
                FROM ingestion_jobs
                WHERE status IN ('pending', 'processing')
            )
            UPDATE ingestion_jobs AS jobs
            SET
                status = 'failed',
                completed_at = COALESCE(jobs.completed_at, CURRENT_TIMESTAMP),
                failure_message = COALESCE(
                    jobs.failure_message,
                    'Superseded while enforcing the one-active-job invariant. Start a new ingestion job to retry.'
                ),
                updated_at = CURRENT_TIMESTAMP
            FROM ranked_active_jobs AS ranked
            WHERE jobs.id = ranked.id
              AND ranked.active_rank > 1
            """
        )
    )
    op.create_index(
        INDEX_NAME,
        "ingestion_jobs",
        ["document_id"],
        unique=True,
        postgresql_where=sa.text("status IN ('pending', 'processing')"),
    )


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.drop_index(INDEX_NAME, table_name="ingestion_jobs")
