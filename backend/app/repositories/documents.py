"""Owner-scoped document and ingestion job queries."""

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..models import Document, IngestionJob


class DocumentRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def list_active(self, user_id: str) -> list[Document]:
        statement = (
            select(Document)
            .where(Document.user_id == user_id, Document.status != "deleted")
            .order_by(Document.uploaded_at.desc(), Document.id)
        )
        return list(self.session.scalars(statement))

    def list_retrievable_ids(self, user_id: str) -> list[str]:
        """Return only completed document IDs eligible for RAG retrieval."""

        statement = (
            select(Document.id)
            .where(
                Document.user_id == user_id,
                Document.status == "completed",
            )
            .order_by(Document.id)
        )
        return list(self.session.scalars(statement))

    def get_owned(self, user_id: str, document_id: str) -> Document | None:
        return self.session.scalar(
            select(Document).where(
                Document.id == document_id,
                Document.user_id == user_id,
            )
        )

    def find_duplicate(self, user_id: str, sha256: str) -> Document | None:
        return self.session.scalar(
            select(Document).where(
                Document.user_id == user_id,
                Document.sha256 == sha256,
                Document.status != "deleted",
            )
        )

    def add(self, document: Document) -> Document:
        self.session.add(document)
        self.session.flush()
        return document

    def create_job(self, document_id: str) -> IngestionJob:
        job = IngestionJob(document_id=document_id, status="pending")
        self.session.add(job)
        self.session.flush()
        return job

    def get_job_owned(self, user_id: str, job_id: str) -> IngestionJob | None:
        statement = (
            select(IngestionJob)
            .join(Document, Document.id == IngestionJob.document_id)
            .where(IngestionJob.id == job_id, Document.user_id == user_id)
        )
        return self.session.scalar(statement)

    def latest_job(self, document_id: str) -> IngestionJob | None:
        statement = (
            select(IngestionJob)
            .where(IngestionJob.document_id == document_id)
            .order_by(IngestionJob.created_at.desc())
            .limit(1)
        )
        return self.session.scalar(statement)
