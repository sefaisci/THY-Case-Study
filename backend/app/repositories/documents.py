"""Owner-scoped document and ingestion job queries."""

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..models import Document, IngestionJob


class DocumentRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def list_active(self, user_id: str) -> list[Document]:
        statement = (
            select(Document)
            .where(Document.user_id == user_id, Document.status != "deleted")
            .order_by(Document.uploaded_at.desc(), Document.id)
        )
        return list(await self.session.scalars(statement))

    async def list_retrievable_ids(self, user_id: str) -> list[str]:
        """Return only completed document IDs eligible for RAG retrieval."""

        statement = (
            select(Document.id)
            .where(
                Document.user_id == user_id,
                Document.status == "completed",
            )
            .order_by(Document.id)
        )
        return list(await self.session.scalars(statement))

    async def get_owned(self, user_id: str, document_id: str) -> Document | None:
        return await self.session.scalar(
            select(Document).where(
                Document.id == document_id,
                Document.user_id == user_id,
            )
        )

    async def get_owned_for_update(
        self,
        user_id: str,
        document_id: str,
    ) -> Document | None:
        """Lock one owner-scoped document before any lifecycle mutation.

        Every ingestion and deletion transaction acquires this row before it
        locks or mutates an ingestion job.  The shared lock order prevents the
        document/job deadlock inversion that otherwise becomes possible when
        background workers and delete requests race.
        """

        statement = (
            select(Document)
            .where(
                Document.id == document_id,
                Document.user_id == user_id,
            )
            .with_for_update()
            .execution_options(populate_existing=True)
        )
        return await self.session.scalar(statement)

    async def get_for_update(self, document_id: str) -> Document | None:
        """Lock a document by its backend-resolved identifier."""

        statement = (
            select(Document)
            .where(Document.id == document_id)
            .with_for_update()
            .execution_options(populate_existing=True)
        )
        return await self.session.scalar(statement)

    async def find_duplicate(self, user_id: str, sha256: str) -> Document | None:
        return await self.session.scalar(
            select(Document).where(
                Document.user_id == user_id,
                Document.sha256 == sha256,
                Document.status != "deleted",
            )
        )

    async def add(self, document: Document) -> Document:
        self.session.add(document)
        await self.session.flush()
        return document

    async def create_job(self, document_id: str) -> IngestionJob:
        job = IngestionJob(document_id=document_id, status="pending")
        self.session.add(job)
        await self.session.flush()
        return job

    async def get_job_document_id(self, job_id: str) -> str | None:
        """Resolve the parent without locking the job ahead of the document."""

        return await self.session.scalar(
            select(IngestionJob.document_id).where(IngestionJob.id == job_id)
        )

    async def get_job_for_update(
        self,
        job_id: str,
        *,
        document_id: str,
    ) -> IngestionJob | None:
        """Lock a job after its parent document row has been locked."""

        statement = (
            select(IngestionJob)
            .where(
                IngestionJob.id == job_id,
                IngestionJob.document_id == document_id,
            )
            .with_for_update()
            .execution_options(populate_existing=True)
        )
        return await self.session.scalar(statement)

    async def get_active_job_for_update(
        self,
        document_id: str,
    ) -> IngestionJob | None:
        """Return the one pending/processing job after locking its document."""

        statement = (
            select(IngestionJob)
            .where(
                IngestionJob.document_id == document_id,
                IngestionJob.status.in_(("pending", "processing")),
            )
            .order_by(IngestionJob.created_at, IngestionJob.id)
            .limit(1)
            .with_for_update()
            .execution_options(populate_existing=True)
        )
        return await self.session.scalar(statement)

    async def get_job_owned(self, user_id: str, job_id: str) -> IngestionJob | None:
        statement = (
            select(IngestionJob)
            .join(Document, Document.id == IngestionJob.document_id)
            .where(IngestionJob.id == job_id, Document.user_id == user_id)
        )
        return await self.session.scalar(statement)

    async def list_jobs_owned(
        self,
        user_id: str,
        job_ids: list[str],
    ) -> list[IngestionJob]:
        """Return requested jobs in request order, restricted to one owner."""

        unique_ids = list(dict.fromkeys(job_ids))
        statement = (
            select(IngestionJob)
            .join(Document, Document.id == IngestionJob.document_id)
            .where(IngestionJob.id.in_(unique_ids), Document.user_id == user_id)
        )
        jobs_by_id = {
            job.id: job for job in await self.session.scalars(statement)
        }
        return [jobs_by_id[job_id] for job_id in unique_ids if job_id in jobs_by_id]

    async def latest_job(self, document_id: str) -> IngestionJob | None:
        statement = (
            select(IngestionJob)
            .where(IngestionJob.document_id == document_id)
            .order_by(IngestionJob.created_at.desc())
            .limit(1)
        )
        return await self.session.scalar(statement)
