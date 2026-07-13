"""Persistent ingestion jobs executed through the existing model pipelines."""

from __future__ import annotations

import logging
import asyncio
import inspect
import weakref
from collections.abc import Callable
from pathlib import Path

from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from model.document_processing import DocumentSource
from model.ingestion import IngestionSettings, create_connected_ingestion_coordinator
from model.usage import ModelUsage

from ..config import PROJECT_ROOT, Settings
from ..exceptions import AppError, ConflictError, NotFoundError
from ..models import Document, IngestionJob, utc_now
from ..repositories import DocumentRepository
from .pricing import PricingRegistry
from .usage import UsageService

logger = logging.getLogger(__name__)

_INGESTION_LIMITERS: weakref.WeakKeyDictionary[
    asyncio.AbstractEventLoop,
    dict[int, asyncio.Semaphore],
] = weakref.WeakKeyDictionary()


def _ingestion_limiter(limit: int) -> asyncio.Semaphore:
    """Return one process-wide limiter per event loop and configured capacity."""

    loop = asyncio.get_running_loop()
    by_limit = _INGESTION_LIMITERS.setdefault(loop, {})
    return by_limit.setdefault(limit, asyncio.Semaphore(limit))


class IngestionService:
    def __init__(
        self,
        session: AsyncSession,
        settings: Settings,
        *,
        session_factory: async_sessionmaker[AsyncSession]
        | Callable[[], AsyncSession]
        | None = None,
        coordinator_factory: Callable = create_connected_ingestion_coordinator,
    ) -> None:
        self.session = session
        self.settings = settings
        self.repository = DocumentRepository(session)
        self.session_factory = session_factory
        self.coordinator_factory = coordinator_factory

    async def start(self, *, user_id: str, document_ids: list[str]) -> list[IngestionJob]:
        requested_ids = list(dict.fromkeys(document_ids))
        locked_documents: dict[str, Document] = {}
        try:
            # Lock in a stable global order so overlapping multi-document
            # requests cannot deadlock by acquiring document rows in reverse.
            for document_id in sorted(requested_ids):
                document = await self.repository.get_owned_for_update(
                    user_id,
                    document_id,
                )
                if document is None or document.status == "deleted":
                    raise NotFoundError("Document not found.", code="document_not_found")
                if document.status in {"processing", "deletion_pending"}:
                    raise ConflictError(
                        f"Document {document.filename!r} cannot start ingestion while status is {document.status!r}.",
                        code="document_not_ingestible",
                    )
                active_job = await self.repository.get_active_job_for_update(document.id)
                if active_job is not None:
                    raise ConflictError(
                        f"Document {document.filename!r} already has an active ingestion job.",
                        code="ingestion_already_active",
                    )
                locked_documents[document_id] = document

            jobs_by_document: dict[str, IngestionJob] = {}
            for document_id in requested_ids:
                document = locked_documents[document_id]
                document.status = "pending"
                document.error_message = None
                jobs_by_document[document_id] = await self.repository.create_job(document.id)
            await self.session.commit()
            return [jobs_by_document[document_id] for document_id in requested_ids]
        except IntegrityError as exc:
            # The PostgreSQL partial unique index is the final cross-process
            # guard if two requests reached the insert before either observed
            # the other's active job.
            await self.session.rollback()
            raise ConflictError(
                "One or more documents already have an active ingestion job.",
                code="ingestion_already_active",
            ) from exc
        except Exception:
            await self.session.rollback()
            raise

    async def get_job(self, *, user_id: str, job_id: str) -> IngestionJob:
        job = await self.repository.get_job_owned(user_id, job_id)
        if job is None:
            raise NotFoundError("Ingestion job not found.", code="ingestion_job_not_found")
        return job

    async def get_jobs(self, *, user_id: str, job_ids: list[str]) -> list[IngestionJob]:
        unique_ids = list(dict.fromkeys(job_ids))
        jobs = await self.repository.list_jobs_owned(user_id, unique_ids)
        if len(jobs) != len(unique_ids):
            raise NotFoundError("One or more ingestion jobs were not found.", code="ingestion_job_not_found")
        return jobs

    async def process_jobs(self, job_ids: list[str]) -> None:
        """Process one batch concurrently while respecting the process-wide limit."""

        await asyncio.gather(*(self.process_job(job_id) for job_id in job_ids))

    async def process_job(self, job_id: str) -> None:
        if self.session_factory is None:
            raise RuntimeError("A session factory is required for background ingestion.")
        try:
            async with _ingestion_limiter(self.settings.ingestion_job_concurrency):
                async with self.session_factory() as session:
                    await self._process_job_with_session(session, job_id)
        except asyncio.CancelledError:
            # Cancellation can arrive while this job is still waiting for a
            # process slot, before _process_job_with_session can claim it.
            try:
                await asyncio.shield(self._mark_cancelled_with_new_session(job_id))
            except Exception:
                logger.exception(
                    "Failed to persist cancelled queued ingestion state",
                    extra={"event": "queued_ingestion_cancellation_state_failed", "job_id": job_id},
                )
            raise

    async def _process_job_with_session(self, session: AsyncSession, job_id: str) -> None:
        repository = DocumentRepository(session)
        document, job = await self._lock_document_and_job(repository, job_id)
        if job is None:
            await session.rollback()
            return
        if job.status != "pending":
            # Duplicate background delivery is intentionally a no-op.  Only
            # the transaction that changes pending -> processing owns work.
            await session.commit()
            return
        if document is None or document.status in {"deletion_pending", "deleted"}:
            job.status = "failed"
            job.completed_at = utc_now()
            job.failure_message = "Document is unavailable because deletion has started."
            await session.commit()
            return
        if document.status == "processing":
            job.status = "failed"
            job.completed_at = utc_now()
            job.failure_message = "Another ingestion worker already owns this document."
            await session.commit()
            return

        job.status = "processing"
        job.started_at = utc_now()
        job.completed_at = None
        job.failure_message = None
        job.total_pages = 0
        job.processed_pages = 0
        document.status = "processing"
        document.error_message = None
        await session.commit()
        events: list[ModelUsage] = []
        coordinator = None
        try:
            ingestion_settings = self._ingestion_settings(document)

            async def progress_callback(total_pages: int, processed_pages: int) -> None:
                try:
                    await self._persist_progress(
                        session,
                        job_id,
                        total_pages=total_pages,
                        processed_pages=processed_pages,
                    )
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.warning(
                        "Failed to persist ingestion progress",
                        exc_info=True,
                        extra={
                            "event": "ingestion_progress_failed",
                            "job_id": job_id,
                        },
                    )

            coordinator_result = self.coordinator_factory(
                ingestion_settings,
                usage_callback=events.append,
                progress_callback=progress_callback,
            )
            coordinator = (
                await coordinator_result
                if inspect.isawaitable(coordinator_result)
                else coordinator_result
            )
            source = DocumentSource(
                path=Path(document.storage_path),
                document_id=document.id,
                document_name=document.filename,
                document_type=document.file_extension,
                source_sha256=document.sha256,
            )
            pipeline = (
                coordinator.semantic_pipeline
                if document.ingestion_method == "semantic"
                else coordinator.docling_pipeline
            )
            result = await pipeline.process_document(source, document.user_id)
            document, job = await self._lock_document_and_job(repository, job_id)
            if job is None or job.status != "processing":
                await session.rollback()
                return
            if document is None:
                job.status = "failed"
                job.completed_at = utc_now()
                job.failure_message = "Document no longer exists."
                await session.commit()
                return

            await self._persist_job_usage(session, document, job, events)
            job.total_pages = max(
                job.total_pages,
                result.total_locations,
                result.processed_locations,
            )
            job.processed_pages = min(
                job.total_pages,
                max(job.processed_pages, result.processed_locations),
            )
            job.chunk_count = result.chunk_count
            job.point_count = result.point_count
            job.completed_at = utc_now()
            if document.status in {"deletion_pending", "deleted"}:
                job.status = "failed"
                job.failure_message = "Ingestion finished after document deletion had started."
            elif result.status == "completed":
                job.status = "completed"
                job.processed_pages = job.total_pages
                document.status = "completed"
                job.failure_message = None
                document.error_message = None
            else:
                failure = "; ".join(item.message for item in result.failures)[:2000]
                job.status = "failed"
                document.status = "failed"
                job.failure_message = failure or "Ingestion did not complete."
                document.error_message = job.failure_message
            await session.commit()
            logger.info(
                "Ingestion job finished",
                extra={
                    "event": "ingestion_finished",
                    "user_id": document.user_id,
                    "document_id": document.id,
                },
            )
        except asyncio.CancelledError:
            try:
                await asyncio.shield(
                    self._finish_failed_job(
                        session,
                        job_id,
                        events,
                        "Ingestion worker was cancelled before completion. Start a new ingestion job to retry.",
                    )
                )
            except Exception:
                logger.exception(
                    "Failed to persist cancelled ingestion state",
                    extra={"event": "ingestion_cancellation_state_failed", "job_id": job_id},
                )
            raise
        except Exception as exc:
            await self._finish_failed_job(session, job_id, events, str(exc)[:2000])
            logger.exception(
                "Ingestion job failed",
                extra={
                    "event": "ingestion_failed",
                    "job_id": job_id,
                },
            )
        finally:
            if coordinator is not None:
                close = getattr(coordinator, "aclose", None)
                if close is not None:
                    close_result = close()
                    if inspect.isawaitable(close_result):
                        await asyncio.shield(close_result)

    async def _lock_document_and_job(
        self,
        repository: DocumentRepository,
        job_id: str,
    ) -> tuple[Document | None, IngestionJob | None]:
        """Acquire lifecycle rows in the canonical document -> job order."""

        document_id = await repository.get_job_document_id(job_id)
        if document_id is None:
            return None, None
        document = await repository.get_for_update(document_id)
        if document is None:
            return None, None
        job = await repository.get_job_for_update(job_id, document_id=document_id)
        return document, job

    async def _finish_failed_job(
        self,
        session: AsyncSession,
        job_id: str,
        events: list[ModelUsage],
        message: str,
    ) -> None:
        """Persist a terminal worker failure without reviving deleted content."""

        await session.rollback()
        repository = DocumentRepository(session)
        document, job = await self._lock_document_and_job(repository, job_id)
        if job is None or job.status not in {"pending", "processing"}:
            await session.rollback()
            return
        if document is not None:
            try:
                await self._persist_job_usage(session, document, job, events)
            except Exception:
                await session.rollback()
                document, job = await self._lock_document_and_job(repository, job_id)
                logger.exception(
                    "Failed to persist ingestion usage after job failure",
                    extra={"event": "ingestion_usage_failed", "job_id": job_id},
                )
        if job is None or job.status not in {"pending", "processing"}:
            await session.rollback()
            return
        job.status = "failed"
        job.completed_at = utc_now()
        job.failure_message = message[:2000]
        if document is not None and document.status in {"pending", "processing"}:
            document.status = "failed"
            document.error_message = job.failure_message
        await session.commit()

    async def _mark_cancelled_with_new_session(self, job_id: str) -> None:
        """Finalize cancellation even when it occurred before worker claim."""

        if self.session_factory is None:
            return
        async with self.session_factory() as session:
            await self._finish_failed_job(
                session,
                job_id,
                [],
                "Ingestion worker was cancelled before completion. Start a new ingestion job to retry.",
            )

    @staticmethod
    async def _persist_progress(
        session: AsyncSession,
        job_id: str,
        *,
        total_pages: int,
        processed_pages: int,
    ) -> None:
        """Persist one monotonic job-scoped progress snapshot for API polling."""

        bounded_total = max(0, total_pages)
        bounded_processed = min(max(0, processed_pages), bounded_total)
        try:
            job = await session.get(IngestionJob, job_id)
            if job is None or job.status != "processing":
                await session.rollback()
                return
            job.total_pages = max(job.total_pages, bounded_total)
            job.processed_pages = min(
                job.total_pages,
                max(job.processed_pages, bounded_processed),
            )
            await session.commit()
        except BaseException:
            await session.rollback()
            raise

    async def _persist_job_usage(
        self,
        session: AsyncSession,
        document: Document,
        job: IngestionJob,
        events: list[ModelUsage],
    ) -> None:
        usage_service = UsageService(
            session,
            PricingRegistry(self.settings.pricing_registry_path),
        )
        usage_records = await usage_service.persist_events(
            events,
            user_id=document.user_id,
            operation="ingestion",
            reasoning_effort=document.semantic_reasoning_effort,
            document_id=document.id,
            ingestion_job_id=job.id,
        )
        recorded_stages = {item.stage for item in usage_records}
        for stage in ("document_processing", "semantic_chunking", "embeddings"):
            if stage not in recorded_stages:
                usage_records.append(
                    await usage_service.record_not_applicable(
                        user_id=document.user_id,
                        operation="ingestion",
                        stage=stage,
                        document_id=document.id,
                        ingestion_job_id=job.id,
                    )
                )
        totals = usage_service.totals(usage_records)
        job.input_tokens = totals.input_tokens
        job.cached_input_tokens = totals.cached_input_tokens
        job.output_tokens = totals.output_tokens
        job.reasoning_tokens = totals.reasoning_tokens
        job.total_tokens = totals.total_tokens
        job.cost_usd = totals.cost_usd

    def _ingestion_settings(self, document: Document) -> IngestionSettings:
        if document.ingestion_method == "semantic" and (
            not document.semantic_model or not document.semantic_reasoning_effort
        ):
            raise AppError(
                "Stored semantic ingestion configuration is incomplete.",
                code="semantic_configuration_missing",
            )
        return IngestionSettings(
            project_root=PROJECT_ROOT,
            page_image_dir=self.settings.page_image_dir,
            processing_dir=self.settings.processing_dir,
            openai_api_key=self.settings.openai_api_key,
            openai_base_url=self.settings.openai_base_url,
            qdrant_url=self.settings.qdrant_url,
            qdrant_api_key=self.settings.qdrant_api_key,
            semantic_collection=self.settings.qdrant_collection_semantic,
            docling_collection=self.settings.qdrant_collection_docling,
            dense_vector_name=self.settings.qdrant_dense_vector_name,
            sparse_vector_name=self.settings.qdrant_sparse_vector_name,
            dense_vector_size=self.settings.qdrant_dense_vector_size,
            embedding_model=self.settings.embedding_model,
            semantic_model=document.semantic_model or "",
            semantic_reasoning_effort=document.semantic_reasoning_effort or "low",
            conversion_dpi=self.settings.doc_conversion_dpi,
            fixed_chunk_size_tokens=self.settings.fixed_chunk_size_tokens,
            fixed_chunk_overlap_tokens=self.settings.fixed_chunk_overlap_tokens,
            embedding_batch_size=self.settings.embedding_batch_size,
            llm_request_timeout_seconds=self.settings.llm_request_timeout_seconds,
            semantic_flush_batch_size=self.settings.semantic_flush_batch_size,
            semantic_page_max_concurrency=self.settings.semantic_page_max_concurrency,
            document_max_concurrency=self.settings.document_max_concurrency,
            sparse_encoder_provider=self.settings.sparse_encoder_provider,
            sparse_encoder_model=self.settings.sparse_encoder_model,
            sparse_encoder_cache_dir=self.settings.sparse_encoder_cache_dir,
        )
