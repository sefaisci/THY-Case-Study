"""Persistent ingestion jobs executed through the existing model pipelines."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

from sqlalchemy.orm import Session, sessionmaker

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


class IngestionService:
    def __init__(
        self,
        session: Session,
        settings: Settings,
        *,
        session_factory: sessionmaker | Callable[[], Session] | None = None,
        coordinator_factory: Callable = create_connected_ingestion_coordinator,
    ) -> None:
        self.session = session
        self.settings = settings
        self.repository = DocumentRepository(session)
        self.session_factory = session_factory
        self.coordinator_factory = coordinator_factory

    def start(self, *, user_id: str, document_ids: list[str]) -> list[IngestionJob]:
        jobs: list[IngestionJob] = []
        for document_id in list(dict.fromkeys(document_ids)):
            document = self.repository.get_owned(user_id, document_id)
            if document is None or document.status == "deleted":
                raise NotFoundError("Document not found.", code="document_not_found")
            if document.status in {"processing", "deletion_pending"}:
                raise ConflictError(
                    f"Document {document.filename!r} cannot start ingestion while status is {document.status!r}.",
                    code="document_not_ingestible",
                )
            document.status = "pending"
            document.error_message = None
            jobs.append(self.repository.create_job(document.id))
        self.session.commit()
        return jobs

    def get_job(self, *, user_id: str, job_id: str) -> IngestionJob:
        job = self.repository.get_job_owned(user_id, job_id)
        if job is None:
            raise NotFoundError("Ingestion job not found.", code="ingestion_job_not_found")
        return job

    def process_job(self, job_id: str) -> None:
        if self.session_factory is None:
            raise RuntimeError("A session factory is required for background ingestion.")
        session = self.session_factory()
        try:
            self._process_job_with_session(session, job_id)
        finally:
            session.close()

    def _process_job_with_session(self, session: Session, job_id: str) -> None:
        repository = DocumentRepository(session)
        job = session.get(IngestionJob, job_id)
        if job is None:
            return
        document = session.get(Document, job.document_id)
        if document is None or document.status == "deleted":
            job.status = "failed"
            job.failure_message = "Document no longer exists."
            session.commit()
            return

        job.status = "processing"
        job.started_at = utc_now()
        document.status = "processing"
        document.error_message = None
        session.commit()
        events: list[ModelUsage] = []
        try:
            ingestion_settings = self._ingestion_settings(document)
            coordinator = self.coordinator_factory(
                ingestion_settings,
                usage_callback=events.append,
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
            result = pipeline.process_document(source, document.user_id)
            self._persist_job_usage(session, document, job, events)
            job.chunk_count = result.chunk_count
            job.point_count = result.point_count
            job.completed_at = utc_now()
            if result.status == "completed":
                job.status = "completed"
                document.status = "completed"
                job.failure_message = None
                document.error_message = None
            else:
                failure = "; ".join(item.message for item in result.failures)[:2000]
                job.status = "failed"
                document.status = "failed"
                job.failure_message = failure or "Ingestion did not complete."
                document.error_message = job.failure_message
            session.commit()
            logger.info(
                "Ingestion job finished",
                extra={
                    "event": "ingestion_finished",
                    "user_id": document.user_id,
                    "document_id": document.id,
                },
            )
        except Exception as exc:
            session.rollback()
            job = session.get(IngestionJob, job_id)
            document = session.get(Document, job.document_id) if job else None
            if job is not None and document is not None:
                try:
                    self._persist_job_usage(session, document, job, events)
                except Exception:
                    session.rollback()
                    job = session.get(IngestionJob, job_id)
                    document = session.get(Document, job.document_id) if job else None
                    logger.exception(
                        "Failed to persist ingestion usage after job failure",
                        extra={"event": "ingestion_usage_failed", "job_id": job_id},
                    )
            if job is not None:
                job.status = "failed"
                job.completed_at = utc_now()
                job.failure_message = str(exc)[:2000]
            if document is not None:
                document.status = "failed"
                document.error_message = str(exc)[:2000]
            session.commit()
            logger.exception(
                "Ingestion job failed",
                extra={
                    "event": "ingestion_failed",
                    "document_id": document.id if document else None,
                },
            )

    def _persist_job_usage(
        self,
        session: Session,
        document: Document,
        job: IngestionJob,
        events: list[ModelUsage],
    ) -> None:
        usage_service = UsageService(
            session,
            PricingRegistry(self.settings.pricing_registry_path),
        )
        usage_records = usage_service.persist_events(
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
                    usage_service.record_not_applicable(
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
        )
