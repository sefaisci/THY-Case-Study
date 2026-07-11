"""Document upload, owner-scoped listing, and idempotent deletion."""

from __future__ import annotations

import uuid
import asyncio
import logging
from pathlib import Path
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from model.vector_store import QdrantChunkStore

from ..config import Settings
from ..exceptions import AppError, ConflictError, NotFoundError, ProviderError
from ..models import Document, utc_now
from ..repositories import DocumentRepository
from ..schemas.documents import DocumentDeleteResponse, DocumentResponse
from .model_catalog import ModelCatalogService
from .storage import UploadStorage

logger = logging.getLogger(__name__)


class DocumentService:
    def __init__(
        self,
        session: AsyncSession,
        settings: Settings,
        *,
        model_catalog: ModelCatalogService | None = None,
        qdrant_store: QdrantChunkStore | None = None,
    ) -> None:
        self.session = session
        self.settings = settings
        self.repository = DocumentRepository(session)
        self.model_catalog = model_catalog
        self.storage = UploadStorage(
            settings.upload_dir,
            max_size_bytes=settings.max_upload_size_mb * 1024 * 1024,
            allowed_extensions=settings.allowed_extensions,
        )
        self._qdrant_store = qdrant_store

    async def list_documents(self, user_id: str) -> list[DocumentResponse]:
        return [
            DocumentResponse.model_validate(item)
            for item in await self.repository.list_active(user_id)
        ]

    async def upload_many(
        self,
        *,
        user_id: str,
        files: list[Any],
        ingestion_method: str,
        semantic_model: str | None,
        semantic_reasoning_effort: str | None,
    ) -> list[Document]:
        if not files:
            raise AppError("Select at least one file to upload.", code="files_required", status_code=422)
        if ingestion_method not in {"semantic", "docling"}:
            raise AppError("Invalid ingestion method.", code="invalid_ingestion_method", status_code=422)
        if ingestion_method == "semantic":
            if not semantic_model or not semantic_reasoning_effort:
                raise AppError(
                    "Semantic model and reasoning effort are required for semantic chunking.",
                    code="semantic_configuration_required",
                    status_code=422,
                )
            if self.model_catalog is None:
                raise ProviderError("OpenAI model validation is unavailable.")
            await self.model_catalog.validate(semantic_model, semantic_reasoning_effort)
        else:
            semantic_model = None
            semantic_reasoning_effort = None

        created: list[Document] = []
        stored_paths: list[Path] = []
        in_flight_store: asyncio.Task[Any] | None = None
        try:
            for upload in files:
                document_id = str(uuid.uuid4())
                await upload.seek(0)
                in_flight_store = asyncio.create_task(
                    asyncio.to_thread(
                        self.storage.store,
                        stream=upload.file,
                        original_filename=upload.filename or "",
                        declared_mime_type=getattr(upload, "content_type", None),
                        user_id=user_id,
                        document_id=document_id,
                    )
                )
                # A shield lets cancellation cleanup await the non-cancellable
                # worker thread and discover the exact path it created.
                stored = await asyncio.shield(in_flight_store)
                in_flight_store = None
                stored_paths.append(stored.path)
                duplicate = await self.repository.find_duplicate(user_id, stored.sha256)
                if duplicate is not None:
                    await asyncio.to_thread(self.storage.remove, stored.path)
                    stored_paths.remove(stored.path)
                    raise ConflictError(
                        f"A document with identical content already exists as {duplicate.filename!r}.",
                        code="duplicate_document",
                    )
                collection = (
                    self.settings.qdrant_collection_semantic
                    if ingestion_method == "semantic"
                    else self.settings.qdrant_collection_docling
                )
                document = Document(
                    id=document_id,
                    user_id=user_id,
                    filename=stored.filename,
                    storage_path=str(stored.path),
                    mime_type=stored.mime_type,
                    file_extension=stored.extension,
                    file_size_bytes=stored.size_bytes,
                    sha256=stored.sha256,
                    ingestion_method=ingestion_method,
                    semantic_model=semantic_model,
                    semantic_reasoning_effort=semantic_reasoning_effort,
                    collection_name=collection,
                    status="pending",
                )
                created.append(await self.repository.add(document))
            await self.session.commit()
            return created
        except asyncio.CancelledError:
            if in_flight_store is not None:
                try:
                    stored = await asyncio.shield(in_flight_store)
                    stored_paths.append(stored.path)
                except Exception:
                    logger.exception(
                        "Upload storage task failed while cancellation was being handled",
                        extra={"event": "cancelled_upload_store_failed", "user_id": user_id},
                    )
            try:
                await asyncio.shield(self.session.rollback())
            except Exception:
                logger.exception(
                    "Database rollback failed while upload cancellation was being handled",
                    extra={"event": "cancelled_upload_rollback_failed", "user_id": user_id},
                )
            try:
                await asyncio.shield(self._remove_upload_paths(stored_paths))
            except Exception:
                logger.exception(
                    "Upload artifact cleanup failed after cancellation",
                    extra={"event": "cancelled_upload_cleanup_failed", "user_id": user_id},
                )
            raise
        except Exception:
            try:
                await self.session.rollback()
            except Exception:
                logger.exception(
                    "Database rollback failed after upload request failure",
                    extra={"event": "failed_upload_rollback_failed", "user_id": user_id},
                )
            try:
                await self._remove_upload_paths(stored_paths)
            except Exception:
                logger.exception(
                    "Upload artifact cleanup failed after request failure",
                    extra={"event": "failed_upload_cleanup_failed", "user_id": user_id},
                )
            raise

    async def delete(self, *, user_id: str, document_id: str) -> DocumentDeleteResponse:
        document = await self.repository.get_owned_for_update(user_id, document_id)
        if document is None:
            raise NotFoundError("Document not found.", code="document_not_found")
        if document.status == "deleted":
            await self.session.commit()
            return DocumentDeleteResponse(
                document_id=document.id,
                status="deleted",
                message="Document was already deleted.",
            )
        if document.status == "deletion_pending" and document.error_message is None:
            await self.session.rollback()
            raise ConflictError(
                "Document deletion is already in progress.",
                code="document_deletion_in_progress",
            )
        active_job = await self.repository.get_active_job_for_update(document.id)
        if active_job is not None:
            await self.session.rollback()
            raise ConflictError(
                "Document cannot be deleted while ingestion is pending or processing.",
                code="document_ingestion_active",
            )

        document.status = "deletion_pending"
        document.error_message = None
        await self.session.commit()
        deleted_points = 0
        store: QdrantChunkStore | None = None
        owns_store = False
        try:
            store = self._qdrant_store
            if store is None:
                store = self._create_qdrant_store()
                owns_store = True
            for collection in (
                self.settings.qdrant_collection_semantic,
                self.settings.qdrant_collection_docling,
            ):
                deleted_points += await store.delete_user_document_points(
                    collection_name=collection,
                    user_id=user_id,
                    document_id=document.id,
                )
            await asyncio.to_thread(self.storage.remove, document.storage_path)
            document = await self.repository.get_owned_for_update(user_id, document_id)
            if document is None:
                raise NotFoundError("Document not found.", code="document_not_found")
            if document.status == "deleted":
                await self.session.commit()
                return DocumentDeleteResponse(
                    document_id=document.id,
                    status="deleted",
                    deleted_points=deleted_points,
                    message="Document was already deleted.",
                )
            if document.status != "deletion_pending":
                raise ConflictError(
                    "Document lifecycle changed before deletion could be finalized.",
                    code="document_deletion_state_changed",
                )
            document.status = "deleted"
            document.deleted_at = utc_now()
            document.error_message = None
            await self.session.commit()
            return DocumentDeleteResponse(
                document_id=document.id,
                status="deleted",
                deleted_points=deleted_points,
                message="Document deleted successfully.",
            )
        except asyncio.CancelledError:
            try:
                await asyncio.shield(
                    self._mark_deletion_retryable(
                        user_id,
                        document_id,
                        "Deletion was cancelled and can be retried.",
                    )
                )
            except Exception:
                logger.exception(
                    "Failed to persist cancelled deletion state",
                    extra={"event": "deletion_cancellation_state_failed", "document_id": document_id},
                )
            raise
        except Exception as exc:
            await self._mark_deletion_retryable(
                user_id,
                document_id,
                f"Deletion failed and can be retried: {str(exc)[:1000]}",
            )
            raise ProviderError(
                "Document deletion did not complete. The document remains in a retryable deletion_pending state.",
                code="document_deletion_failed",
            ) from exc
        finally:
            if owns_store and store is not None:
                try:
                    await asyncio.shield(store.aclose())
                except (Exception, asyncio.CancelledError):
                    # Resource cleanup must never replace the domain result or
                    # the provider/cancellation error already being handled.
                    logger.exception(
                        "Failed to close the document deletion Qdrant client",
                        extra={"event": "document_delete_qdrant_close_failed"},
                    )

    async def _remove_upload_paths(self, paths: list[Path]) -> None:
        """Remove every known upload artifact exactly once."""

        errors: list[Exception] = []
        for path in dict.fromkeys(paths):
            try:
                await asyncio.to_thread(self.storage.remove, path)
            except Exception as exc:
                errors.append(exc)
        if errors:
            raise errors[0]

    async def _mark_deletion_retryable(
        self,
        user_id: str,
        document_id: str,
        message: str,
    ) -> None:
        """Fail closed without overwriting a newer terminal lifecycle state."""

        await self.session.rollback()
        document = await self.repository.get_owned_for_update(user_id, document_id)
        if document is None or document.status == "deleted":
            await self.session.rollback()
            return
        if document.status == "deletion_pending":
            document.error_message = message[:1200]
            await self.session.commit()
            return
        await self.session.rollback()

    def _create_qdrant_store(self) -> QdrantChunkStore:
        return QdrantChunkStore(
            url=self.settings.qdrant_url,
            api_key=self.settings.qdrant_api_key,
            dense_vector_name=self.settings.qdrant_dense_vector_name,
            sparse_vector_name=self.settings.qdrant_sparse_vector_name,
            dense_vector_size=self.settings.qdrant_dense_vector_size,
        )
