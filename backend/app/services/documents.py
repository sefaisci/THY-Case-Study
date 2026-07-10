"""Document upload, owner-scoped listing, and idempotent deletion."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from model.vector_store import QdrantChunkStore

from ..config import Settings
from ..exceptions import AppError, ConflictError, NotFoundError, ProviderError
from ..models import Document, utc_now
from ..repositories import DocumentRepository
from ..schemas.documents import DocumentDeleteResponse, DocumentResponse
from .model_catalog import ModelCatalogService
from .storage import UploadStorage


class DocumentService:
    def __init__(
        self,
        session: Session,
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

    def list_documents(self, user_id: str) -> list[DocumentResponse]:
        return [DocumentResponse.model_validate(item) for item in self.repository.list_active(user_id)]

    def upload_many(
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
            self.model_catalog.validate(semantic_model, semantic_reasoning_effort)
        else:
            semantic_model = None
            semantic_reasoning_effort = None

        created: list[Document] = []
        stored_paths: list[Path] = []
        try:
            for upload in files:
                document_id = str(uuid.uuid4())
                upload.file.seek(0)
                stored = self.storage.store(
                    stream=upload.file,
                    original_filename=upload.filename or "",
                    declared_mime_type=getattr(upload, "content_type", None),
                    user_id=user_id,
                    document_id=document_id,
                )
                stored_paths.append(stored.path)
                duplicate = self.repository.find_duplicate(user_id, stored.sha256)
                if duplicate is not None:
                    self.storage.remove(stored.path)
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
                created.append(self.repository.add(document))
            self.session.commit()
            return created
        except Exception:
            self.session.rollback()
            for path in stored_paths:
                self.storage.remove(path)
            raise

    def delete(self, *, user_id: str, document_id: str) -> DocumentDeleteResponse:
        document = self.repository.get_owned(user_id, document_id)
        if document is None:
            raise NotFoundError("Document not found.", code="document_not_found")
        if document.status == "deleted":
            return DocumentDeleteResponse(
                document_id=document.id,
                status="deleted",
                message="Document was already deleted.",
            )

        document.status = "deletion_pending"
        document.error_message = None
        self.session.commit()
        deleted_points = 0
        try:
            store = self._qdrant_store or self._create_qdrant_store()
            for collection in (
                self.settings.qdrant_collection_semantic,
                self.settings.qdrant_collection_docling,
            ):
                deleted_points += store.delete_user_document_points(
                    collection_name=collection,
                    user_id=user_id,
                    document_id=document.id,
                )
            self.storage.remove(document.storage_path)
            document.status = "deleted"
            document.deleted_at = utc_now()
            document.error_message = None
            self.session.commit()
            return DocumentDeleteResponse(
                document_id=document.id,
                status="deleted",
                deleted_points=deleted_points,
                message="Document deleted successfully.",
            )
        except Exception as exc:
            document.status = "deletion_pending"
            document.error_message = f"Deletion failed and can be retried: {str(exc)[:1000]}"
            self.session.commit()
            raise ProviderError(
                "Document deletion did not complete. The document remains in a retryable deletion_pending state.",
                code="document_deletion_failed",
            ) from exc

    def _create_qdrant_store(self) -> QdrantChunkStore:
        return QdrantChunkStore(
            url=self.settings.qdrant_url,
            api_key=self.settings.qdrant_api_key,
            dense_vector_name=self.settings.qdrant_dense_vector_name,
            sparse_vector_name=self.settings.qdrant_sparse_vector_name,
            dense_vector_size=self.settings.qdrant_dense_vector_size,
        )
