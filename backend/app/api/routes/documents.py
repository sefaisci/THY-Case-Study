"""Owner-scoped document upload, listing, and deletion routes."""

from typing import Annotated, Literal

from fastapi import APIRouter, File, Form, UploadFile, status

from ...schemas.documents import (
    DocumentDeleteResponse,
    DocumentResponse,
    MultiUploadResponse,
)
from ...services import DocumentService
from ..dependencies import (
    ApplicationSettings,
    CurrentUser,
    DatabaseSession,
    ModelCatalog,
)

router = APIRouter(prefix="/documents", tags=["documents"])


@router.get("", response_model=list[DocumentResponse], summary="List the current user's documents")
async def list_documents(
    user: CurrentUser,
    session: DatabaseSession,
    settings: ApplicationSettings,
) -> list[DocumentResponse]:
    return DocumentService(session, settings).list_documents(user.id)


@router.post(
    "/upload",
    response_model=MultiUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Validate and persist multiple source documents without starting ingestion",
)
async def upload_documents(
    user: CurrentUser,
    session: DatabaseSession,
    settings: ApplicationSettings,
    model_catalog: ModelCatalog,
    files: Annotated[list[UploadFile], File(description="PDF, DOCX, or PPTX source files")],
    ingestion_method: Annotated[Literal["semantic", "docling"], Form()],
    semantic_model: Annotated[str | None, Form()] = None,
    semantic_reasoning_effort: Annotated[
        Literal["low", "medium", "high"] | None,
        Form(),
    ] = None,
) -> MultiUploadResponse:
    service = DocumentService(
        session,
        settings,
        model_catalog=model_catalog,
    )
    documents = service.upload_many(
        user_id=user.id,
        files=files,
        ingestion_method=ingestion_method,
        semantic_model=semantic_model,
        semantic_reasoning_effort=semantic_reasoning_effort,
    )
    return MultiUploadResponse(
        documents=[DocumentResponse.model_validate(item) for item in documents],
        message="Files uploaded and are pending ingestion.",
    )


@router.delete(
    "/{document_id}",
    response_model=DocumentDeleteResponse,
    summary="Delete one owned document from Qdrant, file storage, and active metadata",
)
async def delete_document(
    document_id: str,
    user: CurrentUser,
    session: DatabaseSession,
    settings: ApplicationSettings,
) -> DocumentDeleteResponse:
    return DocumentService(session, settings).delete(
        user_id=user.id,
        document_id=document_id,
    )
