"""Document upload, ingestion, listing, and deletion schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

DocumentStatus = Literal[
    "pending",
    "processing",
    "completed",
    "failed",
    "deletion_pending",
    "deleted",
]
IngestionMethod = Literal["semantic", "docling"]


class DocumentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    filename: str
    mime_type: str
    file_extension: str
    file_size_bytes: int
    sha256: str
    uploaded_at: datetime
    ingestion_method: IngestionMethod
    semantic_model: str | None = None
    semantic_reasoning_effort: str | None = None
    collection_name: str
    status: DocumentStatus
    error_message: str | None = None


class MultiUploadResponse(BaseModel):
    documents: list[DocumentResponse]
    message: str


class IngestionStartRequest(BaseModel):
    document_ids: list[str] = Field(min_length=1, max_length=50)


class IngestionStatusRequest(BaseModel):
    job_ids: list[str] = Field(min_length=1, max_length=100)


class IngestionJobResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    document_id: str
    status: Literal["pending", "processing", "completed", "failed"]
    total_pages: int = Field(ge=0)
    processed_pages: int = Field(ge=0)
    progress_percent: int = Field(default=0, ge=0, le=100)
    chunk_count: int
    point_count: int
    started_at: datetime | None = None
    completed_at: datetime | None = None
    failure_message: str | None = None
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    total_tokens: int
    cost_usd: float
    message: str | None = None


class IngestionBatchResponse(BaseModel):
    jobs: list[IngestionJobResponse]
    message: str


class IngestionStatusResponse(BaseModel):
    jobs: list[IngestionJobResponse]


class DocumentDeleteResponse(BaseModel):
    document_id: str
    status: Literal["deleted", "deletion_pending"]
    deleted_points: int = 0
    message: str
