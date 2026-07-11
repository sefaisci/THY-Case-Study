"""Requests and results for selectable multi-document ingestion."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

IngestionMethod = Literal["semantic", "docling"]


class LocationFailure(BaseModel):
    """Bounded page or slide failure information."""

    location_number: int | None = None
    message: str


class DocumentIngestionResult(BaseModel):
    """Outcome for one source document."""

    document_id: str
    document_name: str
    document_type: Literal["pdf", "docx", "pptx"]
    method: IngestionMethod
    status: Literal["completed", "partial", "failed", "skipped"]
    total_locations: int = Field(default=0, ge=0)
    processed_locations: int = 0
    chunk_count: int = 0
    point_count: int = 0
    failures: list[LocationFailure] = Field(default_factory=list)


class IngestionRequest(BaseModel):
    """One optional-path, selectable-method ingestion request."""

    method: IngestionMethod
    user_id: str = Field(min_length=1)
    pdf_path: Path | None = None
    docx_path: Path | None = None
    pptx_path: Path | None = None
    filename_allowlist: set[str] | None = None


class IngestionRunResult(BaseModel):
    """Aggregate outcome for a multi-format ingestion run."""

    method: IngestionMethod
    user_id: str
    discovered_count: int
    completed_count: int = 0
    partial_count: int = 0
    failed_count: int = 0
    skipped_count: int = 0
    chunk_count: int = 0
    point_count: int = 0
    documents: list[DocumentIngestionResult] = Field(default_factory=list)
