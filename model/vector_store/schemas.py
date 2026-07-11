"""Normalized chunk record shared by semantic and Docling ingestion."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

CollectionName = str
DocumentType = Literal["pdf", "docx", "pptx"]


class ChunkRecord(BaseModel):
    """Citation-ready Qdrant payload before vectorization."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    collection_name: CollectionName = Field(min_length=1)
    collection_type: Literal["semantic", "docling_fixed"]
    user_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    document_name: str = Field(min_length=1)
    document_type: DocumentType
    chunk_id: str = Field(min_length=1)
    chunk_index: int = Field(ge=0)
    source_pipeline: Literal["semantic_image_chunking", "docling_fixed_chunking"]
    source_excerpt: str = Field(min_length=1, max_length=2000)
    text: str = Field(min_length=1)
    created_at: str
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    embedding_model: str
    sparse_encoder_version: str
    page_number: int | None = Field(default=None, ge=1)
    slide_number: int | None = Field(default=None, ge=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_location(self) -> "ChunkRecord":
        """Require the correct page/slide location for the document type."""

        if self.document_type == "pptx":
            if self.slide_number is None or self.page_number is not None:
                raise ValueError("PPTX chunks require slide_number only.")
        elif self.page_number is None or self.slide_number is not None:
            raise ValueError("PDF and DOCX chunks require page_number only.")
        return self

    def payload(self) -> dict[str, Any]:
        """Return a flat Qdrant payload with normalized metadata."""

        data = self.model_dump(exclude={"metadata"}, exclude_none=True)
        data.update(self.metadata)
        data.update(
            {
                "filename": self.document_name,
                "ingestion_method": (
                    "semantic" if self.collection_type == "semantic" else "docling"
                ),
                "chunk_text": self.text,
                "source_location": {
                    "page_number": self.page_number,
                    "slide_number": self.slide_number,
                },
                "citation_metadata": {
                    "document_id": self.document_id,
                    "filename": self.document_name,
                    "chunk_id": self.chunk_id,
                    "page_number": self.page_number,
                    "slide_number": self.slide_number,
                    "source_excerpt": self.source_excerpt,
                    "source_collection": self.collection_name,
                },
            }
        )
        return data
