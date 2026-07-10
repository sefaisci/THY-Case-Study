"""Shared schemas for document discovery and page rendering."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

DocumentType = Literal["pdf", "docx", "pptx"]


class DocumentSource(BaseModel):
    """Stable identity and filesystem metadata for one source document."""

    model_config = ConfigDict(frozen=True)

    path: Path
    document_id: str = Field(min_length=1)
    document_name: str = Field(min_length=1)
    document_type: DocumentType
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class DiscoveredInputs(BaseModel):
    """Deterministically ordered documents discovered from optional inputs."""

    documents: list[DocumentSource] = Field(default_factory=list)
    supplied_paths: dict[DocumentType, Path | None] = Field(default_factory=dict)


class RenderedPage(BaseModel):
    """One page or slide image with citation-ready source metadata."""

    model_config = ConfigDict(frozen=True)

    document_id: str
    document_name: str
    document_type: DocumentType
    source_path: Path
    source_sha256: str
    image_path: Path
    image_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    mime_type: Literal["image/png"] = "image/png"
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    dpi: int = Field(gt=0)
    page_number: int | None = Field(default=None, ge=1)
    slide_number: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_location(self) -> "RenderedPage":
        """Require exactly one location whose kind matches the source type."""

        if self.document_type == "pptx":
            if self.slide_number is None or self.page_number is not None:
                raise ValueError("PPTX artifacts require slide_number only.")
        elif self.page_number is None or self.slide_number is not None:
            raise ValueError("PDF and DOCX artifacts require page_number only.")
        return self

    @property
    def location_number(self) -> int:
        """Return the source page or slide number."""

        return self.slide_number or self.page_number or 0
