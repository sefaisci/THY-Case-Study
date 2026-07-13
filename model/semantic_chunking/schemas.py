"""Strict, flat structured-output schemas for semantic page chunking."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SemanticChunk(BaseModel):
    """One flat, variable-length semantic unit from the current page image."""

    model_config = ConfigDict(extra="forbid", strict=True)

    chunk_key: str = Field(min_length=1, max_length=120, pattern=r"^[a-zA-Z0-9_-]+$")
    title: str = Field(min_length=1, max_length=240)
    text: str = Field(min_length=1)
    keywords: list[str] = Field(default_factory=list, max_length=30)
    relationships: list[str] = Field(default_factory=list, max_length=30)
    source_excerpt: str = Field(min_length=1, max_length=1500)
    confidence: float = Field(ge=0.0, le=1.0)


class SemanticPageResult(BaseModel):
    """Validated semantic result for one page or slide image."""

    model_config = ConfigDict(extra="forbid", strict=True)

    page_number: int = Field(ge=1)
    page_classification: Literal["content", "blank"]
    page_summary: str = Field(min_length=1, max_length=2000)
    chunks: list[SemanticChunk] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list, max_length=20)

    @model_validator(mode="after")
    def validate_page_contract(self) -> "SemanticPageResult":
        """Require explicit page coverage and unambiguous chunk identifiers."""

        if self.page_classification == "content" and not self.chunks:
            raise ValueError("A content page must contain at least one semantic chunk.")
        if self.page_classification == "blank" and self.chunks:
            raise ValueError("A blank page must not contain semantic chunks.")

        keys = [chunk.chunk_key for chunk in self.chunks]
        if len(keys) != len(set(keys)):
            raise ValueError("Semantic chunk keys must be unique within a page or slide.")
        return self
