"""Chat session, message, citation, and RAG response schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .usage import UsageTotals


class ChatSessionCreate(BaseModel):
    title: str | None = Field(default=None, max_length=200)


class ChatSessionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    session_identifier: str
    title: str
    created_at: datetime
    last_activity_at: datetime


class CitationResponse(BaseModel):
    filename: str
    document_id: str
    page_number: int | None = None
    slide_number: int | None = None
    chunk_id: str
    source_excerpt: str
    retrieval_score: float
    ingestion_method: Literal["semantic", "docling"]
    source_collection: Literal["semantic_chunks", "docling_fixed_chunks"]
    source_pipeline: str


class ChatMessageResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    role: Literal["user", "assistant"]
    content: str
    citations: list[CitationResponse] = Field(default_factory=list)
    model: str | None = None
    reasoning_effort: str | None = None
    latency_ms: int | None = Field(default=None, ge=0)
    created_at: datetime


class ChatMessageRequest(BaseModel):
    question: str = Field(min_length=1, max_length=20_000)
    chat_model: str = Field(min_length=1, max_length=120)
    chat_reasoning_effort: Literal["low", "medium", "high"]
    collection_scope: Literal["semantic", "docling", "both"] = "both"


class ChatTurnResponse(BaseModel):
    user_message: ChatMessageResponse
    assistant_message: ChatMessageResponse
    no_answer: bool
    checked_collections: list[str]
    request_usage: UsageTotals
    session_usage: UsageTotals
    total_usage: UsageTotals
