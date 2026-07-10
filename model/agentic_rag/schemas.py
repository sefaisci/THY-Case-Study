"""Shared schemas for the agentic RAG graph."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import TypedDict

CollectionName = Literal["semantic_chunks", "docling_fixed_chunks"]
CollectionScope = Literal["semantic", "docling", "both"]
ResponseMode = Literal["grounded", "conversational"]
GroundingStatus = Literal["grounded", "weak", "unsupported"]
HallucinationRisk = Literal["low", "medium", "high"]


class NodeError(BaseModel):
    """Recoverable node error captured in graph state."""

    node: str
    message: str


class RetrievalPlan(BaseModel):
    """Collection and scoring plan for one retrieval turn."""

    collections: list[CollectionName] = Field(default_factory=list)
    top_k: int = 12
    rerank_top_k: int = 6
    dense_weight: float = 0.65
    sparse_weight: float = 0.35
    reason: str = ""


class RetrievalQueryRewrite(BaseModel):
    """Standalone and English retrieval forms of one user question."""

    standalone_query: str = Field(min_length=1, max_length=2_000)
    english_query: str = Field(min_length=1, max_length=2_000)


class RetrievedChunk(BaseModel):
    """Citation-ready chunk returned by a retrieval adapter."""

    model_config = ConfigDict(extra="allow")

    user_id: str
    document_id: str
    document_name: str
    document_type: str
    chunk_id: str
    collection_name: CollectionName
    source_pipeline: str
    source_excerpt: str
    text: str
    retrieval_score: float
    created_at: str | None = None
    page_number: int | None = None
    slide_number: int | None = None
    collection_type: str | None = None
    rerank_score: float | None = None

    @property
    def display_location(self) -> str:
        """Return a human-readable page or slide location."""

        if self.page_number is not None:
            return f"page {self.page_number}"
        if self.slide_number is not None:
            return f"slide {self.slide_number}"
        return "unknown location"

    @property
    def effective_score(self) -> float:
        """Return the rerank score when available, otherwise the retrieval score."""

        return self.rerank_score if self.rerank_score is not None else self.retrieval_score


class Citation(BaseModel):
    """Source attribution attached to a final answer."""

    document_name: str
    document_id: str
    page_number: int | None = None
    slide_number: int | None = None
    chunk_id: str
    source_excerpt: str
    retrieval_score: float
    collection_name: CollectionName
    ingestion_method: Literal["semantic", "docling"]
    source_pipeline: str
    grounding_indicator: GroundingStatus = "grounded"


class ConversationTurn(BaseModel):
    """One session-scoped chat turn supplied as short-term context."""

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=20_000)


class CitationValidationResult(BaseModel):
    """Exact chunk-ID safety validation with non-blocking coverage diagnostics."""

    is_valid: bool
    cited_chunk_ids: list[str] = Field(default_factory=list)
    unknown_chunk_ids: list[str] = Field(default_factory=list)
    cross_user_chunk_ids: list[str] = Field(default_factory=list)
    missing_citation_sentences: list[str] = Field(default_factory=list)


class ClaimEvaluation(BaseModel):
    """Claim-level evidence decision returned by the grounding evaluator."""

    claim_text: str
    cited_chunk_ids: list[str] = Field(default_factory=list)
    supported: bool
    evidence_rationale: str


class ReflectionResult(BaseModel):
    """Grounding critique produced before final response generation."""

    is_grounded: bool
    hallucination_risk: HallucinationRisk
    decision: Literal["accept", "revise", "no_answer"]
    claims: list[ClaimEvaluation] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)
    missing_citations: list[str] = Field(default_factory=list)
    notes: str = ""


class RagRequest(BaseModel):
    """External request shape for a single RAG question."""

    question: str = Field(min_length=1, max_length=20_000)
    user_id: str = Field(min_length=1)
    thread_id: str | None = None
    collection_scope: CollectionScope = "both"
    conversation_history: list[ConversationTurn] = Field(default_factory=list)


class RagResponse(BaseModel):
    """External response shape returned by the graph runner."""

    answer: str
    citations: list[Citation] = Field(default_factory=list)
    no_answer: bool = False
    checked_collections: list[CollectionName] = Field(default_factory=list)
    citation_validation: CitationValidationResult | None = None
    reflection: ReflectionResult | None = None
    errors: list[NodeError] = Field(default_factory=list)


def append_node_errors(
    current: list[NodeError] | None, update: list[NodeError] | None
) -> list[NodeError]:
    """Append node errors while preserving previous graph state."""

    return (current or []) + (update or [])


class RagState(TypedDict, total=False):
    """LangGraph state shared by all RAG nodes and subgraphs."""

    user_id: str
    thread_id: str | None
    collection_scope: CollectionScope
    conversation_history: list[ConversationTurn]
    question: str
    normalized_question: str
    retrieval_query: str
    query_intent: str
    documents_available: bool
    response_mode: ResponseMode
    retrieval_plan: RetrievalPlan
    retrieved_chunks: list[RetrievedChunk]
    reranked_chunks: list[RetrievedChunk]
    evidence_sufficient: bool
    draft_answer: str
    citations: list[Citation]
    citation_validation: CitationValidationResult
    reflection: ReflectionResult
    final_answer: str
    response: RagResponse
    checked_collections: list[CollectionName]
    no_answer: bool
    errors: Annotated[list[NodeError], append_node_errors]
