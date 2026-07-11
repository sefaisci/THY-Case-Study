"""Shared schemas for the agentic RAG graph."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import TypedDict

# Collection names are configured through the environment. Keeping the alias
# open prevents ingestion and retrieval from diverging when operators use
# non-default Qdrant collection names.
CollectionName = str
CollectionScope = Literal["semantic", "docling", "both"]
ResponseMode = Literal["grounded", "conversational"]
GroundingStatus = Literal["grounded", "weak", "unsupported"]
HallucinationRisk = Literal["low", "medium", "high"]
QueryVariantKind = Literal[
    "verbatim",
    "standalone",
    "english",
    "keywords",
    "source_style",
]


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
    """Four generated retrieval forms paired with the verbatim user question."""

    standalone_query: str = Field(min_length=1, max_length=2_000)
    english_query: str = Field(min_length=1, max_length=2_000)
    keyword_query: str = Field(min_length=1, max_length=2_000)
    source_style_query: str = Field(min_length=1, max_length=2_000)


class QueryVariant(BaseModel):
    """One stable query-map input used for independent embedding and retrieval."""

    id: str = Field(min_length=1, max_length=80)
    kind: QueryVariantKind
    text: str = Field(min_length=1, max_length=4_000)
    weight: float = Field(gt=0.0, le=1.0)


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
    fusion_score: float = 0.0
    matched_variant_ids: list[str] = Field(default_factory=list)

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


class RetrievalAdapterResult(BaseModel):
    """One async adapter call result with collection-isolated failures."""

    chunks: list[RetrievedChunk] = Field(default_factory=list)
    errors: list[NodeError] = Field(default_factory=list)


class VariantRetrievalResult(BaseModel):
    """Map result for one uniquely identified query variant."""

    variant: QueryVariant
    chunks: list[RetrievedChunk] = Field(default_factory=list)
    errors: list[NodeError] = Field(default_factory=list)


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


def merge_variant_results(
    current: list[VariantRetrievalResult] | None,
    update: list[VariantRetrievalResult] | None,
) -> list[VariantRetrievalResult]:
    """Merge parallel query-map results deterministically and idempotently.

    The reducer is associative and commutative because duplicate variants and hits
    are selected by stable extrema rather than arrival order. Replaying the same
    update is idempotent because both variants and hits are keyed before output.
    """

    by_variant: dict[str, VariantRetrievalResult] = {}
    for result in [*(current or []), *(update or [])]:
        result = _normalize_variant_result(result)
        existing = by_variant.get(result.variant.id)
        if existing is None:
            by_variant[result.variant.id] = result
            continue

        variant = min(
            (existing.variant, result.variant),
            key=lambda item: (item.kind, item.text.casefold(), item.text, -item.weight),
        )
        chunks: dict[tuple[str, str], RetrievedChunk] = {}
        for chunk in [*existing.chunks, *result.chunks]:
            key = (chunk.collection_name, chunk.chunk_id)
            selected = chunks.get(key)
            if selected is None or _chunk_reducer_key(chunk) < _chunk_reducer_key(selected):
                chunks[key] = chunk
        errors = {
            (error.node, error.message): error
            for error in [*existing.errors, *result.errors]
        }
        by_variant[result.variant.id] = VariantRetrievalResult(
            variant=variant,
            chunks=sorted(chunks.values(), key=_chunk_reducer_key),
            errors=[errors[key] for key in sorted(errors)],
        )

    return [by_variant[key] for key in sorted(by_variant)]


def _normalize_variant_result(
    result: VariantRetrievalResult,
) -> VariantRetrievalResult:
    """Canonicalize one branch result before it enters the CRDT-like reducer."""

    chunks: dict[tuple[str, str], RetrievedChunk] = {}
    for chunk in result.chunks:
        key = (chunk.collection_name, chunk.chunk_id)
        selected = chunks.get(key)
        if selected is None or _chunk_reducer_key(chunk) < _chunk_reducer_key(selected):
            chunks[key] = chunk
    errors = {(error.node, error.message): error for error in result.errors}
    return VariantRetrievalResult(
        variant=result.variant.model_copy(deep=True),
        chunks=sorted(chunks.values(), key=_chunk_reducer_key),
        errors=[errors[key] for key in sorted(errors)],
    )


def _chunk_reducer_key(chunk: RetrievedChunk) -> tuple:
    """Return a total ordering that prefers the strongest duplicate hit."""

    return (
        -chunk.retrieval_score,
        chunk.collection_name,
        chunk.document_id,
        chunk.chunk_id,
        chunk.document_name,
        chunk.model_dump_json(),
    )


class RagState(TypedDict, total=False):
    """LangGraph state shared by all RAG nodes and subgraphs."""

    user_id: str
    thread_id: str | None
    collection_scope: CollectionScope
    conversation_history: list[ConversationTurn]
    question: str
    normalized_question: str
    retrieval_query: str
    query_variants: list[QueryVariant]
    query_variant: QueryVariant
    query_intent: str
    documents_available: bool
    response_mode: ResponseMode
    retrieval_plan: RetrievalPlan
    variant_results: Annotated[list[VariantRetrievalResult], merge_variant_results]
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
