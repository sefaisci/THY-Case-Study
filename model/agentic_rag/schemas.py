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
ResponseMode = Literal[
    "grounded",
    "hybrid",
    "general_knowledge",
    "conversational",
]
GroundingStatus = Literal["grounded", "weak", "unsupported"]
HallucinationRisk = Literal["low", "medium", "high"]
QuestionCoverage = Literal["full", "partial", "none"]
QueryVariantKind = Literal["verbatim", "standalone"]
RerankSupport = Literal["direct", "partial", "none"]


class NodeError(BaseModel):
    """Recoverable node error captured in graph state."""

    node: str
    message: str


class RetrievalPlan(BaseModel):
    """Collection and scoring plan for one retrieval turn."""

    collections: list[CollectionName] = Field(default_factory=list)
    prefetch_k: int = Field(default=20, gt=0)
    collection_k: int = Field(default=15, gt=0)
    candidate_k: int = Field(default=30, gt=0)
    rerank_top_k: int = Field(default=6, gt=0)
    dense_weight: float = Field(default=0.65, ge=0.0)
    sparse_weight: float = Field(default=0.35, ge=0.0)
    reason: str = ""


class OpenAIRerankCandidate(BaseModel):
    """One model-scored candidate from the OpenAI reranker."""

    chunk_id: str = Field(
        min_length=5,
        max_length=9,
        pattern=r"^c[0-9]{4,8}$",
    )
    relevance_score: float = Field(ge=0.0, le=1.0)
    support: RerankSupport


class OpenAIRerankResult(BaseModel):
    """Structured provider response for one reranking turn."""

    sufficient_evidence: bool
    ranked_candidates: list[OpenAIRerankCandidate] = Field(default_factory=list)


class RetrievalQueryRewrite(BaseModel):
    """One standalone retrieval query for a referential follow-up."""

    standalone_query: str = Field(min_length=1, max_length=20_000)


class QueryVariant(BaseModel):
    """One stable query-map input used for independent embedding and retrieval."""

    id: str = Field(min_length=1, max_length=80)
    kind: QueryVariantKind
    text: str = Field(min_length=1, max_length=20_000)
    weight: float = Field(gt=0.0, le=1.0)


class RetrievedChunk(BaseModel):
    """Citation-ready chunk returned by a retrieval adapter."""

    model_config = ConfigDict(extra="allow")

    user_id: str
    document_id: str
    document_name: str
    document_type: str
    chunk_id: str
    evidence_id: str | None = None
    collection_name: CollectionName
    source_pipeline: str
    source_excerpt: str
    text: str
    retrieval_score: float
    retrieval_rank: int | None = Field(default=None, ge=0)
    created_at: str | None = None
    page_number: int | None = None
    slide_number: int | None = None
    collection_type: str | None = None
    sparse_encoder_version: str = ""
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


class RerankAdapterResult(BaseModel):
    """Validated reranker result consumed by the graph."""

    chunks: list[RetrievedChunk] = Field(default_factory=list)
    sufficient_evidence: bool = False


class Citation(BaseModel):
    """Source attribution attached to a final answer."""

    document_name: str
    document_id: str
    page_number: int | None = None
    slide_number: int | None = None
    chunk_id: str
    evidence_id: str | None = Field(default=None, exclude=True)
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
    attempted_collections: list[CollectionName] = Field(default_factory=list)
    successful_collections: list[CollectionName] = Field(default_factory=list)

    @property
    def retrieval_succeeded(self) -> bool:
        """Return whether at least one selected collection completed normally."""

        return bool(self.successful_collections)


class VariantRetrievalResult(BaseModel):
    """Map result for one uniquely identified query variant."""

    variant: QueryVariant
    chunks: list[RetrievedChunk] = Field(default_factory=list)
    errors: list[NodeError] = Field(default_factory=list)
    attempted_collections: list[CollectionName] = Field(default_factory=list)
    successful_collections: list[CollectionName] = Field(default_factory=list)


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
    question_coverage: QuestionCoverage
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
    citation_validation: CitationValidationResult | None = Field(
        default=None,
        exclude=True,
    )
    reflection: ReflectionResult | None = Field(default=None, exclude=True)
    errors: list[NodeError] = Field(default_factory=list, exclude=True)


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
            attempted_collections=sorted(
                set(existing.attempted_collections)
                | set(result.attempted_collections)
            ),
            successful_collections=sorted(
                set(existing.successful_collections)
                | set(result.successful_collections)
            ),
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
        attempted_collections=sorted(set(result.attempted_collections)),
        successful_collections=sorted(set(result.successful_collections)),
    )


def _chunk_reducer_key(chunk: RetrievedChunk) -> tuple:
    """Preserve explicit adapter rank with a stable legacy fallback."""

    stable_key = (
        -chunk.retrieval_score,
        chunk.collection_name,
        chunk.document_id,
        chunk.chunk_id,
        chunk.document_name,
        chunk.model_dump_json(),
    )
    if chunk.retrieval_rank is None:
        return (1, *stable_key)
    return (0, chunk.retrieval_rank, *stable_key)


class RagState(TypedDict, total=False):
    """LangGraph state shared by all RAG nodes and subgraphs."""

    user_id: str
    thread_id: str | None
    collection_scope: CollectionScope
    conversation_history: list[ConversationTurn]
    question: str
    normalized_question: str
    retrieval_query: str
    retrieval_turn_sequence: int
    query_variants: list[QueryVariant]
    query_variant: QueryVariant
    query_intent: str
    documents_available: bool
    response_mode: ResponseMode | None
    retrieval_plan: RetrievalPlan
    variant_results: Annotated[list[VariantRetrievalResult], merge_variant_results]
    retrieved_chunks: list[RetrievedChunk]
    reranked_chunks: list[RetrievedChunk]
    retrieval_succeeded: bool
    successful_retrieval_collections: list[CollectionName]
    failed_retrieval_collections: list[CollectionName]
    evidence_sufficient: bool
    draft_answer: str
    grounded_draft: str
    grounded_repair_attempted: bool
    grounded_repair_feedback: str
    grounded_answer: str
    general_knowledge_answer: str
    fallback_eligible: bool
    fallback_reason: str | None
    citations: list[Citation]
    citation_validation: CitationValidationResult | None
    reflection: ReflectionResult | None
    final_answer: str
    response: RagResponse | None
    checked_collections: list[CollectionName]
    no_answer: bool
    errors: Annotated[list[NodeError], append_node_errors]
