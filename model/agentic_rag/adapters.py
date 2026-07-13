"""Adapter interfaces and local implementations for the RAG graph."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import logging
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from model.vector_store import (
    SparseEncoder,
    build_user_documents_filter,
    build_user_filter,
    create_sparse_encoder,
)
from model.usage import UsageCallback, emit_usage, usage_from_response

from .schemas import (
    ClaimEvaluation,
    CollectionName,
    NodeError,
    ReflectionResult,
    RerankAdapterResult,
    RetrievalAdapterResult,
    RetrievalPlan,
    RetrievalQueryRewrite,
    RetrievedChunk,
)
from .settings import RagSettings

_BACKGROUND_CLEANUP_TASKS: set[asyncio.Task[Any]] = set()
logger = logging.getLogger(__name__)


class LlmAdapter(Protocol):
    """Text generation boundary used by graph nodes."""

    async def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        reasoning_effort: str | None = None,
        usage_stage: str = "answer_generation",
    ) -> str:
        """Return model output for a grounded prompt."""


class EmbeddingAdapter(Protocol):
    """Embedding boundary used by retrieval adapters."""

    async def embed_query(
        self,
        text: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> list[float]:
        """Return a dense vector for a query string."""


class QueryRewriterAdapter(Protocol):
    """Boundary for resolving one referential follow-up query."""

    async def rewrite(self, question: str) -> RetrievalQueryRewrite:
        """Return one faithful standalone retrieval query."""


class HybridRetrievalAdapter(Protocol):
    """Retrieval boundary for dense plus sparse capable search."""

    async def retrieve(
        self,
        *,
        query: str,
        dense_vector: list[float],
        plan: RetrievalPlan,
        user_id: str,
    ) -> RetrievalAdapterResult:
        """Return user-scoped retrieval candidates."""


class RerankerAdapter(Protocol):
    """Replaceable reranker boundary."""

    async def rerank(
        self,
        *,
        question: str,
        chunks: list[RetrievedChunk],
        limit: int,
        user_id: str,
    ) -> RerankAdapterResult:
        """Return validated chunks and an explicit evidence decision."""


class GroundingEvaluatorAdapter(Protocol):
    """Claim-evidence evaluation boundary used after citation validation."""

    async def evaluate(
        self,
        *,
        question: str,
        draft_answer: str,
        cited_chunks: list[RetrievedChunk],
    ) -> ReflectionResult:
        """Return a structured claim-level grounding decision."""


def _balanced_take(
    chunks: list[RetrievedChunk],
    collections: list[CollectionName],
    limit: int,
) -> list[RetrievedChunk]:
    """Reserve equal pre-rerank capacity for every collection with hits.

    Callers provide candidates in their final deterministic ranking. Configured
    collection order decides quota allocation and interleaving; unused capacity
    is backfilled in the caller's ranking rather than by an unrelated raw score.
    """

    if limit <= 0:
        return []
    configured_collections = list(dict.fromkeys(collections))
    configured_set = set(configured_collections)
    representatives: dict[tuple[str, str], RetrievedChunk] = {}
    ordered_keys: list[tuple[str, str]] = []
    for chunk in chunks:
        if chunk.collection_name not in configured_set:
            continue
        key = (chunk.collection_name, chunk.chunk_id)
        current = representatives.get(key)
        if current is None:
            ordered_keys.append(key)
            representatives[key] = chunk
        elif _retrieved_chunk_representative_key(
            chunk
        ) < _retrieved_chunk_representative_key(current):
            representatives[key] = chunk
    normalized_chunks = [representatives[key] for key in ordered_keys]
    grouped = {
        collection: [
            chunk
            for chunk in normalized_chunks
            if chunk.collection_name == collection
        ]
        for collection in configured_collections
    }
    active = [
        collection
        for collection in configured_collections
        if grouped[collection]
    ]
    if not active:
        return []

    base, remainder = divmod(limit, len(active))
    quotas = {
        collection: base + (1 if index < remainder else 0)
        for index, collection in enumerate(active)
    }
    selected: list[RetrievedChunk] = []
    selected_keys: set[tuple[str, str]] = set()
    for rank in range(max(quotas.values(), default=0)):
        for collection in active:
            candidates = grouped[collection]
            if rank >= quotas[collection] or rank >= len(candidates):
                continue
            chunk = candidates[rank]
            selected.append(chunk)
            selected_keys.add((chunk.collection_name, chunk.chunk_id))

    remaining = [
        chunk
        for chunk in normalized_chunks
        if (chunk.collection_name, chunk.chunk_id) not in selected_keys
    ]
    return [*selected, *remaining][:limit]


def _retrieved_chunk_representative_key(chunk: RetrievedChunk) -> tuple:
    """Return a total server-metadata ordering for duplicate retrieval hits."""

    return (
        -chunk.retrieval_score,
        chunk.collection_name,
        chunk.document_id,
        chunk.chunk_id,
        chunk.document_name,
        chunk.document_type,
        chunk.page_number if chunk.page_number is not None else -1,
        chunk.slide_number if chunk.slide_number is not None else -1,
        chunk.source_pipeline,
        chunk.created_at or "",
        chunk.model_dump_json(),
    )


@dataclass(frozen=True)
class RagAdapters:
    """Adapter bundle injected into LangGraph node closures."""

    llm: LlmAdapter
    embedding: EmbeddingAdapter
    retrieval: HybridRetrievalAdapter
    reranker: RerankerAdapter
    grounding: GroundingEvaluatorAdapter
    query_rewriter: QueryRewriterAdapter | None = None
    _owned_clients: tuple[Any, ...] = field(
        default_factory=tuple,
        repr=False,
        compare=False,
    )
    _closed: bool = field(default=False, init=False, repr=False, compare=False)

    async def aclose(self) -> None:
        """Close only provider clients created by the adapter factory.

        Caller-supplied clients are deliberately excluded because their lifecycle
        belongs to the caller. Closing is idempotent and attempts every owned
        client before reporting any cleanup failures.
        """

        if self._closed:
            return
        object.__setattr__(self, "_closed", True)
        failures: list[Exception] = []
        seen: set[int] = set()
        for client in self._owned_clients:
            identity = id(client)
            if identity in seen:
                continue
            seen.add(identity)
            close = getattr(client, "close", None)
            if close is None:
                close = getattr(client, "aclose", None)
            if close is None:
                continue
            try:
                result = close()
                if inspect.isawaitable(result):
                    await result
            except Exception as exc:  # pragma: no cover - provider cleanup failure
                failures.append(exc)
        if failures:
            raise ExceptionGroup("Failed to close owned RAG provider clients.", failures)

    async def __aenter__(self) -> RagAdapters:
        """Support explicit async context-manager ownership."""

        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        """Close factory-owned clients when leaving an async context."""

        del exc_type, exc, traceback
        await self.aclose()


class FakeEmbeddingAdapter:
    """Deterministic embedding adapter for offline notebook smoke tests."""

    def __init__(self, vector_size: int = 32) -> None:
        self.vector_size = vector_size

    async def embed_query(
        self,
        text: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> list[float]:
        """Create a deterministic pseudo-vector without external calls."""

        del metadata
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        values = []
        for index in range(self.vector_size):
            byte = digest[index % len(digest)]
            values.append((byte / 255.0) * 2.0 - 1.0)
        return values


class FakeLlmAdapter:
    """Grounded fake LLM used when API keys or services are unavailable."""

    async def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        reasoning_effort: str | None = None,
        usage_stage: str = "answer_generation",
    ) -> str:
        """Produce a deterministic answer from provided evidence lines."""

        del reasoning_effort, usage_stage
        if "general model knowledge assistant" in system_prompt:
            try:
                payload = json.loads(user_prompt)
            except json.JSONDecodeError:
                payload = {}
            question = str(payload.get("current_question", "")).strip()
            return f"General model knowledge response for: {question}"
        if "helpful conversational assistant" in system_prompt:
            try:
                payload = json.loads(user_prompt)
            except json.JSONDecodeError:
                payload = {}
            history = payload.get("session_history", [])
            current_message = str(payload.get("current_user_message", "")).strip()
            prior_user_messages = [
                str(item.get("content", "")).strip()
                for item in history
                if item.get("role") == "user" and str(item.get("content", "")).strip()
            ]
            history_question = bool(
                re.search(
                    r"(?:earlier|previous|before|daha önce|daha once|önceki|onceki|geçmiş|gecmis)",
                    current_message.casefold(),
                )
            )
            if history_question:
                if prior_user_messages:
                    quoted = "; ".join(prior_user_messages)
                    return f"Earlier in this chat, you wrote: {quoted}"
                return "There are no earlier user messages in this chat session."
            return f"I can help with your question: {current_message}"
        try:
            payload = json.loads(user_prompt)
        except json.JSONDecodeError:
            payload = {}
        evidence = payload.get("document_evidence", [])
        if not evidence:
            return "The uploaded documents do not contain enough evidence to answer this question."
        first = evidence[0]
        metadata = first.get("server_owned_metadata", {})
        chunk_marker = str(metadata.get("citation_marker", "")).strip()
        first_evidence = str(first.get("untrusted_source_excerpt", "")).strip()
        if not chunk_marker or not first_evidence:
            return "The uploaded documents do not contain enough evidence to answer this question."
        return (
            "Based only on the retrieved document evidence, "
            f"{first_evidence} {chunk_marker}"
        )


class FakeGroundingEvaluatorAdapter:
    """Deterministic claim evaluator for offline graph tests."""

    async def evaluate(
        self,
        *,
        question: str,
        draft_answer: str,
        cited_chunks: list[RetrievedChunk],
    ) -> ReflectionResult:
        del question
        cited_ids = [
            chunk.evidence_id or chunk.chunk_id for chunk in cited_chunks
        ]
        grounded = bool(draft_answer.strip() and cited_ids)
        return ReflectionResult(
            is_grounded=grounded,
            hallucination_risk="low" if grounded else "high",
            decision="accept" if grounded else "no_answer",
            question_coverage="full" if grounded else "none",
            claims=(
                [
                    ClaimEvaluation(
                        claim_text=draft_answer.strip(),
                        cited_chunk_ids=cited_ids,
                        supported=True,
                        evidence_rationale="Offline fake evaluator received validated cited evidence.",
                    )
                ]
                if grounded
                else []
            ),
            unsupported_claims=[] if grounded else [draft_answer.strip()] if draft_answer.strip() else [],
            missing_citations=[] if cited_ids else ["No validated cited evidence was supplied."],
            notes="Deterministic offline grounding evaluation.",
        )


@dataclass
class FakeHybridRetrievalAdapter:
    """In-memory user-scoped retriever for graph and notebook smoke tests."""

    chunks: list[RetrievedChunk] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.chunks:
            return
        self.chunks = [
            RetrievedChunk(
                user_id="local-demo-user",
                document_id="demo-doc-1",
                document_name="demo-self-service-chatbot.pdf",
                document_type="pdf",
                page_number=1,
                chunk_id="semantic-demo-1",
                collection_name="semantic_chunks",
                collection_type="semantic",
                source_pipeline="semantic_image_chunking",
                source_excerpt=(
                    "The self-service chatbot supports PDF, Word, and PowerPoint "
                    "uploads with citation-ready chunks."
                ),
                text=(
                    "The self-service chatbot supports PDF, Word, and PowerPoint "
                    "uploads. Documents are processed through semantic chunking "
                    "and Docling fixed-size chunking before retrieval."
                ),
                retrieval_score=0.82,
                created_at="2026-07-09T00:00:00Z",
            ),
            RetrievedChunk(
                user_id="local-demo-user",
                document_id="demo-doc-1",
                document_name="demo-self-service-chatbot.pdf",
                document_type="pdf",
                page_number=2,
                chunk_id="docling-demo-1",
                collection_name="docling_fixed_chunks",
                collection_type="docling_fixed",
                source_pipeline="docling_fixed_chunking",
                source_excerpt=(
                    "Qdrant retrieval must apply a user_id payload filter before "
                    "chunks are used for answer generation."
                ),
                text=(
                    "Qdrant dense and sparse hybrid retrieval uses two collections: "
                    "semantic_chunks and docling_fixed_chunks. Every request applies "
                    "a user_id payload filter."
                ),
                retrieval_score=0.78,
                created_at="2026-07-09T00:00:00Z",
            ),
            RetrievedChunk(
                user_id="another-user",
                document_id="private-doc",
                document_name="private.pdf",
                document_type="pdf",
                page_number=1,
                chunk_id="private-1",
                collection_name="semantic_chunks",
                collection_type="semantic",
                source_pipeline="semantic_image_chunking",
                source_excerpt="This private chunk must never appear for another user.",
                text="This private chunk belongs to another user.",
                retrieval_score=0.99,
                created_at="2026-07-09T00:00:00Z",
            ),
        ]

    async def retrieve(
        self,
        *,
        query: str,
        dense_vector: list[float],
        plan: RetrievalPlan,
        user_id: str,
    ) -> RetrievalAdapterResult:
        """Return only chunks that belong to the requested user."""

        del dense_vector
        query_terms = _tokenize(query)
        candidates = [
            chunk
            for chunk in self.chunks
            if chunk.user_id == user_id and chunk.collection_name in plan.collections
        ]
        scored = []
        for chunk in candidates:
            chunk_terms = _tokenize(f"{chunk.text} {chunk.source_excerpt}")
            overlap = len(query_terms & chunk_terms)
            if query_terms and overlap == 0:
                adjusted_score = chunk.retrieval_score * 0.50
            else:
                adjusted_score = min(1.0, chunk.retrieval_score + (overlap * 0.03))
            scored.append(chunk.model_copy(update={"retrieval_score": adjusted_score}))
        scored.sort(key=lambda item: item.retrieval_score, reverse=True)
        return RetrievalAdapterResult(
            chunks=scored[: plan.candidate_k],
            attempted_collections=list(plan.collections),
            successful_collections=list(plan.collections),
        )


class NoOpRerankerAdapter:
    """Reranker that preserves retrieval order."""

    async def rerank(
        self,
        *,
        question: str,
        chunks: list[RetrievedChunk],
        limit: int,
        user_id: str,
    ) -> RerankAdapterResult:
        """Return the first owned candidates without changing order."""

        del question
        selected = [chunk for chunk in chunks if chunk.user_id == user_id][:limit]
        return RerankAdapterResult(
            chunks=selected,
            sufficient_evidence=bool(selected),
        )


class HeuristicRerankerAdapter:
    """Lightweight token-overlap reranker for offline experiments."""

    async def rerank(
        self,
        *,
        question: str,
        chunks: list[RetrievedChunk],
        limit: int,
        user_id: str,
    ) -> RerankAdapterResult:
        """Reorder owned chunks by token overlap and retrieval score."""

        question_terms = _meaningful_tokens(question)
        reranked = []
        for chunk in chunks:
            if chunk.user_id != user_id:
                continue
            chunk_terms = _meaningful_tokens(f"{chunk.text} {chunk.source_excerpt}")
            overlap = len(question_terms & chunk_terms)
            # Preserve semantic retrieval as the primary signal. Lexical overlap is
            # only a small tie-breaker because it is not reliable cross-lingually.
            rerank_score = chunk.retrieval_score + (min(overlap, 3) * 0.005)
            reranked.append(
                chunk.model_copy(
                    update={"rerank_score": min(1.0, rerank_score)}
                )
            )
        reranked.sort(
            key=lambda item: (
                -item.fusion_score,
                -item.effective_score,
                item.collection_name,
                item.document_id,
                item.chunk_id,
            )
        )
        selected = reranked[:limit]
        return RerankAdapterResult(
            chunks=selected,
            sufficient_evidence=bool(selected),
        )


class OpenAIEmbeddingAdapter:
    """OpenAI embedding adapter for real retrieval experiments."""

    def __init__(
        self,
        settings: RagSettings,
        *,
        client: Any | None = None,
        usage_callback: UsageCallback | None = None,
    ) -> None:
        if client is None:
            from openai import AsyncOpenAI

            kwargs = {"api_key": settings.openai_api_key}
            if settings.openai_base_url:
                kwargs["base_url"] = settings.openai_base_url
            client = AsyncOpenAI(**kwargs)
        self._client = client
        self._model = settings.embedding_model
        self._usage_callback = usage_callback

    async def embed_query(
        self,
        text: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> list[float]:
        """Create an embedding vector through the OpenAI API."""

        response = await self._client.embeddings.create(model=self._model, input=text)
        emit_usage(
            self._usage_callback,
            usage_from_response(
                response,
                stage="retrieval_embedding",
                fallback_model=self._model,
                metadata=metadata,
            ),
        )
        return list(response.data[0].embedding)


class OpenAIQueryRewriterAdapter:
    """Responses API adapter for one standalone referential query rewrite."""

    _SYSTEM_PROMPT = (
        "Rewrite the supplied referential follow-up as one standalone document-search "
        "query. Preserve the user's language, entities, filenames, symbols, numbers, "
        "and intent. Do not answer the question, translate it, add facts, generate "
        "keywords, or produce multiple alternatives."
    )

    def __init__(
        self,
        settings: RagSettings,
        *,
        client: Any | None = None,
        usage_callback: UsageCallback | None = None,
    ) -> None:
        if client is None:
            from openai import AsyncOpenAI

            kwargs = {"api_key": settings.openai_api_key}
            if settings.openai_base_url:
                kwargs["base_url"] = settings.openai_base_url
            client = AsyncOpenAI(**kwargs)
        self._client = client
        self._model = settings.self_service_llm_model
        self._reasoning_effort = settings.self_service_reasoning_effort
        self._timeout = settings.llm_request_timeout_seconds
        self._usage_callback = usage_callback

    async def rewrite(self, question: str) -> RetrievalQueryRewrite:
        """Return one structured standalone query."""

        response = await self._client.responses.parse(
            model=self._model,
            reasoning={"effort": self._reasoning_effort},
            instructions=self._SYSTEM_PROMPT,
            input=json.dumps({"question": question}, ensure_ascii=False),
            text_format=RetrievalQueryRewrite,
            max_output_tokens=300,
            timeout=self._timeout,
        )
        emit_usage(
            self._usage_callback,
            usage_from_response(
                response,
                stage="retrieval_query_rewrite",
                fallback_model=self._model,
            ),
        )
        parsed = getattr(response, "output_parsed", None)
        if parsed is None:
            raise RuntimeError("OpenAI returned no parsed retrieval query rewrite.")
        if isinstance(parsed, RetrievalQueryRewrite):
            return parsed
        return RetrievalQueryRewrite.model_validate(parsed)


class OpenAILlmAdapter:
    """OpenAI Responses API adapter for grounded answer generation."""

    _SUPPORTED_REASONING_EFFORTS = {"minimal", "low", "medium", "high", "xhigh"}

    def __init__(
        self,
        settings: RagSettings,
        *,
        client: Any | None = None,
        usage_callback: UsageCallback | None = None,
    ) -> None:
        if client is None:
            from openai import AsyncOpenAI

            kwargs = {"api_key": settings.openai_api_key}
            if settings.openai_base_url:
                kwargs["base_url"] = settings.openai_base_url
            client = AsyncOpenAI(**kwargs)
        self._client = client
        self._model = settings.self_service_llm_model
        self._timeout = settings.llm_request_timeout_seconds
        self._usage_callback = usage_callback

    async def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        reasoning_effort: str | None = None,
        usage_stage: str = "answer_generation",
    ) -> str:
        """Generate a response while safely ignoring unsupported reasoning effort."""

        kwargs = {
            "model": self._model,
            "instructions": system_prompt,
            "input": user_prompt,
            "timeout": self._timeout,
        }
        if reasoning_effort in self._SUPPORTED_REASONING_EFFORTS:
            kwargs["reasoning"] = {"effort": reasoning_effort}
        response = await self._client.responses.create(**kwargs)
        emit_usage(
            self._usage_callback,
            usage_from_response(
                response,
                stage=usage_stage,
                fallback_model=self._model,
            ),
        )
        output_text = getattr(response, "output_text", None)
        if output_text:
            return str(output_text)
        return _extract_response_text(response)


class GroundingEvaluationError(RuntimeError):
    """Safe terminal error for an unusable structured grounding response."""

    def __init__(
        self,
        message: str,
        *,
        response_status: str | None = None,
        response_reason: str | None = None,
    ) -> None:
        super().__init__(message)
        self.response_status = response_status
        self.response_reason = response_reason


class GroundingEvaluationRefusal(GroundingEvaluationError):
    """Terminal policy refusal that must not be retried."""


class OpenAIGroundingEvaluatorAdapter:
    """Structured claim-evidence evaluator with bounded provider recovery."""

    _SUPPORTED_REASONING_EFFORTS = {"minimal", "low", "medium", "high", "xhigh"}

    _SYSTEM_PROMPT = (
        "You are a strict claim-evidence grounding evaluator. Evaluate only whether each factual "
        "claim in the draft is supported by the supplied cited chunk content. Do not use "
        "external knowledge. A claim is supported only when its cited chunk directly entails it. "
        "A single exact citation marker after a coherent multiline list or project tree may cover "
        "that entire immediately preceding group; do not report each line as missing a citation "
        "solely because the marker is at the end of the group. The cited chunk must still support "
        "every factual item in that group. Report genuinely missing citations, unsupported claims, "
        "and hallucination risk. Independently classify question_coverage as full when the grounded "
        "draft answers the complete user question, partial when it safely answers only part, or none "
        "when it provides no useful answer. Choose accept only when every material factual claim is "
        "directly supported; otherwise choose no_answer or revise."
    )

    def __init__(
        self,
        settings: RagSettings,
        client: Any | None = None,
        usage_callback: UsageCallback | None = None,
    ) -> None:
        if client is None:
            from openai import AsyncOpenAI

            kwargs = {"api_key": settings.openai_api_key}
            if settings.openai_base_url:
                kwargs["base_url"] = settings.openai_base_url
            client = AsyncOpenAI(**kwargs)
        self._client = client
        self._model = settings.self_service_llm_model
        reasoning_effort = settings.grounding_reasoning_effort.strip().casefold()
        if reasoning_effort not in self._SUPPORTED_REASONING_EFFORTS:
            raise ValueError(
                f"Unsupported grounding reasoning effort: {settings.grounding_reasoning_effort}"
            )
        if not 0 <= settings.grounding_max_retries <= 3:
            raise ValueError("grounding_max_retries must be between zero and three.")
        self._reasoning_effort = reasoning_effort
        self._max_retries = settings.grounding_max_retries
        self._timeout = settings.llm_request_timeout_seconds
        self._usage_callback = usage_callback

    async def evaluate(
        self,
        *,
        question: str,
        draft_answer: str,
        cited_chunks: list[RetrievedChunk],
    ) -> ReflectionResult:
        """Evaluate bounded cited evidence and parse a strict decision schema."""

        evidence = [
            {
                "chunk_id": chunk.evidence_id or chunk.chunk_id,
                "document_name": chunk.document_name,
                "location": chunk.display_location,
                "collection": chunk.collection_name,
                "source_excerpt": chunk.source_excerpt[:1500],
                "chunk_text": chunk.text[:4000],
            }
            for chunk in cited_chunks
        ]
        request_input = json.dumps(
            {
                "question": question,
                "draft_answer": draft_answer,
                "cited_evidence": evidence,
            },
            ensure_ascii=False,
        )
        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                response = await self._client.responses.parse(
                    model=self._model,
                    reasoning={"effort": self._reasoning_effort},
                    instructions=self._SYSTEM_PROMPT,
                    input=request_input,
                    text_format=ReflectionResult,
                    max_output_tokens=6000 * (attempt + 1),
                    timeout=self._timeout,
                )
                emit_usage(
                    self._usage_callback,
                    usage_from_response(
                        response,
                        stage="retrieval_grounding",
                        fallback_model=self._model,
                        metadata={"attempt": attempt + 1},
                    ),
                )
                parsed = getattr(response, "output_parsed", None)
                if parsed is None:
                    refusal = _extract_openai_refusal(response)
                    status = _safe_response_field(response, "status")
                    reason = _safe_incomplete_reason(response)
                    if refusal:
                        raise GroundingEvaluationRefusal(
                            "OpenAI refused the grounding evaluation.",
                            response_status=status,
                            response_reason="refusal",
                        )
                    raise GroundingEvaluationError(
                        "OpenAI returned no parsed grounding evaluation.",
                        response_status=status,
                        response_reason=reason or "missing_parsed_output",
                    )
                return (
                    parsed
                    if isinstance(parsed, ReflectionResult)
                    else ReflectionResult.model_validate(parsed)
                )
            except asyncio.CancelledError:
                raise
            except GroundingEvaluationRefusal:
                raise
            except Exception as exc:
                last_error = exc
                if attempt >= self._max_retries:
                    break
                logger.warning(
                    "Retrying incomplete grounding evaluation.",
                    extra={
                        "event": "grounding_evaluation_retry",
                        "attempt": attempt + 1,
                        "max_attempts": self._max_retries + 1,
                        "exception_type": type(exc).__name__,
                        "provider_status": getattr(exc, "response_status", None),
                        "provider_reason": getattr(exc, "response_reason", None),
                    },
                )
        if isinstance(last_error, GroundingEvaluationError):
            raise last_error
        raise GroundingEvaluationError(
            f"Grounding evaluation failed after {self._max_retries + 1} attempt(s)."
        ) from last_error


def _safe_response_field(response: object, name: str) -> str | None:
    """Return one bounded provider status field without response content."""

    value = getattr(response, name, None)
    return str(value)[:100] if value is not None else None


def _safe_incomplete_reason(response: object) -> str | None:
    """Return the provider's bounded incomplete reason, when present."""

    details = getattr(response, "incomplete_details", None)
    reason = getattr(details, "reason", None)
    return str(reason)[:100] if reason is not None else None


def _extract_openai_refusal(response: object) -> str:
    """Detect a refusal without logging or surfacing provider output text."""

    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            refusal = getattr(content, "refusal", None)
            if refusal:
                return "refusal"
    return ""


class QdrantHybridRetrievalAdapter:
    """Qdrant retrieval adapter with mandatory user_id payload filtering."""

    def __init__(
        self,
        settings: RagSettings,
        *,
        client: Any | None = None,
        sparse_encoder: SparseEncoder | None = None,
    ) -> None:
        if client is None:
            from qdrant_client import AsyncQdrantClient

            client = AsyncQdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key,
            )
        self._client = client
        self._dense_vector_name = settings.dense_vector_name
        self._sparse_vector_name = settings.sparse_vector_name
        self._sparse_encoder = (
            sparse_encoder
            if sparse_encoder is not None
            else create_sparse_encoder(
                settings.sparse_encoder_provider,
                settings.sparse_encoder_model,
                settings.sparse_encoder_cache_dir,
            )
        )
        self._allowed_document_ids = settings.allowed_document_ids

    async def retrieve(
        self,
        *,
        query: str,
        dense_vector: list[float],
        plan: RetrievalPlan,
        user_id: str,
    ) -> RetrievalAdapterResult:
        """Run scoped collection queries concurrently with mandatory owner filters."""

        from qdrant_client import models

        if self._allowed_document_ids == ():
            return RetrievalAdapterResult(
                attempted_collections=list(plan.collections),
                successful_collections=list(plan.collections),
            )
        owner_filter = (
            build_user_documents_filter(user_id, self._allowed_document_ids)
            if self._allowed_document_ids is not None
            else build_user_filter(user_id)
        )
        user_filter = models.Filter(
            must=[
                *(owner_filter.must or []),
                models.FieldCondition(
                    key="sparse_encoder_version",
                    match=models.MatchValue(value=self._sparse_encoder.version),
                ),
            ]
        )
        sparse_vector = await asyncio.to_thread(
            self._sparse_encoder.encode_query,
            query,
        )
        chunks_by_collection: dict[CollectionName, list[RetrievedChunk]] = {}
        errors: list[NodeError] = []
        successful_collections: list[CollectionName] = []

        async def query_collection(collection: CollectionName):
            response = await self._client.query_points(
                collection_name=collection,
                prefetch=[
                    models.Prefetch(
                        query=dense_vector,
                        using=self._dense_vector_name,
                        filter=user_filter,
                        limit=plan.prefetch_k,
                    ),
                    models.Prefetch(
                        query=sparse_vector,
                        using=self._sparse_vector_name,
                        filter=user_filter,
                        limit=plan.prefetch_k,
                    ),
                ],
                query=models.RrfQuery(
                    rrf=models.Rrf(
                        weights=_normalized_rrf_weights(
                            plan.dense_weight,
                            plan.sparse_weight,
                        ),
                    )
                ),
                limit=plan.collection_k,
                with_payload=True,
                with_vectors=False,
            )
            return collection, response.points

        outcomes = await asyncio.gather(
            *(query_collection(collection) for collection in plan.collections),
            return_exceptions=True,
        )
        for collection, outcome in zip(plan.collections, outcomes, strict=True):
            if isinstance(outcome, BaseException):
                logger.warning(
                    "Hybrid retrieval collection query failed.",
                    extra={
                        "event": "hybrid_retrieval_collection_failed",
                        "exception_type": type(outcome).__name__,
                        "collection": collection,
                    },
                )
                errors.append(
                    NodeError(
                        node=f"hybrid_retrieval:{collection}",
                        message="Retrieval failed.",
                    )
                )
                continue
            returned_collection, points = outcome
            successful_collections.append(returned_collection)
            owned_chunks: list[RetrievedChunk] = []
            for hit_index, hit in enumerate(points):
                try:
                    chunk = _qdrant_hit_to_chunk(returned_collection, hit)
                except Exception as exc:
                    logger.warning(
                        "Malformed retrieval payload was dropped.",
                        extra={
                            "event": "hybrid_retrieval_payload_dropped",
                            "exception_type": type(exc).__name__,
                            "collection": returned_collection,
                            "result_index": hit_index,
                        },
                    )
                    errors.append(
                        NodeError(
                            node=f"hybrid_retrieval:{returned_collection}:payload",
                            message=(
                                "Dropped malformed retrieval payload at result "
                                f"index {hit_index}."
                            ),
                        )
                    )
                    continue
                if chunk.user_id != user_id:
                    continue
                if (
                    self._allowed_document_ids is not None
                    and chunk.document_id not in self._allowed_document_ids
                ):
                    continue
                if chunk.sparse_encoder_version != self._sparse_encoder.version:
                    continue
                owned_chunks.append(chunk)
            deduplicated: dict[tuple[str, str], RetrievedChunk] = {}
            for chunk in owned_chunks:
                key = (chunk.collection_name, chunk.chunk_id)
                current = deduplicated.get(key)
                if current is None or _retrieved_chunk_representative_key(
                    chunk
                ) < _retrieved_chunk_representative_key(current):
                    deduplicated[key] = chunk
            ranked = sorted(
                deduplicated.values(),
                key=lambda item: (
                    -item.retrieval_score,
                    item.document_id,
                    item.chunk_id,
                ),
            )
            chunks_by_collection[returned_collection] = ranked[: plan.collection_k]

        all_candidates = [
            chunk
            for collection in plan.collections
            for chunk in chunks_by_collection.get(collection, [])
        ]
        balanced = _balanced_take(
            all_candidates,
            plan.collections,
            plan.candidate_k,
        )
        return RetrievalAdapterResult(
            chunks=balanced,
            errors=sorted(errors, key=lambda item: (item.node, item.message)),
            attempted_collections=list(plan.collections),
            successful_collections=sorted(set(successful_collections)),
        )


def create_fake_adapters(settings: RagSettings | None = None) -> RagAdapters:
    """Create deterministic adapters for offline graph execution."""

    settings = settings or RagSettings.from_env()
    vector_size = 32
    return RagAdapters(
        llm=FakeLlmAdapter(),
        embedding=FakeEmbeddingAdapter(vector_size=vector_size),
        retrieval=FakeHybridRetrievalAdapter(),
        reranker=(
            HeuristicRerankerAdapter()
            if settings.effective_reranker_provider == "heuristic"
            else NoOpRerankerAdapter()
        ),
        grounding=FakeGroundingEvaluatorAdapter(),
        query_rewriter=None,
    )


def create_openai_qdrant_adapters(
    settings: RagSettings | None = None,
    *,
    usage_callback: UsageCallback | None = None,
    openai_client: Any | None = None,
    qdrant_client: Any | None = None,
) -> RagAdapters:
    """Create real OpenAI and Qdrant adapters for connected experiments."""

    settings = settings or RagSettings.from_env()
    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY must be set before using real OpenAI adapters.")

    reranker_provider = settings.effective_reranker_provider
    if reranker_provider not in {"noop", "heuristic", "openai"}:
        raise ValueError(
            f"Unsupported reranker provider: {settings.reranker_provider!r}."
        )
    sparse_encoder = create_sparse_encoder(
        settings.sparse_encoder_provider,
        settings.sparse_encoder_model,
        settings.sparse_encoder_cache_dir,
    )
    openai_reranker_type = None
    if reranker_provider == "openai":
        from .rerankers import (
            OpenAIRerankerAdapter,
            validate_openai_reranker_settings,
        )

        validate_openai_reranker_settings(settings)
        openai_reranker_type = OpenAIRerankerAdapter

    owned_clients: list[Any] = []
    try:
        if openai_client is None:
            from openai import AsyncOpenAI

            openai_kwargs = {"api_key": settings.openai_api_key}
            if settings.openai_base_url:
                openai_kwargs["base_url"] = settings.openai_base_url
            openai_client = AsyncOpenAI(**openai_kwargs)
            owned_clients.append(openai_client)
        if qdrant_client is None:
            from qdrant_client import AsyncQdrantClient

            qdrant_client = AsyncQdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key,
            )
            owned_clients.append(qdrant_client)
        reranker: RerankerAdapter
        if reranker_provider == "noop":
            reranker = NoOpRerankerAdapter()
        elif reranker_provider == "heuristic":
            reranker = HeuristicRerankerAdapter()
        else:
            assert openai_reranker_type is not None
            reranker = openai_reranker_type(
                settings,
                client=openai_client,
                usage_callback=usage_callback,
            )
        return RagAdapters(
            llm=OpenAILlmAdapter(
                settings,
                client=openai_client,
                usage_callback=usage_callback,
            ),
            embedding=OpenAIEmbeddingAdapter(
                settings,
                client=openai_client,
                usage_callback=usage_callback,
            ),
            retrieval=QdrantHybridRetrievalAdapter(
                settings,
                client=qdrant_client,
                sparse_encoder=sparse_encoder,
            ),
            reranker=reranker,
            grounding=OpenAIGroundingEvaluatorAdapter(
                settings,
                client=openai_client,
                usage_callback=usage_callback,
            ),
            query_rewriter=OpenAIQueryRewriterAdapter(
                settings,
                client=openai_client,
                usage_callback=usage_callback,
            ),
            _owned_clients=tuple(owned_clients),
        )
    except Exception:
        _close_owned_clients_best_effort(owned_clients)
        raise


async def _await_client_closures(awaitables: list[Any]) -> None:
    """Await client cleanup without allowing one failure to cancel another."""

    await asyncio.gather(*awaitables, return_exceptions=True)


async def _await_one_client_closure(awaitable: Any) -> None:
    """Await one scheduled close operation without surfacing cleanup errors."""

    try:
        await awaitable
    except Exception:
        return


def _close_owned_clients_best_effort(clients: list[Any]) -> None:
    """Close now outside an event loop or schedule cleanup inside one."""

    awaitables: list[Any] = []
    seen: set[int] = set()
    for client in clients:
        identity = id(client)
        if identity in seen:
            continue
        seen.add(identity)
        close = getattr(client, "close", None)
        if close is None:
            close = getattr(client, "aclose", None)
        if close is None:
            continue
        try:
            result = close()
        except Exception:
            continue
        if inspect.isawaitable(result):
            awaitables.append(result)
    if not awaitables:
        return

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(_await_client_closures(awaitables))
    else:
        for awaitable in awaitables:
            task = loop.create_task(_await_one_client_closure(awaitable))
            _BACKGROUND_CLEANUP_TASKS.add(task)
            task.add_done_callback(_BACKGROUND_CLEANUP_TASKS.discard)


def _qdrant_hit_to_chunk(
    collection_name: CollectionName,
    hit: object,
) -> RetrievedChunk:
    """Validate one server payload without coercing malformed metadata."""

    raw_payload = getattr(hit, "payload", None)
    if not isinstance(raw_payload, Mapping):
        raise ValueError("Retrieval payload must be a mapping.")
    payload = dict(raw_payload)
    document_type = _required_payload_text(payload, "document_type")
    if document_type not in {"pdf", "docx", "pptx"}:
        raise ValueError("Retrieval payload has an invalid document type.")
    page_number = _optional_positive_payload_int(payload, "page_number")
    slide_number = _optional_positive_payload_int(payload, "slide_number")
    if document_type == "pptx":
        if slide_number is None or page_number is not None:
            raise ValueError("PPTX retrieval payload has an invalid location.")
    elif page_number is None or slide_number is not None:
        raise ValueError("Document retrieval payload has an invalid location.")

    score_value = getattr(hit, "score", None)
    if isinstance(score_value, bool) or not isinstance(score_value, (int, float)):
        raise ValueError("Retrieval payload has an invalid score.")
    score = float(score_value)
    if not math.isfinite(score):
        raise ValueError("Retrieval payload has a non-finite score.")

    collection_type = _optional_payload_text(payload, "collection_type")
    created_at = _optional_payload_text(payload, "created_at")
    return RetrievedChunk(
        user_id=_required_payload_text(payload, "user_id"),
        document_id=_required_payload_text(payload, "document_id"),
        document_name=_required_payload_text(payload, "document_name"),
        document_type=document_type,
        page_number=page_number,
        slide_number=slide_number,
        chunk_id=_required_payload_text(payload, "chunk_id"),
        collection_name=collection_name,
        collection_type=collection_type,
        sparse_encoder_version=_required_payload_text(
            payload,
            "sparse_encoder_version",
        ),
        source_pipeline=_required_payload_text(payload, "source_pipeline"),
        source_excerpt=_required_payload_text(payload, "source_excerpt"),
        text=_required_payload_text(payload, "text"),
        retrieval_score=min(1.0, max(0.0, score)),
        created_at=created_at,
    )


def _required_payload_text(payload: Mapping[str, Any], field_name: str) -> str:
    """Return one non-empty string field or raise a content-free error."""

    value = payload.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Retrieval payload field {field_name!r} is invalid.")
    return value


def _optional_payload_text(
    payload: Mapping[str, Any],
    field_name: str,
) -> str | None:
    """Validate one optional string without converting arbitrary values."""

    value = payload.get(field_name)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Retrieval payload field {field_name!r} is invalid.")
    return value


def _optional_positive_payload_int(
    payload: Mapping[str, Any],
    field_name: str,
) -> int | None:
    """Validate one optional one-based page or slide number."""

    value = payload.get(field_name)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"Retrieval payload field {field_name!r} is invalid.")
    return value


def _extract_response_text(response: object) -> str:
    output = getattr(response, "output", []) or []
    parts: list[str] = []
    for item in output:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if text:
                parts.append(str(text))
    return "\n".join(parts).strip()


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[^\W_]+", text.casefold(), flags=re.UNICODE))


_RERANK_STOPWORDS = {
    "a",
    "about",
    "an",
    "and",
    "are",
    "can",
    "could",
    "give",
    "how",
    "is",
    "me",
    "of",
    "or",
    "please",
    "the",
    "to",
    "what",
    "which",
    "alabilir",
    "bir",
    "bilgi",
    "bu",
    "hakkında",
    "ile",
    "misin",
    "miyim",
    "nedir",
    "ve",
    "veya",
    "verebilir",
}


def _meaningful_tokens(text: str) -> set[str]:
    return {token for token in _tokenize(text) if token not in _RERANK_STOPWORDS}


def _normalized_rrf_weights(dense_weight: float, sparse_weight: float) -> list[float]:
    """Preserve weight ratios while retaining Qdrant's two-source RRF score scale."""

    dense = max(0.0, dense_weight)
    sparse = max(0.0, sparse_weight)
    total = dense + sparse
    if total == 0:
        return [1.0, 1.0]
    scale = 2.0 / total
    return [dense * scale, sparse * scale]
