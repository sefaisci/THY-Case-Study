"""Adapter interfaces and local implementations for the RAG graph."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from model.vector_store import (
    StableHashSparseEncoder,
    build_user_documents_filter,
    build_user_filter,
)
from model.usage import UsageCallback, emit_usage, usage_from_response

from .schemas import (
    ClaimEvaluation,
    CollectionName,
    NodeError,
    ReflectionResult,
    RetrievalAdapterResult,
    RetrievalPlan,
    RetrievalQueryRewrite,
    RetrievedChunk,
)
from .settings import RagSettings


class LlmAdapter(Protocol):
    """Text generation boundary used by graph nodes."""

    async def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        reasoning_effort: str | None = None,
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
    """Boundary for producing a standalone bilingual retrieval query."""

    async def rewrite(self, question: str) -> RetrievalQueryRewrite:
        """Return four faithful generated retrieval forms."""


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
    ) -> list[RetrievedChunk]:
        """Return chunks ordered by evidence quality."""


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
    ) -> str:
        """Produce a deterministic answer from provided evidence lines."""

        del reasoning_effort
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
        evidence_lines = [line.strip() for line in user_prompt.splitlines() if re.match(r"^\[[^\]]+\]$", line.strip())]
        if not evidence_lines:
            return "The uploaded documents do not contain enough evidence to answer this question."
        chunk_marker = evidence_lines[0]
        text_match = re.search(r"Source excerpt:\s*(.+)", user_prompt)
        first_evidence = text_match.group(1).strip() if text_match else "the evidence supports the answer."
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
        cited_ids = [chunk.chunk_id for chunk in cited_chunks]
        grounded = bool(draft_answer.strip() and cited_ids)
        return ReflectionResult(
            is_grounded=grounded,
            hallucination_risk="low" if grounded else "high",
            decision="accept" if grounded else "no_answer",
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
        return RetrievalAdapterResult(chunks=scored[: plan.top_k])


class NoOpRerankerAdapter:
    """Reranker that preserves retrieval order."""

    async def rerank(
        self,
        *,
        question: str,
        chunks: list[RetrievedChunk],
        limit: int,
    ) -> list[RetrievedChunk]:
        """Return the first candidates without changing order."""

        del question
        return chunks[:limit]


class HeuristicRerankerAdapter:
    """Lightweight token-overlap reranker for offline experiments."""

    async def rerank(
        self,
        *,
        question: str,
        chunks: list[RetrievedChunk],
        limit: int,
    ) -> list[RetrievedChunk]:
        """Reorder chunks by token overlap and retrieval score."""

        question_terms = _meaningful_tokens(question)
        reranked = []
        for chunk in chunks:
            chunk_terms = _meaningful_tokens(f"{chunk.text} {chunk.source_excerpt}")
            overlap = len(question_terms & chunk_terms)
            # Preserve semantic retrieval as the primary signal. Lexical overlap is
            # only a small tie-breaker because it is not reliable cross-lingually.
            rerank_score = chunk.retrieval_score + (min(overlap, 3) * 0.005)
            reranked.append(chunk.model_copy(update={"rerank_score": min(1.0, rerank_score)}))
        reranked.sort(
            key=lambda item: (
                -item.fusion_score,
                -item.effective_score,
                item.collection_name,
                item.document_id,
                item.chunk_id,
            )
        )
        return reranked[:limit]


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
    """Responses API adapter for four distinct generated retrieval queries."""

    _SYSTEM_PROMPT = (
        "Produce exactly four distinct retrieval forms for the supplied question; do "
        "not answer it. standalone_query must resolve references while preserving the "
        "user's language and intent. english_query must be a faithful English form. "
        "keyword_query must preserve entities, filenames, symbols, numbers, and useful "
        "technical terms for lexical retrieval. source_style_query must resemble a short "
        "passage likely to occur in a relevant source, without asserting or introducing "
        "facts absent from the question. All four values must be meaningfully distinct."
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
        """Return four structured generated query variants."""

        response = await self._client.responses.parse(
            model=self._model,
            reasoning={"effort": self._reasoning_effort},
            instructions=self._SYSTEM_PROMPT,
            input=json.dumps({"question": question}, ensure_ascii=False),
            text_format=RetrievalQueryRewrite,
            max_output_tokens=1_000,
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
                stage="answer_generation",
                fallback_model=self._model,
            ),
        )
        output_text = getattr(response, "output_text", None)
        if output_text:
            return str(output_text)
        return _extract_response_text(response)


class OpenAIGroundingEvaluatorAdapter:
    """Structured GPT-5.5 claim-evidence evaluator with no external knowledge."""

    _SYSTEM_PROMPT = (
        "You are a strict claim-evidence grounding evaluator. Evaluate only whether each factual "
        "claim in the draft is supported by the supplied cited chunk content. Do not use "
        "external knowledge. A claim is supported only when its cited chunk directly entails it. "
        "A single exact citation marker after a coherent multiline list or project tree may cover "
        "that entire immediately preceding group; do not report each line as missing a citation "
        "solely because the marker is at the end of the group. The cited chunk must still support "
        "every factual item in that group. Report genuinely missing citations, unsupported claims, "
        "and hallucination risk. Choose accept only when every material factual claim is directly "
        "supported; otherwise choose no_answer or revise."
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
        self._reasoning_effort = settings.self_service_reasoning_effort
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
                "chunk_id": chunk.chunk_id,
                "document_name": chunk.document_name,
                "location": chunk.display_location,
                "collection": chunk.collection_name,
                "source_excerpt": chunk.source_excerpt[:1500],
                "chunk_text": chunk.text[:4000],
            }
            for chunk in cited_chunks
        ]
        response = await self._client.responses.parse(
            model=self._model,
            reasoning={"effort": self._reasoning_effort},
            instructions=self._SYSTEM_PROMPT,
            input=json.dumps(
                {
                    "question": question,
                    "draft_answer": draft_answer,
                    "cited_evidence": evidence,
                },
                ensure_ascii=False,
            ),
            text_format=ReflectionResult,
            max_output_tokens=4000,
            timeout=self._timeout,
        )
        emit_usage(
            self._usage_callback,
            usage_from_response(
                response,
                stage="retrieval_grounding",
                fallback_model=self._model,
            ),
        )
        parsed = getattr(response, "output_parsed", None)
        if parsed is None:
            raise RuntimeError("OpenAI returned no parsed grounding evaluation.")
        return parsed if isinstance(parsed, ReflectionResult) else ReflectionResult.model_validate(parsed)


class QdrantHybridRetrievalAdapter:
    """Qdrant retrieval adapter with mandatory user_id payload filtering."""

    def __init__(
        self,
        settings: RagSettings,
        *,
        client: Any | None = None,
        sparse_encoder: StableHashSparseEncoder | None = None,
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
        self._sparse_encoder = sparse_encoder or StableHashSparseEncoder()
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
            return RetrievalAdapterResult()
        user_filter = (
            build_user_documents_filter(user_id, self._allowed_document_ids)
            if self._allowed_document_ids is not None
            else build_user_filter(user_id)
        )
        sparse_vector = self._sparse_encoder.encode(query)
        chunks: list[RetrievedChunk] = []
        errors: list[NodeError] = []
        collection_order = {name: index for index, name in enumerate(plan.collections)}

        async def query_collection(collection: CollectionName):
            response = await self._client.query_points(
                collection_name=collection,
                prefetch=[
                    models.Prefetch(
                        query=dense_vector,
                        using=self._dense_vector_name,
                        filter=user_filter,
                        limit=plan.top_k,
                    ),
                    models.Prefetch(
                        query=sparse_vector,
                        using=self._sparse_vector_name,
                        filter=user_filter,
                        limit=plan.top_k,
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
                limit=plan.top_k,
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
                errors.append(
                    NodeError(
                        node=f"hybrid_retrieval:{collection}",
                        message=str(outcome),
                    )
                )
                continue
            returned_collection, points = outcome
            chunks.extend(
                chunk
                for chunk in _qdrant_hits_to_chunks(returned_collection, points)
                if chunk.user_id == user_id
                and (
                    self._allowed_document_ids is None
                    or chunk.document_id in self._allowed_document_ids
                )
            )
        deduplicated: dict[tuple[str, str], RetrievedChunk] = {}
        for chunk in chunks:
            key = (chunk.collection_name, chunk.chunk_id)
            current = deduplicated.get(key)
            if current is None or chunk.retrieval_score > current.retrieval_score:
                deduplicated[key] = chunk
        merged = list(deduplicated.values())
        merged.sort(
            key=lambda item: (
                -item.retrieval_score,
                collection_order.get(item.collection_name, len(collection_order)),
                item.document_id,
                item.chunk_id,
            )
        )
        return RetrievalAdapterResult(
            chunks=merged[: plan.top_k],
            errors=sorted(errors, key=lambda item: (item.node, item.message)),
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
            if settings.reranker_provider == "heuristic"
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
    owned_clients: list[Any] = []
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
    if settings.enable_reranker and settings.reranker_provider == "heuristic":
        reranker = HeuristicRerankerAdapter()
    else:
        reranker = NoOpRerankerAdapter()
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
        retrieval=QdrantHybridRetrievalAdapter(settings, client=qdrant_client),
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


def _qdrant_hits_to_chunks(
    collection_name: CollectionName, hits: list[object]
) -> list[RetrievedChunk]:
    chunks = []
    for hit in hits:
        payload = dict(getattr(hit, "payload", {}) or {})
        chunks.append(
            RetrievedChunk(
                user_id=str(payload.get("user_id", "")),
                document_id=str(payload.get("document_id", "")),
                document_name=str(payload.get("document_name", "")),
                document_type=str(payload.get("document_type", "")),
                page_number=payload.get("page_number"),
                slide_number=payload.get("slide_number"),
                chunk_id=str(payload.get("chunk_id", getattr(hit, "id", ""))),
                collection_name=collection_name,
                collection_type=payload.get("collection_type"),
                source_pipeline=str(payload.get("source_pipeline", "")),
                source_excerpt=str(payload.get("source_excerpt", "")),
                text=str(payload.get("text", payload.get("content", ""))),
                retrieval_score=min(
                    1.0,
                    max(0.0, float(getattr(hit, "score", 0.0) or 0.0)),
                ),
                created_at=payload.get("created_at"),
            )
        )
    return chunks


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
